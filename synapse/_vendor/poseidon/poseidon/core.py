"""Poseidon's small, trainable shared scene/control core.

This network predicts scene attributes, simulation actions and state changes.
It is not a language model or a pixel/video/mesh decoder. Rendering and fluent
conversation are separate, explicitly identified parts of the runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SCHEMA = "poseidon-tidal-core-v1"
FEATURE_SCHEMA = "blake2b-lemma-words-bigrams-l2-v2"
# Deterministic, closed-vocabulary lemma table. Only inflections/synonyms of the
# controlled scene grammar are folded; all other words hash unchanged.
_LEMMAS = {
    "cubes": "cube", "spheres": "sphere", "balls": "ball", "pyramids": "pyramid",
    "cylinders": "cylinder", "blocks": "block", "objects": "object", "shapes": "shape",
    "orbiting": "orbit", "orbits": "orbit", "orbited": "orbit", "orbital": "orbit",
    "bouncing": "bounce", "bounces": "bounce", "bounced": "bounce", "bouncy": "bounce",
    "spinning": "spin", "spins": "spin", "spun": "spin",
    "rotating": "rotate", "rotates": "rotate", "rotated": "rotate",
    "stationary": "still", "static": "still", "motionless": "still", "remaining": "remain",
    "moving": "move", "moves": "move", "tinier": "tiny", "bigger": "big", "larger": "large",
    "smaller": "small", "coloured": "colored", "colour": "color",
}


def normalize_words(prompt: str) -> list[str]:
    words = re.findall(r"[\w]+", str(prompt).casefold(), flags=re.UNICODE)[:256]
    return [_LEMMAS.get(word, word) for word in words]
SCENE_LABELS = {
    "shape": ("cube", "sphere", "pyramid", "cylinder"),
    "color": ("red", "blue", "green", "yellow", "purple", "cyan"),
    "motion": ("still", "orbit", "bounce", "spin"),
    "count": (1, 2, 3),
    "scale": ("small", "medium", "large"),
}
ACTION_LABELS = ("rest", "forage", "drink", "shelter", "explore", "flee")


@dataclass(frozen=True)
class CoreConfig:
    hash_buckets: int = 512
    hidden_size: int = 128
    experts: int = 4
    recurrent_steps: int = 2
    observation_size: int = 16
    action_count: int = 6
    feature_schema: str = FEATURE_SCHEMA

    def __post_init__(self):
        if not (32 <= self.hash_buckets <= 8192 and 16 <= self.hidden_size <= 1024):
            raise ValueError("unsupported hash or hidden dimensions")
        if not (1 <= self.experts <= 16 and 1 <= self.recurrent_steps <= 8):
            raise ValueError("unsupported expert or recurrent count")
        if self.observation_size != 16 or self.action_count != 6:
            raise ValueError("core contract requires 16 observations and six actions")
        if self.feature_schema != FEATURE_SCHEMA:
            raise ValueError("unsupported feature schema")


def config_hash(config: dict | CoreConfig) -> str:
    value = asdict(config) if isinstance(config, CoreConfig) else config
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=32768)
def _feature_index(token: str, buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"poseidon").digest()
    number = int.from_bytes(digest, "little")
    return number % buckets, 1.0 if (number >> 63) == 0 else -1.0


def hash_features(prompts: list[str], buckets: int = 512) -> torch.Tensor:
    """Stable, signed lemma word/bigram features, independent of Python's hash seed."""
    output = torch.zeros((len(prompts), buckets), dtype=torch.float32)
    for row, prompt in enumerate(prompts):
        words = normalize_words(prompt)
        features = [("w:" + word, 1.0) for word in words]
        features += [("b:" + left + " " + right, 0.7) for left, right in zip(words, words[1:])]
        for token, weight in features:
            index, sign = _feature_index(token, buckets)
            output[row, index] += sign * weight
    return F.normalize(output, p=2, dim=-1)


class TidalCore(nn.Module):
    """A recurrent mixture shared by language-conditioned scenes and control.

    Routing is learned for every example and recurrent step. The optional memory
    argument allows experiments with carried hidden state; the default trained
    protocol resets it for each example, so persistent memory is not claimed.
    """

    def __init__(self, config: CoreConfig | None = None):
        super().__init__()
        self.config = config or CoreConfig()
        cfg = self.config
        h = cfg.hidden_size
        self.text_projection = nn.Linear(cfg.hash_buckets, h, bias=False)
        self.observation_projection = nn.Linear(cfg.observation_size, h)
        self.domain = nn.Embedding(2, h)
        self.input_norm = nn.LayerNorm(h)
        self.router = nn.Linear(h * 2, cfg.experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, h)) for _ in range(cfg.experts))
        self.gate = nn.Linear(h * 2, h)
        self.state_norm = nn.LayerNorm(h)
        self.scene_heads = nn.ModuleDict({name: nn.Linear(h, len(values)) for name, values in SCENE_LABELS.items()})
        self.action_head = nn.Linear(h, cfg.action_count)
        self.action_embedding = nn.Embedding(cfg.action_count, 24)
        self.dynamics_head = nn.Sequential(nn.Linear(h + 24 + cfg.observation_size, h), nn.SiLU(), nn.Linear(h, cfg.observation_size))

    def forward(self, prompts: list[str] | torch.Tensor, observations: torch.Tensor | None = None,
                actions: torch.Tensor | None = None, memory: torch.Tensor | None = None) -> dict[str, Any]:
        device = self.text_projection.weight.device
        features = hash_features(prompts, self.config.hash_buckets).to(device) if isinstance(prompts, list) else prompts.to(device)
        if features.ndim != 2 or features.shape[1] != self.config.hash_buckets:
            raise ValueError("prompt features have incorrect dimensions")
        batch = features.shape[0]
        if observations is None:
            observations = torch.zeros((batch, 16), device=device)
        else:
            observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        if observations.shape != (batch, 16):
            raise ValueError("observations must have shape [batch,16]")
        if not torch.isfinite(observations).all() or not torch.isfinite(features).all():
            raise ValueError("observations and features must be finite")
        domain = (features.abs().sum(-1) == 0).long()
        context = self.input_norm(self.text_projection(features) + self.observation_projection(observations) + self.domain(domain))
        state = torch.zeros_like(context) if memory is None else memory.to(device)
        if state.shape != context.shape or not torch.isfinite(state).all():
            raise ValueError("memory must be finite and match hidden-state dimensions")
        routes = []
        for _ in range(self.config.recurrent_steps):
            joined = torch.cat((context, state), dim=-1)
            weights = torch.softmax(self.router(joined), dim=-1)
            candidates = torch.stack([expert(joined) for expert in self.experts], dim=1)
            update = (candidates * weights.unsqueeze(-1)).sum(dim=1)
            gate = torch.sigmoid(self.gate(joined))
            state = self.state_norm((1 - gate) * state + gate * update + context)
            routes.append(weights)
        logits = self.action_head(state)
        if actions is None:
            actions = logits.argmax(-1)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        if actions.shape != (batch,) or bool(((actions < 0) | (actions >= 6)).any()):
            raise ValueError("actions must be a batch of indices in [0,5]")
        delta = self.dynamics_head(torch.cat((state, self.action_embedding(actions), observations), dim=-1))
        return {"scene": {name: head(state) for name, head in self.scene_heads.items()},
                "action": logits, "delta": delta, "memory": state,
                "routing": torch.stack(routes).mean(0)}


def save_checkpoint(path: str | Path, model: TidalCore, *, receipt: dict,
                    training: dict | None = None, calibration: dict | None = None) -> None:
    """Atomically publish a tensor/primitive-only checkpoint loadable safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": CHECKPOINT_SCHEMA, "config": asdict(model.config),
               "config_hash": config_hash(model.config), "state_dict": model.state_dict(),
               "receipt": receipt, "training": training or {}, "calibration": calibration or {}}
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> tuple[TidalCore, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported core checkpoint schema")
    if config_hash(payload.get("config", {})) != payload.get("config_hash"):
        raise ValueError("checkpoint config hash mismatch")
    model = TidalCore(CoreConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint has non-finite weights")
    temperatures = scene_temperatures(payload)
    if any(not (0.05 <= value <= 20.0) for value in temperatures.values()):
        raise ValueError("checkpoint calibration temperatures out of range")
    return model, payload


def scene_temperatures(payload: dict) -> dict[str, float]:
    """Per-head temperature-scaling factors (Guo et al. 2017); empty if uncalibrated."""
    values = (payload.get("calibration") or {}).get("scene_temperature") or {}
    if set(values) - set(SCENE_LABELS):
        raise ValueError("calibration names an unknown scene head")
    return {name: float(value) for name, value in values.items()}


ACTIVE_CORE_POINTER = "runs/active_core.json"
DEFAULT_CORE = "runs/tidal/core.pt"


def active_core_path(root: str | Path = ".") -> Path:
    """The explicitly promoted checkpoint, else the base Tidal checkpoint.

    Promotion is recorded in runs/active_core.json by a validated selection step
    (for example DAgger) so the served model is always auditable.
    """
    root = Path(root)
    pointer = root / ACTIVE_CORE_POINTER
    if pointer.exists():
        try:
            relative = json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        except (ValueError, KeyError, OSError, TypeError) as error:
            raise ValueError("runs/active_core.json is malformed") from error
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents:
            raise ValueError("active core pointer must stay inside the project root")
        if candidate.exists():
            return candidate
    return root / DEFAULT_CORE


class CoreRuntime:
    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint)
        self.model, self.payload = load_checkpoint(self.path)
        self.temperatures = scene_temperatures(self.payload)
        self.model.eval()

    def calibrated(self, name: str, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperatures.get(name, 1.0)

    @torch.inference_mode()
    def scene(self, prompt: str) -> dict:
        result = self.model([prompt])
        scene, confidences = {}, {}
        for name, logits in result["scene"].items():
            probabilities = self.calibrated(name, logits[0]).softmax(-1)
            index = int(probabilities.argmax())
            scene[name] = SCENE_LABELS[name][index]
            confidences[name] = float(probabilities[index])
        note = ("Temperature-scaled on the dev split (in-distribution calibration only); controlled scene vocabulary."
                if self.temperatures else "Uncalibrated classifier probabilities; controlled scene vocabulary.")
        scene.update(confidence=confidences, routing=result["routing"][0].tolist(),
                     planner="learned-tidal-core", confidence_note=note)
        return scene

    @torch.inference_mode()
    def act(self, observation) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(self.model([""], obs)["action"].argmax(-1)[0])

    @torch.inference_mode()
    def predict_transition(self, observation, action: int) -> list[float]:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        delta = self.model([""], obs, torch.tensor([action]))["delta"]
        return (obs + delta)[0].tolist()

    def diagnostics(self) -> dict:
        receipt = self.payload.get("receipt", {})
        return {"schema": CHECKPOINT_SCHEMA, "checkpoint": str(self.path.resolve()),
                "config": asdict(self.model.config), "config_hash": self.payload["config_hash"],
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "examples_seen": receipt.get("examples_seen", 0), "training_receipt": receipt,
                "calibration": self.payload.get("calibration") or {},
                "capabilities": ["controlled-scene-attributes", "six-action-simulation-policy", "action-conditioned-state-delta"],
                "persistent_memory_validated": False}
