"""Train an isolated dynamics-ensemble candidate and record a paired audit.

The pretrained core stays frozen. New seeded ensemble heads learn one-step
transition deltas from surprise replay mixed with scripted baseline anchors.
The audit checks reactive policy retention, same-transition dynamics, scene
retention and a disagreement/error diagnostic. It never activates a candidate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .adaptation import PromotionAuditor, PromotionCriteria, SurprisePrioritizedBuffer, model_digest
from .core import CoreRuntime
from .ensemble import WorldModelEnsemble
from .media import generate_scene_examples
from .world import TidePool, generate_transitions


def run_adaptation_cycle(
    base_core_path: str = "runs/tidal_dagger/core.pt",
    output_dir: str = "runs/beyond_adapted",
    collect_episodes: int = 15,
    adaptation_steps: int = 64,
    batch_size: int = 32,
    lr: float = 1e-4,
    base_seed: int = 7000,
) -> dict[str, Any]:
    """Produce a candidate and evidence receipt; preserve the active core."""
    for name, value, low, high in (("collect_episodes", collect_episodes, 1, 1000),
                                  ("adaptation_steps", adaptation_steps, 1, 100000),
                                  ("batch_size", batch_size, 2, 1024),
                                  ("base_seed", base_seed, 0, 1_000_000_000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if isinstance(lr, bool) or not math.isfinite(lr) or not 0 < lr <= 1:
        raise ValueError("lr must be finite and in (0, 1]")
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError("candidate output directory must be empty; existing experiments are preserved")
    runtime = CoreRuntime(base_core_path)
    core_model = runtime.model.eval()
    core_before = model_digest(core_model)
    # A newly initialized ensemble is not a promoted or pretrained world model.
    # Preserve process-global torch RNG even though legacy head constructors seed it.
    with torch.random.fork_rng(devices=[]):
        ensemble = WorldModelEnsemble(hidden_size=core_model.config.hidden_size, k=3, base_seed=base_seed + 100)
    incumbent_ensemble = copy.deepcopy(ensemble).eval()
    ensemble_before = model_digest(ensemble)
    buffer = SurprisePrioritizedBuffer(capacity=2000, seed=base_seed + 200)
    collection_seeds = [base_seed + ep for ep in range(collect_episodes)]
    audit_seeds = [base_seed + collect_episodes + 1_000_000 + i for i in range(10)]
    anchors = generate_transitions(max(256, batch_size * 4), seed=base_seed + 2_000_000)
    anchor_seeds = sorted({row["episode_seed"] for row in anchors})
    if set(collection_seeds) & set(audit_seeds) or set(anchor_seeds) & set(audit_seeds):
        raise ValueError("adaptation and audit episode seeds overlap")
    replay_rng = random.Random(base_seed + 300)
    bootstrap_rng = torch.Generator().manual_seed(base_seed + 400)
    collected, surprises = 0, 0

    # Collect with the frozen reactive policy. Random ensemble predictions only
    # score surprise; they do not choose actions or confer validated planning.
    ensemble.eval()
    with torch.inference_mode():
        for ep, seed in enumerate(collection_seeds):
            env = TidePool(seed=seed, scarcity=1.5 + (ep % 4) * 0.6, max_steps=128)
            obs = env.reset(seed)
            done = False
            while not done:
                obs_t = torch.tensor([obs], dtype=torch.float32)
                core_out = core_model(torch.zeros(1, core_model.config.hash_buckets), obs_t)
                action = int(core_out["action"].argmax(-1).item())
                prediction = ensemble(core_out["memory"], obs_t, torch.tensor([action]))
                next_obs, reward, done, _ = env.step(action)
                error = float((prediction["mean_delta"][0] - (torch.tensor(next_obs) - obs_t[0])).square().mean())
                disagreement = float(prediction["disagreement"][0])
                failure = bool(done and not env.alive)
                surprises += bool(disagreement > 0.08 or error > 0.05 or failure)
                buffer.add(obs, action, reward, next_obs, disagreement, error, failure)
                obs = next_obs
                collected += 1

    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=1e-5)
    loss_history = []
    ensemble.train()
    replay_exposures, anchor_exposures = 0, 0
    for _ in range(adaptation_steps):
        replay = buffer.sample(max(1, batch_size // 2))
        reference = replay_rng.choices(anchors, k=batch_size - len(replay))
        observations = [row.observation for row in replay] + [row["obs"] for row in reference]
        actions = [row.action for row in replay] + [row["action"] for row in reference]
        successors = [row.next_observation for row in replay] + [row["next_obs"] for row in reference]
        obs_b = torch.tensor(observations, dtype=torch.float32)
        action_b = torch.tensor(actions, dtype=torch.long)
        targets = torch.tensor(successors, dtype=torch.float32) - obs_b
        with torch.no_grad():
            latent = core_model(torch.zeros(len(observations), core_model.config.hash_buckets), obs_b)["memory"]
        losses = []
        for head in ensemble.heads:
            indices = torch.randint(len(observations), (len(observations),), generator=bootstrap_rng)
            predicted = head(latent[indices], obs_b[indices], action_b[indices])
            losses.append(nn.functional.mse_loss(predicted, targets[indices]))
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite adaptation loss; candidate was not saved")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_history.append(float(loss.detach()))
        replay_exposures += len(replay)
        anchor_exposures += len(reference)
    ensemble.eval()
    core_after = model_digest(core_model)
    if core_before != core_after:
        raise RuntimeError("frozen core changed during ensemble adaptation")
    ensemble_after = model_digest(ensemble)

    scene_examples = generate_scene_examples(256, seed=771122, split="test")
    auditor = PromotionAuditor(PromotionCriteria(min_survival_rate=0.90, max_dynamics_mse=0.045,
                                               min_epistemic_correlation=0.15))
    audit = auditor.audit_candidate(core_model, ensemble, test_seeds=audit_seeds, scarcity=2.0,
                                    incumbent_model=core_model, incumbent_ensemble=incumbent_ensemble,
                                    scene_examples=scene_examples)
    protocol = {"base_seed": base_seed, "collection_seeds": collection_seeds, "audit_seeds": audit_seeds,
                "anchor_episode_seeds": anchor_seeds, "collection_policy": "frozen reactive core argmax",
                "ensemble_initialization": "new independently seeded random heads; no prior ensemble loaded",
                "sampling": "half surprise replay, half scripted baseline anchors; per-head bootstrap with replacement",
                "replay_examples_consumed": replay_exposures, "anchor_examples_consumed": anchor_exposures,
                "bootstrap_head_exposures": (replay_exposures + anchor_exposures) * ensemble.k,
                "scene_seed": 771122, "scene_split": "test", "scene_examples": len(scene_examples),
                "audit_reuse": "fixed per call; repeated selection needs a separate fresh final audit"}
    parameters = {"core_before_sha256": core_before, "core_after_sha256": core_after, "core_changed": False,
                  "ensemble_before_sha256": ensemble_before, "ensemble_after_sha256": ensemble_after,
                  "ensemble_changed": ensemble_before != ensemble_after,
                  "core_parameter_count": sum(p.numel() for p in core_model.parameters()),
                  "adapted_parameter_count": sum(p.numel() for p in ensemble.parameters()),
                  "optimized_component": "dynamics ensemble only"}
    out_path.mkdir(parents=True, exist_ok=True)
    artifact = out_path / "ensemble_adapted.pt"
    temporary = out_path / "ensemble_adapted.pt.tmp"
    torch.save({"schema": "poseidon-ensemble-candidate-v2", "ensemble_state_dict": ensemble.state_dict(),
                "config": {"hidden_size": core_model.config.hidden_size, "k": ensemble.k},
                "core_sha256": core_before, "audit_report": audit, "loss_history": loss_history,
                "protocol": protocol, "parameters": parameters, "activated": False}, temporary)
    os.replace(temporary, artifact)
    receipt = {"schema": "poseidon-adaptation-receipt-v2", "status": "complete", "activated": False,
               "readiness": "candidate_trained_and_audited; deployment and planner benefit unverified",
               "steps_collected": collected, "surprises_stored": surprises, "replay_buffer_size": len(buffer),
               "adaptation_steps": len(loss_history), "initial_loss": loss_history[0], "final_loss": loss_history[-1],
               "audit": audit, "protocol": protocol, "parameters": parameters,
               "base_core_path": str(Path(base_core_path).resolve()),
               "base_core_file_sha256": hashlib.sha256(Path(base_core_path).read_bytes()).hexdigest(),
               "artifact_path": str(artifact.resolve()), "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
               "limits": ["Surprise weights and failure scores are heuristic; probabilities are not calibrated.",
                          "The policy core did not change; this run cannot establish improved reactive survival.",
                          "Loss values use resampled batches and are not a held-out learning curve."]}
    receipt_tmp = out_path / "audit_receipt.json.tmp"
    receipt_tmp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(receipt_tmp, out_path / "audit_receipt.json")
    return receipt
