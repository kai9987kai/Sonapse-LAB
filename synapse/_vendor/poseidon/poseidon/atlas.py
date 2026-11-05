"""Bounded action-conditioned residual memory for one-step Tidal predictions.

Only fitting can branch simulator snapshots. Serving accepts the same sixteen
observable values as the reactive core. Held-out error radii describe errors in
this synthetic fit; they are not calibrated failure probabilities or guarantees.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import torch

from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION, teacher_action

ATLAS_SCHEMA = "poseidon-counterfactual-atlas-v2"
MAX_ARTIFACT_BYTES = 48 * 1024 * 1024
_LIMITS = [
    "One-step residual retrieval in the synthetic TidePool environment.",
    "Support distance and held-out vital error quantiles are descriptive diagnostics, not failure probabilities.",
    "No simulator snapshot, episode seed or teacher is available to plan(observation).",
    "Fitting is an offline sidecar; the neural checkpoint and active-core pointer are unchanged.",
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return float(value)


def _observation(value):
    if not isinstance(value, (list, tuple)) or len(value) != OBS_SIZE:
        raise ValueError(f"observation must contain {OBS_SIZE} normalized values")
    return [_number(x, "observation", 0, 1) for x in value]


def _seeds(values, name):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 256:
        raise ValueError(f"{name} requires 1 to 256 distinct seeds")
    result = [_integer(seed, name, 0, 2**63 - 1) for seed in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct seeds")
    return result


def _model_digest(core):
    digest = hashlib.sha256()
    for name, tensor in sorted(core.model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parameter_stamp(core):
    """Cheap serving-time guard against ordinary in-place updates/replacement."""
    return tuple((name, id(tensor), tensor._version)
                 for name, tensor in list(core.model.named_parameters()) + list(core.model.named_buffers()))


@dataclass(frozen=True)
class AtlasConfig:
    k_neighbors: int = 5
    support_radius: float = 0.25
    quantile: float = 0.90
    policy_weight: float = 0.15
    margin_penalty: float = 2.0
    max_vital_error: float = 0.20
    override_margin: float = 0.02
    max_samples: int = 12000
    scarcity_levels: tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self):
        _integer(self.k_neighbors, "k_neighbors", 1, 64)
        _integer(self.max_samples, "max_samples", 6, 24000)
        _number(self.support_radius, "support_radius", 0.000001, 1)
        _number(self.quantile, "quantile", 0.5, 1)
        _number(self.policy_weight, "policy_weight", 0, 2)
        _number(self.margin_penalty, "margin_penalty", 0, 20)
        _number(self.max_vital_error, "max_vital_error", 0, 1)
        _number(self.override_margin, "override_margin", 0, 5)
        if not isinstance(self.scarcity_levels, (list, tuple)) or not 1 <= len(self.scarcity_levels) <= 8:
            raise ValueError("scarcity_levels requires 1 to 8 levels")
        levels = tuple(_number(x, "scarcity", 0.5, 4) for x in self.scarcity_levels)
        if len(set(levels)) != len(levels):
            raise ValueError("scarcity_levels must be distinct")
        object.__setattr__(self, "scarcity_levels", levels)


def matched_branches(snapshot):
    """Six one-step interventions from one frozen anchor; never modify it."""
    anchor = TidePool.from_snapshot(snapshot)
    if anchor.done:
        raise ValueError("Cannot branch a terminal snapshot")
    observation = anchor.observe()
    stable = anchor.snapshot()
    anchor_id = _digest(stable)
    rows = []
    for action in range(N_ACTIONS):
        branch = TidePool.from_snapshot(stable)
        next_obs, _, _, _ = branch.step(action)
        rows.append({"sample_id": f"{anchor_id}:{action}", "anchor_id": anchor_id,
                     "observation": observation[:], "action": action,
                     "next_observation": next_obs})
    return rows


@torch.inference_mode()
def _base(core, observations, actions):
    obs = torch.tensor(observations, dtype=torch.float32)
    ids = torch.tensor(actions, dtype=torch.long)
    predictions, logits = [], []
    for start in range(0, len(obs), 256):
        batch = obs[start:start + 256]
        result = core.model([""] * len(batch), batch, ids[start:start + 256])
        if not torch.isfinite(result["delta"]).all() or not torch.isfinite(result["action"]).all():
            raise ValueError("Core returned non-finite predictions")
        predictions.append((batch + result["delta"]).clamp(0, 1))
        logits.append(result["action"])
    predicted, policy = torch.cat(predictions), torch.cat(logits)
    if not torch.isfinite(predicted).all() or not torch.isfinite(policy).all():
        raise ValueError("Core returned non-finite predictions")
    return predicted, policy


def _collect(core, seeds, anchors, max_steps, config):
    rows = []
    ticks = {min(max_steps - 1, i * max_steps // anchors) for i in range(anchors)}
    for index, seed in enumerate(seeds):
        env = TidePool(seed=seed, scarcity=config.scarcity_levels[index % len(config.scarcity_levels)], max_steps=max_steps)
        rng = random.Random(seed ^ 0x41544C4153)
        while not env.done:
            obs = env.observe()
            if env.tick in ticks:
                rows.extend(matched_branches(env.snapshot()))
            draw = rng.random()
            action = teacher_action(obs) if draw < 0.7 else core.act(obs) if draw < 0.9 else rng.randrange(N_ACTIONS)
            env.step(action)
    predicted, _ = _base(core, [r["observation"] for r in rows], [r["action"] for r in rows])
    for row, base in zip(rows, predicted.tolist()):
        row["base_observation"] = base
    return rows


def _quantile(values, level):
    ordered = sorted(values)
    # Observed order statistic, with no asymptotic or coverage claim.
    return ordered[min(len(ordered) - 1, max(0, math.ceil(level * len(ordered)) - 1))]


class CounterfactualAtlas:
    def __init__(self, core, payload):
        if not isinstance(payload, dict) or payload.get("schema") != ATLAS_SCHEMA:
            raise ValueError("Unsupported atlas schema")
        try:
            serialized = _canonical(payload)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("Atlas must contain finite JSON primitives") from exc
        if len(serialized.encode()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Atlas exceeds artifact size limit")
        data = json.loads(serialized)
        expected = data.pop("sha256", None)
        if expected != _digest(data):
            raise ValueError("Atlas checksum mismatch")
        if set(data) != {"schema", "world_version", "checkpoint_sha256", "model_sha256", "config", "partition", "records", "calibration", "limits"}:
            raise ValueError("Unexpected atlas fields")
        if data["world_version"] != WORLD_VERSION:
            raise ValueError("Atlas world version mismatch")
        if data["checkpoint_sha256"] != hashlib.sha256(Path(core.path).read_bytes()).hexdigest():
            raise ValueError("Atlas checkpoint mismatch")
        if data["model_sha256"] != _model_digest(core):
            raise ValueError("Atlas in-memory model mismatch")
        try:
            self.config = AtlasConfig(**data["config"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid atlas configuration") from exc
        partition = data["partition"]
        if not isinstance(partition, dict) or set(partition) != {"train_seeds", "calibration_seeds", "anchors_per_episode", "max_steps", "path_policy"}:
            raise ValueError("Invalid atlas partition")
        training = _seeds(partition["train_seeds"], "train_seeds")
        calibration = _seeds(partition["calibration_seeds"], "calibration_seeds")
        if set(training) & set(calibration):
            raise ValueError("Train and calibration seeds must be disjoint")
        _integer(partition["anchors_per_episode"], "anchors_per_episode", 1, 256)
        _integer(partition["max_steps"], "max_steps", 1, 1024)
        records = data["records"]
        if not isinstance(records, list) or not 6 <= len(records) <= self.config.max_samples:
            raise ValueError("Atlas record count exceeds bounds")
        sample_ids, anchors = set(), {}
        for row in records:
            if not isinstance(row, dict) or set(row) != {"sample_id", "anchor_id", "observation", "action", "next_observation", "base_observation"}:
                raise ValueError("Invalid atlas record fields")
            action = _integer(row["action"], "action", 0, 5)
            anchor = row["anchor_id"]
            if not isinstance(anchor, str) or len(anchor) != 64 or any(c not in "0123456789abcdef" for c in anchor):
                raise ValueError("Invalid anchor identity")
            if row["sample_id"] != f"{anchor}:{action}" or row["sample_id"] in sample_ids:
                raise ValueError("Invalid or duplicate sample identity")
            sample_ids.add(row["sample_id"])
            for key in ("observation", "next_observation", "base_observation"):
                _observation(row[key])
            previous, actions = anchors.setdefault(anchor, (row["observation"], set()))
            if previous != row["observation"]:
                raise ValueError("Matched anchor observations differ")
            actions.add(action)
        if any(actions != set(range(N_ACTIONS)) for _, actions in anchors.values()):
            raise ValueError("Every anchor requires all six matched actions")
        summary = data["calibration"]
        if not isinstance(summary, dict) or set(summary) != {"memory", "base", "metric"}:
            raise ValueError("Invalid calibration summary")
        for mode in ("memory", "base"):
            items = summary[mode]
            if not isinstance(items, list) or len(items) != N_ACTIONS:
                raise ValueError("Calibration requires all six actions")
            for action, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != {"action", "count", "supported_count", "error_radius", "mean_error"} or item["action"] != action:
                    raise ValueError("Invalid calibration action")
                count = _integer(item["count"], "calibration count", 1, self.config.max_samples)
                _integer(item["supported_count"], "supported_count", 0, count)
                _number(item["error_radius"], "error_radius", 0, 1)
                _number(item["mean_error"], "mean_error", 0, 1)
        data["sha256"] = expected
        self.core = core
        self._parameter_stamp = _parameter_stamp(core)
        self._payload = data
        self._records = [[r for r in records if r["action"] == action] for action in range(N_ACTIONS)]
        self._observations = [torch.tensor([r["observation"] for r in group], dtype=torch.float32) for group in self._records]
        self._residuals = [torch.tensor([[n - b for n, b in zip(r["next_observation"], r["base_observation"])] for r in group], dtype=torch.float32) for group in self._records]

    @property
    def artifact(self):
        return copy.deepcopy(self._payload)

    @classmethod
    def fit(cls, core, *, train_seeds, calibration_seeds, anchors_per_episode=24, max_steps=128, config=None):
        config = config or AtlasConfig()
        if not isinstance(config, AtlasConfig):
            raise ValueError("config must be AtlasConfig")
        training = _seeds(train_seeds, "train_seeds")
        calibration = _seeds(calibration_seeds, "calibration_seeds")
        if set(training) & set(calibration):
            raise ValueError("Train and calibration seeds must be disjoint")
        _integer(anchors_per_episode, "anchors_per_episode", 1, 256)
        _integer(max_steps, "max_steps", 1, 1024)
        if anchors_per_episode > max_steps or max(len(training), len(calibration)) * anchors_per_episode * N_ACTIONS > config.max_samples:
            raise ValueError("Requested fitting samples exceed bounded budget")
        if core.model.training:
            raise ValueError("Atlas requires the core in evaluation mode")
        checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
        model_hash = _model_digest(core)
        records = _collect(core, training, anchors_per_episode, max_steps, config)
        held_out = _collect(core, calibration, anchors_per_episode, max_steps, config)
        calibration_summary = {"metric": "max-absolute-error-first-five-vital-channels", "memory": [], "base": []}
        # Construct a validated temporary view using conservative finite radii.
        for mode in ("memory", "base"):
            calibration_summary[mode] = [{"action": a, "count": 1, "supported_count": 0, "error_radius": 1.0, "mean_error": 1.0} for a in range(N_ACTIONS)]
        payload = {"schema": ATLAS_SCHEMA, "world_version": WORLD_VERSION,
                   "checkpoint_sha256": checkpoint_hash, "model_sha256": model_hash,
                   "config": asdict(config), "records": records,
                   "partition": {"train_seeds": training, "calibration_seeds": calibration,
                                 "anchors_per_episode": anchors_per_episode, "max_steps": max_steps,
                                 "path_policy": "local-rng:teacher-0.7/core-0.2/random-0.1"},
                   "calibration": calibration_summary, "limits": _LIMITS}
        payload["sha256"] = _digest(payload)
        atlas = cls(core, payload)
        errors = {mode: [[] for _ in range(N_ACTIONS)] for mode in ("memory", "base")}
        supported = [0] * N_ACTIONS
        raw_errors = []
        for row in held_out:
            action = row["action"]
            prediction, distance, _ = atlas._correct(row["observation"], action, row["base_observation"])
            support = distance <= config.support_radius
            supported[action] += int(support)
            base_error = max(abs(n - b) for n, b in zip(row["next_observation"][:5], row["base_observation"][:5]))
            memory_error = max(abs(n - p) for n, p in zip(row["next_observation"][:5], prediction[:5]))
            errors["base"][action].append(base_error)
            errors["memory"][action].append(memory_error)
            raw_errors.append({"sample_id": row["sample_id"], "action": action, "support_distance": distance,
                               "supported": support, "base_error": base_error, "memory_error": memory_error})
        for mode in ("memory", "base"):
            calibration_summary[mode] = [{"action": action, "count": len(values), "supported_count": supported[action],
                                          "error_radius": _quantile(values, config.quantile),
                                          "mean_error": sum(values) / len(values)} for action, values in enumerate(errors[mode])]
        if checkpoint_hash != hashlib.sha256(Path(core.path).read_bytes()).hexdigest() or model_hash != _model_digest(core):
            raise ValueError("Core changed during atlas fitting")
        payload.pop("sha256")
        payload["sha256"] = _digest(payload)
        atlas = cls(core, payload)
        receipt = {"schema": "poseidon-atlas-fit-v1", "artifact_sha256": payload["sha256"],
                   "checkpoint_sha256": checkpoint_hash, "training_samples": len(records),
                   "calibration_samples": len(held_out), "partition": copy.deepcopy(payload["partition"]),
                   "calibration": copy.deepcopy(calibration_summary), "calibration_rows": raw_errors,
                   "limits": _LIMITS[:]}
        return atlas, receipt

    def _correct(self, observation, action, base):
        query = torch.tensor(observation, dtype=torch.float32)
        distances = ((self._observations[action] - query).square().mean(dim=1)).sqrt()
        indices = torch.argsort(distances, stable=True)[:self.config.k_neighbors]
        nearest = distances[indices]
        weights = 1.0 / (nearest + 1e-6)
        weights /= weights.sum()
        correction = (self._residuals[action][indices] * weights[:, None]).sum(dim=0)
        corrected = (torch.tensor(base) + correction).clamp(0, 1).tolist()
        return corrected, float(nearest[0]), [self._records[action][int(i)]["sample_id"] for i in indices]

    @torch.inference_mode()
    def plan(self, observation, *, use_memory=True):
        obs = _observation(observation)
        if type(use_memory) is not bool:
            raise ValueError("use_memory must be boolean")
        if self.core.model.training:
            raise ValueError("Atlas requires the core in evaluation mode")
        if self._parameter_stamp != _parameter_stamp(self.core):
            raise ValueError("Core weights changed after atlas binding")
        bases, policy_logits = _base(self.core, [obs] * N_ACTIONS, list(range(N_ACTIONS)))
        probabilities = policy_logits[0].softmax(-1).tolist()
        policy_action = int(policy_logits[0].argmax())
        mode = "memory" if use_memory else "base"
        candidates = []
        for action, base in enumerate(bases.tolist()):
            corrected, distance, sources = self._correct(obs, action, base)
            predicted = corrected if use_memory else base
            calibration = self._payload["calibration"][mode][action]
            radius = calibration["error_radius"]
            lower = [max(0.0, predicted[i] - radius) for i in range(4)]
            upper_exposure = min(1.0, predicted[4] + radius)
            health, energy, hydration, stamina = lower
            vitality = 2.0 * health + 1.6 * energy + 1.9 * hydration + 0.8 * stamina - 1.2 * upper_exposure
            reserve_shortfall = max(0, 0.12 - energy) + 1.2 * max(0, 0.12 - hydration) + max(0, 0.08 - stamina) + max(0, upper_exposure - 0.8)
            value = vitality - self.config.margin_penalty * reserve_shortfall + self.config.policy_weight * probabilities[action]
            # An override must beat the incumbent's optimistic value using the
            # challenger pessimistic value. Radii are empirical, not guarantees.
            high_health, high_energy, high_hydration, high_stamina = [min(1.0, predicted[i] + radius) for i in range(4)]
            low_exposure = max(0.0, predicted[4] - radius)
            upper_shortfall = max(0, 0.12 - high_energy) + 1.2 * max(0, 0.12 - high_hydration) + max(0, 0.08 - high_stamina) + max(0, low_exposure - 0.8)
            upper_value = (2.0 * high_health + 1.6 * high_energy + 1.9 * high_hydration + 0.8 * high_stamina
                           - 1.2 * low_exposure - self.config.margin_penalty * upper_shortfall
                           + self.config.policy_weight * probabilities[action])
            candidates.append({"action": action, "action_name": ACTIONS[action],
                               "predicted_observation": predicted, "base_observation": base,
                               "support_distance": distance, "error_radius": radius,
                               "supported": distance <= self.config.support_radius,
                               "trusted": distance <= self.config.support_radius and calibration["supported_count"] > 0 and radius <= self.config.max_vital_error,
                               "calibration_count": calibration["count"],
                               "supported_calibration_count": calibration["supported_count"],
                               "value": value, "upper_value": upper_value, "source_ids": sources if use_memory else []})
        valid = [row for row in candidates if row["trusted"]]
        chosen = max(valid, key=lambda row: (row["value"], -row["action"]))["action"] if valid else policy_action
        proposal = chosen
        reason = None
        if not valid:
            if not any(row["supported"] for row in candidates):
                reason = "outside_fitted_support"
            elif not any(row["supported"] and row["supported_calibration_count"] > 0 for row in candidates):
                reason = "no_supported_calibration"
            else:
                reason = "prediction_error_exceeds_budget"
        elif chosen != policy_action:
            if not candidates[policy_action]["trusted"]:
                reason = "incumbent_prediction_not_supported"
                chosen = policy_action
            elif candidates[chosen]["value"] <= candidates[policy_action]["upper_value"] + self.config.override_margin:
                reason = "counterfactual_advantage_unresolved"
                chosen = policy_action
        return {"action": chosen, "action_name": ACTIONS[chosen], "policy_action": policy_action,
                "proposed_action": proposal, "override_accepted": chosen != policy_action,
                "empirical_advantage_margin": candidates[proposal]["value"] - candidates[policy_action]["upper_value"],
                "trusted": bool(valid) and reason is None, "fallback_reason": reason, "use_memory": use_memory,
                "backend": "counterfactual-atlas-v2", "horizon": 1,
                "candidates": candidates, "artifact_sha256": self._payload["sha256"],
                "error_note": "Held-out vital prediction-error quantile; not a failure probability or safety guarantee."}

    def act(self, observation):
        return self.plan(observation)["action"]

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _canonical(self._payload).encode()
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("Atlas exceeds artifact size limit")
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return str(target.resolve())

    @classmethod
    def load(cls, core, path):
        target = Path(path)
        if target.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Atlas exceeds artifact size limit")
        try:
            with target.open("rb") as handle:
                data = handle.read(MAX_ARTIFACT_BYTES + 1)
            if len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("Atlas exceeds artifact size limit")
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError("Invalid atlas JSON") from exc
        return cls(core, payload)


fit_atlas = CounterfactualAtlas.fit
