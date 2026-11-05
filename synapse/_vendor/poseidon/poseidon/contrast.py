"""Selective residual memory with episode-calibrated paired action margins.

Fitting can branch frozen synthetic snapshots. Planning sees only the sixteen
normalized observations. Calibration is descriptive: no coverage, safety, return
improvement, or automatic model promotion is claimed.
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
from statistics import fmean
import tempfile

import torch

from .atlas import (AtlasConfig, MAX_ARTIFACT_BYTES, _base, _canonical, _digest,
                    _integer, _model_digest, _number, _observation,
                    _parameter_stamp, _quantile, _seeds, matched_branches)
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION, teacher_action

CONTRAST_SCHEMA = "poseidon-contrast-atlas-v3"
_MODES = ("memory", "unfiltered", "base")
_VITAL_METRIC = "max-absolute-error-first-five-vital-channels"
_PAIRED_METRIC = "episode-max-positive-action-advantage-overestimation"
_SELECTION_METRIC = "episode-balanced-per-channel-MSE"
_PATH_POLICY = "local-rng:teacher-0.7/core-0.2/random-0.1"
_LIMITS = [
    "One-step residual retrieval in the synthetic TidePool environment.",
    "Feature selection and calibration use disjoint episode seeds; correlated branches are not independent samples.",
    "Paired margins use episode maxima and are descriptive diagnostics, not conformal coverage or safety guarantees.",
    "One-step reserve utility is not episode return; accepted overrides do not establish survival improvement.",
    "Planning receives observations only, with no simulator snapshot, episode seed or teacher.",
    "The neural checkpoint and active-core pointer are unchanged; this sidecar never promotes itself.",
]


def reserve_utility(observation, policy_probability=0.0, *, policy_weight=.15, margin_penalty=2.0):
    """Observable one-step reserve score, shared with model-independent replay."""
    obs = _observation(observation)
    probability = _number(policy_probability, "policy_probability", 0, 1)
    weight = _number(policy_weight, "policy_weight", 0, 2)
    penalty = _number(margin_penalty, "margin_penalty", 0, 20)
    health, energy, hydration, stamina, exposure = obs[:5]
    shortfall = (max(0.0, .12 - energy) + 1.2 * max(0.0, .12 - hydration)
                 + max(0.0, .08 - stamina) + max(0.0, exposure - .8))
    return (2.0 * health + 1.6 * energy + 1.9 * hydration + .8 * stamina
            - 1.2 * exposure - penalty * shortfall + weight * probability)


@dataclass(frozen=True)
class ContrastConfig(AtlasConfig):
    shrinkage_grid: tuple[float, ...] = (0.0, .25, .5, .75, 1.0)
    min_selection_gain: float = .02

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.shrinkage_grid, (tuple, list)) or not 2 <= len(self.shrinkage_grid) <= 16:
            raise ValueError("shrinkage_grid requires 2 to 16 distinct coefficients")
        grid = tuple(_number(value, "shrinkage coefficient", 0, 1) for value in self.shrinkage_grid)
        if len(set(grid)) != len(grid) or 0.0 not in grid or 1.0 not in grid:
            raise ValueError("shrinkage_grid must contain distinct coefficients including zero and one")
        object.__setattr__(self, "shrinkage_grid", tuple(sorted(grid)))
        _number(self.min_selection_gain, "min_selection_gain", 0, 1)


def _partitions(train, selection, calibration):
    groups = tuple(_seeds(values, name) for values, name in (
        (train, "train_seeds"), (selection, "selection_seeds"),
        (calibration, "calibration_seeds")))
    if any(set(groups[left]) & set(groups[right]) for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Train, selection and calibration episode seeds must be disjoint")
    return groups


def _collect(core, seeds, anchors, max_steps, config):
    rows = []
    ticks = {min(max_steps - 1, i * max_steps // anchors) for i in range(anchors)}
    for index, seed in enumerate(seeds):
        env = TidePool(seed=seed, scarcity=config.scarcity_levels[index % len(config.scarcity_levels)], max_steps=max_steps)
        rng = random.Random(seed ^ 0x41544C4153)
        while not env.done:
            obs = env.observe()
            if env.tick in ticks:
                branches = matched_branches(env.snapshot())
                for row in branches:
                    row["episode_seed"] = seed
                rows.extend(branches)
            draw = rng.random()
            action = teacher_action(obs) if draw < .7 else core.act(obs) if draw < .9 else rng.randrange(N_ACTIONS)
            env.step(action)
    predictions, logits = _base(core, [row["observation"] for row in rows], [row["action"] for row in rows])
    policies = {}
    for row, prediction, probability in zip(rows, predictions.tolist(), logits.softmax(-1).tolist()):
        row["base_observation"] = prediction
        policies[row["anchor_id"]] = probability
    return rows, policies


def _select_features(rows, raw_predictions, seeds, config):
    """Choose shrinkage using equal episode weight, before any calibration."""
    grouped = {seed: [] for seed in seeds}
    for row, raw in zip(rows, raw_predictions):
        grouped[row["episode_seed"]].append((row, raw))
    if len(rows) != len(raw_predictions) or any(not group for group in grouped.values()):
        raise ValueError("Feature selection requires every episode and prediction")
    alphas, base_errors, admitted_errors = [], [], []
    for channel in range(OBS_SIZE):
        errors = {}
        for alpha in config.shrinkage_grid:
            errors[alpha] = fmean(fmean(
                (max(0.0, min(1.0, row["base_observation"][channel]
                               + alpha * (raw[channel] - row["base_observation"][channel])))
                 - row["next_observation"][channel]) ** 2
                for row, raw in group) for group in grouped.values())
        baseline = errors[0.0]
        selected = min(config.shrinkage_grid, key=lambda alpha: (errors[alpha], alpha))
        gain = (baseline - errors[selected]) / baseline if baseline > 0 else 0.0
        if errors[selected] >= baseline or gain < config.min_selection_gain:
            selected = 0.0
        alphas.append(selected)
        base_errors.append(baseline)
        admitted_errors.append(errors[selected])
    return {"metric": _SELECTION_METRIC, "episode_count": len(seeds), "sample_count": len(rows),
            "alphas": alphas, "base_mse": base_errors, "admitted_mse": admitted_errors}


class ContrastAtlas:
    def __init__(self, core, payload):
        try:
            self._initialize(core, payload)
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError("Malformed contrast artifact") from error

    def _initialize(self, core, payload):
        if not isinstance(payload, dict) or payload.get("schema") != CONTRAST_SCHEMA:
            raise ValueError("Unsupported contrast schema")
        try:
            serialized = _canonical(payload)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("Contrast artifact requires finite JSON primitives") from error
        if len(serialized.encode()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Contrast artifact exceeds size limit")
        data = json.loads(serialized)
        expected = data.pop("sha256", None)
        if expected != _digest(data):
            raise ValueError("Contrast artifact checksum mismatch")
        if set(data) != {"schema", "world_version", "checkpoint_sha256", "model_sha256", "config",
                         "partition", "records", "selection", "calibration", "limits"}:
            raise ValueError("Unexpected contrast artifact fields")
        if data["world_version"] != WORLD_VERSION:
            raise ValueError("Contrast world version mismatch")
        if data["checkpoint_sha256"] != hashlib.sha256(Path(core.path).read_bytes()).hexdigest():
            raise ValueError("Contrast checkpoint mismatch")
        if data["model_sha256"] != _model_digest(core):
            raise ValueError("Contrast in-memory model mismatch")
        self.config = ContrastConfig(**data["config"])
        if data["limits"] != _LIMITS:
            raise ValueError("Invalid contrast diagnostic limits")
        partition = data["partition"]
        if not isinstance(partition, dict) or set(partition) != {
            "train_seeds", "selection_seeds", "calibration_seeds", "anchors_per_episode", "max_steps", "path_policy"
        }:
            raise ValueError("Invalid contrast partition")
        training, selection, calibration = _partitions(
            partition["train_seeds"], partition["selection_seeds"], partition["calibration_seeds"])
        anchors_per_episode = _integer(partition["anchors_per_episode"], "anchors_per_episode", 1, 256)
        max_steps = _integer(partition["max_steps"], "max_steps", 1, 1024)
        if anchors_per_episode > max_steps or partition["path_policy"] != _PATH_POLICY:
            raise ValueError("Invalid contrast collection protocol")
        if max(map(len, (training, selection, calibration))) * anchors_per_episode * N_ACTIONS > self.config.max_samples:
            raise ValueError("Contrast partition exceeds sample budget")
        records = data["records"]
        if not isinstance(records, list) or not N_ACTIONS <= len(records) <= self.config.max_samples:
            raise ValueError("Invalid contrast record count")
        ids, anchors, episodes = set(), {}, set()
        for row in records:
            if not isinstance(row, dict) or set(row) != {
                "sample_id", "anchor_id", "episode_seed", "observation", "action", "next_observation", "base_observation"
            }:
                raise ValueError("Invalid contrast record fields")
            action = _integer(row["action"], "action", 0, N_ACTIONS - 1)
            seed = _integer(row["episode_seed"], "episode_seed", 0, 2**63 - 1)
            if seed not in training:
                raise ValueError("Residual memory contains a nontraining episode")
            episodes.add(seed)
            anchor = row["anchor_id"]
            if not isinstance(anchor, str) or len(anchor) != 64 or any(c not in "0123456789abcdef" for c in anchor):
                raise ValueError("Invalid contrast anchor identity")
            if row["sample_id"] != f"{anchor}:{action}" or row["sample_id"] in ids:
                raise ValueError("Invalid or duplicate contrast sample identity")
            ids.add(row["sample_id"])
            for field in ("observation", "next_observation", "base_observation"):
                _observation(row[field])
            observation, episode_seed, actions = anchors.setdefault(anchor, (row["observation"], seed, set()))
            if observation != row["observation"] or episode_seed != seed:
                raise ValueError("Matched anchor provenance differs")
            actions.add(action)
        if any(actions != set(range(N_ACTIONS)) for _, _, actions in anchors.values()):
            raise ValueError("Every contrast anchor requires all six matched actions")
        if episodes != set(training) or any(sum(item[1] == seed for item in anchors.values()) > anchors_per_episode for seed in training):
            raise ValueError("Contrast training episode provenance mismatch")
        self._validate_selection(data["selection"], selection, anchors_per_episode)
        self._validate_calibration(data["calibration"], calibration, anchors_per_episode)
        data["sha256"] = expected
        self.core = core
        self._parameter_stamp = _parameter_stamp(core)
        stat = Path(core.path).stat()
        self._checkpoint_stamp = (stat.st_size, stat.st_mtime_ns)
        self._payload = data
        self._records = [[row for row in records if row["action"] == action] for action in range(N_ACTIONS)]
        self._observations = [torch.tensor([row["observation"] for row in group], dtype=torch.float32) for group in self._records]
        self._residuals = [torch.tensor([[n - b for n, b in zip(row["next_observation"], row["base_observation"])]
                                      for row in group], dtype=torch.float32) for group in self._records]

    def _validate_selection(self, selected, seeds, anchors):
        if not isinstance(selected, dict) or set(selected) != {
            "metric", "episode_count", "sample_count", "alphas", "base_mse", "admitted_mse"
        } or selected["metric"] != _SELECTION_METRIC:
            raise ValueError("Invalid contrast feature selection")
        if _integer(selected["episode_count"], "selection episodes", 1, 256) != len(seeds):
            raise ValueError("Selection episode count mismatch")
        count = _integer(selected["sample_count"], "selection samples", N_ACTIONS * len(seeds), len(seeds) * anchors * N_ACTIONS)
        if count % N_ACTIONS:
            raise ValueError("Selection requires complete anchors")
        for name in ("alphas", "base_mse", "admitted_mse"):
            if not isinstance(selected[name], list) or len(selected[name]) != OBS_SIZE:
                raise ValueError("Selection requires sixteen channels")
            for value in selected[name]:
                _number(value, name, 0, 1)
        for alpha, baseline, admitted in zip(selected["alphas"], selected["base_mse"], selected["admitted_mse"]):
            if alpha not in self.config.shrinkage_grid:
                raise ValueError("Selection coefficient is outside the frozen grid")
            if alpha == 0 and admitted != baseline:
                raise ValueError("Base feature error mismatch")
            if alpha > 0 and (baseline <= 0 or admitted >= baseline or (baseline - admitted) / baseline + 1e-12 < self.config.min_selection_gain):
                raise ValueError("Feature correction lacks required selection improvement")

    def _validate_calibration(self, summary, seeds, anchors):
        if not isinstance(summary, dict) or set(summary) != {"metric", "memory", "unfiltered", "base", "paired"} or summary["metric"] != _VITAL_METRIC:
            raise ValueError("Invalid contrast vital calibration")
        paired = summary["paired"]
        if not isinstance(paired, dict) or set(paired) != {"metric", *_MODES} or paired["metric"] != _PAIRED_METRIC:
            raise ValueError("Invalid contrast paired calibration")
        counts = []
        for mode in _MODES:
            items = summary[mode]
            if not isinstance(items, list) or len(items) != N_ACTIONS:
                raise ValueError("Calibration requires all six actions")
            for action, item in enumerate(items):
                if not isinstance(item, dict) or set(item) != {"action", "count", "supported_count", "error_radius", "mean_error"}:
                    raise ValueError("Invalid contrast calibration action")
                if _integer(item["action"], "calibration action", 0, N_ACTIONS - 1) != action:
                    raise ValueError("Calibration action order mismatch")
                count = _integer(item["count"], "calibration count", len(seeds), len(seeds) * anchors)
                counts.append(count)
                _integer(item["supported_count"], "supported_count", 0, count)
                _number(item["error_radius"], "vital error radius", 0, 1)
                _number(item["mean_error"], "vital mean error", 0, 1)
            item = paired[mode]
            if not isinstance(item, dict) or set(item) != {"episode_count", "anchor_count", "error_radius", "mean_error", "episode_max_errors"}:
                raise ValueError("Invalid paired mode calibration")
            if _integer(item["episode_count"], "paired episodes", 1, 256) != len(seeds):
                raise ValueError("Paired episode count mismatch")
            if _integer(item["anchor_count"], "paired anchors", len(seeds), len(seeds) * anchors) != items[0]["count"]:
                raise ValueError("Paired anchor count mismatch")
            errors = item["episode_max_errors"]
            if not isinstance(errors, list) or len(errors) != len(seeds):
                raise ValueError("Paired calibration requires one maximum per episode")
            for error in errors:
                _number(error, "episode maximum advantage error", 0, 100)
            _number(item["error_radius"], "advantage radius", 0, 100)
            _number(item["mean_error"], "mean advantage error", 0, 100)
            if item["error_radius"] != _quantile(errors, self.config.quantile) or item["mean_error"] != fmean(errors):
                raise ValueError("Paired calibration arithmetic mismatch")
        if len(set(counts)) != 1:
            raise ValueError("Calibrated action anchor counts differ")

    @property
    def artifact(self):
        return copy.deepcopy(self._payload)

    def _assert_core(self):
        if self.core.model.training:
            raise ValueError("Contrast requires the core in evaluation mode")
        if self._parameter_stamp != _parameter_stamp(self.core):
            raise ValueError("Core weights changed after contrast binding")
        try:
            stat = Path(self.core.path).stat()
        except OSError as error:
            raise ValueError("Contrast checkpoint is unavailable") from error
        if self._checkpoint_stamp != (stat.st_size, stat.st_mtime_ns):
            if hashlib.sha256(Path(self.core.path).read_bytes()).hexdigest() != self._payload["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after contrast binding")
            self._checkpoint_stamp = (stat.st_size, stat.st_mtime_ns)

    def _retrieve(self, observation, action, base):
        query = torch.tensor(observation, dtype=torch.float32)
        distances = ((self._observations[action] - query).square().mean(dim=1)).sqrt()
        indices = torch.argsort(distances, stable=True)[:self.config.k_neighbors]
        nearest = distances[indices]
        weights = 1.0 / (nearest + 1e-6)
        weights /= weights.sum()
        correction = (self._residuals[action][indices] * weights[:, None]).sum(dim=0)
        raw = (torch.tensor(base) + correction).tolist()
        return raw, float(nearest[0]), [self._records[action][int(index)]["sample_id"] for index in indices]

    @staticmethod
    def _admit(base, raw, alphas):
        return [max(0.0, min(1.0, b + alpha * (r - b))) for b, r, alpha in zip(base, raw, alphas)]

    def utility(self, observation, policy_probability=0.0):
        return reserve_utility(observation, policy_probability, policy_weight=self.config.policy_weight,
                               margin_penalty=self.config.margin_penalty)

    @classmethod
    def fit(cls, core, *, train_seeds, selection_seeds, calibration_seeds,
            anchors_per_episode=24, max_steps=128, config=None):
        config = ContrastConfig() if config is None else config
        if not isinstance(config, ContrastConfig):
            raise ValueError("config must be ContrastConfig")
        training, selection, calibration = _partitions(train_seeds, selection_seeds, calibration_seeds)
        _integer(anchors_per_episode, "anchors_per_episode", 1, 256)
        _integer(max_steps, "max_steps", 1, 1024)
        if anchors_per_episode > max_steps or max(map(len, (training, selection, calibration))) * anchors_per_episode * N_ACTIONS > config.max_samples:
            raise ValueError("Requested contrast samples exceed bounded budget")
        if core.model.training:
            raise ValueError("Contrast requires the core in evaluation mode")
        checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
        model_hash = _model_digest(core)
        records, _ = _collect(core, training, anchors_per_episode, max_steps, config)
        selection_rows, _ = _collect(core, selection, anchors_per_episode, max_steps, config)
        held_out, policies = _collect(core, calibration, anchors_per_episode, max_steps, config)
        selected = {"metric": _SELECTION_METRIC, "episode_count": len(selection), "sample_count": len(selection_rows),
                    "alphas": [0.0] * OBS_SIZE, "base_mse": [0.0] * OBS_SIZE, "admitted_mse": [0.0] * OBS_SIZE}
        summary = {"metric": _VITAL_METRIC, "paired": {"metric": _PAIRED_METRIC}}
        for mode in _MODES:
            summary[mode] = [{"action": action, "count": len(calibration), "supported_count": 0, "error_radius": 1.0, "mean_error": 1.0} for action in range(N_ACTIONS)]
            summary["paired"][mode] = {"episode_count": len(calibration), "anchor_count": len(calibration),
                                        "error_radius": 1.0, "mean_error": 1.0,
                                        "episode_max_errors": [1.0] * len(calibration)}
        payload = {"schema": CONTRAST_SCHEMA, "world_version": WORLD_VERSION,
                   "checkpoint_sha256": checkpoint_hash, "model_sha256": model_hash,
                   "config": asdict(config), "partition": {
                       "train_seeds": training, "selection_seeds": selection, "calibration_seeds": calibration,
                       "anchors_per_episode": anchors_per_episode, "max_steps": max_steps, "path_policy": _PATH_POLICY},
                   "records": records, "selection": selected, "calibration": summary, "limits": _LIMITS}
        payload["sha256"] = _digest(payload)
        temporary = cls(core, payload)
        raw_selection = [temporary._retrieve(row["observation"], row["action"], row["base_observation"])[0]
                         for row in selection_rows]
        payload["selection"] = _select_features(selection_rows, raw_selection, selection, config)
        alphas = {"memory": payload["selection"]["alphas"], "unfiltered": [1.0] * OBS_SIZE, "base": [0.0] * OBS_SIZE}
        errors = {mode: [[] for _ in range(N_ACTIONS)] for mode in _MODES}
        supported = [0] * N_ACTIONS
        raw_errors, anchors = [], {}
        for row in held_out:
            action = row["action"]
            raw, distance, _ = temporary._retrieve(row["observation"], action, row["base_observation"])
            support = distance <= config.support_radius
            supported[action] += int(support)
            predictions = {mode: temporary._admit(row["base_observation"], raw, alphas[mode]) for mode in _MODES}
            values = {mode: max(abs(actual - predicted) for actual, predicted in zip(row["next_observation"][:5], prediction[:5]))
                      for mode, prediction in predictions.items()}
            for mode in _MODES:
                errors[mode][action].append(values[mode])
            raw_errors.append({"sample_id": row["sample_id"], "anchor_id": row["anchor_id"],
                               "episode_seed": row["episode_seed"], "action": action,
                               "support_distance": distance, "supported": support,
                               "base_error": values["base"], "memory_error": values["memory"],
                               "unfiltered_error": values["unfiltered"]})
            anchors.setdefault(row["anchor_id"], []).append((row, predictions))
        episode_errors = {mode: {seed: 0.0 for seed in calibration} for mode in _MODES}
        anchor_scores = []
        for anchor_id, group in anchors.items():
            probability = policies[anchor_id]
            incumbent = max(range(N_ACTIONS), key=lambda action: (probability[action], -action))
            actual = [temporary.utility(row["next_observation"], probability[row["action"]]) for row, _ in group]
            seed = group[0][0]["episode_seed"]
            score = {"anchor_id": anchor_id, "episode_seed": seed, "policy_action": incumbent}
            for mode in _MODES:
                predicted = [temporary.utility(prediction[mode], probability[row["action"]]) for row, prediction in group]
                error = max(0.0, max((predicted[action] - predicted[incumbent])
                                    - (actual[action] - actual[incumbent]) for action in range(N_ACTIONS)))
                episode_errors[mode][seed] = max(episode_errors[mode][seed], error)
                score[mode + "_overestimation"] = error
            anchor_scores.append(score)
        for mode in _MODES:
            summary[mode] = [{"action": action, "count": len(values), "supported_count": supported[action],
                              "error_radius": _quantile(values, config.quantile), "mean_error": fmean(values)}
                             for action, values in enumerate(errors[mode])]
            maximums = [episode_errors[mode][seed] for seed in calibration]
            summary["paired"][mode] = {"episode_count": len(calibration), "anchor_count": len(anchors),
                                        "error_radius": _quantile(maximums, config.quantile),
                                        "mean_error": fmean(maximums), "episode_max_errors": maximums}
        if checkpoint_hash != hashlib.sha256(Path(core.path).read_bytes()).hexdigest() or model_hash != _model_digest(core):
            raise ValueError("Core changed during contrast fitting")
        payload.pop("sha256")
        payload["sha256"] = _digest(payload)
        atlas = cls(core, payload)
        receipt = {"schema": "poseidon-contrast-fit-v1", "artifact_sha256": payload["sha256"],
                   "checkpoint_sha256": checkpoint_hash, "model_sha256": model_hash,
                   "training_samples": len(records), "selection_samples": len(selection_rows),
                   "calibration_samples": len(held_out), "training_anchors": len(records) // N_ACTIONS,
                   "selection_anchors": len(selection_rows) // N_ACTIONS, "calibration_anchors": len(anchors),
                   "partition": copy.deepcopy(payload["partition"]), "selection": copy.deepcopy(payload["selection"]),
                   "calibration": copy.deepcopy(summary), "calibration_rows": raw_errors,
                   "paired_anchor_rows": anchor_scores, "limits": _LIMITS[:]}
        return atlas, receipt

    @torch.inference_mode()
    def plan(self, observation, *, use_memory=True, use_selection=True, gate="paired"):
        obs = _observation(observation)
        if type(use_memory) is not bool or type(use_selection) is not bool:
            raise ValueError("use_memory and use_selection must be boolean")
        if gate not in ("paired", "absolute"):
            raise ValueError("gate must be paired or absolute")
        self._assert_core()
        bases, logits = _base(self.core, [obs] * N_ACTIONS, list(range(N_ACTIONS)))
        probabilities = logits[0].softmax(-1).tolist()
        incumbent = int(logits[0].argmax())
        mode = "base" if not use_memory else "memory" if use_selection else "unfiltered"
        alphas = self._payload["selection"]["alphas"] if mode == "memory" else [1.0 if mode == "unfiltered" else 0.0] * OBS_SIZE
        candidates = []
        for action, base in enumerate(bases.tolist()):
            raw, distance, sources = self._retrieve(obs, action, base)
            prediction = self._admit(base, raw, alphas)
            calibration = self._payload["calibration"][mode][action]
            radius = calibration["error_radius"]
            lower, upper = prediction[:], prediction[:]
            for index in range(4):
                lower[index] = max(0.0, prediction[index] - radius)
                upper[index] = min(1.0, prediction[index] + radius)
            lower[4] = min(1.0, prediction[4] + radius)
            upper[4] = max(0.0, prediction[4] - radius)
            supported = distance <= self.config.support_radius
            candidates.append({"action": action, "action_name": ACTIONS[action],
                               "predicted_observation": prediction, "base_observation": base,
                               "policy_probability": probabilities[action], "feature_alphas": alphas[:],
                               "support_distance": distance, "error_radius": radius,
                               "supported": supported,
                               "trusted": supported and calibration["supported_count"] > 0 and radius <= self.config.max_vital_error,
                               "calibration_count": calibration["count"],
                               "supported_calibration_count": calibration["supported_count"],
                               "point_value": self.utility(prediction, probabilities[action]),
                               "value": self.utility(lower, probabilities[action]),
                               "upper_value": self.utility(upper, probabilities[action]),
                               "source_ids": sources if use_memory else []})
        valid = [candidate for candidate in candidates if candidate["trusted"]]
        rank = "point_value" if gate == "paired" else "value"
        proposal = max(valid, key=lambda row: (row[rank], -row["action"]))["action"] if valid else incumbent
        radius = self._payload["calibration"]["paired"][mode]["error_radius"]
        point_advantage = candidates[proposal]["point_value"] - candidates[incumbent]["point_value"]
        margin = (point_advantage - radius if gate == "paired"
                  else candidates[proposal]["value"] - candidates[incumbent]["upper_value"])
        chosen, reason = proposal, None
        if not valid:
            if not any(row["supported"] for row in candidates):
                reason = "outside_fitted_support"
            elif not any(row["supported"] and row["supported_calibration_count"] > 0 for row in candidates):
                reason = "no_supported_calibration"
            else:
                reason = "prediction_error_exceeds_budget"
        elif proposal != incumbent:
            if not candidates[incumbent]["trusted"]:
                reason = "incumbent_prediction_not_supported"
                chosen = incumbent
            elif margin <= self.config.override_margin:
                reason = "counterfactual_advantage_unresolved"
                chosen = incumbent
        return {"action": chosen, "action_name": ACTIONS[chosen], "policy_action": incumbent,
                "proposed_action": proposal, "override_accepted": chosen != incumbent,
                "point_advantage": point_advantage, "advantage_error_radius": radius,
                "empirical_advantage_margin": margin, "trusted": bool(valid) and reason is None,
                "fallback_reason": reason, "use_memory": use_memory, "use_selection": use_selection,
                "feature_alphas": alphas[:], "calibration_mode": mode, "gate": gate,
                "policy_probabilities": probabilities,
                "backend": "contrast-atlas-v3", "horizon": 1, "candidates": candidates,
                "artifact_sha256": self._payload["sha256"],
                "error_note": "Empirical episode-max advantage error and action vital errors; no coverage or safety guarantee."}

    def act(self, observation):
        return self.plan(observation)["action"]

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _canonical(self._payload).encode()
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("Contrast artifact exceeds size limit")
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(target.resolve())

    @classmethod
    def load(cls, core, path):
        target = Path(path)
        if target.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Contrast artifact exceeds size limit")
        try:
            with target.open("rb") as stream:
                data = stream.read(MAX_ARTIFACT_BYTES + 1)
            if len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("Contrast artifact exceeds size limit")
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("Invalid contrast JSON") from error
        return cls(core, payload)


fit_contrast = ContrastAtlas.fit
