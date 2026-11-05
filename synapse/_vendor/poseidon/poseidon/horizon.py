"""Multi-horizon trajectory advantage with depletion awareness and empirical gating.

Offline fitting branches synthetic simulator snapshots for H steps under the core policy.
Planning sees only the sixteen normalized observations.
Calibration is descriptive: no safety, conformal coverage, or asymptotic claim.
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
from typing import Sequence

import torch

from .atlas import (
    MAX_ARTIFACT_BYTES,
    _canonical,
    _digest,
    _integer,
    _model_digest,
    _number,
    _observation,
    _parameter_stamp,
    _quantile,
    _seeds,
)
from .core import CoreRuntime
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION, teacher_action

HORIZON_SCHEMA = "poseidon-horizon-atlas-v4"
_MODES = ("memory", "no_memory", "uncalibrated")
_METRIC = "episode-max-horizon-advantage-overestimation"
_PATH_POLICY = "local-rng:teacher-0.7/core-0.2/random-0.1"
_LIMITS = [
    "Multi-horizon rollout advantage retrieval in the synthetic TidePool environment.",
    "Fitting uses offline simulator rollouts under the frozen core policy; planning receives observations only.",
    "Empirical error radii use episode maxima and are descriptive diagnostics, not conformal coverage guarantees.",
    "Depletion detection relies on local observable resource channels and macro action definitions.",
    "Planning receives observations only, with no simulator snapshot, episode seed or teacher.",
    "The neural checkpoint and active-core pointer are unchanged; this candidate never promotes itself.",
]


@dataclass(frozen=True)
class HorizonConfig:
    horizon: int = 16
    k_neighbors: int = 16
    support_radius: float = 0.25
    quantile: float = 0.90
    override_margin: float = 0.03
    min_harvest_resource: float = 0.04
    max_samples: int = 12000
    scarcity_levels: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)

    def __post_init__(self):
        _integer(self.horizon, "horizon", 2, 64)
        _integer(self.k_neighbors, "k_neighbors", 1, 64)
        _integer(self.max_samples, "max_samples", 6, 24000)
        _number(self.support_radius, "support_radius", 0.000001, 1.0)
        _number(self.quantile, "quantile", 0.5, 1.0)
        _number(self.override_margin, "override_margin", 0.0, 5.0)
        _number(self.min_harvest_resource, "min_harvest_resource", 0.0, 0.5)
        if not isinstance(self.scarcity_levels, (list, tuple)) or not 1 <= len(self.scarcity_levels) <= 8:
            raise ValueError("scarcity_levels requires 1 to 8 levels")
        levels = tuple(_number(x, "scarcity", 0.5, 4.0) for x in self.scarcity_levels)
        if len(set(levels)) != len(levels):
            raise ValueError("scarcity_levels must be distinct")
        object.__setattr__(self, "scarcity_levels", levels)


def _partitions(train: Sequence[int], calibration: Sequence[int]) -> tuple[list[int], list[int]]:
    groups = tuple(_seeds(values, name) for values, name in (
        (train, "train_seeds"), (calibration, "calibration_seeds")
    ))
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Train and calibration episode seeds must be disjoint")
    return groups[0], groups[1]


def _rollout_returns(snapshot: dict, core: CoreRuntime, horizon: int) -> list[float]:
    envs = [TidePool.from_snapshot(snapshot) for _ in range(N_ACTIONS)]
    totals = [0.0] * N_ACTIONS
    obs_list = [None] * N_ACTIONS
    for a, env in enumerate(envs):
        o, r, _, _ = env.step(a)
        totals[a] += r
        obs_list[a] = o
    for _ in range(horizon - 1):
        live = [i for i, env in enumerate(envs) if not env.done]
        if not live:
            break
        obs_tensor = torch.tensor([obs_list[i] for i in live], dtype=torch.float32)
        acts = core.model([""] * len(live), obs_tensor)["action"].argmax(-1).tolist()
        for i, a in zip(live, acts):
            o, r, _, _ = envs[i].step(a)
            totals[i] += r
            obs_list[i] = o
    return totals


def _collect_horizon(
    core: CoreRuntime,
    seeds: list[int],
    anchors: int,
    max_steps: int,
    config: HorizonConfig,
) -> tuple[list[dict], dict[str, list[float]], dict[str, int]]:
    rows = []
    policies = {}
    incumbents = {}
    ticks = {min(max_steps - 1, i * max_steps // anchors) for i in range(anchors)}
    for index, seed in enumerate(seeds):
        scarcity = config.scarcity_levels[index % len(config.scarcity_levels)]
        env = TidePool(seed=seed, scarcity=scarcity, max_steps=max_steps)
        rng = random.Random(seed ^ 0x484F52495A4F4E)
        while not env.done:
            obs = env.observe()
            if env.tick in ticks:
                stable = env.snapshot()
                anchor_id = _digest(stable)
                g = _rollout_returns(stable, core, config.horizon)
                obs_t = torch.tensor([obs], dtype=torch.float32)
                logits = core.model([""], obs_t)["action"][0]
                probs = logits.softmax(-1).tolist()
                incumbent = int(logits.argmax())
                policies[anchor_id] = probs
                incumbents[anchor_id] = incumbent
                for a in range(N_ACTIONS):
                    branch = TidePool.from_snapshot(stable)
                    next_obs, reward, _, _ = branch.step(a)
                    adv = g[a] - g[incumbent]
                    rows.append({
                        "sample_id": f"{anchor_id}:{a}",
                        "anchor_id": anchor_id,
                        "episode_seed": seed,
                        "observation": obs[:],
                        "action": a,
                        "next_observation": next_obs,
                        "horizon_return": g[a],
                        "horizon_advantage": adv,
                        "policy_action": incumbent,
                    })
            draw = rng.random()
            action = teacher_action(obs) if draw < 0.7 else core.act(obs) if draw < 0.9 else rng.randrange(N_ACTIONS)
            env.step(action)
    return rows, policies, incumbents


class HorizonAtlas:
    def __init__(self, core: CoreRuntime, payload: dict):
        try:
            self._initialize(core, payload)
        except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            raise ValueError("Malformed horizon artifact") from error

    def _initialize(self, core: CoreRuntime, payload: dict):
        if not isinstance(payload, dict) or payload.get("schema") != HORIZON_SCHEMA:
            raise ValueError("Unsupported horizon schema")
        try:
            serialized = _canonical(payload)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("Horizon artifact requires finite JSON primitives") from error
        if len(serialized.encode()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Horizon artifact exceeds size limit")
        data = json.loads(serialized)
        expected = data.pop("sha256", None)
        if expected != _digest(data):
            raise ValueError("Horizon artifact checksum mismatch")
        if set(data) != {
            "schema", "world_version", "checkpoint_sha256", "model_sha256", "config",
            "partition", "records", "calibration", "limits"
        }:
            raise ValueError("Unexpected horizon artifact fields")
        if data["world_version"] != WORLD_VERSION:
            raise ValueError("Horizon world version mismatch")
        if data["checkpoint_sha256"] != hashlib.sha256(Path(core.path).read_bytes()).hexdigest():
            raise ValueError("Horizon checkpoint mismatch")
        if data["model_sha256"] != _model_digest(core):
            raise ValueError("Horizon in-memory model mismatch")
        self.config = HorizonConfig(**data["config"])
        if data["limits"] != _LIMITS:
            raise ValueError("Invalid horizon diagnostic limits")
        partition = data["partition"]
        if not isinstance(partition, dict) or set(partition) != {
            "train_seeds", "calibration_seeds", "anchors_per_episode", "max_steps", "path_policy"
        }:
            raise ValueError("Invalid horizon partition")
        training, calibration = _partitions(partition["train_seeds"], partition["calibration_seeds"])
        anchors_per_episode = _integer(partition["anchors_per_episode"], "anchors_per_episode", 1, 256)
        max_steps = _integer(partition["max_steps"], "max_steps", 1, 1024)
        if anchors_per_episode > max_steps or partition["path_policy"] != _PATH_POLICY:
            raise ValueError("Invalid horizon collection protocol")
        if max(len(training), len(calibration)) * anchors_per_episode * N_ACTIONS > self.config.max_samples:
            raise ValueError("Horizon partition exceeds sample budget")
        records = data["records"]
        if not isinstance(records, list) or not N_ACTIONS <= len(records) <= self.config.max_samples:
            raise ValueError("Invalid horizon record count")
        ids, anchors, episodes = set(), {}, set()
        for row in records:
            if not isinstance(row, dict) or set(row) != {
                "sample_id", "anchor_id", "episode_seed", "observation", "action",
                "next_observation", "horizon_return", "horizon_advantage", "policy_action"
            }:
                raise ValueError("Invalid horizon record fields")
            action = _integer(row["action"], "action", 0, N_ACTIONS - 1)
            seed = _integer(row["episode_seed"], "episode_seed", 0, 2**63 - 1)
            if seed not in training:
                raise ValueError("Residual memory contains a nontraining episode")
            episodes.add(seed)
            anchor = row["anchor_id"]
            if not isinstance(anchor, str) or len(anchor) != 64 or any(c not in "0123456789abcdef" for c in anchor):
                raise ValueError("Invalid horizon anchor identity")
            if row["sample_id"] != f"{anchor}:{action}" or row["sample_id"] in ids:
                raise ValueError("Invalid or duplicate horizon sample identity")
            ids.add(row["sample_id"])
            for field in ("observation", "next_observation"):
                _observation(row[field])
            _number(row["horizon_return"], "horizon_return", -1000.0, 1000.0)
            _number(row["horizon_advantage"], "horizon_advantage", -1000.0, 1000.0)
            _integer(row["policy_action"], "policy_action", 0, N_ACTIONS - 1)
            observation, episode_seed, actions = anchors.setdefault(anchor, (row["observation"], seed, set()))
            if observation != row["observation"] or episode_seed != seed:
                raise ValueError("Matched anchor provenance differs")
            actions.add(action)
        if any(actions != set(range(N_ACTIONS)) for _, _, actions in anchors.values()):
            raise ValueError("Every horizon anchor requires all six matched actions")
        if episodes != set(training) or any(sum(item[1] == s for item in anchors.values()) > anchors_per_episode for s in training):
            raise ValueError("Horizon training episode provenance mismatch")
        self._validate_calibration(data["calibration"], calibration, anchors_per_episode)
        data["sha256"] = expected
        self.core = core
        self._parameter_stamp = _parameter_stamp(core)
        stat = Path(core.path).stat()
        self._checkpoint_stamp = (stat.st_size, stat.st_mtime_ns)
        self._payload = data

        # Pre-index training observation anchors and action advantages for fast tensor lookups
        anchor_obs = []
        anchor_adv = []
        seen_anchors = set()
        for row in records:
            aid = row["anchor_id"]
            if aid not in seen_anchors:
                seen_anchors.add(aid)
                anchor_obs.append(row["observation"])
        self._anchor_obs_tensor = torch.tensor(anchor_obs, dtype=torch.float32)
        # 6 advantage values per anchor
        adv_map = {}
        for row in records:
            adv_map.setdefault(row["anchor_id"], [0.0] * N_ACTIONS)[row["action"]] = row["horizon_advantage"]
        ordered_adv = [adv_map[row["anchor_id"]] for row in records if row["action"] == 0]
        self._anchor_adv_tensor = torch.tensor(ordered_adv, dtype=torch.float32)
        self._anchor_count = len(anchor_obs)

    def _validate_calibration(self, summary: dict, seeds: list[int], anchors: int):
        if not isinstance(summary, dict) or set(summary) != {"metric", *_MODES} or summary["metric"] != _METRIC:
            raise ValueError("Invalid horizon calibration schema")
        for mode in _MODES:
            item = summary[mode]
            if not isinstance(item, dict) or set(item) != {
                "episode_count", "anchor_count", "error_radius", "mean_error", "episode_max_errors"
            }:
                raise ValueError(f"Invalid {mode} calibration schema")
            if _integer(item["episode_count"], "calibration episodes", 1, 256) != len(seeds):
                raise ValueError("Calibration episode count mismatch")
            _integer(item["anchor_count"], "anchor_count", len(seeds), len(seeds) * anchors)
            errors = item["episode_max_errors"]
            if not isinstance(errors, list) or len(errors) != len(seeds):
                raise ValueError("Calibration requires one maximum per episode")
            for err in errors:
                _number(err, "episode maximum error", 0.0, 100.0)
            _number(item["error_radius"], "error_radius", 0.0, 100.0)
            _number(item["mean_error"], "mean_error", 0.0, 100.0)
            expected_radius = 0.0 if mode == "uncalibrated" else _quantile(errors, self.config.quantile)
            if abs(item["error_radius"] - expected_radius) > 1e-6 or abs(item["mean_error"] - fmean(errors)) > 1e-6:
                raise ValueError("Calibration arithmetic mismatch")

    @property
    def artifact(self) -> dict:
        return copy.deepcopy(self._payload)

    def _assert_core(self):
        if self.core.model.training:
            raise ValueError("Horizon requires the core in evaluation mode")
        if self._parameter_stamp != _parameter_stamp(self.core):
            raise ValueError("Core weights changed after horizon binding")
        try:
            stat = Path(self.core.path).stat()
        except OSError as error:
            raise ValueError("Horizon checkpoint is unavailable") from error
        if self._checkpoint_stamp != (stat.st_size, stat.st_mtime_ns):
            if hashlib.sha256(Path(self.core.path).read_bytes()).hexdigest() != self._payload["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after horizon binding")
            self._checkpoint_stamp = (stat.st_size, stat.st_mtime_ns)

    def _retrieve_advantage(self, observation: list[float]) -> tuple[list[float], float]:
        query = torch.tensor(observation, dtype=torch.float32)
        dist = ((self._anchor_obs_tensor - query).square().mean(dim=1)).sqrt()
        sorted_idx = torch.argsort(dist, stable=True)[:self.config.k_neighbors]
        nearest_d = float(dist[sorted_idx[0]])
        weights = 1.0 / (dist[sorted_idx] + 1e-4)
        weights = weights / weights.sum()
        pred_adv = (self._anchor_adv_tensor[sorted_idx] * weights[:, None]).sum(dim=0).tolist()
        return pred_adv, nearest_d

    @classmethod
    def fit(
        cls,
        core: CoreRuntime,
        *,
        train_seeds: list[int],
        calibration_seeds: list[int],
        anchors_per_episode: int = 24,
        max_steps: int = 128,
        config: HorizonConfig | None = None,
    ) -> tuple[HorizonAtlas, dict]:
        config = HorizonConfig() if config is None else config
        if not isinstance(config, HorizonConfig):
            raise ValueError("config must be HorizonConfig")
        training, calibration = _partitions(train_seeds, calibration_seeds)
        _integer(anchors_per_episode, "anchors_per_episode", 1, 256)
        _integer(max_steps, "max_steps", 1, 1024)
        if anchors_per_episode > max_steps:
            raise ValueError("anchors_per_episode cannot exceed max_steps")
        if max(len(training), len(calibration)) * anchors_per_episode * N_ACTIONS > config.max_samples:
            raise ValueError("Requested horizon samples exceed bounded budget")
        if core.model.training:
            raise ValueError("Horizon requires the core in evaluation mode")
        checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
        model_hash = _model_digest(core)

        # 1. Collect training partition
        records, _, _ = _collect_horizon(core, training, anchors_per_episode, max_steps, config)

        # 2. Collect calibration partition
        cal_records, _, cal_incumbents = _collect_horizon(core, calibration, anchors_per_episode, max_steps, config)

        # Create temporary instance to predict calibration advantages
        summary = {"metric": _METRIC}
        for mode in _MODES:
            summary[mode] = {
                "episode_count": len(calibration),
                "anchor_count": len(cal_incumbents),
                "error_radius": 0.0 if mode == "uncalibrated" else 1.0,
                "mean_error": 1.0,
                "episode_max_errors": [1.0] * len(calibration),
            }
        payload = {
            "schema": HORIZON_SCHEMA,
            "world_version": WORLD_VERSION,
            "checkpoint_sha256": checkpoint_hash,
            "model_sha256": model_hash,
            "config": asdict(config),
            "partition": {
                "train_seeds": training,
                "calibration_seeds": calibration,
                "anchors_per_episode": anchors_per_episode,
                "max_steps": max_steps,
                "path_policy": _PATH_POLICY,
            },
            "records": records,
            "calibration": summary,
            "limits": _LIMITS,
        }
        payload["sha256"] = _digest(payload)
        temp_atlas = cls(core, payload)

        # Compute overestimation errors across calibration anchors
        cal_anchors = {}
        for r in cal_records:
            cal_anchors.setdefault(r["anchor_id"], []).append(r)

        episode_errors = {mode: {s: 0.0 for s in calibration} for mode in _MODES}
        cal_anchor_details = []
        for aid, group in cal_anchors.items():
            obs = group[0]["observation"]
            seed = group[0]["episode_seed"]
            inc = cal_incumbents[aid]
            actual_adv = [0.0] * N_ACTIONS
            for r in group:
                actual_adv[r["action"]] = r["horizon_advantage"]

            pred_adv_mem, d = temp_atlas._retrieve_advantage(obs)
            pred_adv_mem[inc] = 0.0

            predictions = {
                "memory": pred_adv_mem,
                "no_memory": [0.0] * N_ACTIONS,
                "uncalibrated": pred_adv_mem,
            }

            err_detail = {"anchor_id": aid, "episode_seed": seed, "policy_action": inc}
            for mode in _MODES:
                preds = predictions[mode]
                overest = max(0.0, max((preds[a] - preds[inc]) - actual_adv[a] for a in range(N_ACTIONS)))
                episode_errors[mode][seed] = max(episode_errors[mode][seed], overest)
                err_detail[f"{mode}_overestimation"] = overest
            cal_anchor_details.append(err_detail)

        for mode in _MODES:
            err_list = [episode_errors[mode][s] for s in calibration]
            rad = 0.0 if mode == "uncalibrated" else _quantile(err_list, config.quantile)
            summary[mode] = {
                "episode_count": len(calibration),
                "anchor_count": len(cal_anchors),
                "error_radius": rad,
                "mean_error": fmean(err_list),
                "episode_max_errors": err_list,
            }

        if checkpoint_hash != hashlib.sha256(Path(core.path).read_bytes()).hexdigest() or model_hash != _model_digest(core):
            raise ValueError("Core changed during horizon fitting")

        payload.pop("sha256")
        payload["sha256"] = _digest(payload)
        atlas = cls(core, payload)
        receipt = {
            "schema": "poseidon-horizon-fit-v1",
            "artifact_sha256": payload["sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "model_sha256": model_hash,
            "training_samples": len(records),
            "calibration_samples": len(cal_records),
            "training_anchors": len(records) // N_ACTIONS,
            "calibration_anchors": len(cal_anchors),
            "partition": copy.deepcopy(payload["partition"]),
            "calibration": copy.deepcopy(summary),
            "calibration_anchor_rows": cal_anchor_details,
            "limits": _LIMITS[:],
        }
        return atlas, receipt

    @torch.inference_mode()
    def plan(
        self,
        observation: Sequence[float],
        *,
        use_memory: bool = True,
        gate: str = "calibrated",
    ) -> dict:
        obs = _observation(observation)
        if type(use_memory) is not bool:
            raise ValueError("use_memory must be boolean")
        if gate not in ("calibrated", "uncalibrated"):
            raise ValueError("gate must be calibrated or uncalibrated")
        self._assert_core()

        obs_t = torch.tensor([obs], dtype=torch.float32)
        logits = self.core.model([""], obs_t)["action"][0]
        probabilities = logits.softmax(-1).tolist()
        incumbent = int(logits.argmax())

        raw_adv, dist = self._retrieve_advantage(obs)
        mode = "memory" if (use_memory and gate == "calibrated") else (
            "uncalibrated" if (use_memory and gate == "uncalibrated") else "no_memory"
        )
        adv_scores = [0.0] * N_ACTIONS if not use_memory else raw_adv[:]
        adv_scores[incumbent] = 0.0

        # Depletion trap detection from observable state:
        food, water, scent = obs[6], obs[7], obs[13]
        threat, energy, hyd, stam = obs[5], obs[1], obs[2], obs[3]

        depletion_trap = False
        if incumbent == 1 and food < self.config.min_harvest_resource:
            depletion_trap = True
            adv_scores[1] = -0.5
        elif incumbent == 2 and water < self.config.min_harvest_resource:
            depletion_trap = True
            adv_scores[2] = -0.5

        # Vital state filter:
        candidates = []
        for a in range(N_ACTIONS):
            adv = adv_scores[a]
            safe = True
            if threat > 0.68 and a not in (3, 5):
                safe = False
            if a == 1 and food < 0.02:
                safe = False
            if a == 2 and water < 0.02:
                safe = False
            if energy < 0.10 and a in (0, 3):
                safe = False
            if hyd < 0.10 and a in (0, 3):
                safe = False
            if stam < 0.05 and a in (4, 5) and threat <= 0.68:
                safe = False
            candidates.append({
                "action": a,
                "action_name": ACTIONS[a],
                "advantage": adv,
                "safe": safe,
                "policy_probability": probabilities[a],
            })

        valid = [c for c in candidates if c["safe"]]
        best_cand = max(valid, key=lambda c: (c["advantage"], -c["action"])) if valid else candidates[incumbent]
        proposal = best_cand["action"]
        point_adv = best_cand["advantage"]

        supported = dist <= self.config.support_radius
        radius = self._payload["calibration"][mode]["error_radius"]
        margin = point_adv - radius

        chosen = incumbent
        reason = None

        if proposal != incumbent:
            if depletion_trap:
                chosen = proposal
                reason = "depletion_override"
            elif not supported:
                reason = "outside_fitted_support"
            elif margin > self.config.override_margin:
                chosen = proposal
                reason = "advantage_confirmed"
            else:
                reason = "insufficient_margin"

        return {
            "action": chosen,
            "action_name": ACTIONS[chosen],
            "policy_action": incumbent,
            "proposed_action": proposal,
            "override_accepted": chosen != incumbent,
            "point_advantage": point_adv,
            "advantage_error_radius": radius,
            "margin": margin,
            "supported": supported,
            "support_distance": dist,
            "depletion_trap": depletion_trap,
            "fallback_reason": reason,
            "use_memory": use_memory,
            "gate": gate,
            "calibration_mode": mode,
            "policy_probabilities": probabilities,
            "backend": "horizon-atlas-v4",
            "horizon": self.config.horizon,
            "candidates": candidates,
            "artifact_sha256": self._payload["sha256"],
        }

    def act(self, observation: Sequence[float]) -> int:
        return self.plan(observation)["action"]

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _canonical(self._payload).encode()
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("Horizon artifact exceeds size limit")
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
    def load(cls, core: CoreRuntime, path: str | Path) -> HorizonAtlas:
        target = Path(path)
        if target.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Horizon artifact exceeds size limit")
        try:
            with target.open("rb") as stream:
                data = stream.read(MAX_ARTIFACT_BYTES + 1)
            if len(data) > MAX_ARTIFACT_BYTES:
                raise ValueError("Horizon artifact exceeds size limit")
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValueError("Invalid horizon JSON") from error
        return cls(core, payload)


fit_horizon = HorizonAtlas.fit
