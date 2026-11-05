"""Topological cognitive mapping, navigational memory, and multi-horizon empirical advantage.

Maintains an on-the-fly episodic cognitive map of the 6x6 toroidal TidePool world from 16-d observations.
Tracks discovered patches, terrain fingerprints, replenishment dynamics, and graph topology.
Directs goal-oriented navigation to remembered resource hotspots when local patches deplete,
overriding blind exploratory wandering.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
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
from .horizon import (
    HORIZON_SCHEMA,
    HorizonAtlas,
    HorizonConfig,
    _collect_horizon,
    _partitions,
    _rollout_returns,
)
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION, teacher_action

ODYSSEY_SCHEMA = "poseidon-odyssey-atlas-v5"
_MODES = ("memory", "no_memory", "uncalibrated")
_METRIC = "episode-max-odyssey-advantage-overestimation"
_PATH_POLICY = "local-rng:teacher-0.7/core-0.2/random-0.1"
_LIMITS = [
    "Episodic topological cognitive mapping and multi-horizon advantage in synthetic TidePool.",
    "Observations are strictly 16-d normalized vectors; no global simulator pointers, seeds, or ground-truth coordinates.",
    "Cognitive map tracks patch fingerprints, observed yields, replenishment decay, and topological transitions.",
    "Waypoint navigation evaluates net travel cost versus expected replenishment before initiating transit.",
    "Descriptive empirical calibration error radii bound multi-horizon advantage claims.",
    "Core neural weights and checkpoint remain strictly frozen; Odyssey acts as an external cognitive controller.",
]


@dataclass
class CognitivePatch:
    patch_id: str
    terrain: float
    shelter: float
    first_seen_tick: int
    last_seen_tick: int
    visit_count: int
    observed_food: float
    observed_water: float
    food_cap: float
    water_cap: float
    last_threat: float
    relative_x: int | None = None
    relative_y: int | None = None

    def estimated_food(self, current_tick: int, scarcity: float = 2.5) -> float:
        elapsed = max(0, current_tick - self.last_seen_tick)
        rate = 0.007 / max(0.5, scarcity)
        return min(self.food_cap, self.observed_food + elapsed * rate)

    def estimated_water(self, current_tick: int, scarcity: float = 2.5) -> float:
        elapsed = max(0, current_tick - self.last_seen_tick)
        rate = 0.014 / max(0.5, scarcity)
        return min(self.water_cap, self.observed_water + elapsed * rate)

    def to_dict(self, current_tick: int = 0, scarcity: float = 2.5) -> dict:
        return {
            "patch_id": self.patch_id,
            "terrain": round(self.terrain, 6),
            "shelter": round(self.shelter, 6),
            "first_seen_tick": self.first_seen_tick,
            "last_seen_tick": self.last_seen_tick,
            "visit_count": self.visit_count,
            "observed_food": round(self.observed_food, 4),
            "observed_water": round(self.observed_water, 4),
            "food_cap": round(self.food_cap, 4),
            "water_cap": round(self.water_cap, 4),
            "estimated_food": round(self.estimated_food(current_tick, scarcity), 4),
            "estimated_water": round(self.estimated_water(current_tick, scarcity), 4),
            "last_threat": round(self.last_threat, 4),
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
        }


class CognitiveMap:
    """Episodic topological memory graph and replenishment tracker."""

    def __init__(self, scarcity: float = 2.5):
        self.scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
        self.patches: dict[str, CognitivePatch] = {}
        self.adjacency: dict[str, set[str]] = {}
        self.current_patch_id: str | None = None
        self.prev_patch_id: str | None = None
        self.prev_action: int | None = None
        self.current_tick: int = 0
        self.total_edges: int = 0

    def reset(self, scarcity: float | None = None) -> None:
        if scarcity is not None:
            self.scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
        self.patches.clear()
        self.adjacency.clear()
        self.current_patch_id = None
        self.prev_patch_id = None
        self.prev_action = None
        self.current_tick = 0
        self.total_edges = 0

    @staticmethod
    def fingerprint(terrain: float, shelter: float) -> str:
        return f"{round(float(terrain), 6):.6f}_{round(float(shelter), 6):.6f}"

    def update(self, observation: Sequence[float], tick: int | None = None) -> CognitivePatch:
        obs = _observation(observation)
        if tick is not None:
            self.current_tick = _integer(tick, "tick", 0, 100000)
        else:
            # Estimate tick from progress (index 15) if max_steps is ~256 or increment
            self.current_tick += 1

        food, water, shelter = obs[6], obs[7], obs[8]
        terrain, threat = obs[12], obs[5]
        pid = self.fingerprint(terrain, shelter)

        # Topological edge discovery
        if self.prev_action in (4, 5) and self.prev_patch_id and self.prev_patch_id != pid:
            self.add_edge(self.prev_patch_id, pid)

        if pid in self.patches:
            patch = self.patches[pid]
            patch.last_seen_tick = self.current_tick
            patch.visit_count += 1
            patch.observed_food = food
            patch.observed_water = water
            patch.food_cap = max(patch.food_cap, food)
            patch.water_cap = max(patch.water_cap, water)
            patch.last_threat = threat
        else:
            patch = CognitivePatch(
                patch_id=pid,
                terrain=terrain,
                shelter=shelter,
                first_seen_tick=self.current_tick,
                last_seen_tick=self.current_tick,
                visit_count=1,
                observed_food=food,
                observed_water=water,
                food_cap=max(0.20, food),
                water_cap=max(0.20, water),
                last_threat=threat,
                relative_x=0 if not self.patches else None,
                relative_y=0 if not self.patches else None,
            )
            self.patches[pid] = patch
            self.adjacency.setdefault(pid, set())

        self.current_patch_id = pid
        return patch

    def record_action(self, action: int) -> None:
        self.prev_action = _integer(action, "action", 0, 5)
        self.prev_patch_id = self.current_patch_id

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.adjacency:
            self.adjacency[u] = set()
        if v not in self.adjacency:
            self.adjacency[v] = set()
        if v not in self.adjacency[u]:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            self.total_edges += 1

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        if source_id == target_id:
            return [source_id]
        if source_id not in self.adjacency or target_id not in self.adjacency:
            return None
        visited = {source_id}
        queue = [[source_id]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            for neighbor in sorted(self.adjacency.get(node, ())):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def shortest_distance(self, source_id: str, target_id: str) -> int:
        path = self.shortest_path(source_id, target_id)
        return len(path) - 1 if path is not None else 999

    def evaluate_waypoints(self, observation: Sequence[float]) -> list[dict]:
        if not self.current_patch_id or len(self.patches) < 2:
            return []
        obs = _observation(observation)
        energy, hyd, stam, exposure = obs[1], obs[2], obs[3], obs[4]
        curr_id = self.current_patch_id

        need_food = max(0.0, 0.70 - energy)
        need_water = max(0.0, 0.70 - hyd)
        need_shelter = max(0.0, exposure - 0.40)

        results = []
        for pid, patch in self.patches.items():
            if pid == curr_id:
                continue
            dist = self.shortest_distance(curr_id, pid)
            if dist > 6:
                continue
            est_food = patch.estimated_food(self.current_tick, self.scarcity)
            est_water = patch.estimated_water(self.current_tick, self.scarcity)

            # Travel cost
            cost_energy = dist * 0.038
            cost_hyd = dist * 0.037
            cost_stam = dist * 0.112

            viable = bool(
                energy - cost_energy > 0.06
                and hyd - cost_hyd > 0.06
                and (stam >= cost_stam or stam >= 0.15)
            )

            gain = (
                need_food * min(0.60, est_food)
                + need_water * min(0.70, est_water)
                + need_shelter * patch.shelter
            )
            net_score = gain - (dist * 0.075) - (0.35 * patch.last_threat)

            results.append({
                "patch_id": pid,
                "distance": dist,
                "estimated_food": est_food,
                "estimated_water": est_water,
                "shelter": patch.shelter,
                "threat": patch.last_threat,
                "viable": viable,
                "gain": gain,
                "travel_cost": cost_energy + cost_hyd + cost_stam,
                "score": net_score,
            })
        results.sort(key=lambda item: (-item["score"], item["distance"]))
        return results

    def state_summary(self) -> dict:
        curr = self.patches.get(self.current_patch_id) if self.current_patch_id else None
        return {
            "discovered_patches": len(self.patches),
            "discovered_edges": self.total_edges,
            "current_patch": curr.to_dict(self.current_tick, self.scarcity) if curr else None,
            "current_tick": self.current_tick,
            "scarcity": self.scarcity,
        }


@dataclass(frozen=True)
class OdysseyConfig:
    horizon: int = 16
    k_neighbors: int = 16
    support_radius: float = 0.25
    quantile: float = 0.90
    override_margin: float = 0.03
    min_harvest_resource: float = 0.04
    max_samples: int = 12000
    scarcity_levels: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    scarcity: float = 2.5
    min_stamina_for_transit: float = 0.20
    waypoint_threshold: float = 0.05

    def __post_init__(self):
        _integer(self.horizon, "horizon", 2, 64)
        _integer(self.k_neighbors, "k_neighbors", 1, 64)
        _integer(self.max_samples, "max_samples", 6, 24000)
        _number(self.support_radius, "support_radius", 0.000001, 1.0)
        _number(self.quantile, "quantile", 0.5, 1.0)
        _number(self.override_margin, "override_margin", 0.0, 5.0)
        _number(self.min_harvest_resource, "min_harvest_resource", 0.0, 0.5)
        _number(self.scarcity, "scarcity", 0.5, 4.0)
        _number(self.min_stamina_for_transit, "min_stamina_for_transit", 0.05, 0.60)
        _number(self.waypoint_threshold, "waypoint_threshold", 0.0, 2.0)
        if not isinstance(self.scarcity_levels, (list, tuple)) or not 1 <= len(self.scarcity_levels) <= 8:
            raise ValueError("scarcity_levels requires 1 to 8 levels")
        levels = tuple(_number(x, "scarcity", 0.5, 4.0) for x in self.scarcity_levels)
        if len(set(levels)) != len(levels):
            raise ValueError("scarcity_levels must be distinct")
        object.__setattr__(self, "scarcity_levels", levels)


class OdysseyAtlas:
    """Odyssey: Spatial Cognitive Mapping & Navigational Memory over Multi-Horizon Atlas."""

    def __init__(self, core: CoreRuntime, payload: dict):
        self._initialize(core, payload)

    def _initialize(self, core: CoreRuntime, payload: dict):
        if not isinstance(payload, dict) or payload.get("schema") != ODYSSEY_SCHEMA:
            raise ValueError("Payload does not match Odyssey Atlas schema")
        if payload.get("world_version") != WORLD_VERSION:
            raise ValueError("World version mismatch")
        digest = payload.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Missing artifact checksum")
        copied = copy.deepcopy(payload)
        copied.pop("sha256", None)
        if _digest(copied) != digest:
            raise ValueError("Payload checksum mismatch")

        self.core = core
        self.config = OdysseyConfig(**payload["config"])
        self._payload = copy.deepcopy(payload)
        self._parameter_stamp = _parameter_stamp(core)
        try:
            stat = Path(core.path).stat()
            self._checkpoint_stamp = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            self._checkpoint_stamp = None

        if payload["checkpoint_sha256"] != hashlib.sha256(Path(core.path).read_bytes()).hexdigest():
            raise ValueError("Core checkpoint sha256 mismatch")
        if payload["model_sha256"] != _model_digest(core):
            raise ValueError("Core model weight hash mismatch")

        records = payload["records"]
        if not isinstance(records, list) or len(records) > self.config.max_samples or len(records) % N_ACTIONS != 0:
            raise ValueError("Records must be complete 6-action bundles within sample budget")

        partition = payload["partition"]
        _partitions(partition["train_seeds"], partition["calibration_seeds"])
        anchors = _integer(partition["anchors_per_episode"], "anchors_per_episode", 1, 256)
        _integer(partition["max_steps"], "max_steps", 1, 1024)
        self._validate_calibration(payload["calibration"], partition["calibration_seeds"], anchors)

        # Index anchor observations and advantages
        seen_anchors = set()
        anchor_obs = []
        for row in records:
            aid = row["anchor_id"]
            if aid not in seen_anchors:
                seen_anchors.add(aid)
                anchor_obs.append(row["observation"])
        self._anchor_obs_tensor = torch.tensor(anchor_obs, dtype=torch.float32)
        adv_map = {}
        for row in records:
            adv_map.setdefault(row["anchor_id"], [0.0] * N_ACTIONS)[row["action"]] = row["horizon_advantage"]
        ordered_adv = [adv_map[row["anchor_id"]] for row in records if row["action"] == 0]
        self._anchor_adv_tensor = torch.tensor(ordered_adv, dtype=torch.float32)
        self._anchor_count = len(anchor_obs)

        # Episodic stateful cognitive map
        self.cognitive_map = CognitiveMap(scarcity=self.config.scarcity)
        self._last_progress = -1.0
        self._step_counter = 0

    def _validate_calibration(self, summary: dict, seeds: list[int], anchors: int):
        if not isinstance(summary, dict) or set(summary) != {"metric", *_MODES} or summary["metric"] != _METRIC:
            raise ValueError("Invalid odyssey calibration schema")
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
            raise ValueError("Odyssey requires the core in evaluation mode")
        if self._parameter_stamp != _parameter_stamp(self.core):
            raise ValueError("Core weights changed after odyssey binding")
        try:
            stat = Path(self.core.path).stat()
        except OSError as error:
            raise ValueError("Odyssey checkpoint is unavailable") from error
        if self._checkpoint_stamp != (stat.st_size, stat.st_mtime_ns):
            if hashlib.sha256(Path(self.core.path).read_bytes()).hexdigest() != self._payload["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after odyssey binding")
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

    def reset(self, scarcity: float | None = None) -> None:
        """Reset the episodic cognitive map for a new episode."""
        self.cognitive_map.reset(self.cognitive_map.scarcity if scarcity is None else scarcity)
        self._last_progress = -1.0
        self._step_counter = 0

    @classmethod
    def fit(
        cls,
        core: CoreRuntime,
        *,
        train_seeds: list[int],
        calibration_seeds: list[int],
        anchors_per_episode: int = 24,
        max_steps: int = 128,
        config: OdysseyConfig | None = None,
    ) -> tuple[OdysseyAtlas, dict]:
        config = OdysseyConfig() if config is None else config
        if not isinstance(config, OdysseyConfig):
            raise ValueError("config must be OdysseyConfig")
        training, calibration = _partitions(train_seeds, calibration_seeds)
        _integer(anchors_per_episode, "anchors_per_episode", 1, 256)
        _integer(max_steps, "max_steps", 1, 1024)
        if anchors_per_episode > max_steps:
            raise ValueError("anchors_per_episode cannot exceed max_steps")
        if max(len(training), len(calibration)) * anchors_per_episode * N_ACTIONS > config.max_samples:
            raise ValueError("Requested odyssey samples exceed bounded budget")
        if core.model.training:
            raise ValueError("Odyssey requires the core in evaluation mode")
        checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
        model_hash = _model_digest(core)

        # Convert config to HorizonConfig for underlying multi-horizon collection
        h_config = HorizonConfig(
            horizon=config.horizon,
            k_neighbors=config.k_neighbors,
            support_radius=config.support_radius,
            quantile=config.quantile,
            override_margin=config.override_margin,
            min_harvest_resource=config.min_harvest_resource,
            max_samples=config.max_samples,
            scarcity_levels=config.scarcity_levels,
        )

        records, _, _ = _collect_horizon(core, training, anchors_per_episode, max_steps, h_config)
        cal_records, _, cal_incumbents = _collect_horizon(core, calibration, anchors_per_episode, max_steps, h_config)

        summary = {"metric": _METRIC}
        for mode in _MODES:
            summary[mode] = {
                "episode_count": len(calibration),
                "anchor_count": len(cal_incumbents),
                "error_radius": 0.0 if mode == "uncalibrated" else 1.0,
                "mean_error": 1.0,
                "episode_max_errors": [1.0] * len(calibration),
            }

        conf_dict = asdict(config)
        conf_dict["scarcity_levels"] = list(config.scarcity_levels)
        payload = {
            "schema": ODYSSEY_SCHEMA,
            "world_version": WORLD_VERSION,
            "checkpoint_sha256": checkpoint_hash,
            "model_sha256": model_hash,
            "config": conf_dict,
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

            for mode in _MODES:
                use_mem = mode != "no_memory"
                gate_type = "uncalibrated" if mode == "uncalibrated" else "calibrated"
                pred_adv, d = temp_atlas._retrieve_advantage(obs)
                eff_adv = pred_adv if use_mem else [0.0] * N_ACTIONS
                eff_adv[inc] = 0.0
                cand = max(range(N_ACTIONS), key=lambda a: (eff_adv[a], -a))
                overestimate = max(0.0, eff_adv[cand] - actual_adv[cand])
                if overestimate > episode_errors[mode][seed]:
                    episode_errors[mode][seed] = overestimate

            cal_anchor_details.append({
                "anchor_id": aid, "episode_seed": seed, "policy_action": inc,
                "actual_advantages": actual_adv,
            })

        for mode in _MODES:
            err_list = [episode_errors[mode][s] for s in calibration]
            radius = 0.0 if mode == "uncalibrated" else _quantile(err_list, config.quantile)
            summary[mode] = {
                "episode_count": len(calibration),
                "anchor_count": len(cal_incumbents),
                "error_radius": radius,
                "mean_error": fmean(err_list),
                "episode_max_errors": err_list,
            }

        if checkpoint_hash != hashlib.sha256(Path(core.path).read_bytes()).hexdigest() or model_hash != _model_digest(core):
            raise ValueError("Core changed during odyssey fitting")

        payload.pop("sha256", None)
        payload["calibration"] = summary
        payload["sha256"] = _digest(payload)
        atlas = cls(core, payload)
        receipt = {
            "schema": "poseidon-odyssey-fit-v1",
            "checkpoint_sha256": checkpoint_hash,
            "model_sha256": model_hash,
            "artifact_sha256": payload["sha256"],
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

    @classmethod
    def from_horizon(cls, horizon_atlas: HorizonAtlas, scarcity: float = 2.5) -> OdysseyAtlas:
        """Upgrade an existing fitted HorizonAtlas into an OdysseyAtlas."""
        h_artifact = horizon_atlas.artifact
        h_conf = h_artifact["config"]
        od_conf = OdysseyConfig(
            horizon=h_conf["horizon"],
            k_neighbors=h_conf["k_neighbors"],
            support_radius=h_conf["support_radius"],
            quantile=h_conf["quantile"],
            override_margin=h_conf["override_margin"],
            min_harvest_resource=h_conf["min_harvest_resource"],
            max_samples=h_conf["max_samples"],
            scarcity_levels=tuple(h_conf["scarcity_levels"]),
            scarcity=scarcity,
        )
        payload = copy.deepcopy(h_artifact)
        payload["schema"] = ODYSSEY_SCHEMA
        od_dict = asdict(od_conf)
        od_dict["scarcity_levels"] = list(od_conf.scarcity_levels)
        payload["config"] = od_dict
        payload["calibration"]["metric"] = _METRIC
        payload["limits"] = _LIMITS[:]
        payload.pop("sha256", None)
        payload["sha256"] = _digest(payload)
        return cls(horizon_atlas.core, payload)

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

        # Automatic episode boundary detection
        progress = obs[15]
        if progress == 0.0 or progress < self._last_progress:
            self.reset()
        self._last_progress = progress
        self._step_counter += 1

        # Update cognitive map
        curr_patch = self.cognitive_map.update(obs, tick=self._step_counter)

        # Core policy evaluation
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

        # Depletion trap detection
        food, water, shelter = obs[6], obs[7], obs[8]
        threat, energy, hyd, stam, exposure = obs[5], obs[1], obs[2], obs[3], obs[4]
        weather_sev = obs[9]

        depletion_trap = False
        if incumbent == 1 and food < self.config.min_harvest_resource:
            depletion_trap = True
            adv_scores[1] = -0.5
        elif incumbent == 2 and water < self.config.min_harvest_resource:
            depletion_trap = True
            adv_scores[2] = -0.5

        # Waypoint evaluation from cognitive map
        waypoints = self.cognitive_map.evaluate_waypoints(obs)
        best_wp = waypoints[0] if waypoints and waypoints[0]["score"] > self.config.waypoint_threshold else None

        pretransit_rest = False
        storm_shelter = False
        nav_bias = [0.0] * N_ACTIONS

        # Pre-transit rest guard: need to travel to waypoint, but stamina is inadequate
        if best_wp and (depletion_trap or min(energy, hyd) < 0.50):
            if stam < self.config.min_stamina_for_transit and threat <= 0.68 and min(energy, hyd) > 0.12:
                pretransit_rest = True
                nav_bias[0] += 0.35
            else:
                # Directional macro action towards goal
                if exposure > 0.60 or weather_sev > 0.60:
                    nav_bias[5] += 0.30  # flee towards shelter
                else:
                    nav_bias[4] += 0.25 + 0.15 * min(1.0, best_wp["score"])

        # Storm shelter guard
        if exposure > 0.70 and weather_sev > 0.60 and shelter > 0.50:
            storm_shelter = True
            nav_bias[3] += 0.40

        # Integrate cognitive navigational bias into candidate advantages
        for a in range(N_ACTIONS):
            adv_scores[a] += nav_bias[a]

        # Vital state filter
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
            if pretransit_rest:
                chosen = proposal
                reason = "pretransit_rest_override"
            elif storm_shelter:
                chosen = proposal
                reason = "storm_shelter_override"
            elif depletion_trap:
                chosen = proposal
                reason = "depletion_waypoint_override" if best_wp else "depletion_override"
            elif not supported:
                reason = "outside_fitted_support"
            elif margin > self.config.override_margin:
                chosen = proposal
                reason = "advantage_confirmed"
            else:
                reason = "insufficient_margin"

        # Record chosen action into cognitive map for transition tracking
        self.cognitive_map.record_action(chosen)

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
            "backend": "odyssey-atlas-v5",
            "horizon": self.config.horizon,
            "candidates": candidates,
            "artifact_sha256": self._payload["sha256"],
            "cognitive_map": self.cognitive_map.state_summary(),
            "best_waypoint": best_wp,
            "pretransit_rest": pretransit_rest,
            "storm_shelter": storm_shelter,
        }

    def act(self, observation: Sequence[float]) -> int:
        return self.plan(observation)["action"]

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _canonical(self._payload).encode()
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ValueError("Odyssey artifact exceeds size limit")
        target.write_bytes(content)
        return str(target)

    @classmethod
    def load(cls, core: CoreRuntime, path: str | Path) -> OdysseyAtlas:
        target = Path(path)
        if not target.is_file():
            raise ValueError(f"Odyssey artifact does not exist: {target}")
        raw = target.read_bytes()
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError("Odyssey artifact exceeds size limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Corrupt Odyssey artifact") from error
        return cls(core, payload)
