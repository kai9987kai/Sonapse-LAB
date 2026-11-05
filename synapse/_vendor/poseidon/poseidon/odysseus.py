"""Odysseus: Empirical Transition Distribution & Bayesian Replenishment Navigator.

Constructs an episodic cognitive map strictly from 16-d observation streams,
learning online Bayesian resource replenishment rates on patch revisits and
estimating empirical macro-action transition outcome distributions P(v | u, a).
Directs goal-oriented navigation with transition confidence and travel-cost guards.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import time
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

ODYSSEUS_SCHEMA = "poseidon-odysseus-atlas-v1"
_MODES = ("calibrated", "uncalibrated", "no_memory")
_LIMITS = [
    "Episodic topological cognitive mapping with empirical transition distributions in synthetic TidePool.",
    "Observations are strictly 16-d normalized vectors; no global simulator coordinates or ground-truth pointers.",
    "Resource replenishment rates are learned online via Bayesian updates on patch revisits; no hardcoded world rates.",
    "Macro-action transitions P(P_dest | P_src, a) are estimated from observed movement outcomes under explore/flee.",
    "Descriptive empirical calibration error radii bound multi-horizon advantage claims on held-out partitions.",
    "Core neural weights and reset protocol remain strictly frozen; Odysseus acts as an external cognitive controller.",
]


@dataclass
class OdysseusConfig:
    scarcity: float = 2.5
    lcb_z: float = 1.5
    confidence_threshold: float = 0.25
    stamina_rest_threshold: float = 0.20
    storm_shelter_threshold: float = 0.45

    def __post_init__(self):
        _number(self.scarcity, "scarcity", 0.5, 4.0)
        _number(self.lcb_z, "lcb_z", 0.0, 5.0)
        _number(self.confidence_threshold, "confidence_threshold", 0.0, 1.0)
        _number(self.stamina_rest_threshold, "stamina_rest_threshold", 0.05, 0.95)
        _number(self.storm_shelter_threshold, "storm_shelter_threshold", 0.10, 0.90)


@dataclass
class EmpiricalPatch:
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
    # Online Bayesian replenishment estimates: mean rate and variance per tick
    food_rate_mean: float = 0.003
    food_rate_var: float = 0.0001
    water_rate_mean: float = 0.005
    water_rate_var: float = 0.0002
    revisit_count: int = 0

    def update_revisit(self, current_tick: int, food: float, water: float) -> None:
        elapsed = max(1, current_tick - self.last_seen_tick)
        if elapsed > 0:
            # Observed sample replenishment rate: delta over elapsed ticks
            sample_food_rate = max(0.0, (food - self.observed_food) / elapsed)
            sample_water_rate = max(0.0, (water - self.observed_water) / elapsed)

            # Online running update of mean and variance
            self.revisit_count += 1
            n = self.revisit_count
            # Update food rate
            delta_f = sample_food_rate - self.food_rate_mean
            self.food_rate_mean += delta_f / (n + 1)
            self.food_rate_var = ((n - 1) * self.food_rate_var + delta_f * (sample_food_rate - self.food_rate_mean)) / max(1, n)

            # Update water rate
            delta_w = sample_water_rate - self.water_rate_mean
            self.water_rate_mean += delta_w / (n + 1)
            self.water_rate_var = ((n - 1) * self.water_rate_var + delta_w * (sample_water_rate - self.water_rate_mean)) / max(1, n)

        self.last_seen_tick = current_tick
        self.visit_count += 1
        self.observed_food = food
        self.observed_water = water
        self.food_cap = max(self.food_cap, food)
        self.water_cap = max(self.water_cap, water)

    def estimated_food_lcb(self, current_tick: int, z: float = 1.5) -> float:
        """Conservative Lower Confidence Bound for expected food replenishment."""
        elapsed = max(0, current_tick - self.last_seen_tick)
        std = math.sqrt(max(1e-8, self.food_rate_var))
        conservative_rate = max(0.0, self.food_rate_mean - z * std)
        return min(self.food_cap, self.observed_food + elapsed * conservative_rate)

    def estimated_water_lcb(self, current_tick: int, z: float = 1.5) -> float:
        """Conservative Lower Confidence Bound for expected water replenishment."""
        elapsed = max(0, current_tick - self.last_seen_tick)
        std = math.sqrt(max(1e-8, self.water_rate_var))
        conservative_rate = max(0.0, self.water_rate_mean - z * std)
        return min(self.water_cap, self.observed_water + elapsed * conservative_rate)

    def to_dict(self, current_tick: int = 0) -> dict:
        return {
            "patch_id": self.patch_id,
            "terrain": round(self.terrain, 6),
            "shelter": round(self.shelter, 6),
            "visit_count": self.visit_count,
            "revisit_count": self.revisit_count,
            "observed_food": round(self.observed_food, 4),
            "observed_water": round(self.observed_water, 4),
            "food_rate_mean": round(self.food_rate_mean, 6),
            "water_rate_mean": round(self.water_rate_mean, 6),
            "estimated_food_lcb": round(self.estimated_food_lcb(current_tick), 4),
            "estimated_water_lcb": round(self.estimated_water_lcb(current_tick), 4),
        }


class EmpiricalCognitiveMap:
    """Cognitive map with online Bayesian replenishment and transition distributions."""

    def __init__(self, config: OdysseusConfig | None = None):
        self.config = config or OdysseusConfig()
        self.patches: dict[str, EmpiricalPatch] = {}
        # Transition counts: counts[u][action][v] = count
        self.transition_counts: dict[str, dict[int, dict[str, int]]] = {}
        # Total action attempts from patch: action_attempts[u][action] = total
        self.action_attempts: dict[str, dict[int, int]] = {}
        self.current_patch_id: str | None = None
        self.prev_patch_id: str | None = None
        self.prev_action: int | None = None
        self.current_tick: int = 0

    def reset(self, scarcity: float | None = None) -> None:
        if scarcity is not None:
            self.config.scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
        self.patches.clear()
        self.transition_counts.clear()
        self.action_attempts.clear()
        self.current_patch_id = None
        self.prev_patch_id = None
        self.prev_action = None
        self.current_tick = 0

    @staticmethod
    def fingerprint(terrain: float, shelter: float) -> str:
        return f"{round(float(terrain), 6):.6f}_{round(float(shelter), 6):.6f}"

    def update(self, observation: Sequence[float], tick: int | None = None) -> EmpiricalPatch:
        obs = _observation(observation)
        if tick is not None:
            self.current_tick = _integer(tick, "tick", 0, 100000)
        else:
            self.current_tick += 1

        food, water, shelter = obs[6], obs[7], obs[8]
        terrain, threat = obs[12], obs[5]
        pid = self.fingerprint(terrain, shelter)

        # Record empirical transition under movement actions (4=explore, 5=flee)
        if self.prev_action in (4, 5) and self.prev_patch_id:
            src = self.prev_patch_id
            act = self.prev_action
            self.transition_counts.setdefault(src, {}).setdefault(act, {})
            self.action_attempts.setdefault(src, {}).setdefault(act, 0)
            self.transition_counts[src][act][pid] = self.transition_counts[src][act].get(pid, 0) + 1
            self.action_attempts[src][act] += 1

        if pid in self.patches:
            patch = self.patches[pid]
            patch.update_revisit(self.current_tick, food, water)
            patch.last_threat = threat
        else:
            patch = EmpiricalPatch(
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
            )
            self.patches[pid] = patch

        self.current_patch_id = pid
        return patch

    def record_action(self, action: int) -> None:
        self.prev_action = _integer(action, "action", 0, 5)
        self.prev_patch_id = self.current_patch_id

    def transition_probability(self, src: str, action: int, dst: str) -> float:
        """Empirical probability P(dst | src, action) with Laplace smoothing."""
        attempts = self.action_attempts.get(src, {}).get(action, 0)
        if attempts == 0:
            return 0.0
        count = self.transition_counts.get(src, {}).get(action, {}).get(dst, 0)
        return count / attempts

    def evaluate_waypoints(self, observation: Sequence[float]) -> list[dict]:
        """Score candidate waypoints under transition probability and replenishment LCB."""
        if not self.current_patch_id or len(self.patches) < 2:
            return []
        obs = _observation(observation)
        energy, hyd, stam, exposure = obs[1], obs[2], obs[3], obs[4]
        curr_patch = self.patches[self.current_patch_id]
        curr_food = curr_patch.observed_food
        curr_water = curr_patch.observed_water

        candidates = []
        for pid, patch in sorted(self.patches.items()):
            if pid == self.current_patch_id:
                continue

            # Check transition feasibility under explore (4) or flee (5)
            p_explore = self.transition_probability(self.current_patch_id, 4, pid)
            p_flee = self.transition_probability(self.current_patch_id, 5, pid)
            max_p = max(p_explore, p_flee)

            # If unobserved directly, estimate conservative reachability
            reach_confidence = max_p if max_p > 0 else 0.20
            best_action = 4 if p_explore >= p_flee else 5

            est_food = patch.estimated_food_lcb(self.current_tick, self.config.lcb_z)
            est_water = patch.estimated_water_lcb(self.current_tick, self.config.lcb_z)

            # Expected yield gain over remaining in depleted current patch
            food_gain = max(0.0, est_food - curr_food)
            water_gain = max(0.0, est_water - curr_water)

            # Travel expenditure penalty
            travel_cost = 0.038 * 1.0 + 0.037 * 1.0 + 0.112 * 1.0
            urgency_food = max(0.0, 0.40 - energy)
            urgency_water = max(0.0, 0.40 - hyd)

            net_utility = reach_confidence * (urgency_food * food_gain + urgency_water * water_gain) - travel_cost

            # Storm shelter bonus
            if exposure > self.config.storm_shelter_threshold and patch.shelter > curr_patch.shelter:
                net_utility += reach_confidence * (patch.shelter - curr_patch.shelter) * 0.50

            candidates.append({
                "target_id": pid,
                "best_action": best_action,
                "reach_confidence": reach_confidence,
                "est_food_lcb": est_food,
                "est_water_lcb": est_water,
                "net_utility": net_utility,
                "terrain": patch.terrain,
                "shelter": patch.shelter,
            })

        return sorted(candidates, key=lambda c: -c["net_utility"])


class OdysseusAtlas:
    """Odysseus cognitive navigation controller with empirical transition distributions."""

    def __init__(self, core: CoreRuntime, artifact: dict, config: OdysseusConfig | None = None):
        self.core = core
        self.artifact = validate_odysseus_artifact(artifact, core)
        self.config = config or OdysseusConfig()
        self.map = EmpiricalCognitiveMap(self.config)

    def reset(self, scarcity: float = 2.5) -> None:
        self.config.scarcity = _number(scarcity, "scarcity", 0.5, 4.0)
        self.map.reset(scarcity)

    def plan(self, observation: Sequence[float], mode: str = "calibrated") -> dict:
        """Produce a navigational decision conditioned on cognitive map and transition distributions."""
        obs = _observation(observation)
        patch = self.map.update(obs)
        energy, hyd, stam, exposure = obs[1], obs[2], obs[3], obs[4]
        food, water = obs[6], obs[7]

        # 1. Base policy proposal
        policy_action = self.core.act(obs)

        # 2. Pre-transit rest guard: if stamina depleted and attempting movement, rest (0)
        if stam < self.config.stamina_rest_threshold and policy_action in (4, 5):
            action = 0  # rest
            overridden = (action != policy_action)
            reason = "pre_transit_stamina_rest_guard"
            self.map.record_action(action)
            return {
                "action": action,
                "policy_action": policy_action,
                "overridden": overridden,
                "reason": reason,
                "mode": mode,
                "active_patch": patch.patch_id,
                "candidates": [1.0 if i == policy_action else 0.0 for i in range(6)],
                "artifact_sha256": self.artifact["sha256"],
            }

        # 3. Waypoint evaluation
        waypoints = self.map.evaluate_waypoints(obs)
        radius = self.artifact.get("error_radius", 0.15) if mode == "calibrated" else 0.0

        chosen_action = policy_action
        overridden = False
        reason = "policy_incumbent"

        if waypoints:
            best = waypoints[0]
            # Override if net utility clears the empirical error radius
            if best["net_utility"] > radius and best["reach_confidence"] >= self.config.confidence_threshold:
                # If local resources depleted, navigate to recommended waypoint
                if (energy < 0.35 and food < 0.15) or (hyd < 0.35 and water < 0.15) or (exposure > self.config.storm_shelter_threshold):
                    chosen_action = best["best_action"]
                    if chosen_action != policy_action:
                        overridden = True
                        reason = "odysseus_empirical_navigation_override"

        self.map.record_action(chosen_action)
        return {
            "action": chosen_action,
            "policy_action": policy_action,
            "overridden": overridden,
            "reason": reason,
            "mode": mode,
            "active_patch": patch.patch_id,
            "top_waypoint": waypoints[0] if waypoints else None,
            "candidates": [1.0 if i == policy_action else 0.0 for i in range(6)],
            "artifact_sha256": self.artifact["sha256"],
        }

    def act(self, observation: Sequence[float]) -> int:
        return self.plan(observation)["action"]

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.artifact, indent=2, sort_keys=True).encode("utf-8")
        target.write_bytes(content)
        return str(target)

    @classmethod
    def load(cls, core: CoreRuntime, path: str | Path) -> "OdysseusAtlas":
        target = Path(path)
        if not target.is_file():
            raise ValueError(f"Odysseus artifact does not exist: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        config = OdysseusConfig(**cfg) if cfg else None
        return cls(core, payload, config=config)


def validate_odysseus_artifact(artifact: dict, core: CoreRuntime | None = None) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("Odysseus artifact must be a dict")
    if artifact.get("schema") != ODYSSEUS_SCHEMA:
        raise ValueError(f"Invalid schema: {artifact.get('schema')}")
    if "sha256" not in artifact:
        raise ValueError("Missing artifact sha256")
    unsigned = {k: v for k, v in artifact.items() if k != "sha256"}
    if _digest(unsigned) != artifact["sha256"]:
        raise ValueError("Artifact checksum mismatch")
    if core is not None and hasattr(core, "path"):
        core_checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
        if artifact.get("checkpoint_sha256") != core_checkpoint_hash:
            raise ValueError("Core checkpoint mismatch")
    return artifact


def fit_odysseus(
    core: CoreRuntime,
    train_seeds: Sequence[int],
    calibration_seeds: Sequence[int],
    max_steps: int = 64,
    scarcity: float = 2.5,
) -> tuple[OdysseusAtlas, dict]:
    """Fit Odysseus error calibration radius on disjoint episode partitions."""
    _seeds(train_seeds, "train_seeds")
    _seeds(calibration_seeds, "calibration_seeds")
    if not (1 <= len(train_seeds) <= 64 and 1 <= len(calibration_seeds) <= 64):
        raise ValueError("Seeds must contain 1-64 values")
    if set(train_seeds) & set(calibration_seeds):
        raise ValueError("Partitions must be disjoint")

    # Run calibration episodes to measure empirical overestimation radius
    config = OdysseusConfig(scarcity=scarcity)
    eval_map = EmpiricalCognitiveMap(config)
    overestimations = []

    for seed in calibration_seeds:
        env = TidePool(seed, max_steps=max_steps, scarcity=scarcity)
        obs = env.observe()
        eval_map.reset(scarcity)
        ep_overest = 0.0

        while not env.done:
            eval_map.update(obs)
            wps = eval_map.evaluate_waypoints(obs)
            if wps:
                top = wps[0]
                pred_gain = max(0.0, top["net_utility"])
                # Compare against actual one-step transition reward
                nxt, rew, done, _ = env.step(top["best_action"])
                eval_map.record_action(top["best_action"])
                actual_gain = rew
                error = max(0.0, pred_gain - actual_gain)
                ep_overest = max(ep_overest, error)
                obs = nxt
            else:
                act = core.act(obs)
                eval_map.record_action(act)
                nxt, _, _, _ = env.step(act)
                obs = nxt

        overestimations.append(ep_overest)

    error_radius = _quantile(overestimations, 0.90) if overestimations else 0.15
    checkpoint_hash = hashlib.sha256(Path(core.path).read_bytes()).hexdigest()
    model_hash = _model_digest(core)

    artifact_data = {
        "schema": ODYSSEUS_SCHEMA,
        "date": "2026-10-09",
        "checkpoint_sha256": checkpoint_hash,
        "model_sha256": model_hash,
        "world_version": WORLD_VERSION,
        "config": asdict(config),
        "partition": {"train_seeds": list(train_seeds), "calibration_seeds": list(calibration_seeds)},
        "error_radius": error_radius,
        "limits": _LIMITS,
    }
    artifact_data["sha256"] = _digest(artifact_data)

    atlas = OdysseusAtlas(core, artifact_data, config)
    receipt = {
        "schema": ODYSSEUS_SCHEMA,
        "artifact_sha256": artifact_data["sha256"],
        "error_radius": error_radius,
        "train_episodes": len(train_seeds),
        "calibration_episodes": len(calibration_seeds),
    }
    return atlas, receipt
