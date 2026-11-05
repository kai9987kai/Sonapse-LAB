"""Identity-gated residual control with observed directed topology.

TRIDENT is an opt-in TidePool controller. It scores residual-corrected
one-step futures from the frozen Tidal dynamics head, admits a travel action
only when that directed step has been observed from the current fingerprint,
and overrides the incumbent only when the intact residual history changes the
ranking relative to erased and time-shifted copies of the same history.

These are classical software calculations on synthetic observations. They are
not biological memory evidence, hardware navigation, or a trained replacement
for the Tidal policy.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Sequence

from .planning import evaluate_predicted_observation
from .world import ACTIONS, N_ACTIONS, OBS_SIZE

TRIDENT_SCHEMA = "poseidon-trident-controller-v1"
TRIDENT_MODES = ("intact", "erased", "shifted", "no_topology", "no_residual", "ungated")
TRAVEL_ACTIONS = frozenset({4, 5})
_LIMITS = (
    "Synthetic TidePool control; residual identity and observed topology are local software state.",
    "Directed travel is admitted only after an executed observation of that fingerprint-action pair, except during early discovery.",
    "Replenishment rates are estimated from return visits; sparse revisits abstain from waypoint scoring.",
    "Identity gating compares intact, erased and shifted residual copies at the same observation; it is not a causal theorem.",
    "The frozen Tidal checkpoint is unchanged and is never promoted by this controller.",
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fingerprint(observation: Sequence[float]) -> str:
    return f"{round(float(observation[12]), 6):.6f}_{round(float(observation[8]), 6):.6f}"


def _shift_residual(residual: list[float], offset: int = 5) -> list[float]:
    n = len(residual)
    k = offset % n
    return residual[-k:] + residual[:-k]


def _admissibility(observation: Sequence[float], minimum: float = 0.04) -> list[dict]:
    candidates = []
    for action, label in enumerate(ACTIONS):
        reason = None
        if action == 1 and observation[6] < minimum:
            reason = "depleted_food"
        elif action == 2 and observation[7] < minimum:
            reason = "depleted_water"
        elif action in TRAVEL_ACTIONS and (observation[3] < 0.12 or observation[1] < 0.08 or observation[2] < 0.08):
            reason = "insufficient_travel_reserves"
        candidates.append({"action": action, "label": label, "admissible": reason is None, "reason": reason})
    return candidates


@dataclass(frozen=True)
class TridentConfig:
    leak: float = 0.35
    override_margin: float = 0.05
    identity_margin: float = 0.02
    discovery_visits: int = 2
    min_replenish_samples: int = 2
    residual_horizon: int = 8
    min_harvest_resource: float = 0.04

    def __post_init__(self):
        if isinstance(self.leak, bool) or not isinstance(self.leak, (int, float)) or not 0 < self.leak <= 1:
            raise ValueError("leak must be in (0, 1]")
        for name in ("override_margin", "identity_margin", "min_harvest_resource"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if type(self.discovery_visits) is not int or not 1 <= self.discovery_visits <= 8:
            raise ValueError("discovery_visits must be an integer in [1, 8]")
        if type(self.min_replenish_samples) is not int or not 1 <= self.min_replenish_samples <= 16:
            raise ValueError("min_replenish_samples must be an integer in [1, 16]")
        if type(self.residual_horizon) is not int or not 2 <= self.residual_horizon <= 32:
            raise ValueError("residual_horizon must be an integer in [2, 32]")


class TridentController:
    """Causal residual identity + observed directed topology around frozen Tidal."""

    def __init__(self, core, config: TridentConfig | None = None, mode: str = "intact"):
        if mode not in TRIDENT_MODES:
            raise ValueError("Unknown TRIDENT mode")
        self.core = core
        self.config = config or TridentConfig()
        self.mode = mode
        self.reset()

    def reset(self, scarcity: float = 1.0, seed: int = 0, max_steps: int = 256) -> None:
        if isinstance(scarcity, bool) or not isinstance(scarcity, (int, float)) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4:
            raise ValueError("scarcity must be finite and between 0.5 and 4")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2^63")
        if type(max_steps) is not int or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be an integer in [1, 10000]")
        self.scarcity = float(scarcity)
        self.seed = seed
        self.max_steps = max_steps
        self.step = 0
        self._ewma = [0.0] * OBS_SIZE
        self._buffer: list[list[float]] = []
        self._patches: dict[str, dict] = {}
        self._transitions: dict[tuple[str, int], dict[str, int]] = {}
        self._current_fp: str | None = None
        self._pending_transition = None
        self.last_decision = None

    def _residual_for_mode(self) -> list[float]:
        if self.mode in ("erased", "no_residual"):
            return [0.0] * OBS_SIZE
        if self.mode == "shifted":
            return _shift_residual(self._ewma)
        return list(self._ewma)

    def _directed_support(self, fingerprint: str, action: int) -> tuple[bool, int]:
        counts = self._transitions.get((fingerprint, action), {})
        support = sum(counts.values())
        visits = int(self._patches.get(fingerprint, {}).get("visits", 0))
        if action not in TRAVEL_ACTIONS:
            return True, support
        if visits < self.config.discovery_visits:
            return True, support
        return support > 0, support

    def _estimated_yield(self, fingerprint: str) -> tuple[float | None, float | None]:
        patch = self._patches.get(fingerprint)
        if patch is None or patch["replenish_samples"] < self.config.min_replenish_samples:
            return None, None
        elapsed = max(0, self.step - patch["last_tick"])
        food_rate = patch["replenish_food_sum"] / patch["replenish_samples"]
        water_rate = patch["replenish_water_sum"] / patch["replenish_samples"]
        food = min(patch["food_cap"], patch["last_food"] + max(0.0, food_rate) * elapsed)
        water = min(patch["water_cap"], patch["last_water"] + max(0.0, water_rate) * elapsed)
        return food, water

    def _destination_yield(self, fingerprint: str, action: int) -> float | None:
        counts = self._transitions.get((fingerprint, action), {})
        total = sum(counts.values())
        if total <= 0:
            return None
        expected = 0.0
        used = 0
        for dest, count in counts.items():
            food, water = self._estimated_yield(dest)
            if food is None or water is None:
                continue
            expected += count * (food + water)
            used += count
        if used <= 0:
            return None
        return expected / used

    def _score(self, observation: Sequence[float], action: int, residual: Sequence[float]) -> float:
        predicted = self.core.predict_transition(observation, action)
        corrected = [_clamp(predicted[i] + residual[i]) for i in range(OBS_SIZE)]
        vitality = evaluate_predicted_observation(corrected, action)
        destination = self._destination_yield(_fingerprint(observation), action)
        if destination is not None and action in TRAVEL_ACTIONS:
            vitality += 0.35 * destination
        return vitality

    def plan(self, observation: Sequence[float]) -> dict:
        if self._pending_transition is not None:
            raise RuntimeError("observe_transition must record the previous action outcome before planning again")
        if len(observation) != OBS_SIZE:
            raise ValueError(f"Observation must have {OBS_SIZE} elements")
        obs = [float(x) for x in observation]
        if any(not math.isfinite(value) for value in obs):
            raise ValueError("Observation must be finite")
        fingerprint = _fingerprint(obs)
        policy = int(self.core.act(obs))
        residual = self._residual_for_mode()
        intact = list(self._ewma) if self.mode not in ("erased", "no_residual") else residual
        erased = [0.0] * OBS_SIZE
        shifted = _shift_residual(intact)
        candidates = _admissibility(obs, self.config.min_harvest_resource)
        scores = {"intact": [], "erased": [], "shifted": []}
        for action in range(N_ACTIONS):
            supported, support = self._directed_support(fingerprint, action)
            if self.mode != "no_topology" and not supported:
                candidates[action]["admissible"] = False
                candidates[action]["reason"] = "unobserved_directed_step"
            candidates[action]["directed_support"] = support
            candidates[action]["directed_supported"] = supported
            scores["intact"].append(self._score(obs, action, intact))
            scores["erased"].append(self._score(obs, action, erased))
            scores["shifted"].append(self._score(obs, action, shifted))
            candidates[action]["intact_score"] = round(scores["intact"][-1], 6)
            candidates[action]["erased_score"] = round(scores["erased"][-1], 6)
            candidates[action]["shifted_score"] = round(scores["shifted"][-1], 6)
        valid = [item["action"] for item in candidates if item["admissible"]]
        ranking = scores["intact"]
        if self.mode == "erased":
            ranking = scores["erased"]
        elif self.mode == "shifted":
            ranking = scores["shifted"]
        elif self.mode == "no_residual":
            ranking = scores["erased"]
        proposal = policy
        if valid:
            proposal = max(valid, key=lambda action: (ranking[action], action == policy, -action))
        remaining = self.max_steps - self.step
        intact_adv = ranking[proposal] - ranking[policy] if self.mode == "intact" else scores["intact"][proposal] - scores["intact"][policy]
        erased_adv = scores["erased"][proposal] - scores["erased"][policy]
        shifted_adv = scores["shifted"][proposal] - scores["shifted"][policy]
        chosen, reason = policy, None
        if remaining < 2:
            reason = "insufficient_remaining_horizon"
        elif proposal == policy:
            reason = "policy_preferred"
        elif not candidates[proposal]["admissible"]:
            reason = "inadmissible_action"
        elif self.mode == "ungated":
            chosen, reason = proposal, None
        elif intact_adv <= self.config.override_margin:
            reason = "insufficient_intact_margin"
        elif self.mode == "intact" and intact_adv <= erased_adv + self.config.identity_margin:
            reason = "identity_not_specific_vs_erased"
        elif self.mode == "intact" and intact_adv <= shifted_adv + self.config.identity_margin:
            reason = "identity_not_specific_vs_shifted"
        elif self.mode in ("erased", "shifted", "no_residual", "no_topology"):
            if ranking[proposal] - ranking[policy] > self.config.override_margin:
                chosen, reason = proposal, None
            else:
                reason = "insufficient_intact_margin"
        else:
            chosen, reason = proposal, None
        decision = {
            "schema": TRIDENT_SCHEMA,
            "action": int(chosen),
            "action_name": ACTIONS[chosen],
            "policy_action": policy,
            "proposed_action": int(proposal),
            "overridden": chosen != policy,
            "mode": self.mode,
            "fingerprint": fingerprint,
            "step": self.step,
            "remaining_steps": remaining,
            "fallback_reason": reason,
            "intact_advantage": round(float(intact_adv), 6),
            "erased_advantage": round(float(erased_adv), 6),
            "shifted_advantage": round(float(shifted_adv), 6),
            "identity_specific": bool(
                intact_adv > erased_adv + self.config.identity_margin
                and intact_adv > shifted_adv + self.config.identity_margin
            ),
            "residual_norm": round(math.sqrt(sum(value * value for value in residual)), 6),
            "patch_count": len(self._patches),
            "directed_edges": sum(sum(counts.values()) for counts in self._transitions.values()),
            "candidates": candidates,
            "limits": list(_LIMITS),
            "backend": "trident-identity-directed-v1",
        }
        self._pending_transition = {"observation": obs, "action": int(chosen), "fingerprint": fingerprint}
        self.last_decision = copy.deepcopy(decision)
        return decision

    def observe_transition(self, observation, action, reward, next_observation, info=None) -> None:
        pending = self._pending_transition
        if pending is None:
            raise RuntimeError("plan must precede observe_transition")
        if int(action) != pending["action"]:
            raise ValueError("Observed action does not match the pending TRIDENT decision")
        before = [float(x) for x in observation]
        after = [float(x) for x in next_observation]
        if before != pending["observation"]:
            raise ValueError("Observed observation does not match the pending TRIDENT decision")
        if len(after) != OBS_SIZE or any(not math.isfinite(value) for value in after):
            raise ValueError("Next observation must be a finite 16-vector")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("Reward must be a finite number")
        predicted = self.core.predict_transition(before, int(action))
        residual = [after[i] - predicted[i] for i in range(OBS_SIZE)]
        leak = self.config.leak
        self._ewma = [(1.0 - leak) * self._ewma[i] + leak * residual[i] for i in range(OBS_SIZE)]
        self._buffer.append(residual)
        if len(self._buffer) > self.config.residual_horizon:
            self._buffer.pop(0)
        origin = pending["fingerprint"]
        destination = _fingerprint(after)
        if origin not in self._patches:
            self._patches[origin] = {
                "visits": 0, "last_tick": self.step, "last_food": before[6], "last_water": before[7],
                "food_on_leave": before[6], "water_on_leave": before[7],
                "food_cap": before[6], "water_cap": before[7],
                "replenish_food_sum": 0.0, "replenish_water_sum": 0.0, "replenish_samples": 0,
            }
        patch = self._patches[origin]
        patch["visits"] += 1
        patch["last_food"] = before[6]
        patch["last_water"] = before[7]
        patch["food_cap"] = max(patch["food_cap"], before[6])
        patch["water_cap"] = max(patch["water_cap"], before[7])
        patch["last_tick"] = self.step
        if destination != origin:
            patch["food_on_leave"] = before[6]
            patch["water_on_leave"] = before[7]
            key = (origin, int(action))
            self._transitions.setdefault(key, {})
            self._transitions[key][destination] = self._transitions[key].get(destination, 0) + 1
            if destination in self._patches:
                returned = self._patches[destination]
                elapsed = max(1, self.step - returned["last_tick"])
                returned["replenish_food_sum"] += (after[6] - returned["food_on_leave"]) / elapsed
                returned["replenish_water_sum"] += (after[7] - returned["water_on_leave"]) / elapsed
                returned["replenish_samples"] += 1
        if destination not in self._patches:
            self._patches[destination] = {
                "visits": 0, "last_tick": self.step + 1, "last_food": after[6], "last_water": after[7],
                "food_on_leave": after[6], "water_on_leave": after[7],
                "food_cap": after[6], "water_cap": after[7],
                "replenish_food_sum": 0.0, "replenish_water_sum": 0.0, "replenish_samples": 0,
            }
        else:
            dest = self._patches[destination]
            dest["last_tick"] = self.step + 1
            dest["last_food"] = after[6]
            dest["last_water"] = after[7]
            dest["food_cap"] = max(dest["food_cap"], after[6])
            dest["water_cap"] = max(dest["water_cap"], after[7])
        self._current_fp = destination
        self.step += 1
        self._pending_transition = None
        if info is not None and not isinstance(info, dict):
            raise ValueError("info must be a mapping when provided")

    def act(self, observation):
        return self.plan(observation)["action"]
