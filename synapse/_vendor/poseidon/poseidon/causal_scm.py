"""POSEIDON CAUSAL-SCM: Structural Causal Model & Pearl's do-Calculus Engine.

Inspired by Causeway and Pearl's Causal Hierarchy:
1. Structural Causal Model (SCM):
   Decomposes state transitions into an explicit Structural Causal Graph:
     Exogenous Variables U: {weather_noise, scarcity_pressure, terrain_hazard}
     Endogenous State X: {position, vitals, resource_patches}
     Action Intervention: do(a_t)
   where vitals update according to structural equations:
     vitals_{t+1} := f_{vitals}(vitals_t, do(a_t), patch_t, U_t) + \\epsilon

2. Pearl's do-Calculus Interventional Queries:
   Evaluates P(s_{t+1} | do(a_t), s_t), distinguishing correlation from true causation.
   Prevents spurious associations (e.g. associating rainfall with agent action).

3. Counterfactual Regret Attribution:
   For any realized outcome (s_t, a_t, r_t, s_{t+1}):
   Computes counterfactual outcomes s_{t+1}^{a'} under identical noise realization U:
     Regret(a, a') = Value(s_{t+1}^{a'}) - Value(s_{t+1}^a)
   Isolates whether failure was caused by agent policy or unpreventable environmental shock.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass(frozen=True)
class CausalGraphNode:
    name: str
    parents: tuple[str, ...]
    is_exogenous: bool = False


# Canonical Causal DAG for TidePool dynamics
CAUSAL_DAG: dict[str, tuple[str, ...]] = {
    # Exogenous nodes (uncontrolled environment drivers)
    "weather_noise": (),
    "scarcity_pressure": (),
    "terrain_hazard": (),
    # Endogenous state
    "position": ("action", "terrain_hazard"),
    "energy": ("action", "energy", "weather_noise"),
    "hydration": ("action", "hydration", "weather_noise"),
    "health": ("action", "energy", "hydration", "weather_noise", "terrain_hazard"),
    "stamina": ("action", "stamina", "terrain_hazard"),
    "threat": ("action", "position", "weather_noise"),
}


@dataclass
class CausalAttribution:
    action: int
    action_name: str
    direct_causal_effect: float
    environmental_shock: float
    counterfactual_regrets: dict[int, float]
    primary_causal_driver: str
    preventable_by_policy: bool


class StructuralCausalModel:
    """Explicit Structural Causal Model for the TidePool POMDP domain."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # Vital channel indices in standard 16-D observation
        self.idx_health = 0
        self.idx_energy = 1
        self.idx_hydration = 2
        self.idx_stamina = 3
        self.idx_exposure = 4
        self.idx_threat = 5
        self.idx_food_local = 6
        self.idx_water_local = 7
        self.idx_shelter_local = 8

    def estimate_exogenous_noise(
        self,
        obs: Sequence[float],
        next_obs: Sequence[float],
        action: int,
    ) -> dict[str, float]:
        """Infers exogenous noise realization U_t that produced transition from obs to next_obs."""
        o = list(obs)
        n = list(next_obs)

        # Expected baseline action metabolic impacts
        expected_energy_drain = 0.02
        expected_hydration_drain = 0.025
        if action == 1:  # forage
            expected_energy_drain = -max(0.0, o[self.idx_food_local] * 0.4)
        elif action == 2:  # drink
            expected_hydration_drain = -max(0.0, o[self.idx_water_local] * 0.5)
        elif action == 0:  # rest
            expected_energy_drain = 0.005
            expected_hydration_drain = 0.005

        realized_energy_delta = n[self.idx_energy] - o[self.idx_energy]
        realized_hydration_delta = n[self.idx_hydration] - o[self.idx_hydration]
        realized_health_delta = n[self.idx_health] - o[self.idx_health]

        # Exogenous shocks represent environmental divergence from expected physics
        weather_shock = max(0.0, n[self.idx_exposure] - o[self.idx_exposure])
        threat_shock = max(0.0, n[self.idx_threat] - o[self.idx_threat])
        unmodeled_health_shock = -(realized_health_delta) if realized_health_delta < -0.05 else 0.0

        return {
            "weather_shock": float(weather_shock),
            "threat_shock": float(threat_shock),
            "unmodeled_health_shock": float(unmodeled_health_shock),
            "energy_residual": float(realized_energy_delta - (-expected_energy_drain)),
            "hydration_residual": float(realized_hydration_delta - (-expected_hydration_drain)),
        }

    def interventional_prediction(
        self,
        obs: Sequence[float],
        do_action: int,
        exogenous_noise: dict[str, float] | None = None,
    ) -> list[float]:
        """Computes structural do-calculus forward projection: E[s_{t+1} | do(a), s_t, U]."""
        if not self.enabled:
            return list(obs)
        s = list(obs)
        u = exogenous_noise or {"weather_shock": 0.0, "threat_shock": 0.0, "energy_residual": 0.0, "hydration_residual": 0.0}
        pred = list(s)

        # 1. Structural equation for Energy:
        if do_action == 1:  # forage
            gain = s[self.idx_food_local] * 0.45
            cost = 0.015
            pred[self.idx_energy] = min(1.0, s[self.idx_energy] + gain - cost + u.get("energy_residual", 0.0) * 0.5)
            pred[self.idx_food_local] = max(0.0, s[self.idx_food_local] * 0.4)
        elif do_action == 0:  # rest
            pred[self.idx_energy] = max(0.0, s[self.idx_energy] - 0.008)
        else:
            cost = 0.022
            pred[self.idx_energy] = max(0.0, s[self.idx_energy] - cost + u.get("energy_residual", 0.0) * 0.5)

        # 2. Structural equation for Hydration:
        if do_action == 2:  # drink
            gain = s[self.idx_water_local] * 0.50
            cost = 0.010
            pred[self.idx_hydration] = min(1.0, s[self.idx_hydration] + gain - cost + u.get("hydration_residual", 0.0) * 0.5)
            pred[self.idx_water_local] = max(0.0, s[self.idx_water_local] * 0.3)
        elif do_action == 0:  # rest
            pred[self.idx_hydration] = max(0.0, s[self.idx_hydration] - 0.010)
        else:
            cost = 0.025
            pred[self.idx_hydration] = max(0.0, s[self.idx_hydration] - cost + u.get("hydration_residual", 0.0) * 0.5)

        # 3. Structural equation for Stamina:
        if do_action == 0:  # rest
            pred[self.idx_stamina] = min(1.0, s[self.idx_stamina] + 0.15)
        elif do_action in (4, 5):  # explore or flee
            pred[self.idx_stamina] = max(0.0, s[self.idx_stamina] - 0.06)
        else:
            pred[self.idx_stamina] = max(0.0, s[self.idx_stamina] - 0.02)

        # 4. Structural equation for Exposure / Weather:
        if do_action == 3:  # shelter
            pred[self.idx_exposure] = max(0.0, s[self.idx_exposure] - 0.25)
        else:
            pred[self.idx_exposure] = min(1.0, s[self.idx_exposure] + u.get("weather_shock", 0.0))

        # 5. Structural equation for Health:
        starvation = max(0.0, 0.10 - pred[self.idx_energy]) * 2.0
        dehydration = max(0.0, 0.10 - pred[self.idx_hydration]) * 2.5
        exposure_dmg = max(0.0, pred[self.idx_exposure] - 0.70) * 1.5
        threat_dmg = pred[self.idx_threat] * 0.5

        damage = starvation + dehydration + exposure_dmg + threat_dmg
        pred[self.idx_health] = max(0.0, min(1.0, s[self.idx_health] - damage))

        return pred

    def attribute_causality(
        self,
        obs: Sequence[float],
        action_taken: int,
        realized_next_obs: Sequence[float],
    ) -> CausalAttribution:
        """Computes Pearlian Counterfactual Regret Attribution for an executed transition."""
        if not self.enabled:
            return CausalAttribution(
                action=action_taken,
                action_name=ACTIONS[action_taken],
                direct_causal_effect=0.0,
                environmental_shock=0.0,
                counterfactual_regrets={a: 0.0 for a in range(N_ACTIONS)},
                primary_causal_driver="exogenous",
                preventable_by_policy=False,
            )
        u = self.estimate_exogenous_noise(obs, realized_next_obs, action_taken)

        cf_outcomes = {}
        cf_values = {}
        for a in range(N_ACTIONS):
            cf_state = self.interventional_prediction(obs, do_action=a, exogenous_noise=u)
            cf_outcomes[a] = cf_state
            cf_val = cf_state[self.idx_health] + 0.5 * min(cf_state[self.idx_energy], cf_state[self.idx_hydration])
            cf_values[a] = cf_val

        realized_val = realized_next_obs[self.idx_health] + 0.5 * min(
            realized_next_obs[self.idx_energy], realized_next_obs[self.idx_hydration]
        )

        regrets = {a: float(cf_values[a] - realized_val) for a in range(N_ACTIONS)}
        best_cf_action = max(regrets.keys(), key=lambda a: regrets[a])
        max_regret = regrets[best_cf_action]

        env_shock_severity = u["weather_shock"] + u["threat_shock"] + u["unmodeled_health_shock"]
        preventable = max_regret > 0.05 and (max_regret > env_shock_severity * 0.5)

        if env_shock_severity > 0.30 and not preventable:
            driver = "exogenous_environmental_shock"
        elif realized_next_obs[self.idx_hydration] < 0.15:
            driver = "endogenous_dehydration"
        elif realized_next_obs[self.idx_energy] < 0.15:
            driver = "endogenous_starvation"
        else:
            driver = "controlled_metabolism"

        return CausalAttribution(
            action=action_taken,
            action_name=ACTIONS[action_taken],
            direct_causal_effect=float(realized_val - cf_values[0]),
            environmental_shock=float(env_shock_severity),
            counterfactual_regrets=regrets,
            primary_causal_driver=driver,
            preventable_by_policy=preventable,
        )
