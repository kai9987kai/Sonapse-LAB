"""Project PROMETHEUS Discovery Lab: Intrinsic Curiosity & Controllability Engine.

Implements the CASD (Causal Adaptive Skill Discovery) composite reward objective:
    r_t = w_g * G_t + w_i * I_t + w_c * C_t - w_a * A_t - w_f * F_t

Where:
    G_t: Progress toward environment-level goal (reaching opposite bank / new territory)
    I_t: Information gain / learning progress (epistemic uncertainty reduction)
    C_t: Controllability gain (state change resulting directly from agent action)
    A_t: Metabolic & action effort cost
    F_t: Failure penalty (falling in water, sinking, structural collapse)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CuriosityDecomposition:
    total_composite_reward: float
    goal_progress_g: float
    information_gain_i: float
    controllability_c: float
    action_cost_a: float
    failure_penalty_f: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_composite_reward": round(float(self.total_composite_reward), 4),
            "goal_progress_g": round(float(self.goal_progress_g), 4),
            "information_gain_i": round(float(self.information_gain_i), 4),
            "controllability_c": round(float(self.controllability_c), 4),
            "action_cost_a": round(float(self.action_cost_a), 4),
            "failure_penalty_f": round(float(self.failure_penalty_f), 4),
        }


class DiscoveryCuriosityEngine:
    """Computes non-trivial intrinsic curiosity rewards based on learning progress and object control."""

    def __init__(
        self,
        weight_goal: float = 1.0,
        weight_info: float = 0.4,
        weight_control: float = 0.3,
        weight_action_cost: float = 0.02,
        weight_failure: float = 1.5,
        state_dim: int = 24,
    ):
        self.w_g = float(weight_goal)
        self.w_i = float(weight_info)
        self.w_c = float(weight_control)
        self.w_a = float(weight_action_cost)
        self.w_f = float(weight_failure)
        self.state_dim = state_dim

        # Running novelty and density estimators
        self.visited_states: List[np.ndarray] = []
        self.prediction_errors: List[float] = []

    def compute_reward(
        self,
        prev_obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        model_epistemic_variance: float,
        model_prediction_error: float,
        env_reward: float,
        info: Dict[str, Any],
    ) -> CuriosityDecomposition:
        """Evaluates r_t across the 5 CASD terms."""
        # 1. Goal Progress G_t: Forward movement across the river channel toward right bank
        delta_x = float(next_obs[0] - prev_obs[0])
        g_t = max(0.0, delta_x * 2.0)
        if info.get("reached_goal", False):
            g_t += 5.0

        # 2. Information Gain I_t: Epistemic uncertainty that can be resolved
        # Modeled as bounded epistemic variance * prediction error (Plan2Explore principle)
        i_t = float(min(2.0, math.sqrt(max(0.0, model_epistemic_variance)) * (1.0 + model_prediction_error)))

        # 3. Controllability C_t: Did the agent's action directly change object or environment state?
        # Nonzero displacement of nearby objects or successful connection/catalysis
        c_t = 0.0
        event = info.get("event", "")
        if "grasped" in event:
            c_t += 0.3
        elif "released" in event:
            c_t += 0.4
        elif "pushed" in event:
            c_t += 0.3
        elif "connected" in event:
            c_t += 0.8
        elif "catalytic_curing" in event:
            c_t += 1.2

        # 4. Action Cost A_t: Energy/metabolic expenditure
        a_t = 0.05 if action in (5, 6, 7, 9) else 0.02

        # 5. Failure Penalty F_t: Falling into deep water or severe damage
        f_t = 0.0
        if "fell_in_deep_water" in event or "drowned" in event:
            f_t = 1.0

        total = (self.w_g * g_t) + (self.w_i * i_t) + (self.w_c * c_t) - (self.w_a * a_t) - (self.w_f * f_t)

        return CuriosityDecomposition(
            total_composite_reward=total,
            goal_progress_g=g_t,
            information_gain_i=i_t,
            controllability_c=c_t,
            action_cost_a=a_t,
            failure_penalty_f=f_t,
        )
