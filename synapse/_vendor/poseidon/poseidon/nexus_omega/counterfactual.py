"""PROJECT NEXUS-Ω: Pearl Level 3 Counterfactual SCM & Regret Engine.

Implements structural causal abduction, noise inversion, counterfactual
interventions do(A = a'), and policy credit/blame attribution.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import (
    CounterfactualQuery,
    CounterfactualResult,
)


class CounterfactualEngine:
    """Pearl Level 3 Counterfactual Structural Causal Model for POMDP Worlds."""

    def __init__(self, action_dim: int = 6, obs_dim: int = 16):
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    def predict_factual(self, state: np.ndarray, action: int) -> np.ndarray:
        """Computes expected physical transition f(s, a)."""
        s = state.copy()
        pred = s.copy()

        # Action 0: Rest (recovers stamina, minimal metabolic drain)
        if action == 0:
            pred[0] = min(1.0, s[0] + 0.01)   # health slight regen if nourished
            pred[1] = max(0.0, s[1] - 0.005)  # energy minimal drain
            pred[2] = max(0.0, s[2] - 0.008)  # hydration minimal drain
            pred[3] = min(1.0, s[3] + 0.20)   # stamina significant recovery
        # Action 1: Forage (eats local food)
        elif action == 1:
            local_food = s[6] if len(s) > 6 else 0.5
            food_gain = local_food * 0.40
            pred[1] = min(1.0, s[1] + food_gain - 0.02)
            pred[2] = max(0.0, s[2] - 0.025)
            pred[3] = max(0.0, s[3] - 0.03)
            if len(pred) > 6:
                pred[6] = max(0.0, s[6] * 0.4)
        # Action 2: Drink (drinks local water)
        elif action == 2:
            local_water = s[7] if len(s) > 7 else 0.5
            water_gain = local_water * 0.45
            pred[1] = max(0.0, s[1] - 0.02)
            pred[2] = min(1.0, s[2] + water_gain - 0.015)
            pred[3] = max(0.0, s[3] - 0.03)
            if len(pred) > 7:
                pred[7] = max(0.0, s[7] * 0.3)
        # Action 3: Shelter (protects against exposure / hazard)
        elif action == 3:
            pred[1] = max(0.0, s[1] - 0.015)
            pred[2] = max(0.0, s[2] - 0.02)
            pred[3] = min(1.0, s[3] + 0.05)
            if len(pred) > 4:
                pred[4] = max(0.0, s[4] - 0.30)  # exposure reduction
        # Action 4: Explore (locomotion through space)
        elif action == 4:
            pred[1] = max(0.0, s[1] - 0.03)
            pred[2] = max(0.0, s[2] - 0.035)
            pred[3] = max(0.0, s[3] - 0.06)
        # Action 5: Flee / Evade
        elif action == 5:
            pred[1] = max(0.0, s[1] - 0.04)
            pred[2] = max(0.0, s[2] - 0.045)
            pred[3] = max(0.0, s[3] - 0.08)
            if len(pred) > 5:
                pred[5] = max(0.0, s[5] - 0.40)  # threat reduction

        # General vital homeostasis and starvation penalty
        if pred[1] < 0.10:
            pred[0] = max(0.0, pred[0] - 0.08)
        if pred[2] < 0.10:
            pred[0] = max(0.0, pred[0] - 0.10)
        if len(pred) > 4 and pred[4] > 0.70:
            pred[0] = max(0.0, pred[0] - 0.06)

        return np.clip(pred, 0.0, 1.0)

    def abduct_exogenous_noise(
        self,
        factual_state: np.ndarray,
        factual_action: int,
        realized_next_state: np.ndarray,
    ) -> np.ndarray:
        """Step 1 (Abduction): Invert structural equation to isolate historical noise U_t."""
        expected_next = self.predict_factual(factual_state, factual_action)
        noise = realized_next_state - expected_next
        return noise

    def evaluate_counterfactual(
        self,
        factual_state: np.ndarray,
        factual_action: int,
        realized_next_state: np.ndarray,
        counterfactual_action: int,
    ) -> CounterfactualResult:
        """Pearl Level 3 Counterfactual Evaluation:
        
        1. Abduction:  u_t = s_{t+1} - f(s_t, a_t)
        2. Action:     do(A = a')
        3. Prediction: s_{t+1}^{do(A = a')} = clip(f(s_t, a') + u_t, 0, 1)
        """
        u = self.abduct_exogenous_noise(factual_state, factual_action, realized_next_state)
        cf_expected = self.predict_factual(factual_state, counterfactual_action)
        cf_predicted = np.clip(cf_expected + u, 0.0, 1.0)

        # Viability value function V(s): health + 0.5 * min(energy, hydration)
        def value_fn(s: np.ndarray) -> float:
            return float(s[0] + 0.5 * min(s[1], s[2]))

        realized_val = value_fn(realized_next_state)
        cf_val = value_fn(cf_predicted)
        regret = float(cf_val - realized_val)

        noise_norm = float(np.linalg.norm(u))
        if regret > 0.05 and noise_norm < 0.35:
            attribution = "policy_error"
            preventable = True
        elif noise_norm >= 0.35 and regret <= 0.05:
            attribution = "exogenous_shock"
            preventable = False
        else:
            attribution = "unavoidable_attrition"
            preventable = False

        query = CounterfactualQuery(
            factual_state=factual_state,
            factual_action=factual_action,
            factual_next_state=realized_next_state,
            counterfactual_action=counterfactual_action,
            exogenous_noise=u,
        )

        return CounterfactualResult(
            query=query,
            counterfactual_predicted_state=cf_predicted,
            regret=regret,
            preventable_by_policy=preventable,
            attribution=attribution,
        )

    def find_optimal_counterfactual(
        self,
        factual_state: np.ndarray,
        factual_action: int,
        realized_next_state: np.ndarray,
    ) -> Tuple[int, CounterfactualResult]:
        """Scans all candidate actions to find the best counterfactual intervention."""
        best_action = factual_action
        best_result = self.evaluate_counterfactual(
            factual_state, factual_action, realized_next_state, factual_action
        )

        for a in range(self.action_dim):
            if a == factual_action:
                continue
            res = self.evaluate_counterfactual(
                factual_state, factual_action, realized_next_state, a
            )
            if res.regret > best_result.regret:
                best_result = res
                best_action = a

        return best_action, best_result
