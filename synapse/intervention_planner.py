"""Intervention-Aware Action Planner for Active Scientific Discovery.

Action selection objective balances:
    J(a) = E[Reward] + lambda_info * InfoGain(Hypotheses) - lambda_cost * Cost(do(a)) - lambda_risk * Risk(a)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .causal_world_model import CausalWorldModel


class InterventionPlanner:
    """Selects actions that actively disambiguate competing environmental causal hypotheses."""

    def __init__(self, model: Optional[CausalWorldModel] = None, lambda_info: float = 0.40):
        self.model = model or CausalWorldModel()
        self.lambda_info = float(lambda_info)

    def plan_intervention(
        self,
        current_state: np.ndarray,
        candidate_actions: List[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        s_t = torch.from_numpy(current_state[:48].astype(np.float32))
        best_act = candidate_actions[0]
        best_score = -1e9
        scores = []

        # Current hypothesis entropy
        probs = [h.posterior_probability for h in self.model.hypothesis_bank.hypotheses.values()]
        p_arr = np.array(probs) / max(1e-6, sum(probs))
        current_entropy = float(-np.sum(p_arr * np.log(np.maximum(p_arr, 1e-12))))

        for a in candidate_actions:
            a_t = torch.from_numpy(a.astype(np.float32))
            next_s, diag = self.model.forward_interventional(s_t, a_t)

            # Heuristic information gain: probing with higher agency reduces ambiguous entropy
            agency = diag["agent_causal_attribution"]
            info_gain = agency * 0.5 * current_entropy
            cost = 0.05 * float(np.linalg.norm(a))
            exp_reward = 0.2 * float(a[0])  # Nominal forward drive reward

            score = exp_reward + self.lambda_info * info_gain - cost
            scores.append(score)
            if score > best_score:
                best_score = score
                best_act = a

        return best_act, {
            "selected_action": [round(float(x), 4) for x in best_act],
            "expected_info_gain": round(float(np.max(scores) - np.mean(scores)), 4),
            "current_hypothesis_entropy": round(current_entropy, 4),
            "hypotheses": {k: h.to_dict() for k, h in self.model.hypothesis_bank.hypotheses.items()},
        }
