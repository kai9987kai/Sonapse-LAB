"""PROJECT NEXUS-Ω: Active Inference with Model-Score Guarded Recovery.

Formulates action selection as variational Expected Free Energy (EFE) optimization
with an uncalibrated uncertainty penalty and explicit heuristic action guard:
1. Pragmatic Instrumental Value:
   Mahalanobis divergence from homeostatic viability kernel (health, energy, hydration, stamina).
2. Epistemic Curiosity Value:
   Information gain / variance reduction over exogenous field and latent causal regime.
3. Dynamic Curiosity Regulation:
   Suppressed when physiological safety is endangered; unconstrained during surplus.
4. Heuristic Guard:
   Uses ensemble disagreement and vital/exposure thresholds to override selected
   actions. Its score is not a distribution-free bound or safety guarantee.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np

from poseidon.nexus_omega.contracts import (
    CausalEFEDecision,
    FactoredPredictionOutput,
    FactoredState,
)
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel


class CausalEFEPlanner:
    """Plans by Expected Free Energy under uncalibrated, model-score heuristics."""

    def __init__(
        self,
        world_model: FactoredCausalWorldModel,
        n_actions: int = 4,
        base_epistemic_weight: float = 0.50,
        conformal_confidence_quantile: float = 1.96,
        safety_threshold: float = -1.20,
        exposure_index: Optional[int] = None,
        safety_fallback_actions: Optional[Sequence[int]] = None,
        calm_recovery_action: Optional[int] = None,
    ):
        self.world_model = world_model
        self.n_actions = n_actions
        self.base_epistemic_weight = base_epistemic_weight
        self.conformal_quantile = conformal_confidence_quantile
        self.safety_threshold = safety_threshold
        if exposure_index is not None and (type(exposure_index) is not int or exposure_index < 0):
            raise ValueError("exposure_index must be a non-negative integer or None")
        self.exposure_index = exposure_index
        if safety_fallback_actions is not None:
            fallback_actions = tuple(safety_fallback_actions)
            if not fallback_actions or any(
                type(action) is not int or not 0 <= action < n_actions
                for action in fallback_actions
            ):
                raise ValueError("safety_fallback_actions must contain valid action indices")
            self.safety_fallback_actions: Optional[Tuple[int, ...]] = fallback_actions
        else:
            self.safety_fallback_actions = None
        if calm_recovery_action is not None and (
            type(calm_recovery_action) is not int or not 0 <= calm_recovery_action < n_actions
        ):
            raise ValueError("calm_recovery_action must be a valid action index or None")
        self.calm_recovery_action = calm_recovery_action

        # Target homeostatic vital prior: health=1.0, energy=0.9, hydration=0.9, stamina=0.9
        self.s_target = np.array([1.0, 0.90, 0.90, 0.90], dtype=np.float32)
        # Precision weighting: health is penalized most heavily
        self.precision_weights = np.array([4.0, 2.5, 2.5, 1.5], dtype=np.float32)

        self.last_decision: Optional[CausalEFEDecision] = None

    def compute_pragmatic_value(self, predicted_endo: np.ndarray, predicted_reward: float = 0.0) -> float:
        """Expected utility combining extrinsic reward and homeostatic vital alignment."""
        diff = predicted_endo[:3] - self.s_target[:3]
        vital_penalty = np.sum(self.precision_weights[:3] * (diff ** 2))
        return float(predicted_reward - vital_penalty)

    def compute_epistemic_drive(self, current_obs: np.ndarray) -> float:
        """Modulates epistemic curiosity based on current physiological safety margin."""
        health = max(0.0, min(1.0, float(current_obs[0])))
        energy = max(0.0, min(1.0, float(current_obs[1])))
        hydration = max(0.0, min(1.0, float(current_obs[2])))

        # When vital reserves are safe, exploration blooms; under starvation, curiosity shuts down
        vital_safety = health * min(energy, hydration)
        epistemic_weight = max(0.02, self.base_epistemic_weight * vital_safety)
        return float(epistemic_weight)

    def compute_empowerment(
        self,
        predicted_endo: np.ndarray,
        predicted_exo: np.ndarray,
        mem: np.ndarray,
    ) -> float:
        """Estimates future behavioral freedom across 1-step successor actions."""
        succ_states = []
        obs_hypo = np.concatenate([predicted_endo, predicted_exo], axis=0)
        for a_succ in range(self.n_actions):
            pred = self.world_model.predict_factored(obs_hypo, mem, a_succ)
            succ_states.append(pred.endogenous_pred)

        # Variance of reachable vital states across actions
        stack = np.stack(succ_states, axis=0)
        dispersion = float(np.mean(np.var(stack, axis=0)))
        return dispersion

    def plan_action(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        action_names: Optional[Sequence[str]] = None,
    ) -> CausalEFEDecision:
        """Evaluates all actions under Expected Free Energy and enforces Conformal Guard."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(obs_arr) > self.world_model.d_obs:
            obs_arr = obs_arr[:self.world_model.d_obs]

        names = list(action_names or ["forage", "shelter", "explore", "rest"])

        best_action = 0
        best_efe = -float("inf")
        best_prag = 0.0
        best_epistemic = 0.0
        best_weight = 0.0
        best_empowerment = 0.0
        best_lcb = -float("inf")

        epistemic_weight = self.compute_epistemic_drive(obs_arr)

        action_scores = []

        for a in range(self.n_actions):
            pred = self.world_model.predict_factored(obs_arr, mem, a)
            prag = self.compute_pragmatic_value(pred.endogenous_pred, pred.predicted_reward)
            epistemic = pred.exogenous_uncertainty + pred.regime_uncertainty
            empowerment = self.compute_empowerment(pred.endogenous_pred, pred.exogenous_pred, mem)

            # Expected Free Energy Objective (to be maximized)
            # G = Pragmatic + beta * Epistemic + 0.2 * Empowerment
            efe = prag + epistemic_weight * epistemic + 0.20 * empowerment

            # Conformal Return Lower Bound: penalize by total epistemic uncertainty
            conformal_sigma = math.sqrt(max(1e-6, pred.epistemic_uncertainty_total))
            lcb = efe - self.conformal_quantile * conformal_sigma

            action_scores.append({
                "action": a,
                "efe": efe,
                "prag": prag,
                "epistemic": epistemic,
                "empowerment": empowerment,
                "lcb": lcb,
                "predicted_reward": float(pred.predicted_reward),
                "predicted_endogenous": [float(x) for x in pred.endogenous_pred],
                "predicted_exogenous": [float(x) for x in pred.exogenous_pred],
                "epistemic_uncertainty": float(pred.epistemic_uncertainty_total),
            })

            if efe > best_efe:
                best_efe = efe
                best_action = a
                best_prag = prag
                best_epistemic = epistemic
                best_weight = epistemic_weight
                best_empowerment = empowerment
                best_lcb = lcb

        objective_action = best_action

        # Model-score guard: if the selected forage/explore action has a low
        # heuristic score while health/exposure is threatened, choose a fallback.
        conformal_override = False
        recovery_admission = False
        has_high_exposure = (
            self.exposure_index is not None
            and self.exposure_index < len(obs_arr)
            and float(obs_arr[self.exposure_index]) > 0.5
        )
        is_threatened = (obs_arr[0] < 0.6) or has_high_exposure
        if best_lcb < self.safety_threshold and is_threatened and best_action in (0, 2):
            recovery_data = action_scores[self.calm_recovery_action] if self.calm_recovery_action is not None else None
            recovery_state = recovery_data["predicted_endogenous"] if recovery_data is not None else []
            regime = getattr(self.world_model, "z_regime", np.zeros(1, dtype=np.float32))
            storm_belief = (
                (len(obs_arr) > 3 and float(obs_arr[3]) > 0.8)
                or (len(regime) > 0 and float(regime[0]) > 0.1)
                or (len(mem) > 0 and float(mem[0]) > 0.1)
            )
            recovery_admission = bool(
                self.calm_recovery_action == best_action == 0
                and len(obs_arr) > 2
                and float(obs_arr[1]) < 0.50
                and float(obs_arr[2]) > 0.25
                and not storm_belief
                and len(recovery_state) > 1
                and float(recovery_state[0]) >= float(obs_arr[0]) - 0.05
                and float(recovery_state[1]) > float(obs_arr[1]) + 0.05
            )
            if not recovery_admission:
                # Keep the v1 shelter fallback unless a profile supplies alternatives.
                # Among configured viable actions, select the strongest model LCB.
                fallback_actions = self.safety_fallback_actions or ((1,) if self.n_actions > 1 else (0,))
                fallback_action = max(fallback_actions, key=lambda action: action_scores[action]["lcb"])
                best_action = fallback_action
                conformal_override = True
                fallback_data = action_scores[fallback_action]
                best_efe = fallback_data["efe"]
                best_prag = fallback_data["prag"]
                best_epistemic = fallback_data["epistemic"]
                best_empowerment = fallback_data["empowerment"]
                best_lcb = fallback_data["lcb"]

        # Ensure world model last_prediction corresponds to chosen action
        self.world_model.predict_factored(obs_arr, mem, best_action)

        action_label = names[best_action] if best_action < len(names) else f"action_{best_action}"

        factored = self.world_model.get_factored_state(obs_arr)
        factored_dict = {
            "endogenous": [round(float(x), 4) for x in factored.endogenous_vitals],
            "exogenous": [round(float(x), 4) for x in factored.exogenous_field],
            "causal_regime": [round(float(x), 4) for x in factored.causal_regime],
        }

        decision = CausalEFEDecision(
            action=best_action,
            action_name=action_label,
            free_energy_objective=best_efe,
            pragmatic_value=best_prag,
            epistemic_value=best_epistemic,
            epistemic_weight=best_weight,
            empowerment_value=best_empowerment,
            conformal_lcb=best_lcb,
            conformal_override=conformal_override,
            recovery_admission=recovery_admission,
            factored_state=factored_dict,
            action_scores=[
                {
                    **score,
                    "selected_by_objective": score["action"] == objective_action,
                    "selected_after_guard": score["action"] == best_action,
                }
                for score in action_scores
            ],
        )

        self.last_decision = decision
        return decision
