"""PROJECT NEXUS-Ω: Matched raw and heuristic-guided rollout comparison.

The included factored predictor is an experimental, untrained initialized
ensemble. Its disagreement penalty is diagnostic and is not calibrated.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .contracts import ImaginedTrajectory, MacroIntent
from .causal_world_model import FactoredCausalWorldModel


class TrajectoryDiffusionImaginer:
    """Generates paired rollouts with identical initial state and action sequence."""

    def __init__(
        self,
        world_model: FactoredCausalWorldModel,
        default_horizon: int = 8,
        diffusion_steps: int = 5,
        guidance_scale: float = 1.25,
        conformal_quantile: float = 1.96,
        base_seed: int = 42,
    ):
        self.world_model = world_model
        self.default_horizon = default_horizon
        # Kept for constructor compatibility; this deterministic paired method
        # does not perform diffusion steps or consume random draws.
        self.diffusion_steps = diffusion_steps
        self.base_seed = base_seed  # Retained as constructor metadata; rollout is deterministic.
        self.guidance_scale = guidance_scale
        self.conformal_quantile = conformal_quantile

        # Targets only named vital channels. Other observation slots are not
        # adjusted without an explicit observation-schema contract.
        self.s_homeo_target = np.array([1.0, 0.90, 0.90, 0.0], dtype=np.float32)

    def _intent_to_action_prior(self, intent: MacroIntent) -> np.ndarray:
        """Returns action selection probability distribution for a given macro-intent."""
        # Actions: 0: Forage, 1: Shelter, 2: Explore, 3: Rest
        if intent == MacroIntent.FORAGE_EXPEDITION:
            return np.array([0.70, 0.05, 0.15, 0.10], dtype=np.float32)
        elif intent == MacroIntent.DIAPAUSE_HIBERNATION:
            return np.array([0.05, 0.70, 0.05, 0.20], dtype=np.float32)
        elif intent == MacroIntent.HABITAT_RELOCATION:
            return np.array([0.15, 0.05, 0.70, 0.10], dtype=np.float32)
        elif intent == MacroIntent.EMERGENCY_FORTIFICATION:
            return np.array([0.05, 0.80, 0.05, 0.10], dtype=np.float32)
        return np.ones(4, dtype=np.float32) / 4.0

    def compute_homeostatic_guidance(self, state: np.ndarray) -> np.ndarray:
        """Computes gradient guidance direction pushing state toward homeostatic envelope."""
        grad = np.zeros_like(state)
        # Endogenous vitals: health, energy, hydration
        for i in range(min(3, len(state))):
            diff = self.s_homeo_target[i] - state[i]
            grad[i] = diff * (2.0 if i == 0 else 1.0)
        # Do not infer semantics for exogenous observation slots by position:
        # index 4 is episode progress in the canonical POMDP, not exposure.
        return grad

    def imagine_trajectory(
        self,
        current_obs: np.ndarray | Sequence[float],
        current_mem: np.ndarray | Sequence[float],
        target_intent: MacroIntent = MacroIntent.FORAGE_EXPEDITION,
        horizon: Optional[int] = None,
    ) -> ImaginedTrajectory:
        """Compares an unmodified model rollout with its heuristic-guided counterpart."""
        H = horizon or self.default_horizon
        obs_curr = np.asarray(current_obs, dtype=np.float32)[:self.world_model.d_obs]
        mem_curr = np.asarray(current_mem, dtype=np.float32)

        action_prior = self._intent_to_action_prior(target_intent)

        # 1. Initialize candidate actions across horizon
        # For storm or severe threat, prioritize shelter
        has_storm = (
            (len(obs_curr) > 3 and obs_curr[3] > 0.8)
            or (len(mem_curr) > 0 and float(mem_curr[0]) > 0.1)
        )
        if has_storm or target_intent in (MacroIntent.DIAPAUSE_HIBERNATION, MacroIntent.EMERGENCY_FORTIFICATION):
            action_seq = [1 if t < 3 else int(np.argmax(action_prior)) for t in range(H)]
        else:
            action_seq = [int(np.argmax(action_prior)) for _ in range(H)]

        # 2. Sequential multi-step rollout using Factored Causal World Model
        guided_states: List[np.ndarray] = []
        guided_rewards: List[float] = []
        guided_epistemic_vars: List[float] = []
        raw_states: List[np.ndarray] = []
        raw_rewards: List[float] = []
        raw_epistemic_vars: List[float] = []
        direct_adjustments: List[np.ndarray] = []
        branch_deltas: List[np.ndarray] = []
        branch_divergence: List[float] = []

        guided_state = obs_curr.copy()
        raw_state = obs_curr.copy()
        mem_t = mem_curr.copy()

        total_guidance_norm = 0.0
        previous_prediction = self.world_model.last_prediction
        previous_action = self.world_model.last_action
        try:
            for t in range(H):
                act = action_seq[t]
                # Both branches start from the same observed state and receive
                # the same action. Each then propagates its own successor state.
                raw_pred = self.world_model.predict_factored(raw_state, mem_t, act)
                guided_pred = self.world_model.predict_factored(guided_state, mem_t, act)

                raw_next = raw_pred.predicted_obs.copy()
                guidance_vec = self.compute_homeostatic_guidance(guided_pred.predicted_obs)
                total_guidance_norm += float(np.linalg.norm(guidance_vec))

                alpha = 0.05 * self.guidance_scale
                direct_adjustment = alpha * guidance_vec
                guided_next = guided_pred.predicted_obs + direct_adjustment

                # Clip vitals to physically valid ranges [0, 1]
                guided_next[0] = max(0.0, min(1.0, guided_next[0]))
                guided_next[1] = max(0.0, min(1.0, guided_next[1]))
                guided_next[2] = max(0.0, min(1.0, guided_next[2]))

                raw_states.append(raw_next.copy())
                raw_rewards.append(float(raw_pred.predicted_reward))
                raw_epistemic_vars.append(float(raw_pred.epistemic_uncertainty_total))
                guided_states.append(guided_next.copy())
                guided_rewards.append(float(guided_pred.predicted_reward))
                guided_epistemic_vars.append(float(guided_pred.epistemic_uncertainty_total))
                direct_adjustments.append(direct_adjustment.copy())
                branch_delta = guided_next - raw_next
                branch_deltas.append(branch_delta.copy())
                branch_divergence.append(float(np.linalg.norm(branch_delta)))

                raw_state = raw_next
                guided_state = guided_next
                # Advance working memory decay
                if len(mem_t) > 0:
                    mem_t[0] = max(0.0, float(mem_t[0]) - 1.0)
        finally:
            # Imagined branches must not overwrite the agent's factual
            # action-conditioned prediction cache used by transition feedback.
            self.world_model.last_prediction = previous_prediction
            self.world_model.last_action = previous_action

        cumulative_r = float(np.sum(guided_rewards))

        # Diagnostic uncertainty penalties; neither branch is calibrated.
        total_epistemic_sigma = math.sqrt(max(1e-6, float(np.sum(guided_epistemic_vars))))
        conformal_lcb = cumulative_r - self.conformal_quantile * total_epistemic_sigma
        raw_cumulative_r = float(np.sum(raw_rewards))
        raw_total_sigma = math.sqrt(max(1e-6, float(np.sum(raw_epistemic_vars))))
        raw_return_penalty = raw_cumulative_r - self.conformal_quantile * raw_total_sigma

        return ImaginedTrajectory(
            horizon=H,
            macro_intent=target_intent,
            states=guided_states,
            actions=action_seq,
            rewards=guided_rewards,
            epistemic_uncertainties=guided_epistemic_vars,
            cumulative_reward=cumulative_r,
            conformal_trajectory_lcb=conformal_lcb,
            score_guidance_norm=float(total_guidance_norm / max(1, H)),
            unguided_states=raw_states,
            unguided_rewards=raw_rewards,
            unguided_epistemic_uncertainties=raw_epistemic_vars,
            direct_guidance_adjustments=direct_adjustments,
            branch_state_deltas=branch_deltas,
            branch_divergence_l2=branch_divergence,
            unguided_cumulative_reward=raw_cumulative_r,
            unguided_return_penalty=raw_return_penalty,
            predictor_evaluations=2 * H,
            ensemble_member_forwards=2 * H * self.world_model.n_members,
        )

    def evaluate_all_intents(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        horizon: Optional[int] = None,
    ) -> Dict[str, ImaginedTrajectory]:
        """Imagines and compares counterfactual trajectories across all four macro-intents."""
        results: Dict[str, ImaginedTrajectory] = {}
        for intent in MacroIntent:
            traj = self.imagine_trajectory(obs, mem, target_intent=intent, horizon=horizon)
            results[intent.value] = traj
        return results
