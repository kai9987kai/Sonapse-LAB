"""PROJECT NEXUS-Ω: Hierarchical Multi-Scale Active Inference Kernel.

Coordinates Level-1 Macro-Intent options (slow time-scale, temporal abstraction)
with Level-0 Micro-Action planning (fast time-scale, active inference) and
score-guided counterfactual trajectory imagination.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from .contracts import (
    CausalEFEDecision,
    HierarchicalEFEDecision,
    ImaginedTrajectory,
    MacroIntent,
)
from .causal_world_model import FactoredCausalWorldModel
from .causal_efe import CausalEFEPlanner
from .trajectory_diffusion import TrajectoryDiffusionImaginer


class HierarchicalEFEPlanner:
    """Two-tier active inference planner coordinating macro-intents with micro-actions."""

    def __init__(
        self,
        world_model: FactoredCausalWorldModel,
        n_actions: int = 4,
        commitment_duration: int = 4,
        base_epistemic_weight: float = 0.40,
        conformal_confidence_quantile: float = 1.96,
        safety_threshold: float = -2.5,
        imagination_horizon: int = 8,
        base_seed: int = 42,
        exposure_index: Optional[int] = None,
    ):
        self.world_model = world_model
        self.n_actions = n_actions
        self.commitment_duration = commitment_duration
        self.safety_threshold = safety_threshold
        self.imagination_horizon = imagination_horizon

        # Base micro-action EFE planner
        self.micro_planner = CausalEFEPlanner(
            world_model=world_model,
            n_actions=n_actions,
            base_epistemic_weight=base_epistemic_weight,
            conformal_confidence_quantile=conformal_confidence_quantile,
            safety_threshold=safety_threshold,
            exposure_index=exposure_index,
        )

        # Counterfactual trajectory diffusion imaginer
        self.imaginer = TrajectoryDiffusionImaginer(
            world_model=world_model,
            default_horizon=imagination_horizon,
            conformal_quantile=conformal_confidence_quantile,
            base_seed=base_seed,
        )

        # Level-1 Macro State
        self.active_macro_intent: MacroIntent = MacroIntent.FORAGE_EXPEDITION
        self.intent_remaining_ticks: int = 0
        self.last_decision: Optional[HierarchicalEFEDecision] = None

    def reset(self) -> None:
        self.active_macro_intent = MacroIntent.FORAGE_EXPEDITION
        self.intent_remaining_ticks = 0
        self.last_decision = None

    def evaluate_macro_efe(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        intent: MacroIntent,
    ) -> Tuple[float, ImaginedTrajectory]:
        """Evaluates Level-1 macro Expected Free Energy via counterfactual trajectory rollout."""
        traj = self.imaginer.imagine_trajectory(obs, mem, target_intent=intent, horizon=self.imagination_horizon)
        # Macro selection uses the unmodified model branch. Guided rollouts are
        # retained as a matched shadow candidate and do not silently become the
        # planner's source of imagined state or reward.
        macro_efe = float(0.70 * traj.unguided_cumulative_reward + 0.30 * traj.unguided_return_penalty)
        return macro_efe, traj

    def select_macro_intent(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
    ) -> Tuple[MacroIntent, float, ImaginedTrajectory]:
        """Selects optimal Level-1 Macro-Intent across competing intentional hypotheses."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        mem_arr = np.asarray(mem, dtype=np.float32)

        # Check emergency interruption triggers:
        # 1. Storm alert pulse or persisting storm in working memory -> force Diapause / Fortification
        has_storm = (
            (len(obs_arr) > 3 and obs_arr[3] > 0.8)
            or (len(mem_arr) > 0 and float(mem_arr[0]) > 0.1)
        )
        # 2. Critical health drop -> force Emergency Fortification
        is_critical_health = (obs_arr[0] < 0.40)

        if has_storm or is_critical_health:
            chosen_intent = MacroIntent.EMERGENCY_FORTIFICATION if is_critical_health else MacroIntent.DIAPAUSE_HIBERNATION
            efe, traj = self.evaluate_macro_efe(obs_arr, mem_arr, chosen_intent)
            self.active_macro_intent = chosen_intent
            self.intent_remaining_ticks = self.commitment_duration
            return chosen_intent, efe, traj

        # If existing commitment is still active and no emergency occurred, sustain intent
        if self.intent_remaining_ticks > 0:
            self.intent_remaining_ticks -= 1
            efe, traj = self.evaluate_macro_efe(obs_arr, mem_arr, self.active_macro_intent)
            return self.active_macro_intent, efe, traj

        # Otherwise evaluate all macro-intents via counterfactual trajectory rollouts
        best_intent = MacroIntent.FORAGE_EXPEDITION
        best_efe = -float("inf")
        best_traj: Optional[ImaginedTrajectory] = None

        for intent in MacroIntent:
            efe, traj = self.evaluate_macro_efe(obs_arr, mem_arr, intent)
            if efe > best_efe:
                best_efe = efe
                best_intent = intent
                best_traj = traj

        self.active_macro_intent = best_intent
        self.intent_remaining_ticks = self.commitment_duration
        assert best_traj is not None
        return best_intent, best_efe, best_traj

    def plan(
        self,
        obs: np.ndarray | Sequence[float],
        mem: np.ndarray | Sequence[float],
        action_names: Optional[Sequence[str]] = None,
    ) -> HierarchicalEFEDecision:
        """Executes full hierarchical decision cycle: Level-1 Intent + Level-0 Action."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        mem_arr = np.asarray(mem, dtype=np.float32)

        # 1. Level-1 Macro Intent Selection
        macro_intent, macro_efe, traj = self.select_macro_intent(obs_arr, mem_arr)

        # 2. Level-0 Micro Action Selection conditioned on macro-intent
        # Standard micro EFE evaluation
        micro_dec = self.micro_planner.plan_action(obs_arr, mem_arr, action_names=action_names)

        # 3. Macro Intent Conditioning:
        # Bias micro-action to align with active macro intent if micro conformal bound is safe
        micro_action = micro_dec.action
        micro_action_name = micro_dec.action_name
        conformal_override = micro_dec.conformal_override

        # If micro conformal guard triggered an override, safety gate takes strict precedence
        if not conformal_override:
            # Macro Intent Alignment
            if macro_intent in (MacroIntent.DIAPAUSE_HIBERNATION, MacroIntent.EMERGENCY_FORTIFICATION):
                # Prioritize shelter or rest
                if micro_action in (0, 2):  # Forage or explore
                    micro_action = 1 if self.n_actions > 1 else 0
                    micro_action_name = "shelter"
            elif macro_intent == MacroIntent.FORAGE_EXPEDITION:
                # If vitals permit and not in storm, ensure foraging
                has_storm = (
                    (len(obs_arr) > 3 and obs_arr[3] > 0.8)
                    or (len(mem_arr) > 0 and float(mem_arr[0]) > 0.1)
                )
                if not has_storm and obs_arr[0] > 0.60 and micro_action == 3:
                    micro_action = 0
                    micro_action_name = "forage"

        decision = HierarchicalEFEDecision(
            macro_intent=macro_intent,
            macro_intent_remaining_ticks=self.intent_remaining_ticks,
            micro_action=micro_action,
            micro_action_name=micro_action_name,
            macro_free_energy=macro_efe,
            micro_free_energy=micro_dec.free_energy_objective,
            conformal_lcb=micro_dec.conformal_lcb,
            conformal_override=conformal_override,
            imagined_trajectory=traj,
            factored_state=micro_dec.factored_state,
            action_scores=[
                {
                    **score,
                    "selected_by_micro_planner": score["action"] == micro_dec.action,
                    "selected_after_hierarchy": score["action"] == micro_action,
                }
                for score in micro_dec.action_scores
            ],
        )

        self.last_decision = decision
        return decision
