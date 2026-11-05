"""Stage 12 Contrastive Causal Memory Fabric.

Implements the counterfactual action-advantage formulation, 24-feature structured
episodic belief latching, and dual-layer safety shielding (one-step predictive
vitality guard and risk-adaptive balance deadband) learned from the alias-tidepool-v2
decision-value memory assays.

Guarantees that memory interventions only override incumbent policies when:
1. The memory-induced counterfactual advantage delta is strictly positive.
2. The predicted one-step resource lower-confidence bound exceeds the vital floor.
3. The risk-adaptive balance deadband does not detect asymmetric resource exhaustion.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ContrastiveMemoryDecision:
    """Decision audit contract for contrastive memory arbitration."""
    candidate_action: int
    baseline_action: int
    final_action: int
    memory_override: bool
    advantage_score: float
    predictive_energy_next: float
    predictive_hydration_next: float
    predictive_guard_active: bool
    predictive_guard_override: bool
    balance_deadband_active: bool
    balance_deadband_override: bool
    energy_lcb: float
    hydration_lcb: float
    latched_cue: float
    timestamp: float = field(default_factory=time.time)

    @property
    def recommended_action(self) -> int:
        return self.final_action

    @property
    def counterfactual_advantage(self) -> float:
        return self.advantage_score

    @property
    def predicted_vitality_energy(self) -> float:
        return self.predictive_energy_next

    @property
    def predicted_vitality_hydration(self) -> float:
        return self.predictive_hydration_next

    @property
    def within_deadband(self) -> bool:
        return self.balance_deadband_active

    @property
    def guard_triggered(self) -> bool:
        return self.predictive_guard_active or self.predictive_guard_override

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_action": int(self.candidate_action),
            "baseline_action": int(self.baseline_action),
            "final_action": int(self.final_action),
            "recommended_action": int(self.final_action),
            "memory_override": bool(self.memory_override),
            "advantage_score": round(float(self.advantage_score), 4),
            "counterfactual_advantage": round(float(self.advantage_score), 4),
            "predictive_energy_next": round(float(self.predictive_energy_next), 4),
            "predictive_hydration_next": round(float(self.predictive_hydration_next), 4),
            "predicted_vitality_energy": round(float(self.predictive_energy_next), 4),
            "predicted_vitality_hydration": round(float(self.predictive_hydration_next), 4),
            "predictive_guard_active": bool(self.predictive_guard_active),
            "predictive_guard_override": bool(self.predictive_guard_override),
            "guard_triggered": bool(self.predictive_guard_active or self.predictive_guard_override),
            "balance_deadband_active": bool(self.balance_deadband_active),
            "balance_deadband_override": bool(self.balance_deadband_override),
            "within_deadband": bool(self.balance_deadband_active),
            "energy_lcb": round(float(self.energy_lcb), 4),
            "hydration_lcb": round(float(self.hydration_lcb), 4),
            "latched_cue": round(float(self.latched_cue), 4),
            "timestamp": round(float(self.timestamp), 4),
        }


class ContrastiveCausalMemoryFabric:
    """24-feature structured belief latch with predictive vitality guard and balance deadband."""

    # World dynamics constants derived from empirical TidePool v2 assays
    MAX_DRAIN = 0.08
    MIN_YIELD = 0.12
    DEFAULT_RESERVE_FLOOR = 0.27
    BALANCE_ACTIVATION = 0.35
    BALANCE_DEADBAND = 0.20
    CONFIDENCE_Z = 1.0  # 1-sigma sensor noise margin
    SENSOR_NOISE_STD = 0.04

    def __init__(
        self,
        reserve_floor: float = DEFAULT_RESERVE_FLOOR,
        balance_activation: float = BALANCE_ACTIVATION,
        balance_deadband: float = BALANCE_DEADBAND,
        sensor_noise_std: float = SENSOR_NOISE_STD,
        confidence_z: float = CONFIDENCE_Z,
        influence_scale: float = 32.0,
        d_obs: int = 8,
        d_working: int = 16,
        **kwargs: Any,
    ) -> None:
        self.d_obs = d_obs
        self.d_working = d_working
        self.reserve_floor = float(reserve_floor)
        self.balance_activation = float(balance_activation)
        self.balance_deadband = float(balance_deadband)
        self.sensor_noise_std = float(sensor_noise_std)
        self.confidence_z = float(confidence_z)
        self.influence_scale = float(influence_scale)

        # Episodic state tracking
        self.latched_cue: float = 0.0
        self.prev_action: Optional[int] = None
        self.prev_reward: float = 0.0
        self.tick: int = 0

    def reset(self) -> None:
        self.latched_cue = 0.0
        self.prev_action = None
        self.prev_reward = 0.0
        self.tick = 0

    def evaluate(
        self,
        obs: Sequence[float],
        last_action: Optional[int] = None,
        last_reward: float = 0.0,
        candidate_action: int = 0,
        baseline_action: int = 0,
        predicted_advantage: float = 0.25,
    ) -> ContrastiveMemoryDecision:
        """Ergonomic evaluation entrypoint computing contrastive arbitration decision."""
        return self.arbitrate(
            obs=obs,
            candidate_action=candidate_action,
            baseline_action=baseline_action,
            predicted_advantage=predicted_advantage,
        )

    def encode_belief(
        self,
        obs: Sequence[float],
        prev_action: Optional[int] = None,
        prev_reward: float = 0.0,
        cue_signal: Optional[float] = None,
    ) -> np.ndarray:
        """Constructs 24-feature structured belief vector:

        [16 raw observations, 6-hot previous action, 1 previous reward, 1 latched signed scent cue].
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        if len(obs_arr) < 16:
            padded = np.zeros(16, dtype=np.float32)
            padded[:len(obs_arr)] = obs_arr
            obs_arr = padded
        else:
            obs_arr = obs_arr[:16]

        # Update signed cue latch if cue channel manifests non-zero
        if cue_signal is not None and abs(float(cue_signal)) > 0.05:
            self.latched_cue = float(cue_signal)
        elif len(obs) > 3 and abs(float(obs[3])) > 0.05:
            self.latched_cue = float(obs[3])

        action_hot = np.zeros(6, dtype=np.float32)
        act = prev_action if prev_action is not None else self.prev_action
        if act is not None and 0 <= act < 6:
            action_hot[act] = 1.0

        rew = float(prev_reward if prev_reward != 0.0 else self.prev_reward)
        belief = np.concatenate([
            obs_arr,
            action_hot,
            np.array([rew, self.latched_cue], dtype=np.float32),
        ])
        return belief

    def predict_resources_next(
        self,
        energy_lcb: float,
        hydration_lcb: float,
        action: int,
    ) -> Tuple[float, float]:
        """Calculates conservative 1-step lower-bound vital state after candidate action."""
        e_next = energy_lcb - self.MAX_DRAIN
        h_next = hydration_lcb - self.MAX_DRAIN
        # Action 0: Forage (yields energy in regime A, maintenance energy in v2)
        # Action 1: Shelter / Drink (yields hydration in regime B, maintenance hydration in v2)
        # Action 2: Explore / Neutral
        # Action 3: Rest (replenishes hydration in phaseworld v2)
        if action == 0:
            e_next += self.MIN_YIELD
        elif action in (1, 3):
            h_next += self.MIN_YIELD
        return max(0.0, e_next), max(0.0, h_next)

    def arbitrate(
        self,
        obs: Sequence[float],
        candidate_action: int,
        baseline_action: int,
        predicted_advantage: float = 0.0,
        energy_override: Optional[float] = None,
        hydration_override: Optional[float] = None,
    ) -> ContrastiveMemoryDecision:
        """Arbitrates between memory candidate action and incumbent baseline action.

        Applies:
        1. Lower-confidence bound sensor noise filtering
        2. One-step predictive resource viability guard
        3. Risk-adaptive resource balance deadband
        """
        self.tick += 1
        obs_arr = np.asarray(obs, dtype=np.float32)

        energy = float(energy_override if energy_override is not None else (obs_arr[1] if len(obs_arr) > 1 else 1.0))
        hydration = float(hydration_override if hydration_override is not None else (obs_arr[2] if len(obs_arr) > 2 else 1.0))

        # 1-sigma Lower Confidence Bound under sensor noise
        noise_margin = self.confidence_z * self.SENSOR_NOISE_STD
        energy_lcb = max(0.0, energy - noise_margin)
        hydration_lcb = max(0.0, hydration - noise_margin)

        # 1. Evaluate one-step predictive resource floors for candidate action
        pred_e, pred_h = self.predict_resources_next(energy_lcb, hydration_lcb, candidate_action)
        candidate_min_resource = min(pred_e, pred_h)

        predictive_guard_active = candidate_min_resource < self.reserve_floor

        # 2. Risk-adaptive resource balance deadband
        # If either resource is below activation floor (0.35), verify balance
        min_reserve = min(energy_lcb, hydration_lcb)
        balance_active = min_reserve <= self.balance_activation
        lower_resource_action = 0 if energy_lcb <= hydration_lcb else 1  # 0=Forage (energy), 1=Rest/Shelter (hydration)

        balance_violated = (
            balance_active
            and abs(energy_lcb - hydration_lcb) > self.balance_deadband
            and candidate_action != lower_resource_action
        )

        # Determine safe restorative action if guard or deadband triggers
        restorative_action = lower_resource_action

        predictive_override = False
        balance_override = False
        final_action = int(candidate_action)

        if predictive_guard_active:
            predictive_override = True
            final_action = restorative_action
        elif balance_violated:
            balance_override = True
            final_action = restorative_action
        elif predicted_advantage <= 0.0:
            # Memory offers no counterfactual advantage: fall back to baseline incumbent
            final_action = int(baseline_action)

        memory_override = (final_action == candidate_action and candidate_action != baseline_action)

        self.prev_action = final_action

        return ContrastiveMemoryDecision(
            candidate_action=int(candidate_action),
            baseline_action=int(baseline_action),
            final_action=int(final_action),
            memory_override=memory_override,
            advantage_score=float(predicted_advantage),
            predictive_energy_next=float(pred_e),
            predictive_hydration_next=float(pred_h),
            predictive_guard_active=predictive_guard_active,
            predictive_guard_override=predictive_override,
            balance_deadband_active=balance_active,
            balance_deadband_override=balance_override,
            energy_lcb=float(energy_lcb),
            hydration_lcb=float(hydration_lcb),
            latched_cue=float(self.latched_cue),
            timestamp=time.time(),
        )
