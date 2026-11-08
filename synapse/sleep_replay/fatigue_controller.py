"""Fatigue, Sleep Pressure, and State Machine Controller.

Implements two-process sleep regulation:
1. Process S: Homeostatic sleep pressure P_t accumulating during wakefulness and dissipating during sleep:
   P_{t+dt} = clip(P_t + dt * (k_w * 1_awake - k_s * P_t * 1_rest), 0.0, 1.0)
   Modulated by adenosine and melatonin levels.
2. Process C: Circadian rhythm modulation.

State machine:
- AWAKE: active sensory processing and behavioral motor execution
- QUIET_REST: reduced motor activity, resting equilibrium
- DROWSY: sensory attenuation, slow wave induction
- SLEEP_REPLAY: offline episodic memory consolidation
- DREAM_GENERATION: counterfactual generative world model exploration
- AWAKENING: gradual arousal recovery

Safety invariant:
Acute threat override: Immediate preemption to AWAKE upon sudden danger.
"""
from __future__ import annotations

import dataclasses
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class SleepState(str, Enum):
    AWAKE = "awake"
    QUIET_REST = "quiet_rest"
    DROWSY = "drowsy"
    SLEEP_REPLAY = "sleep_replay"
    DREAM_GENERATION = "dream_generation"
    AWAKENING = "awakening"


class FatigueController(nn.Module):
    """Tracks homeostatic sleep pressure and governs brain state transitions."""

    def __init__(
        self,
        wake_rate_kw: float = 0.002,     # Accumulation rate per second awake
        sleep_rate_ks: float = 0.008,    # Clearance rate per second resting
        drowsy_threshold: float = 0.70,  # Sleep pressure triggering drowsiness
        sleep_threshold: float = 0.85,   # Sleep pressure triggering sleep
        wake_threshold: float = 0.20,    # Cleared pressure triggering awakening
        dt: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.wake_rate_kw = wake_rate_kw
        self.sleep_rate_ks = sleep_rate_ks
        self.drowsy_threshold = drowsy_threshold
        self.sleep_threshold = sleep_threshold
        self.wake_threshold = wake_threshold
        self.dt = dt

        self.current_state = SleepState.AWAKE
        self.register_buffer("sleep_pressure", torch.tensor(0.10))
        self.register_buffer("circadian_phase", torch.tensor(0.0))  # 0 to 2*pi
        self.register_buffer("time_in_state", torch.tensor(0.0))

        # Replay/dream cycle counter
        self._dream_cycles = 0

    def step(
        self,
        adenosine_level: Optional[float] = None,
        melatonin_level: Optional[float] = None,
        orexin_level: Optional[float] = None,
        threat_level: float = 0.0,
        dt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advances fatigue dynamics and evaluates state transitions."""
        dt = dt_override if dt_override is not None else self.dt
        p = float(self.sleep_pressure.item())

        # Circadian phase progression (24-hour cycle proxy)
        phase = float(self.circadian_phase.item()) + (dt / 86400.0) * 2.0 * np.pi
        self.circadian_phase.copy_(torch.tensor(phase % (2.0 * np.pi)))

        # 1. Acute threat override: Immediate emergency awakening
        if threat_level > 0.60 and self.current_state != SleepState.AWAKE:
            self.current_state = SleepState.AWAKE
            self.time_in_state.zero_()
            receipt = EvidenceReceipt(
                item_id="emergency_awakening_threat",
                kind=EvidenceKind.MODELLED_BIOLOGICAL,
                source="FatigueController",
                metrics={"threat_level": threat_level, "sleep_pressure": p},
            )
            return {
                "state": self.current_state.value,
                "sleep_pressure": p,
                "interrupted_by_threat": True,
                "receipt": receipt.to_dict(),
            }

        # 2. Update homeostatic pressure
        is_sleeping = self.current_state in [SleepState.SLEEP_REPLAY, SleepState.DREAM_GENERATION]
        if is_sleeping or self.current_state == SleepState.QUIET_REST:
            # Dissipation: dP/dt = - k_s * P
            p_next = p - dt * self.sleep_rate_ks * p
        else:
            # Accumulation: dP/dt = k_w
            # Accelerated by adenosine if present
            ad_boost = 1.0 + (adenosine_level if adenosine_level is not None else 0.0)
            p_next = p + dt * self.wake_rate_kw * ad_boost

        # Orexin acts as wakefulness stabilizer, reducing apparent sleepiness
        if orexin_level is not None and orexin_level > 0.6:
            p_next -= dt * 0.001 * orexin_level

        p_next = float(np.clip(p_next, 0.0, 1.0))
        self.sleep_pressure.copy_(torch.tensor(p_next))
        self.time_in_state += dt

        t_state = float(self.time_in_state.item())

        # 3. State transitions
        prev_state = self.current_state
        if self.current_state == SleepState.AWAKE:
            if p_next >= self.drowsy_threshold:
                self.current_state = SleepState.DROWSY
                self.time_in_state.zero_()
        elif self.current_state == SleepState.DROWSY:
            if p_next >= self.sleep_threshold:
                self.current_state = SleepState.SLEEP_REPLAY
                self.time_in_state.zero_()
            elif p_next < self.drowsy_threshold - 0.1:
                self.current_state = SleepState.AWAKE
                self.time_in_state.zero_()
        elif self.current_state == SleepState.SLEEP_REPLAY:
            # Replay for a period, then enter generative dream phase
            if t_state > 15.0:  # 15s replay cycle
                self.current_state = SleepState.DREAM_GENERATION
                self.time_in_state.zero_()
                self._dream_cycles += 1
        elif self.current_state == SleepState.DREAM_GENERATION:
            # Generative dream for a period, then alternate back to replay or awaken
            if t_state > 10.0:
                if p_next <= self.wake_threshold:
                    self.current_state = SleepState.AWAKENING
                    self.time_in_state.zero_()
                else:
                    self.current_state = SleepState.SLEEP_REPLAY
                    self.time_in_state.zero_()
        elif self.current_state == SleepState.AWAKENING:
            if t_state > 3.0:
                self.current_state = SleepState.AWAKE
                self.time_in_state.zero_()

        receipt = EvidenceReceipt(
            item_id="fatigue_controller_step",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="FatigueController",
            sha256_digest=compute_tensor_digest(self.sleep_pressure),
            metrics={
                "state": self.current_state.value,
                "sleep_pressure": p_next,
                "time_in_state": float(self.time_in_state.item()),
                "dream_cycles": self._dream_cycles,
            },
        )

        return {
            "state": self.current_state.value,
            "sleep_pressure": p_next,
            "state_changed": (self.current_state != prev_state),
            "dream_cycles": self._dream_cycles,
            "receipt": receipt.to_dict(),
        }

    def force_state(self, state: SleepState) -> None:
        """Explicitly forces a state for testing or external command."""
        self.current_state = state
        self.time_in_state.zero_()

    def reset(self) -> None:
        self.current_state = SleepState.AWAKE
        self.sleep_pressure.copy_(torch.tensor(0.10))
        self.circadian_phase.zero_()
        self.time_in_state.zero_()
        self._dream_cycles = 0

    reset_state = reset
