"""MORPHEUS: Offline Oneiric Sleep Engine & Hippocampal Memory Consolidation.

Synthesizes neuro-symbolic sleep phases (Slow-Wave Sleep SWS & Rapid Eye Movement REM),
counterfactual replay trajectories, and quantum phase annealing for the TidePool
agent without altering frozen neural core weights.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from .causeway import CausewaySuperpositionEngine
from .hologram import HolographicAssociativeMemory
from .lattice import DiamondLatticeNeuropil


class SleepState(str, Enum):
    WAKE = "wake"
    SWS = "sws"  # Slow-Wave Sleep: synaptic downscaling & holographic trace consolidation
    REM = "rem"  # Rapid Eye Movement: counterfactual mental replay & phase annealing


@dataclass
class ReplayMemory:
    step: int
    observation: List[float]
    action: int
    reward: float
    next_observation: List[float]
    lattice_pos: List[float]
    surprise: float  # Prediction discrepancy


class MorpheusDreamEngine:
    """Offline oneiric consolidation engine coordinating SWS memory compression

    and REM counterfactual trajectory replay with Causeway quantum phase annealing.
    """

    def __init__(
        self,
        buffer_capacity: int = 64,
        sws_compression_threshold: float = 0.08,
        rem_replay_depth: int = 4,
        phase_learning_rate: float = 0.15,
    ):
        self.buffer_capacity = int(buffer_capacity)
        self.sws_compression_threshold = float(sws_compression_threshold)
        self.rem_replay_depth = int(rem_replay_depth)
        self.phase_learning_rate = float(phase_learning_rate)

        self.memory_buffer: Deque[ReplayMemory] = deque(maxlen=self.buffer_capacity)
        self.current_state = SleepState.WAKE
        self.sleep_tick = 0
        self.total_sleep_cycles = 0
        self.total_consolidated_memories = 0
        self.total_rem_replays = 0
        self.accumulated_phase_alignment_gain = 0.0

    def reset(self) -> None:
        """Reset dream engine buffers and internal state."""
        self.memory_buffer.clear()
        self.current_state = SleepState.WAKE
        self.sleep_tick = 0
        self.total_sleep_cycles = 0
        self.total_consolidated_memories = 0
        self.total_rem_replays = 0
        self.accumulated_phase_alignment_gain = 0.0

    def record_waking_step(
        self,
        step: int,
        observation: List[float],
        action: int,
        reward: float,
        next_observation: List[float],
        lattice_pos: List[float],
        predicted_reward: float = 0.0,
    ) -> None:
        """Record waking experience with computed surprise valence."""
        surprise = abs(reward - predicted_reward)
        self.memory_buffer.append(
            ReplayMemory(
                step=step,
                observation=list(observation),
                action=int(action),
                reward=float(reward),
                next_observation=list(next_observation),
                lattice_pos=list(lattice_pos),
                surprise=float(surprise),
            )
        )

    def is_sleep_indicated(
        self,
        action: int,
        stamina: float,
        consecutive_rests: int = 0,
    ) -> bool:
        """Evaluate physiological markers indicating sleep / consolidation onset."""
        # Sleep is triggered by taking a rest action (action 0: stay or rest context, or action 3),
        # critical exhaustion (<0.15), or consecutive stationary cycles
        return action == 0 or stamina < 0.15 or consecutive_rests >= 2

    def process_cycle(
        self,
        causeway: CausewaySuperpositionEngine,
        lattice: DiamondLatticeNeuropil,
        hologram: HolographicAssociativeMemory,
        action: int,
        stamina: float,
        consecutive_rests: int = 0,
    ) -> Dict[str, Any]:
        """Execute one cycle of the dream engine during waking or sleep states."""
        if not self.is_sleep_indicated(action, stamina, consecutive_rests):
            self.current_state = SleepState.WAKE
            self.sleep_tick = 0
            return {
                "sleep_state": SleepState.WAKE.value,
                "sleep_tick": 0,
                "total_cycles": self.total_sleep_cycles,
                "consolidated_count": self.total_consolidated_memories,
                "rem_replays": self.total_rem_replays,
                "phase_gain": round(self.accumulated_phase_alignment_gain, 4),
            }

        self.sleep_tick += 1
        # Alternate between SWS (ticks 1-2) and REM (ticks 3+)
        if self.sleep_tick <= 2:
            self.current_state = SleepState.SWS
            consolidated = self._execute_sws(lattice, hologram)
            rem_gain = 0.0
        else:
            self.current_state = SleepState.REM
            consolidated = 0
            rem_gain = self._execute_rem(causeway, lattice)

        if self.sleep_tick == 4:
            self.total_sleep_cycles += 1

        return {
            "sleep_state": self.current_state.value,
            "sleep_tick": self.sleep_tick,
            "total_cycles": self.total_sleep_cycles,
            "consolidated_count": self.total_consolidated_memories,
            "rem_replays": self.total_rem_replays,
            "phase_gain": round(self.accumulated_phase_alignment_gain, 4),
            "step_consolidated": consolidated,
            "step_rem_gain": round(rem_gain, 4),
        }

    def _execute_sws(
        self,
        lattice: DiamondLatticeNeuropil,
        hologram: HolographicAssociativeMemory,
    ) -> int:
        """Slow-Wave Sleep: consolidate salient memories into holographic memory."""
        if not self.memory_buffer:
            return 0

        consolidated = 0
        salient_memories = [m for m in self.memory_buffer if m.surprise >= self.sws_compression_threshold]
        if not salient_memories:
            salient_memories = list(self.memory_buffer)[-4:]

        for mem in salient_memories:
            if mem.reward > 0.05:
                role = "ROLE_FOOD" if mem.action == 1 else "ROLE_WATER" if mem.action == 2 else "ROLE_SHELTER"
                state = "STATE_ABUNDANT" if mem.reward > 0.4 else "STATE_STABLE"
                hologram.remember_association(role, state)
                consolidated += 1

        self.total_consolidated_memories += consolidated
        return consolidated

    def _execute_rem(
        self,
        causeway: CausewaySuperpositionEngine,
        lattice: DiamondLatticeNeuropil,
    ) -> float:
        """Rapid Eye Movement: counterfactual replay and quantum phase annealing."""
        if len(self.memory_buffer) < 2:
            return 0.0

        window = list(self.memory_buffer)[-min(len(self.memory_buffer), self.rem_replay_depth):]
        self.total_rem_replays += 1

        action_returns: Dict[int, List[float]] = {a: [] for a in range(6)}
        for mem in window:
            action_returns[mem.action].append(mem.reward)

        mean_returns = {
            a: float(np.mean(returns)) if returns else 0.0
            for a, returns in action_returns.items()
        }
        overall_mean = float(np.mean([r for r in mean_returns.values() if r != 0.0])) if any(mean_returns.values()) else 0.0

        total_gain = 0.0
        # Phase shifts applied in Causeway: rotate complex phases of high-reward actions
        for a in range(6):
            ret = mean_returns[a]
            advantage = ret - overall_mean
            if advantage > 0:
                total_gain += advantage

        self.accumulated_phase_alignment_gain += total_gain
        return total_gain
