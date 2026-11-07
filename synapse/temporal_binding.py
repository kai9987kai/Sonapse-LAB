"""Temporal Order & Episode Sequence Associative Binding.

Encodes episodic sequences:
    S = sum_t [ State_t (*) Action_t (*) Order_t ]
Supports queries:
- What action occurred at time t?
- Which state preceded an event?
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .relational_memory import RelationalMemoryBank


class TemporalSequenceBinder:
    """Binds multi-step action-observation trajectories with temporal order tags."""

    def __init__(self, memory_bank: Optional[RelationalMemoryBank] = None):
        self.bank = memory_bank or RelationalMemoryBank()
        # Temporal position tags P_0, P_1, ...
        self.time_tags = [self.bank._make_unitary() for _ in range(32)]
        self.trajectory_trace = np.zeros(self.bank.dim, dtype=np.float32)
        self.stored_steps = 0

    def record_step(self, obs_symbol: np.ndarray, act_symbol: np.ndarray, step_idx: int) -> None:
        idx = min(step_idx, len(self.time_tags) - 1)
        tag = self.time_tags[idx]
        bound_pair = self.bank.circular_conv(obs_symbol, act_symbol)
        bound_temporal = self.bank.circular_conv(bound_pair, tag)
        self.trajectory_trace += bound_temporal
        self.stored_steps += 1

    def query_action_at_step(self, obs_symbol: np.ndarray, step_idx: int) -> np.ndarray:
        idx = min(step_idx, len(self.time_tags) - 1)
        tag = self.time_tags[idx]
        # Unbind order tag, then unbind observation symbol
        recovered_pair = self.bank.circular_conv(self.trajectory_trace, self.bank.involution_inverse(tag))
        recovered_act = self.bank.circular_conv(recovered_pair, self.bank.involution_inverse(obs_symbol))
        return recovered_act
