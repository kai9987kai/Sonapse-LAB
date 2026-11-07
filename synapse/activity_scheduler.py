"""Activity-Driven Neural Scheduling.

Dynamically partitions neural subgraphs into priority tiers based on recent activation
rate and event density to focus compute where state changes are occurring.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

import torch


@dataclass
class ScheduleTier:
    tier_id: int
    update_frequency: int  # 1 = every step, 2 = every other step, 4 = every 4 steps
    node_indices: List[int] = field(default_factory=list)


class ActivityScheduler:
    """Prioritizes recurrent subgraphs based on moving-average event frequency."""

    def __init__(self, num_nodes: int, ema_alpha: float = 0.05, num_tiers: int = 3):
        self.num_nodes = num_nodes
        self.ema_alpha = ema_alpha
        self.num_tiers = num_tiers
        self.activity_ema = torch.zeros(num_nodes)
        self.step_count = 0

        # Frequency mapping: Tier 0 (top active) -> 1, Tier 1 (moderate) -> 2, Tier 2 (idle) -> 4
        self.frequencies = [1, 2, 4]
        self.tiers: List[ScheduleTier] = [
            ScheduleTier(tier_id=i, update_frequency=self.frequencies[min(i, len(self.frequencies) - 1)])
            for i in range(num_tiers)
        ]
        self._repartition()

    def record_activity(self, active_deltas: torch.Tensor) -> None:
        """Update activity EMA with absolute changes."""
        abs_d = torch.abs(active_deltas).detach().cpu()
        if abs_d.shape[0] != self.num_nodes:
            return
        self.activity_ema = (1.0 - self.ema_alpha) * self.activity_ema + self.ema_alpha * abs_d

    def _repartition(self) -> None:
        """Divide nodes into tiers based on quantile activity."""
        if self.num_nodes == 0:
            return

        sorted_indices = torch.argsort(self.activity_ema, descending=True).tolist()
        chunk_size = len(sorted_indices) // self.num_tiers

        for i, tier in enumerate(self.tiers):
            if i == self.num_tiers - 1:
                tier.node_indices = sorted_indices[i * chunk_size :]
            else:
                tier.node_indices = sorted_indices[i * chunk_size : (i + 1) * chunk_size]

    def get_active_nodes_for_step(self, step: int) -> List[int]:
        """Return list of node indices scheduled to update at this step."""
        if step % 50 == 0:
            self._repartition()

        active: List[int] = []
        for tier in self.tiers:
            if step % tier.update_frequency == 0:
                active.extend(tier.node_indices)
        return active
