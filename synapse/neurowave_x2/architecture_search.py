"""Communication-Aware Architecture Search & Pareto Frontier for Sonapse NeuroWave-X2.

Defines typed CommunicationMutation objects, generates bounded structural proposals,
and tracks the non-dominated Pareto frontier across (performance, bits, FLOPs, latency, retention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CommunicationMutation:
    """Typed, reproducible representation of an architectural communication mutation."""

    mutation_id: str
    parent_hash: str
    mutation_type: str  # 'add_route', 'remove_route', 'change_protocol', 'change_modes', 'change_precision'
    target_route: Optional[Tuple[int, int]]
    old_value: Any
    new_value: Any
    parameter_delta: int
    estimated_compute_delta: float
    estimated_bandwidth_delta: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "parent_hash": self.parent_hash,
            "mutation_type": self.mutation_type,
            "target_route": list(self.target_route) if self.target_route else None,
            "old_value": str(self.old_value),
            "new_value": str(self.new_value),
            "parameter_delta": self.parameter_delta,
            "estimated_compute_delta": self.estimated_compute_delta,
            "estimated_bandwidth_delta": self.estimated_bandwidth_delta,
            "reason": self.reason,
        }


@dataclass
class ParetoCandidate:
    candidate_id: str
    config_dict: Dict[str, Any]
    return_score: float  # higher is better
    wire_bits: float      # lower is better
    flops: float          # lower is better
    latency_ms: float     # lower is better
    retention_score: float  # higher is better


class ParetoFrontierTracker:
    """Maintains the multi-objective non-dominated Pareto frontier of evaluated communication configurations."""

    def __init__(self) -> None:
        self.candidates: List[ParetoCandidate] = []

    def add_candidate(self, cand: ParetoCandidate) -> bool:
        """Adds candidate if not strictly dominated by any existing member.
        Prunes members dominated by the new candidate.
        
        Criteria:
            A dominates B if A is >= B in (return, retention) and <= B in (bits, flops, latency)
            with at least one strict inequality.
        """
        for other in self.candidates:
            if self._dominates(other, cand):
                return False  # cand is dominated by other, do not add

        # Remove candidates dominated by the new candidate
        survivors = [c for c in self.candidates if not self._dominates(cand, c)]
        survivors.append(cand)
        self.candidates = survivors
        return True

    @staticmethod
    def _dominates(a: ParetoCandidate, b: ParetoCandidate) -> bool:
        better_or_equal = (
            a.return_score >= b.return_score
            and a.retention_score >= b.retention_score
            and a.wire_bits <= b.wire_bits
            and a.flops <= b.flops
            and a.latency_ms <= b.latency_ms
        )
        strictly_better = (
            a.return_score > b.return_score
            or a.retention_score > b.retention_score
            or a.wire_bits < b.wire_bits
            or a.flops < b.flops
            or a.latency_ms < b.latency_ms
        )
        return better_or_equal and strictly_better

    def get_frontier(self) -> List[ParetoCandidate]:
        return list(self.candidates)
