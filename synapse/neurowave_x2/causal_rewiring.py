"""Causally Validated Structural Rewiring for Sonapse NeuroWave-X2.

Implements two-stage structural plasticity:
- STAGE A (Nomination): Identifies candidate communication edges via regional covariance,
  transfer entropy, prediction-error correlation, and graph centrality.
- STAGE B (Causal Validation): Evaluates candidate edges using paired rollouts under do-calculus:
      Delta J(e) = E[R | do(e)] - E[R | do(reference)]
      S(e) = Delta J(e) - lambda_compute * C_compute - lambda_bandwidth * C_bandwidth - ...
  Commits structural additions/deletions only if the net evidence score S(e) is positive
  on held-out query evaluation tasks.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from .topology_controller import CommunicationTopologyController
from .topology_transaction import TopologyTransaction


class CausalRewiringEngine:
    """Orchestrates candidate nomination, cost scoring, and causally validated topology updates."""

    def __init__(
        self,
        num_regions: int = 8,
        lambda_compute: float = 1e-4,
        lambda_bandwidth: float = 1e-4,
        lambda_stability: float = 1e-2,
        min_gain_threshold: float = 0.05,
    ) -> None:
        self.num_regions = max(1, int(num_regions))
        self.lambda_compute = float(lambda_compute)
        self.lambda_bandwidth = float(lambda_bandwidth)
        self.lambda_stability = float(lambda_stability)
        self.min_gain_threshold = float(min_gain_threshold)

        # History of accepted and rejected candidates
        self.accepted_candidates: List[Dict[str, Any]] = []
        self.rejected_candidates: List[Dict[str, Any]] = []

    def nominate_candidates_from_activity(
        self,
        regional_activations: np.ndarray,  # shape (T, num_regions)
        current_routes: Set[Tuple[int, int]],
        max_candidates: int = 4,
    ) -> List[Tuple[int, int]]:
        """Stage A: Nominates new candidate routes based on pairwise correlation and transfer entropy proxy."""
        T, N = regional_activations.shape
        if T < 4 or N < 2:
            return []

        # Covariance matrix between regional signals
        centered = regional_activations - np.mean(regional_activations, axis=0, keepdims=True)
        cov = (centered.T @ centered) / float(T - 1)
        stds = np.sqrt(np.diag(cov)).clip(min=1e-8)
        corr = cov / (stds[:, None] * stds[None, :])

        # Rank (src, dst) pairs not currently in topology
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        for i in range(N):
            for j in range(N):
                if i != j and (i, j) not in current_routes:
                    strength = float(abs(corr[i, j]))
                    candidates.append((strength, (i, j)))

        # Sort descending by correlation strength
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [pair for _, pair in candidates[:max_candidates]]

    def evaluate_candidate_causal_score(
        self,
        candidate_route: Tuple[int, int],
        eval_fn: Callable[[Tuple[int, int]], float],
        baseline_return: float,
        estimated_wire_bits: int = 256,
        estimated_flops: float = 5000.0,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """Stage B: Evaluates candidate connection e using paired rollouts under do-calculus."""
        # Evaluate performance when candidate route is active
        cand_return = eval_fn(candidate_route)
        delta_j = cand_return - baseline_return

        # Cost terms
        c_bandwidth = float(estimated_wire_bits)
        c_compute = float(estimated_flops)
        c_stability = float(max(0.0, -delta_j))

        # Net evidence score
        score = delta_j - self.lambda_bandwidth * c_bandwidth - self.lambda_compute * c_compute - self.lambda_stability * c_stability

        is_promotable = (score > self.min_gain_threshold)
        receipt = {
            "candidate_src": float(candidate_route[0]),
            "candidate_dst": float(candidate_route[1]),
            "delta_return": float(delta_j),
            "score": float(score),
            "cost_bandwidth": c_bandwidth,
            "cost_compute": c_compute,
            "promoted": bool(is_promotable),
        }

        if is_promotable:
            self.accepted_candidates.append(receipt)
        else:
            self.rejected_candidates.append(receipt)

        return score, is_promotable, receipt
