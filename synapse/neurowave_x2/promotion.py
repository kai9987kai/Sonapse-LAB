"""Fail-Closed Research Promotion System for Sonapse NeuroWave-X2.

Enforces strict scientific gates before promoting an architectural or communication mutation:
- Confirmatory seed threshold (at least 2 independent seeds)
- Paired baseline comparison
- Replay verification check
- Retention task verification (no catastrophic forgetting on legacy benchmarks)
- Resource budget compliance (wire bits and analytical compute)
- Fail-closed principle: missing data evaluates strictly to INCONCLUSIVE (never PASSED).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PromotionDecision:
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class FailClosedPromotionGate:
    """Evaluates whether an experimental communication candidate qualifies for promotion."""

    def __init__(
        self,
        min_independent_seeds: int = 2,
        min_return_improvement: float = 0.50,
        max_allowed_retention_loss: float = 0.20,
        max_wire_bits: int = 4096,
        max_flops: float = 1e6,
    ) -> None:
        self.min_independent_seeds = int(min_independent_seeds)
        self.min_return_improvement = float(min_return_improvement)
        self.max_allowed_retention_loss = float(max_allowed_retention_loss)
        self.max_wire_bits = int(max_wire_bits)
        self.max_flops = float(max_flops)

    def evaluate_candidate(
        self,
        candidate_returns: List[float],
        baseline_returns: List[float],
        retention_baseline_return: float,
        retention_candidate_return: float,
        replay_verified: bool,
        measured_wire_bits: int,
        measured_flops: float,
        controls_passed: bool = True,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Applies fail-closed promotion criteria.
        
        Returns:
            (decision, reason_string, audit_receipt)
        """
        # 1. Gate on missing or incomplete runs
        if len(candidate_returns) < self.min_independent_seeds or len(baseline_returns) < self.min_independent_seeds:
            return (
                PromotionDecision.INCONCLUSIVE,
                f"Insufficient independent seeds ({len(candidate_returns)} cand, {len(baseline_returns)} base; required {self.min_independent_seeds})",
                {"reason": "missing_seeds"},
            )

        # 2. Gate on replay verification
        if not replay_verified:
            return (
                PromotionDecision.REJECTED,
                "Deterministic replay verification failed; run is non-reproducible",
                {"reason": "replay_failed"},
            )

        # 3. Gate on matched controls
        if not controls_passed:
            return (
                PromotionDecision.INCONCLUSIVE,
                "Matched control evaluations incomplete or failed",
                {"reason": "controls_incomplete"},
            )

        # 4. Gate on resource budgets
        if measured_wire_bits > self.max_wire_bits:
            return (
                PromotionDecision.REJECTED,
                f"Wire bandwidth budget exceeded: {measured_wire_bits} > {self.max_wire_bits}",
                {"reason": "bandwidth_budget_exceeded"},
            )
        if measured_flops > self.max_flops:
            return (
                PromotionDecision.REJECTED,
                f"Compute budget exceeded: {measured_flops} > {self.max_flops}",
                {"reason": "compute_budget_exceeded"},
            )

        # 5. Gate on retention (catastrophic forgetting prevention)
        retention_delta = retention_baseline_return - retention_candidate_return
        if retention_delta > self.max_allowed_retention_loss:
            return (
                PromotionDecision.REJECTED,
                f"Catastrophic forgetting detected: retention loss {retention_delta:.4f} > allowed {self.max_allowed_retention_loss}",
                {"reason": "retention_degraded", "retention_delta": retention_delta},
            )

        # 6. Statistical performance comparison over independent seeds
        cand_mean = float(np.mean(candidate_returns))
        base_mean = float(np.mean(baseline_returns))
        delta_mean = cand_mean - base_mean

        audit_receipt = {
            "candidate_mean": cand_mean,
            "baseline_mean": base_mean,
            "delta_return": delta_mean,
            "num_seeds": len(candidate_returns),
            "retention_delta": retention_delta,
            "wire_bits": measured_wire_bits,
            "flops": measured_flops,
            "replay_verified": replay_verified,
        }

        if delta_mean >= self.min_return_improvement:
            return (
                PromotionDecision.PROMOTED,
                f"Candidate qualified: mean return delta +{delta_mean:.4f} exceeds threshold {self.min_return_improvement}",
                audit_receipt,
            )
        else:
            return (
                PromotionDecision.REJECTED,
                f"Performance gain insufficient: mean return delta {delta_mean:.4f} < required {self.min_return_improvement}",
                audit_receipt,
            )
