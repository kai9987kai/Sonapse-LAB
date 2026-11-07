"""Developmental Communication Complexity Hierarchy for Sonapse NeuroWave-X2.

Implements evidence-gated complexity development across 7 stages:
LEVEL 0: No auxiliary communication (pure baseline connectome)
LEVEL 1: Scalar residual communication
LEVEL 2: Low-dimensional vector communication
LEVEL 3: Quantized / Attention channels
LEVEL 4: Graph Laplacian spectral channels
LEVEL 5: Phase-aware spectral wave channels
LEVEL 6: Adaptive protocol mixtures + dendritic liquid integration

Promotions require explicit held-out evidence exceeding declared confidence thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DevelopmentalLevelSpec:
    level: int
    name: str
    allowed_protocols: Tuple[str, ...]
    min_evidence_score: float
    description: str


DEVELOPMENTAL_LEVELS: Dict[int, DevelopmentalLevelSpec] = {
    0: DevelopmentalLevelSpec(
        level=0,
        name="LEVEL_0_ISOLATED",
        allowed_protocols=("no_comm",),
        min_evidence_score=0.0,
        description="Isolated connectome baseline (no auxiliary communication)",
    ),
    1: DevelopmentalLevelSpec(
        level=1,
        name="LEVEL_1_SCALAR",
        allowed_protocols=("no_comm", "scalar"),
        min_evidence_score=0.05,
        description="Scalar bottleneck residual channels",
    ),
    2: DevelopmentalLevelSpec(
        level=2,
        name="LEVEL_2_VECTOR",
        allowed_protocols=("no_comm", "scalar", "vector"),
        min_evidence_score=0.10,
        description="Dense vector message channels",
    ),
    3: DevelopmentalLevelSpec(
        level=3,
        name="LEVEL_3_QUANTIZED_ATTENTION",
        allowed_protocols=("no_comm", "scalar", "vector", "quantized", "attention"),
        min_evidence_score=0.15,
        description="Quantized and attentive routed channels",
    ),
    4: DevelopmentalLevelSpec(
        level=4,
        name="LEVEL_4_SPECTRAL",
        allowed_protocols=("no_comm", "scalar", "vector", "quantized", "attention", "spectral"),
        min_evidence_score=0.20,
        description="Graph Laplacian eigenmode carriers",
    ),
    5: DevelopmentalLevelSpec(
        level=5,
        name="LEVEL_5_PHASE_SPECTRAL",
        allowed_protocols=("no_comm", "scalar", "vector", "quantized", "attention", "spectral", "phase", "phase_spectral"),
        min_evidence_score=0.25,
        description="Phase-coherence CTC + spectral wave multiplexing",
    ),
    6: DevelopmentalLevelSpec(
        level=6,
        name="LEVEL_6_ADAPTIVE_DENDRITIC",
        allowed_protocols=("no_comm", "scalar", "vector", "quantized", "attention", "spectral", "phase", "phase_spectral", "phase_spectral_dendritic"),
        min_evidence_score=0.30,
        description="Full adaptive protocol selection targeting dendritic liquid compartments",
    ),
}


class CommunicationDevelopmentEngine:
    """Controls the progressive promotion and regression of communication complexity."""

    def __init__(self, initial_level: int = 0) -> None:
        self.current_level = min(max(0, int(initial_level)), 6)
        self.promotion_history: List[Dict[str, Any]] = []

    def get_allowed_protocols(self) -> List[str]:
        return list(DEVELOPMENTAL_LEVELS[self.current_level].allowed_protocols)

    def evaluate_promotion(
        self,
        candidate_level: int,
        held_out_score_delta: float,
        cost_penalty: float,
    ) -> Tuple[bool, str]:
        """Evaluates whether to promote communication complexity to candidate_level."""
        if candidate_level <= self.current_level:
            return False, f"Candidate level {candidate_level} <= current level {self.current_level}"
        if candidate_level > 6:
            return False, "Exceeds maximum developmental level 6"

        spec = DEVELOPMENTAL_LEVELS[candidate_level]
        net_score = held_out_score_delta - cost_penalty

        if net_score >= spec.min_evidence_score:
            old_lvl = self.current_level
            self.current_level = candidate_level
            receipt = {
                "from_level": old_lvl,
                "to_level": candidate_level,
                "net_score": float(net_score),
                "threshold": float(spec.min_evidence_score),
                "status": "promoted",
            }
            self.promotion_history.append(receipt)
            return True, f"Promoted from level {old_lvl} to {candidate_level} (net score {net_score:.4f} >= {spec.min_evidence_score})"
        else:
            receipt = {
                "from_level": self.current_level,
                "candidate_level": candidate_level,
                "net_score": float(net_score),
                "threshold": float(spec.min_evidence_score),
                "status": "rejected",
            }
            self.promotion_history.append(receipt)
            return False, f"Promotion rejected: net score {net_score:.4f} < required threshold {spec.min_evidence_score}"
