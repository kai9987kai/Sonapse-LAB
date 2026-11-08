"""Spontaneous Cross-Model Neural Exchange Controller.

Governs peer-to-peer synaptic exchange, low-rank parameter transfer,
and causal validation with atomic snapshot rollback.

STRICT CAUSAL VERIFICATION:
Candidate low-rank updates Delta W = U V^T are evaluated on a shadow validation batch.
If loss increases or reward degrades, an atomic rollback restores previous weights
and records a negative-result receipt.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class ExchangeReceipt:
    exchange_id: str
    timestamp: float
    source_model: str
    target_model: str
    delta_rank: int
    delta_frobenius_norm: float
    validated_improvement: bool
    performance_delta: float
    promoted: bool
    rolled_back: bool
    evidence: EvidenceReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "timestamp": self.timestamp,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "delta_rank": self.delta_rank,
            "delta_frobenius_norm": self.delta_frobenius_norm,
            "validated_improvement": self.validated_improvement,
            "performance_delta": self.performance_delta,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "evidence": self.evidence.to_dict(),
        }


class SpontaneousNeuralExchangeController(nn.Module):
    """Controls cross-model parameter and representation exchange."""

    def __init__(
        self,
        feature_dim: int = 64,
        rank: int = 4,
        exchange_threshold: float = 0.65,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.feature_dim = feature_dim
        self.rank = rank
        self.exchange_threshold = exchange_threshold

        # Exchange gating network: evaluates compatibility of two systems
        self.gate_net = nn.Sequential(
            nn.Linear(2 * feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Low-rank delta generator U in R^{dim x rank}, V in R^{dim x rank}
        self.u_net = nn.Linear(feature_dim, feature_dim * rank, bias=False)
        self.v_net = nn.Linear(feature_dim, feature_dim * rank, bias=False)

        self.exchange_history: List[ExchangeReceipt] = []
        self._exchange_counter = 0

    def evaluate_exchange_readiness(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
    ) -> Tuple[bool, float]:
        """Calculates exchange probability p_{ij} = sigmoid(gate_net([h_i, h_j]))."""
        device = next(self.parameters()).device
        pair = torch.cat([h_source.to(device).flatten()[:self.feature_dim], h_target.to(device).flatten()[:self.feature_dim]], dim=0)
        prob = float(self.gate_net(pair.unsqueeze(0)).item())
        should_exchange = prob >= self.exchange_threshold
        return should_exchange, prob

    def generate_candidate_delta(
        self,
        h_source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates low-rank factors U (dim x rank) and V (dim x rank)."""
        device = next(self.parameters()).device
        x = h_source.to(device).flatten()[:self.feature_dim].unsqueeze(0)
        u = self.u_net(x).view(self.feature_dim, self.rank)
        v = self.v_net(x).view(self.feature_dim, self.rank)
        # Scale to ensure modest, non-destructive initial perturbations
        return u * 0.05, v * 0.05

    def execute_transaction(
        self,
        base_weight: torch.Tensor,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
        eval_fn: Any,  # Callable[[torch.Tensor], float] -> loss/cost (lower is better)
        source_id: str = "donor_subsystem",
        target_id: str = "recipient_connectome",
    ) -> Tuple[torch.Tensor, ExchangeReceipt]:
        """Executes full atomic exchange transaction with shadow validation and rollback."""
        self._exchange_counter += 1
        exchange_id = f"exch_{int(time.time()*1000)}_{self._exchange_counter}"

        # 1. Check readiness
        ready, prob = self.evaluate_exchange_readiness(h_source, h_target)
        if not ready:
            receipt = ExchangeReceipt(
                exchange_id=exchange_id,
                timestamp=time.time(),
                source_model=source_id,
                target_model=target_id,
                delta_rank=self.rank,
                delta_frobenius_norm=0.0,
                validated_improvement=False,
                performance_delta=0.0,
                promoted=False,
                rolled_back=False,
                evidence=EvidenceReceipt(
                    item_id=exchange_id,
                    kind=EvidenceKind.EXPLORATORY_RESULT,
                    source="SpontaneousNeuralExchangeController",
                    metrics={"reason": "threshold_not_met", "prob": prob},
                ),
            )
            self.exchange_history.append(receipt)
            return base_weight, receipt

        # 2. Snapshot current base weight
        snapshot = base_weight.clone()
        base_cost = eval_fn(snapshot)

        # 3. Generate candidate low-rank delta Delta W = U V^T
        u, v = self.generate_candidate_delta(h_source)
        delta_w = torch.matmul(u, v.t())
        frob_norm = float(torch.norm(delta_w, p="fro").item())

        candidate_weight = base_weight + delta_w
        candidate_cost = eval_fn(candidate_weight)
        perf_delta = float(base_cost - candidate_cost)  # Positive = improvement

        # 4. Causal safety check: promote only if performance improved
        if perf_delta > 1e-5:
            # Promote
            promoted = True
            rolled_back = False
            final_weight = candidate_weight
            kind = EvidenceKind.CONFIRMATORY_RESULT
        else:
            # Atomic rollback
            promoted = False
            rolled_back = True
            final_weight = snapshot
            kind = EvidenceKind.CONFIRMATORY_RESULT  # Confirmed negative result

        receipt = ExchangeReceipt(
            exchange_id=exchange_id,
            timestamp=time.time(),
            source_model=source_id,
            target_model=target_id,
            delta_rank=self.rank,
            delta_frobenius_norm=frob_norm,
            validated_improvement=promoted,
            performance_delta=perf_delta,
            promoted=promoted,
            rolled_back=rolled_back,
            evidence=EvidenceReceipt(
                item_id=exchange_id,
                kind=kind,
                source="SpontaneousNeuralExchangeController",
                sha256_digest=compute_tensor_digest(final_weight),
                metrics={
                    "base_cost": base_cost,
                    "candidate_cost": candidate_cost,
                    "perf_delta": perf_delta,
                    "promoted": promoted,
                },
            ),
        )

        self.exchange_history.append(receipt)
        return final_weight, receipt

    def get_history(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.exchange_history]
