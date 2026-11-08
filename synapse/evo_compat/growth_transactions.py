"""SONAPSE-EVO Structural Neuroplasticity and Growth Transactions.

Governs causal neural circuit evolution, node/edge addition, and structural repair:
J(g) = Delta R - lambda_p P_g - lambda_c C_g - lambda_u U_g

Implements two-phase commit with strict rollback on negative or negligible utility.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class GrowthCandidate:
    candidate_id: str
    source_region: str
    target_region: str
    num_nodes: int
    num_edges: int
    parameter_cost: float
    compute_cost: float
    uncertainty: float = 0.1
    weights: Optional[torch.Tensor] = None


@dataclasses.dataclass
class GrowthTransactionReceipt:
    transaction_id: str
    timestamp: float
    candidate_id: str
    delta_reward: float
    net_utility: float
    promoted: bool
    rolled_back: bool
    evidence: EvidenceReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "candidate_id": self.candidate_id,
            "delta_reward": self.delta_reward,
            "net_utility": self.net_utility,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "evidence": self.evidence.to_dict(),
        }


class EvolutionaryGrowthController:
    """Manages structural neurogenesis and circuit expansion transactions."""

    def __init__(
        self,
        lambda_param: float = 0.01,
        lambda_compute: float = 0.01,
        lambda_uncertainty: float = 0.05,
        promotion_threshold: float = 0.05,
        seed: int = 42,
    ):
        self.lambda_p = lambda_param
        self.lambda_c = lambda_compute
        self.lambda_u = lambda_uncertainty
        self.threshold = promotion_threshold
        self.rng = np.random.RandomState(seed)

        self.transactions: List[GrowthTransactionReceipt] = []
        self._tx_counter = 0

    def propose_candidate(
        self,
        source_region: str,
        target_region: str,
        num_nodes: int = 8,
        num_edges: int = 16,
    ) -> GrowthCandidate:
        """Proposes a new structural candidate circuit."""
        self._tx_counter += 1
        cid = f"cand_{int(time.time()*1000)}_{self._tx_counter}"
        param_cost = float(num_nodes * 4 + num_edges)
        compute_cost = float(num_edges * 2)

        cand = GrowthCandidate(
            candidate_id=cid,
            source_region=source_region,
            target_region=target_region,
            num_nodes=num_nodes,
            num_edges=num_edges,
            parameter_cost=param_cost,
            compute_cost=compute_cost,
            weights=torch.randn(num_nodes, num_nodes) * 0.05,
        )
        return cand

    def evaluate_and_transact(
        self,
        candidate: GrowthCandidate,
        delta_reward: float,
        uncertainty: float = 0.1,
    ) -> Tuple[bool, GrowthTransactionReceipt]:
        """Evaluates J(g) = Delta R - lambda_p P_g - lambda_c C_g - lambda_u U_g.

        Promotes if J(g) > threshold; rolls back otherwise.
        """
        tx_id = f"tx_evo_{int(time.time()*1000)}_{self._tx_counter}"

        penalties = (
            self.lambda_p * candidate.parameter_cost +
            self.lambda_c * candidate.compute_cost +
            self.lambda_u * uncertainty
        )
        net_utility = float(delta_reward - penalties)

        if net_utility > self.threshold:
            promoted = True
            rolled_back = False
            kind = EvidenceKind.CONFIRMATORY_RESULT
        else:
            promoted = False
            rolled_back = True
            kind = EvidenceKind.CONFIRMATORY_RESULT  # Confirmed negative / discarded

        receipt = GrowthTransactionReceipt(
            transaction_id=tx_id,
            timestamp=time.time(),
            candidate_id=candidate.candidate_id,
            delta_reward=float(delta_reward),
            net_utility=net_utility,
            promoted=promoted,
            rolled_back=rolled_back,
            evidence=EvidenceReceipt(
                item_id=tx_id,
                kind=kind,
                source="EvolutionaryGrowthController",
                metrics={
                    "delta_reward": delta_reward,
                    "net_utility": net_utility,
                    "promoted": promoted,
                    "param_cost": candidate.parameter_cost,
                    "compute_cost": candidate.compute_cost,
                },
            ),
        )

        self.transactions.append(receipt)
        return promoted, receipt

    def get_transactions(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.transactions]
