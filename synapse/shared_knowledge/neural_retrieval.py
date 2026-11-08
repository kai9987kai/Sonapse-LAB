"""Neural Knowledge Pool and Soft-Attention Retrieval.

Maintains a pool of KnowledgeRecords and performs soft-attention-based retrieval:
s_i = (q^T k_i) / (sqrt(d) * tau)
alpha = softmax(s)
m_t = sum_i alpha_i * v_i
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.shared_knowledge.records import KnowledgeRecord, KnowledgeContentType
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class NeuralKnowledgePool(nn.Module):
    """Structured knowledge pool with continuous attention retrieval."""

    def __init__(
        self,
        dim: int = 64,
        capacity: int = 500,
        temperature: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.dim = dim
        self.capacity = capacity
        self.temperature = temperature

        self.records: List[KnowledgeRecord] = []
        self._record_map: Dict[str, KnowledgeRecord] = {}

    def add_record(self, record: KnowledgeRecord) -> bool:
        """Adds or updates a record in the pool."""
        if len(record.key_vector) != self.dim or len(record.value_vector) != self.dim:
            # Pad or truncate to self.dim
            k = list(record.key_vector)[:self.dim] + [0.0] * max(0, self.dim - len(record.key_vector))
            v = list(record.value_vector)[:self.dim] + [0.0] * max(0, self.dim - len(record.value_vector))
            record.key_vector = k
            record.value_vector = v

        if record.record_id in self._record_map:
            # Update existing
            idx = [i for i, r in enumerate(self.records) if r.record_id == record.record_id][0]
            self.records[idx] = record
            self._record_map[record.record_id] = record
            return True

        if len(self.records) >= self.capacity:
            # Evict least recently accessed record
            lru_idx = int(np.argmin([r.last_accessed_time for r in self.records]))
            evicted = self.records.pop(lru_idx)
            self._record_map.pop(evicted.record_id, None)

        self.records.append(record)
        self._record_map[record.record_id] = record
        return True

    def query_attention(
        self,
        query_vec: torch.Tensor,
        temp_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[KnowledgeRecord, float]], Dict[str, Any]]:
        """Performs soft-attention retrieval over the entire knowledge pool.

        Returns:
            retrieved_vec: (dim,) continuous memory representation m_t
            top_records: List of (record, attention_weight)
            receipt: Cryptographic provenance receipt
        """
        device = query_vec.device
        if not self.records:
            empty_vec = torch.zeros(self.dim, device=device)
            receipt = EvidenceReceipt(
                item_id="empty_pool_query",
                kind=EvidenceKind.TRAINED_COMPUTATIONAL,
                source="NeuralKnowledgePool",
                metrics={"pool_size": 0},
            )
            return empty_vec, [], receipt.to_dict()

        tau = temp_override if temp_override is not None else self.temperature
        q = query_vec.to(device).flatten()[:self.dim]

        # Stack keys and values
        keys = torch.tensor([r.key_vector for r in self.records], dtype=torch.float32, device=device)
        vals = torch.tensor([r.value_vector for r in self.records], dtype=torch.float32, device=device)

        # Dot-product similarities scaled by sqrt(dim) and temperature
        scale = math.sqrt(self.dim) * tau
        scores = torch.matmul(keys, q) / max(scale, 1e-6)

        # Confidence weighting
        conf = torch.tensor([r.confidence for r in self.records], dtype=torch.float32, device=device)
        weighted_scores = scores + torch.log(torch.clamp(conf, min=1e-4))

        # Softmax weights alpha
        alpha = F.softmax(weighted_scores, dim=0)

        # Attentive sum m_t = sum_i alpha_i * v_i
        retrieved_vec = torch.matmul(alpha.unsqueeze(0), vals).squeeze(0)

        # Touch and gather top retrieved records
        top_k = min(3, len(self.records))
        top_vals, top_indices = torch.topk(alpha, k=top_k)

        top_matches = []
        for weight, idx in zip(top_vals.tolist(), top_indices.tolist()):
            rec = self.records[idx]
            rec.touch()
            top_matches.append((rec, float(weight)))

        receipt = EvidenceReceipt(
            item_id="knowledge_attention_retrieval",
            kind=EvidenceKind.TRAINED_COMPUTATIONAL,
            source="NeuralKnowledgePool",
            sha256_digest=compute_tensor_digest(retrieved_vec),
            metrics={
                "pool_size": len(self.records),
                "top_score": float(top_vals[0].item()) if top_k > 0 else 0.0,
                "top_record_id": top_matches[0][0].record_id if top_matches else "none",
            },
        )

        return retrieved_vec, top_matches, receipt.to_dict()

    def get_summary(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]

    def size(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()
        self._record_map.clear()
