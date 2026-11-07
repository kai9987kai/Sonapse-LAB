"""Compositional Retrieval with Codebook Cleanup Memory.

Implements vector-symbol codebook cleanup memory, unbinding, and cosine-similarity
ranking for Holographic Reduced Representations (HRR).

Scientific note: This provides neuro-symbolic associative memory and cleanup lookup
for vector representations; it does not claim human-level episodic recall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.relational_memory import HRRCore, RelationalMemoryBank


@dataclass
class CleanupMatch:
    symbol: str
    score: float
    vector: torch.Tensor
    is_exact_match: bool


class CodebookCleanupMemory(nn.Module):
    """Vector-symbol codebook cleanup memory with cosine-similarity ranking and threshold lookup."""

    def __init__(self, dim: int = 256, similarity_threshold: float = 0.25):
        super().__init__()
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.codebook: Dict[str, torch.Tensor] = {}

    def register_symbol(self, symbol: str, vector: Any) -> None:
        """Register or update a symbol vector in the codebook."""
        if isinstance(vector, np.ndarray):
            v = torch.from_numpy(vector).float()
        else:
            v = vector.detach().clone().float()
        if v.shape[-1] != self.dim:
            raise ValueError(f"Vector dim {v.shape[-1]} does not match codebook dim {self.dim}")
        # Normalize
        norm = torch.norm(v, p=2)
        if norm > 1e-8:
            v = v / norm
        self.codebook[symbol] = v

    def register_symbols(self, symbols: Dict[str, Any]) -> None:
        for k, v in symbols.items():
            self.register_symbol(k, v)

    def lookup(
        self, query_vec: Any, top_k: int = 5, temperature: float = 0.1
    ) -> List[CleanupMatch]:
        """Rank codebook symbols by cosine similarity against noisy query vector."""
        if not self.codebook:
            return []

        if isinstance(query_vec, np.ndarray):
            q = torch.from_numpy(query_vec).float()
        else:
            q = query_vec.detach().clone().float()
        q_norm = torch.norm(q, p=2)
        if q_norm > 1e-8:
            q = q / q_norm

        symbols = list(self.codebook.keys())
        stacked = torch.stack([self.codebook[s] for s in symbols], dim=0)  # [N, D]

        # Cosine similarity
        scores = torch.mv(stacked, q).cpu()  # [N]

        k = min(top_k, len(symbols))
        top_scores, top_indices = torch.topk(scores, k=k)

        matches = []
        for s, idx in zip(top_scores.tolist(), top_indices.tolist()):
            sym = symbols[idx]
            matches.append(
                CleanupMatch(
                    symbol=sym,
                    score=float(s),
                    vector=self.codebook[sym],
                    is_exact_match=bool(s >= 0.99),
                )
            )
        return matches

    def clean(self, query_vec: torch.Tensor, fallback_threshold: Optional[float] = None) -> Tuple[torch.Tensor, Optional[str], float]:
        """Clean a noisy vector by replacing it with its closest codebook symbol if above threshold."""
        thresh = fallback_threshold if fallback_threshold is not None else self.similarity_threshold
        matches = self.lookup(query_vec, top_k=1)
        if matches and matches[0].score >= thresh:
            return matches[0].vector.clone(), matches[0].symbol, matches[0].score
        return query_vec, None, (matches[0].score if matches else 0.0)


class CompositionalRetriever:
    """End-to-end compositional query and retrieval processor."""

    def __init__(self, memory_bank: RelationalMemoryBank, cleanup_memory: CodebookCleanupMemory):
        self.memory = memory_bank
        self.cleanup = cleanup_memory
        self.hrr = memory_bank.hrr

    def query_relation(
        self, role_symbol: str, memory_vector: Optional[torch.Tensor] = None, top_k: int = 3
    ) -> List[CleanupMatch]:
        """Query memory for a role's filler (unbinding role from memory trace)."""
        role_vec = self.memory.symbols.get(role_symbol)
        if role_vec is None:
            if role_symbol in self.cleanup.codebook:
                role_vec = self.cleanup.codebook[role_symbol]
            else:
                return []

        trace = memory_vector if memory_vector is not None else self.memory.read_superposition()
        unbound = self.hrr.unbind(trace, role_vec)
        return self.cleanup.lookup(unbound, top_k=top_k)

    def verify_binding(
        self, role_symbol: str, expected_filler_symbol: str, memory_vector: Optional[torch.Tensor] = None
    ) -> Tuple[bool, float]:
        """Verify whether a role is bound to an expected filler in the given memory trace."""
        matches = self.query_relation(role_symbol, memory_vector=memory_vector, top_k=1)
        if not matches:
            return False, 0.0
        best = matches[0]
        is_correct = (best.symbol == expected_filler_symbol)
        return is_correct, best.score
