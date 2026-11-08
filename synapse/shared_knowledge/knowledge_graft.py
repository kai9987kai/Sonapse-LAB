"""Knowledge Graft Adapter.

Projects retrieved structured memory vectors into recipient connectome units
or WPAC memory_context populations, enforcing zero-gain identity when disabled.
"""
from __future__ import annotations

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


class KnowledgeGraftAdapter(nn.Module):
    """Bridges structured knowledge retrieval into neural state injection."""

    def __init__(
        self,
        knowledge_dim: int = 64,
        target_dim: int = 32,  # Matches WPAC memory_context population or connectome graft port
        enabled: bool = False,
        gain: float = 0.25,
        max_current: float = 0.5,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.knowledge_dim = knowledge_dim
        self.target_dim = target_dim
        self.enabled = enabled
        self.gain = gain
        self.max_current = max_current

        self.proj = nn.Linear(knowledge_dim, target_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight, gain=0.2)

        self.register_buffer("last_injection", torch.zeros(target_dim))

    def project(
        self,
        retrieved_vec: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> torch.Tensor:
        """Projects retrieved knowledge vector into target injection current."""
        is_enabled = enabled_override if enabled_override is not None else self.enabled
        device = self.proj.weight.device

        # 1. Zero-gain invariant
        if not is_enabled or self.gain <= 0.0:
            zero_vec = torch.zeros(self.target_dim, device=device)
            self.last_injection = zero_vec
            return zero_vec

        v = retrieved_vec.to(device).flatten()[:self.knowledge_dim]
        raw = self.gain * self.proj(v.unsqueeze(0)).squeeze(0)

        # Bounded clamping
        clamped = torch.clamp(raw, -self.max_current, self.max_current)
        self.last_injection = clamped.detach()
        return clamped

    def forward(
        self,
        retrieved_vec: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        inj = self.project(retrieved_vec, enabled_override)
        receipt = EvidenceReceipt(
            item_id="knowledge_graft_projection",
            kind=EvidenceKind.TRAINED_COMPUTATIONAL,
            source="KnowledgeGraftAdapter",
            sha256_digest=compute_tensor_digest(inj),
            metrics={
                "enabled": self.enabled,
                "norm": float(torch.norm(inj).item()),
            },
        )
        return {
            "injection": inj,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        self.last_injection.zero_()
