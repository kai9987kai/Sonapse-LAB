"""Pretrained Graft Bridge and Latent Space Alignment.

Aligns frozen donor representations z_D in R^{d_D} to recipient connectome coordinates
z_R in R^{d_R} through orthogonal/regularized projection:
hat{z}_R = clip(gain * P_theta z_D, -B, B)

Enforces zero-gain identity when disabled or unconfigured.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class PretrainedGraftBridge(nn.Module):
    """Aligns and maps donor latents to recipient connectome coordinates."""

    def __init__(
        self,
        donor_dim: int = 128,
        recipient_dim: int = 64,
        enabled: bool = False,
        gain: float = 0.20,
        max_bound: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.donor_dim = donor_dim
        self.recipient_dim = recipient_dim
        self.enabled = enabled
        self.gain = gain
        self.max_bound = max_bound

        self.proj = nn.Linear(donor_dim, recipient_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight, gain=0.2)

        self.register_buffer("last_aligned", torch.zeros(recipient_dim))

    def align(
        self,
        donor_latents: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> torch.Tensor:
        """Projects donor latent vector into recipient space."""
        is_enabled = enabled_override if enabled_override is not None else self.enabled
        device = self.proj.weight.device

        # Zero-gain contract
        if not is_enabled or self.gain <= 0.0:
            zero_vec = torch.zeros(self.recipient_dim, device=device)
            self.last_aligned = zero_vec
            return zero_vec

        z_d = donor_latents.to(device).flatten()[:self.donor_dim]
        aligned = self.gain * self.proj(z_d.unsqueeze(0)).squeeze(0)
        clamped = torch.clamp(aligned, -self.max_bound, self.max_bound)

        self.last_aligned = clamped.detach()
        return clamped

    def forward(
        self,
        donor_latents: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        aligned = self.align(donor_latents, enabled_override)
        receipt = EvidenceReceipt(
            item_id="latent_alignment_step",
            kind=EvidenceKind.TRAINED_COMPUTATIONAL,
            source="PretrainedGraftBridge",
            sha256_digest=compute_tensor_digest(aligned),
            metrics={
                "enabled": self.enabled,
                "norm": float(torch.norm(aligned).item()),
            },
        )
        return {
            "aligned_latents": aligned,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        self.last_aligned.zero_()


class PretrainedGraftBridgeFactory:
    """Factory helper to construct tailored alignment bridges."""

    @staticmethod
    def create_bridge(donor_dim: int, recipient_dim: int, enabled: bool = False, gain: float = 0.20) -> PretrainedGraftBridge:
        return PretrainedGraftBridge(
            donor_dim=donor_dim,
            recipient_dim=recipient_dim,
            enabled=enabled,
            gain=gain,
        )
