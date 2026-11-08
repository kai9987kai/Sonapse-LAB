"""Bounded Neural Pulse Router.

Injects traveling wave and phase signals into connectome regions or the graft bus,
subject to strict biological bounds and zero-gain backward compatibility:
When enabled=False or gain=0.0, output is exactly zero (Delta = 0.0).
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


class BoundedPulseRouter(nn.Module):
    """Safely converts macroscale traveling waves into bounded connectome currents."""

    def __init__(
        self,
        num_regions: int = 8,
        target_dim: int = 64,
        max_local_current: float = 0.25,
        max_global_current: float = 2.0,
        enabled: bool = False,
        gain: float = 0.30,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_regions = num_regions
        self.target_dim = target_dim
        self.max_local = max_local_current
        self.max_global = max_global_current
        self.enabled = enabled
        self.gain = gain

        # Projection from region displacements and phase signals to target currents
        # Input: displacement (N) + phase cos (N) + phase sin (N) = 3N
        self.proj = nn.Linear(3 * num_regions, target_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight, gain=0.2)

        self.register_buffer("last_current", torch.zeros(target_dim))

    def compute_current(
        self,
        wave_displacement: torch.Tensor,
        phases: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> torch.Tensor:
        """Computes bounded injection current vector."""
        is_enabled = enabled_override if enabled_override is not None else self.enabled
        device = self.proj.weight.device

        # 1. Zero-gain invariant
        if not is_enabled or self.gain <= 0.0:
            zero_curr = torch.zeros(self.target_dim, device=device)
            self.last_current = zero_curr
            return zero_curr

        x = wave_displacement.to(device).flatten()
        th = phases.to(device).flatten()
        cos_th = torch.cos(th)
        sin_th = torch.sin(th)

        feat = torch.cat([x, cos_th, sin_th], dim=0)
        raw_current = self.gain * self.proj(feat.unsqueeze(0)).squeeze(0)

        # 2. Local element-wise clamping
        clamped = torch.clamp(raw_current, -self.max_local, self.max_local)

        # 3. Global norm boundary enforcement
        l1_norm = torch.sum(torch.abs(clamped))
        if l1_norm > self.max_global:
            clamped = clamped * (self.max_global / (l1_norm + 1e-6))

        self.last_current = clamped.detach()
        return clamped

    def forward(
        self,
        wave_displacement: torch.Tensor,
        phases: torch.Tensor,
        enabled_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        curr = self.compute_current(wave_displacement, phases, enabled_override)
        receipt = EvidenceReceipt(
            item_id="pulse_current_injection",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="BoundedPulseRouter",
            sha256_digest=compute_tensor_digest(curr),
            metrics={
                "enabled": self.enabled,
                "current_l1": float(torch.sum(torch.abs(curr)).item()),
                "current_max": float(torch.max(torch.abs(curr)).item()),
            },
        )
        return {
            "current": curr,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        self.last_current.zero_()
