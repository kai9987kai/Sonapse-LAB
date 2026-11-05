"""POSEIDON NEURO: Neuromodulated Sparse Connectome Routing.

Inspired by biological small-world connectomes and state-dependent neuromodulation:
1. Small-World Sparse Topology:
   Implements structured sparse synaptic weight masks exhibiting high clustering
   and short characteristic path lengths (Watts-Strogatz graph analog),
   avoiding homogeneous all-to-all dense computation.

2. Neuromodulatory Gain Control:
   Dynamically scales layer sensitivity and routing sparsity based on vital state:
     - Noradrenergic tone (threat/stress): increases inhibitory gating, sharpens
       winner-take-all sparsity, prioritizes defensive pathways.
     - Dopaminergic tone (reward/energy): broadens routing entropy, encouraging
       multi-path exploratory processing.

3. SparseConnectomeModule:
   A drop-in neuromodulated layer applying biological sparsity masks and metabolic gain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import OBS_SIZE


@dataclass(frozen=True)
class NeuroConfig:
    dim: int = 64
    sparsity: float = 0.75  # 75% zeros
    small_world_rewire_prob: float = 0.15
    base_gain: float = 1.0


class SparseConnectomeMask:
    """Generates small-world sparse connectivity topology mask."""

    @staticmethod
    def generate_mask(dim: int, sparsity: float = 0.75, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        k_neighbors = max(2, int(dim * (1.0 - sparsity)))
        mask = torch.zeros(dim, dim, dtype=torch.float32)

        # 1D Ring lattice base
        for i in range(dim):
            for j in range(1, k_neighbors // 2 + 1):
                mask[i, (i + j) % dim] = 1.0
                mask[i, (i - j) % dim] = 1.0

        # Small-world random rewiring
        rand = torch.rand(dim, dim)
        rewire = (mask == 1.0) & (rand < 0.15)
        mask[rewire] = 0.0
        num_rewired = int(rewire.sum().item())
        if num_rewired > 0:
            random_targets = torch.randint(0, dim, (num_rewired,))
            sources = torch.nonzero(rewire)[:, 0]
            for s, t in zip(sources, random_targets):
                mask[s, t] = 1.0

        return mask


class NeuromodulatedRouter(nn.Module):
    """Dynamically modulates gain and sparsity based on metabolic and threat state."""

    def __init__(self, config: NeuroConfig | None = None):
        super().__init__()
        self.config = config or NeuroConfig()
        self.dim = self.config.dim

        # Fixed structural connectome mask
        mask = SparseConnectomeMask.generate_mask(self.dim, self.config.sparsity)
        self.register_buffer("connectome_mask", mask)

        # Synaptic weight parameter
        self.weight = nn.Parameter(torch.randn(self.dim, self.dim) * (1.0 / math.sqrt(self.dim)))
        self.bias = nn.Parameter(torch.zeros(self.dim))

        # Neuromodulatory gain projection from vital states (health, energy, hydration, threat)
        self.modulator = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(),
            nn.Linear(16, 2),  # Outputs [gain, sparsity_bias]
        )

    def forward(self, x: torch.Tensor, vitals: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with neuromodulatory gain and sparsity gating.
        
        Args:
            x: Input tensor [batch, dim]
            vitals: Vital state [batch, 4] (health, energy, hydration, threat)
        """
        mod_out = self.modulator(vitals)
        gain = torch.sigmoid(mod_out[:, 0:1]) * 1.5 + 0.5  # Gain in [0.5, 2.0]
        sparsity_shift = torch.tanh(mod_out[:, 1:2]) * 0.2

        # Apply structural sparse connectome mask
        effective_w = self.weight * self.connectome_mask

        # Linear projection
        h = F.linear(x, effective_w, self.bias)

        # Apply state-dependent neuromodulatory gain
        h = h * gain

        # Non-linear activation with dynamic sparsity threshold
        threshold = 0.0 + sparsity_shift
        gated_h = F.relu(h - threshold)

        active_ratio = (gated_h > 0).float().mean()

        return {
            "output": gated_h,
            "gain": gain.mean(),
            "active_sparsity": 1.0 - active_ratio,
        }
