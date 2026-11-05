"""POSEIDON MORPHOGENESIS: Dynamic Neurogenesis & Synaptic Plasticity Adapter.

Inspired by MOLT (recursive self-reconfiguration & model morphing):
1. Dynamic Morphogenesis:
   Instead of fixed static neural layers, the residual network dynamically modulates
   its structural capacity in response to environmental volatility and prediction error:
   - Neurogenesis: When metacognitive volatility surges, allocates auxiliary hidden units
     to capture new survival dynamics without catastrophic interference.
   - Synaptic Pruning: When homeostatic stability is restored, prunes redundant connections
     (synaptic homeostasis hypothesis / Tononi & Cirelli) maintaining high sparsity.

2. Morphological Preservation Gate:
   Guarantees strict identity initialization around the frozen core policy:
     \\Delta \\pi(a | s) = \\alpha_{morph} \\cdot W_2 \\sigma(W_1 s + b_1)
   Newly sprouted neurons are initialized with zero output weights (W_2 = 0), guaranteeing
   zero performance regression upon creation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import N_ACTIONS, OBS_SIZE


@dataclass
class MorphogenesisEvent:
    step: int
    event_type: str  # "neurogenesis" | "pruning" | "consolidation"
    previous_hidden_dim: int
    new_hidden_dim: int
    sparsity_ratio: float
    trigger_reason: str


class MorphogeneticAdapter(nn.Module):
    """Dynamic morphing neural adapter with online neurogenesis and synaptic pruning."""

    def __init__(
        self,
        in_dim: int = OBS_SIZE,
        out_dim: int = N_ACTIONS,
        initial_hidden: int = 64,
        max_hidden: int = 256,
        min_hidden: int = 16,
        alpha: float = 0.20,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.current_hidden = initial_hidden
        self.max_hidden = max_hidden
        self.min_hidden = min_hidden
        self.alpha = alpha

        # Dynamic layers
        self.w1 = nn.Parameter(torch.randn(initial_hidden, in_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(initial_hidden))
        # Zero-initialized output projection guarantees identity to frozen core
        self.w2 = nn.Parameter(torch.zeros(out_dim, initial_hidden))
        self.b2 = nn.Parameter(torch.zeros(out_dim))

        self.history: list[MorphogenesisEvent] = []
        self.step_counter: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes residual logit perturbation Delta pi(s)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.gelu(F.linear(x, self.w1, self.b1))
        out = F.linear(h, self.w2, self.b2)
        return self.alpha * out

    def trigger_neurogenesis(self, growth_factor: int = 16, reason: str = "volatility_surge") -> bool:
        """Dynamically expands hidden dimension by adding new neurons with zero output projections."""
        if self.current_hidden + growth_factor > self.max_hidden:
            return False

        old_h = self.current_hidden
        new_h = old_h + growth_factor

        # Expand w1 and b1
        new_w1 = torch.cat([
            self.w1.data,
            torch.randn(growth_factor, self.in_dim, device=self.w1.device) * 0.02
        ], dim=0)
        new_b1 = torch.cat([
            self.b1.data,
            torch.zeros(growth_factor, device=self.b1.device)
        ], dim=0)

        # Expand w2 (zero-initialize new columns so net output is 100% unchanged)
        new_w2 = torch.cat([
            self.w2.data,
            torch.zeros(self.out_dim, growth_factor, device=self.w2.device)
        ], dim=1)

        self.w1 = nn.Parameter(new_w1)
        self.b1 = nn.Parameter(new_b1)
        self.w2 = nn.Parameter(new_w2)
        self.current_hidden = new_h

        self.history.append(
            MorphogenesisEvent(
                step=self.step_counter,
                event_type="neurogenesis",
                previous_hidden_dim=old_h,
                new_hidden_dim=new_h,
                sparsity_ratio=self.compute_sparsity(),
                trigger_reason=reason,
            )
        )
        return True

    def trigger_pruning(self, prune_threshold: float = 1e-3, reason: str = "homeostatic_stabilization") -> bool:
        """Prunes low-salience neurons whose output weights are near zero."""
        if self.current_hidden <= self.min_hidden:
            return False

        with torch.no_grad():
            col_norms = torch.norm(self.w2.data, p=2, dim=0)
            keep_mask = col_norms > prune_threshold
            n_keep = int(keep_mask.sum().item())

            if n_keep < self.min_hidden or n_keep == self.current_hidden:
                return False

            old_h = self.current_hidden
            self.w1 = nn.Parameter(self.w1.data[keep_mask, :])
            self.b1 = nn.Parameter(self.b1.data[keep_mask])
            self.w2 = nn.Parameter(self.w2.data[:, keep_mask])
            self.current_hidden = n_keep

            self.history.append(
                MorphogenesisEvent(
                    step=self.step_counter,
                    event_type="pruning",
                    previous_hidden_dim=old_h,
                    new_hidden_dim=n_keep,
                    sparsity_ratio=self.compute_sparsity(),
                    trigger_reason=reason,
                )
            )
            return True

    def compute_sparsity(self) -> float:
        """Computes fraction of near-zero weights in the network."""
        with torch.no_grad():
            w_total = self.w1.numel() + self.w2.numel()
            w_zeros = (self.w1.abs() < 1e-4).sum().item() + (self.w2.abs() < 1e-4).sum().item()
            return float(w_zeros / w_total) if w_total > 0 else 0.0

    def step_metacognitive_trigger(self, epistemic_volatility: float, surprise: float) -> str:
        """Automatically triggers morphogenesis based on online metacognitive signals."""
        self.step_counter += 1
        if epistemic_volatility > 0.35 or surprise > 0.60:
            if self.trigger_neurogenesis(growth_factor=16, reason=f"volatility={epistemic_volatility:.3f}"):
                return "neurogenesis"
        elif epistemic_volatility < 0.08 and surprise < 0.15:
            if self.trigger_pruning(prune_threshold=0.01, reason="low_surprise_consolidation"):
                return "pruning"
        return "stable"
