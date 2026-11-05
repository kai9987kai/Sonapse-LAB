"""Sparse Mixture of Experts with Dynamic Width and Adaptive Recurrent Depth.

Implements Experiment D of Supermix Beyond:
1. Sparse Top-k Gating (k in {1, 2}):
   - Computes router logits, selects top-k experts per token/state, and executes
     only the activated expert networks, saving compute over dense mixture routing.
2. Load-Balancing Auxiliary Loss:
   - Prevents expert collapse using classical Switch/MoSE coefficient balancing.
3. Adaptive Recurrent Depth (Early Exit):
   - Computes router confidence margin M = p_{top1} - p_{top2}.
   - Highly confident/routine decisions exit after step 1; ambiguous/hazardous
     states proceed to full recurrent depth, spending CPU cycles adaptively.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseMoEConfig:
    hidden_size: int = 128
    num_experts: int = 4
    top_k: int = 2
    max_recurrent_steps: int = 3
    confidence_exit_threshold: float = 0.55  # early exit if top-1 margin > threshold
    aux_loss_coeff: float = 0.01


class SparseMoERouter(nn.Module):
    """Computes top-k sparse routing with load balancing."""

    def __init__(self, in_features: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:
            (top_weights, top_indices, confidence_margin, aux_loss)
        """
        logits = self.gate(x)  # [batch, num_experts]
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_weights, top_indices = torch.topk(probs, self.top_k, dim=-1)
        # Re-normalize top-k weights so they sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Confidence margin (for early exit)
        top1_p, top2_p = top_weights[..., 0], (top_weights[..., 1] if self.top_k > 1 else torch.zeros_like(top1_p))
        confidence_margin = top1_p - top2_p

        # Auxiliary load balancing loss: N * sum(f_i * P_i)
        batch_size = x.size(0)
        density = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.top_k):
            density.scatter_add_(0, top_indices[:, i], torch.ones(batch_size, device=x.device))
        density = density / (batch_size * self.top_k)
        mean_probs = probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * mean_probs)

        return top_weights, top_indices, confidence_margin, aux_loss


class SparseExpertLayer(nn.Module):
    """Executes only the selected expert MLPs per sample."""

    def __init__(self, hidden_size: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, top_weights: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        """Dispatches inputs strictly to chosen experts and sums weighted outputs."""
        batch_size = x.size(0)
        out = torch.zeros(batch_size, x.size(-1) // 2, device=x.device, dtype=x.dtype)

        # Iterate over k top ranks
        k = top_weights.size(-1)
        for rank in range(k):
            indices = top_indices[:, rank]
            weights = top_weights[:, rank].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (indices == e_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    out[mask] += weights[mask] * expert_output

        return out


class AdaptiveTidalMoE(nn.Module):
    """Sparse MoE core featuring adaptive depth computation."""

    def __init__(self, config: SparseMoEConfig | None = None):
        super().__init__()
        self.config = config or SparseMoEConfig()
        cfg = self.config
        h = cfg.hidden_size

        self.router = SparseMoERouter(h * 2, num_experts=cfg.num_experts, top_k=cfg.top_k)
        self.expert_layer = SparseExpertLayer(h, num_experts=cfg.num_experts)
        self.gate = nn.Linear(h * 2, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, context: torch.Tensor, state: torch.Tensor | None = None) -> dict[str, Any]:
        """Runs adaptive recurrent computation with confidence-driven early exit."""
        if state is None:
            state = torch.zeros_like(context)

        total_aux_loss = 0.0
        steps_taken = 0

        for step in range(self.config.max_recurrent_steps):
            steps_taken += 1
            joined = torch.cat((context, state), dim=-1)
            top_weights, top_indices, margin, aux_loss = self.router(joined)
            total_aux_loss += aux_loss

            update = self.expert_layer(joined, top_weights, top_indices)
            g = torch.sigmoid(self.gate(joined))
            state = self.norm((1.0 - g) * state + g * update + context)

            # Adaptive Early Exit: if router is decisive, avoid unnecessary recurrent steps
            if margin.mean().item() > self.config.confidence_exit_threshold and step >= 1:
                break

        return {
            "state": state,
            "steps_taken": steps_taken,
            "aux_loss": total_aux_loss,
            "last_margin": float(margin.mean().item()),
        }


def benchmark_sparse_vs_dense(num_samples: int = 500, hidden_size: int = 128) -> dict[str, dict[str, float]]:
    """Compares throughput and compute of Dense MoE vs Sparse Top-2 MoE with Adaptive Depth."""
    inputs = torch.randn(num_samples, hidden_size)

    # 1. Sparse Adaptive Model
    sparse_model = AdaptiveTidalMoE(SparseMoEConfig(hidden_size=hidden_size, top_k=2))
    t0 = time.perf_counter()
    with torch.no_grad():
        sparse_out = sparse_model(inputs)
    t_sparse = (time.perf_counter() - t0) * 1000.0

    # 2. Dense 4-expert baseline
    dense_experts = nn.ModuleList([
        nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        for _ in range(4)
    ])
    dense_gate = nn.Linear(hidden_size * 2, 4)
    t0 = time.perf_counter()
    with torch.no_grad():
        state = torch.zeros_like(inputs)
        for _ in range(3):  # fixed 3 steps
            joined = torch.cat((inputs, state), dim=-1)
            w = F.softmax(dense_gate(joined), dim=-1)
            c = torch.stack([e(joined) for e in dense_experts], dim=1)
            up = (c * w.unsqueeze(-1)).sum(dim=1)
            state = state + up
    t_dense = (time.perf_counter() - t0) * 1000.0

    return {
        "sparse_adaptive": {
            "latency_ms": round(t_sparse, 2),
            "recurrent_steps": float(sparse_out["steps_taken"]),
            "throughput_samples_per_sec": round(num_samples / (t_sparse / 1000.0), 1),
        },
        "dense_baseline": {
            "latency_ms": round(t_dense, 2),
            "recurrent_steps": 3.0,
            "throughput_samples_per_sec": round(num_samples / (t_dense / 1000.0), 1),
        }
    }
