"""Shared Neural Fast-Weight Memory with Universal Test-Time Training (uTTT & Titans).

Grounding:
- Universal Test-Time Training (uTTT, arxiv.org)
- Titans: Learning to Memorize at Test Time (arxiv.org)
Implements a compact fast-weight memory matrix shared across cognitive layers that
updates online via self-supervised inner gradient descent:
    M_{t+1} = (1 - lambda_decay) * M_t - eta_t * grad_M L_inner(M_t, x_t)
where L_inner measures self-supervised forward representation prediction.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalFastWeightMemory(nn.Module):
    """Shared fast-weight associative matrix with test-time gradient adaptation."""

    def __init__(
        self,
        in_dim: int = 32,
        out_dim: int = 32,
        learning_rate: float = 0.05,
        decay_rate: float = 0.01,
        seed: int = 42
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr_base = learning_rate
        self.decay_rate = decay_rate

        # Key and Value projections for inner self-supervised task
        self.phi_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.psi_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.gate_proj = nn.Linear(in_dim, out_dim)

        # Initialize projections deterministically
        torch.manual_seed(seed)
        nn.init.orthogonal_(self.phi_proj.weight)
        nn.init.orthogonal_(self.psi_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        # Fast-weight memory matrix M [out_dim, in_dim]
        # Registered as buffer (adapted online at test time, not updated by offline backward pass)
        self.register_buffer("M", torch.zeros(out_dim, in_dim, dtype=torch.float32))
        self.step_count = 0
        self.total_inner_updates = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        readout, gate = self.read(x)
        return readout, {'gate': gate, 'memory_norm': float(self.M.norm().item())}

    def inner_update(self, x: torch.Tensor, surprisal: float = 0.0) -> Dict[str, Any]:
        return self.step_and_update(x, surprisal=surprisal, adapt=True)

    def get_metrics(self) -> Dict[str, Any]:
        return self.summary()

    @property
    def inner_step_count(self) -> int:
        return self.total_inner_updates

    def reset_memory(self):
        """Reset fast-weight buffer to zero (e.g. at episode boundaries if desired)."""
        self.M.zero_()
        self.step_count = 0

    def read(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from fast memory: returns (memory_readout, gate)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        phi_x = F.normalize(self.phi_proj(x), p=2, dim=-1)  # [B, in_dim]
        # Linear associative recall: v_pred = phi_x @ M.T -> [B, out_dim]
        readout = F.linear(phi_x, self.M)
        gate = torch.sigmoid(self.gate_proj(x))
        return readout.squeeze(0), gate.squeeze(0)

    @torch.no_grad()
    def step_and_update(
        self,
        x: torch.Tensor,
        surprisal: float = 0.0,
        adapt: bool = True
    ) -> Dict[str, Any]:
        """Query memory and perform online test-time inner update."""
        self.step_count += 1
        x_t = torch.as_tensor(x, dtype=torch.float32).flatten()
        if len(x_t) > self.in_dim:
            x_t = x_t[:self.in_dim]
        elif len(x_t) < self.in_dim:
            x_t = F.pad(x_t, (0, self.in_dim - len(x_t)))

        # 1. Read
        readout, gate = self.read(x_t)

        # 2. Self-supervised inner loss: L_inner = 0.5 * ||M @ phi(x) - psi(x)||^2
        phi_x = F.normalize(self.phi_proj(x_t.unsqueeze(0)), p=2, dim=-1).squeeze(0)  # [in_dim]
        psi_x = torch.tanh(self.psi_proj(x_t.unsqueeze(0))).squeeze(0)                # [out_dim]

        pred_v = self.M @ phi_x                                                       # [out_dim]
        err = pred_v - psi_x                                                          # [out_dim]
        inner_loss = 0.5 * float((err ** 2).sum().item())

        # 3. Test-time outer product gradient update
        if adapt:
            # Gradient: grad_M = err outer phi_x
            grad_M = torch.outer(err, phi_x)                                         # [out_dim, in_dim]
            # Modulate learning rate by surprisal
            eta_t = self.lr_base * (1.0 + min(2.0, max(0.0, surprisal)))
            # M_{t+1} = (1 - decay) * M_t - eta_t * grad_M
            self.M.mul_(1.0 - self.decay_rate).add_(grad_M, alpha=-eta_t)
            # Clip fast weights to prevent divergence
            self.M.clamp_(-2.0, 2.0)
            self.total_inner_updates += 1

        return {
            'step': self.step_count,
            'readout': readout.detach().cpu().numpy(),
            'gate': gate.detach().cpu().numpy(),
            'inner_loss': inner_loss,
            'memory_norm': float(self.M.norm().item()),
            'adaptation_active': adapt
        }

    def summary(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'total_inner_updates': self.total_inner_updates,
            'memory_frobenius_norm': float(self.M.norm().item()),
            'memory_mean': float(self.M.mean().item()),
            'memory_sparsity': float((self.M.abs() < 1e-3).float().mean().item())
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'total_inner_updates': self.total_inner_updates,
            'M': self.M.detach().cpu().numpy().tolist(),
            'phi_weight': self.phi_proj.weight.detach().cpu().numpy().tolist(),
            'psi_weight': self.psi_proj.weight.detach().cpu().numpy().tolist(),
            'gate_weight': self.gate_proj.weight.detach().cpu().numpy().tolist(),
            'gate_bias': self.gate_proj.bias.detach().cpu().numpy().tolist(),
        }

    def import_state(self, state: Dict[str, Any]):
        self.step_count = int(state.get('step_count', 0))
        self.total_inner_updates = int(state.get('total_inner_updates', 0))
        self.M.copy_(torch.tensor(state['M'], dtype=torch.float32))
        self.phi_proj.weight.data.copy_(torch.tensor(state['phi_weight'], dtype=torch.float32))
        self.psi_proj.weight.data.copy_(torch.tensor(state['psi_weight'], dtype=torch.float32))
        self.gate_proj.weight.data.copy_(torch.tensor(state['gate_weight'], dtype=torch.float32))
        self.gate_proj.bias.data.copy_(torch.tensor(state['gate_bias'], dtype=torch.float32))
