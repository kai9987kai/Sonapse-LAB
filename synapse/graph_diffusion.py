"""Chebyshev Spectral Graph Diffusion Engine for Scaled Connectomes.

Grounding:
In a 6,500-neuron connectome spanning central brain (FlyWire), motor ventral nerve
cord (MANC), and visual neuropil (Optic-Lobe), signal transport between sensory and
motor neuropils requires multi-hop topological propagation.

Chebyshev polynomial spectral graph filtering (Defferrard et al., NeurIPS 2016;
Chamberlain et al., GRAND Graph Neural Diffusion 2021) evaluates multi-scale heat
kernel diffusion directly in O(E) time:
    L = I - D^{-1/2} A D^{-1/2}
    L_tilde = (2 / lambda_max) * L - I,   lambda_max ~ 2.0
    T_0(L_tilde) x = x
    T_1(L_tilde) x = L_tilde x
    T_k(L_tilde) x = 2 * L_tilde * T_{k-1}(L_tilde) x - T_{k-2}(L_tilde) x
    x_diff = sum_{k=0}^K alpha_k * T_k(L_tilde) x

The module features a zero-initialized residual projection, strictly preserving
backward compatibility and frozen baseline invariance when untrained or disabled.
"""
from __future__ import annotations
import math
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ChebyshevGraphDiffusion(nn.Module):
    """Multi-scale Chebyshev spectral graph diffusion over connectome topologies."""

    def __init__(
        self,
        num_nodes: int = 6500,
        pre: Optional[torch.Tensor] = None,
        post: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        order: int = 4,
        diffusion_coeff: float = 0.25,
        seed: int = 42,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.order = int(order)
        self.diffusion_coeff = float(diffusion_coeff)
        self.seed = int(seed)

        # Build sparse normalized Laplacian
        self._build_laplacian(pre, post, weights)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 7719)
            # Chebyshev spectral filter weights (order + 1 parameters)
            self.filter_weights = nn.Parameter(torch.zeros(self.order + 1))
            # Heat kernel decaying initial filter coefficients
            with torch.no_grad():
                for k in range(self.order + 1):
                    self.filter_weights[k] = math.exp(-0.5 * k) * (0.1 if k > 0 else 1.0)
            # Zero-initialized projection back to residual current
            self.readout_gate = nn.Parameter(torch.zeros(1))

        self.register_buffer("last_diffusion_energy", torch.zeros(1))
        self.register_buffer("last_dispersion", torch.zeros(1))
        self.updates = 0

    def _build_laplacian(
        self,
        pre: Optional[torch.Tensor],
        post: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
    ) -> None:
        """Constructs rescaled normalized Laplacian L_tilde as a sparse tensor."""
        n = self.num_nodes
        if pre is None or post is None or pre.numel() == 0:
            # Fallback identity laplacian
            indices = torch.arange(n, dtype=torch.long)
            i_indices = torch.stack([indices, indices], dim=0)
            self.register_buffer("laplacian_indices", i_indices)
            self.register_buffer("laplacian_values", torch.zeros(n, dtype=torch.float32))
            return

        pre_t = pre.long()
        post_t = post.long()
        if weights is not None:
            w_t = weights.float().abs()
        else:
            w_t = torch.ones(len(pre_t), dtype=torch.float32)

        # Compute degrees (in-degrees + out-degrees)
        deg = torch.zeros(n, dtype=torch.float32)
        deg.scatter_add_(0, post_t, w_t)
        deg.scatter_add_(0, pre_t, w_t)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-5), -0.5)

        # Normalized adjacency: A_norm = D^{-1/2} A D^{-1/2}
        norm_w = deg_inv_sqrt[pre_t] * w_t * deg_inv_sqrt[post_t]

        # Rescaled Laplacian: L_tilde = (2 / lambda_max) * (I - A_norm) - I
        # Assuming lambda_max ~ 2.0: L_tilde = -A_norm
        edge_indices = torch.stack([post_t, pre_t], dim=0)
        self.register_buffer("laplacian_indices", edge_indices)
        self.register_buffer("laplacian_values", -norm_w)

    def _sparse_laplacian_mm(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sparse L_tilde @ x."""
        n = self.num_nodes
        if self.laplacian_indices.numel() == 0 or self.laplacian_values.numel() == 0:
            return torch.zeros_like(x)

        row = self.laplacian_indices[0]
        col = self.laplacian_indices[1]
        val = self.laplacian_values

        # Sparse-dense vector multiplication: out[row] += val * x[col]
        out = torch.zeros(n, dtype=torch.float32, device=x.device)
        product = val * x[col]
        out.scatter_add_(0, row, product)
        return out

    def forward(
        self,
        rates: torch.Tensor,
        neuromod_gain: float = 1.0,
        learn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Applies Chebyshev spectral graph diffusion over neuron rates vector."""
        r = rates.detach().reshape(-1)[: self.num_nodes]
        if r.numel() < self.num_nodes:
            r = F.pad(r, (0, self.num_nodes - r.numel()))

        # Evaluate Chebyshev polynomials: T_0, T_1, ..., T_K
        t_prev = r.clone()
        diff_accum = self.filter_weights[0] * t_prev

        if self.order >= 1:
            t_curr = self._sparse_laplacian_mm(t_prev)
            diff_accum = diff_accum + self.filter_weights[1] * t_curr

            for k in range(2, self.order + 1):
                t_next = 2.0 * self._sparse_laplacian_mm(t_curr) - t_prev
                diff_accum = diff_accum + self.filter_weights[k] * t_next
                t_prev = t_curr
                t_curr = t_next

        # Bounded diffusion residual
        diff_signal = torch.tanh(diff_accum) * float(neuromod_gain)
        # Readout gate (starts at 0.0 for zero-intervention baseline)
        gate = torch.tanh(self.readout_gate)
        residual = gate * diff_signal

        # Telemetry metrics
        energy = float(torch.norm(diff_signal).item())
        dispersion = float(torch.std(diff_signal).item())
        if learn:
            self.last_diffusion_energy.fill_(energy)
            self.last_dispersion.fill_(dispersion)

        telemetry = {
            "enabled": True,
            "chebyshev_order": self.order,
            "diffusion_energy": round(energy, 4),
            "dispersion": round(dispersion, 4),
            "readout_gate": round(float(gate.item()), 5),
            "updates": int(self.updates),
        }
        return residual, telemetry

    def update_from_outcome(self, reward: float, prediction_error: float, lr: float = 1e-3) -> Dict[str, float]:
        """Reward-modulated spectral weight adaptation."""
        with torch.no_grad():
            mod = float(np.clip(reward - 0.5 * prediction_error, -1.0, 1.0))
            self.readout_gate.add_(lr * mod)
            self.readout_gate.clamp_(-0.5, 0.5)
            self.updates += 1
        return {
            "updates": int(self.updates),
            "readout_gate": round(float(self.readout_gate.item()), 5),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "num_nodes": self.num_nodes,
            "order": self.order,
            "readout_gate": round(float(torch.tanh(self.readout_gate).item()), 5),
            "last_energy": round(float(self.last_diffusion_energy.item()), 4),
            "last_dispersion": round(float(self.last_dispersion.item()), 4),
            "updates": int(self.updates),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "state_dict": deepcopy(self.state_dict()),
            "updates": int(self.updates),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "state_dict" in state:
            self.load_state_dict(state["state_dict"])
        self.updates = int(state.get("updates", 0))

    def run_graph_diffusion_assay(self, steps: int = 5, seed: int = 42) -> Dict[str, Any]:
        """Evaluates multi-scale spectral graph diffusion over connectome structure."""
        rng = np.random.default_rng(seed)
        sample_input = torch.from_numpy(rng.standard_normal(self.num_nodes).astype(np.float32))

        energies = []
        dispersions = []
        for _ in range(steps):
            _, diag = self.forward(sample_input, learn=False)
            energies.append(diag["diffusion_energy"])
            dispersions.append(diag["dispersion"])

        return {
            "schema": "chebyshev-spectral-graph-diffusion-v1",
            "status": "completed",
            "num_nodes": self.num_nodes,
            "chebyshev_order": self.order,
            "mean_diffusion_energy": round(float(np.mean(energies)), 4),
            "mean_spatial_dispersion": round(float(np.mean(dispersions)), 4),
            "energy_conservation_ratio": round(float(np.mean(energies) / max(1e-4, float(torch.norm(sample_input).item()))), 4),
            "spectral_radius_bound": 2.0,
        }
