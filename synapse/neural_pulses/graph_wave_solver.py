"""Discrete Damped Graph Wave Solver.

Solves the discrete damped wave equation on the anatomical graph Laplacian:
x_{t+1} = 2 x_t - x_{t-1} - 2 zeta omega_0 dt (x_t - x_{t-1}) - dt^2 (c^2 L + omega_0^2 I) x_t + dt^2 u(t)

Provides sub-critically damped wave propagation across connectome regions,
modeling macroscale cortical/connectomic traveling waves.
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


class GraphWaveSolver(nn.Module):
    """Discrete second-order damped wave equation solver on graph Laplacian."""

    def __init__(
        self,
        num_regions: int = 8,
        velocity_c: float = 1.5,
        damping_zeta: float = 0.25,
        natural_omega0: float = 2.0,
        dt: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_regions = num_regions
        self.velocity_c = velocity_c
        self.damping_zeta = damping_zeta
        self.natural_omega0 = natural_omega0
        self.dt = dt

        # Build anatomical adjacency and Laplacian (ring/small-world graph default)
        adj = torch.zeros(num_regions, num_regions)
        for i in range(num_regions):
            adj[i, (i + 1) % num_regions] = 1.0
            adj[i, (i - 1) % num_regions] = 1.0
            # Cross-hemisphere / hub connections
            adj[i, (i + num_regions // 2) % num_regions] = 0.5

        deg = torch.diag(adj.sum(dim=1))
        laplacian = deg - adj  # L = D - A

        # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
        deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.clamp(adj.sum(dim=1), min=1e-5)))
        laplacian_norm = deg_inv_sqrt @ laplacian @ deg_inv_sqrt

        self.register_buffer("adj", adj)
        self.register_buffer("laplacian", laplacian_norm)

        # Dynamic wave displacement state buffers
        self.register_buffer("x_curr", torch.zeros(num_regions))
        self.register_buffer("x_prev", torch.zeros(num_regions))
        self.register_buffer("velocity", torch.zeros(num_regions))

        # Check Courant stability condition
        eigs = torch.linalg.eigvalsh(self.laplacian)
        max_eig = float(torch.max(eigs).item())
        stiff = (self.velocity_c ** 2) * max_eig + (self.natural_omega0 ** 2)
        dt_max = 2.0 / np.sqrt(max(stiff, 1e-4))
        self.safe_dt = min(self.dt, float(dt_max * 0.8))

    def inject_pulse(self, region_idx: int, amplitude: float) -> None:
        """Adds instantaneous excitation to a region."""
        idx = int(np.clip(region_idx, 0, self.num_regions - 1))
        self.x_curr[idx] += amplitude

    def step(
        self,
        u_t: Optional[torch.Tensor] = None,
        dt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advances the wave equation by one time step."""
        dt = dt_override if dt_override is not None else self.safe_dt
        device = self.x_curr.device

        if u_t is None:
            u_t = torch.zeros(self.num_regions, device=device)
        else:
            u_t = u_t.to(device).flatten()

        x_t = self.x_curr
        x_tm1 = self.x_prev

        # Velocity approximation: v = (x_t - x_tm1) / dt
        v_t = (x_t - x_tm1) / max(dt, 1e-6)

        # Damping term: 2 * zeta * omega0 * dt * (x_t - x_tm1)
        damping_term = 2.0 * self.damping_zeta * self.natural_omega0 * dt * (x_t - x_tm1)

        # Spatial Laplacian operator: (c^2 L + omega0^2 I) x_t
        spatial_term = (self.velocity_c ** 2) * torch.matmul(self.laplacian, x_t) + (self.natural_omega0 ** 2) * x_t

        # Wave recurrence update
        x_next = 2.0 * x_t - x_tm1 - damping_term - (dt ** 2) * spatial_term + (dt ** 2) * u_t

        # Soft boundary clamp to prevent instability under extreme forcing
        x_next = torch.clamp(x_next, -10.0, 10.0)

        # Update buffers
        self.x_prev.copy_(x_t)
        self.x_curr.copy_(x_next)
        self.velocity.copy_((x_next - x_t) / max(dt, 1e-6))

        # Kinetic and potential energy
        ke = 0.5 * torch.sum(self.velocity ** 2)
        pe = 0.5 * (self.velocity_c ** 2) * torch.matmul(x_next, torch.matmul(self.laplacian, x_next)) + 0.5 * (self.natural_omega0 ** 2) * torch.sum(x_next ** 2)
        total_energy = ke + pe

        receipt = EvidenceReceipt(
            item_id="graph_wave_step",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="GraphWaveSolver",
            sha256_digest=compute_tensor_digest(self.x_curr),
            metadata={
                "kinetic_energy": float(ke.item()),
                "potential_energy": float(pe.item()),
                "total_energy": float(total_energy.item()),
                "max_displacement": float(torch.max(torch.abs(self.x_curr)).item()),
            },
        )

        return {
            "displacement": self.x_curr.clone(),
            "velocity": self.velocity.clone(),
            "kinetic_energy": ke,
            "potential_energy": pe,
            "total_energy": total_energy,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        """Clears wave displacement and velocity."""
        self.x_curr.zero_()
        self.x_prev.zero_()
        self.velocity.zero_()

    reset_state = reset
