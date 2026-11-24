"""Hybrid Biological + Learned Communication Basis for Sonapse NeuroWave-X2.

Combines connectome-derived Laplacian modes with task-trained components:
    B_hybrid = orthogonalize([B_graph, B_learned])
Maintains orthogonality, exposes condition numbers, energy concentration,
and provides standard comparison baselines (Identity, Random Orthogonal, DCT, Hadamard).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_dct_basis(num_modes: int, dimension: int) -> np.ndarray:
    """Constructs standard Type-II Discrete Cosine Transform basis matrix (num_modes, dimension)."""
    basis = np.zeros((num_modes, dimension), dtype=np.float32)
    for k in range(num_modes):
        factor = math.sqrt(1.0 / dimension) if k == 0 else math.sqrt(2.0 / dimension)
        for n in range(dimension):
            basis[k, n] = factor * math.cos(math.pi * k * (2 * n + 1) / (2.0 * dimension))
    return basis


def construct_hadamard_basis(num_modes: int, dimension: int) -> np.ndarray:
    """Constructs normalized Hadamard basis matrix (truncated/padded to num_modes, dimension)."""
    n = 2 ** math.ceil(math.log2(max(num_modes, dimension, 2)))
    rows, cols = len(range(n)[:num_modes]), len(range(n)[:dimension])
    basis = np.empty((rows, cols), dtype=np.float32)
    if not basis.size:
        return basis
    basis[0, 0] = 1.0
    # Expand the Sylvester quadrants only where they intersect the requested rectangle.
    side = 1
    while side < max(rows, cols):
        height, width = min(rows, side), min(cols, side)
        extra_rows = min(height, max(0, rows - side))
        extra_cols = min(width, max(0, cols - side))
        if extra_cols:
            basis[:height, side:side + extra_cols] = basis[:height, :extra_cols]
        if extra_rows:
            basis[side:side + extra_rows, :width] = basis[:extra_rows, :width]
            np.negative(basis[:extra_rows, :extra_cols], out=basis[side:side + extra_rows, side:side + extra_cols])
        side *= 2
    basis /= math.sqrt(n)
    return basis


def construct_random_orthogonal_basis(num_modes: int, dimension: int, seed: int = 42) -> np.ndarray:
    """Generates a random orthonormal basis using QR decomposition."""
    rng = np.random.RandomState(seed)
    A = rng.randn(dimension, dimension)
    Q, _ = np.linalg.qr(A)
    return Q[:num_modes, :dimension].astype(np.float32)


def orthogonalize_stacked_bases(
    base_graph: torch.Tensor,  # (K_graph, D)
    base_learned: torch.Tensor,  # (K_learned, D)
) -> torch.Tensor:
    """Gram-Schmidt / QR orthogonalization of learned basis against graph basis.
    Preserves the graph basis directions while finding orthogonal learned extensions.
    """
    K_g, D = base_graph.shape
    K_l, _ = base_learned.shape

    # Project learned vectors onto the orthogonal complement of base_graph
    # P_g = base_graph^T @ base_graph assuming base_graph rows are orthonormal
    # If not orthonormal, perform QR on base_graph first
    Q_g, _ = torch.linalg.qr(base_graph.T)  # Q_g is (D, K_g)
    P_g = torch.matmul(Q_g, Q_g.T)  # Projector onto graph subspace (D, D)

    # Deflate learned components: (I - P_g) @ learned^T
    I = torch.eye(D, device=base_graph.device, dtype=base_graph.dtype)
    deflated = torch.matmul(I - P_g, base_learned.T).T  # (K_l, D)

    # QR on deflated components
    if K_l > 0 and deflated.norm() > 1e-6:
        Q_l, _ = torch.linalg.qr(deflated.T)  # (D, K_l)
        ortho_learned = Q_l[:, :K_l].T
        return torch.cat([base_graph, ortho_learned], dim=0)
    return base_graph


class HybridBasis(nn.Module):
    """Hybrid Communication Basis module managing graph and learned carriers."""

    def __init__(
        self,
        num_graph_modes: int = 4,
        num_learned_modes: int = 4,
        dimension: int = 16,
        basis_type: str = "hybrid",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_graph_modes = int(num_graph_modes)
        self.num_learned_modes = int(num_learned_modes)
        self.dimension = int(dimension)
        self.basis_type = str(basis_type)
        self.total_modes = self.num_graph_modes + self.num_learned_modes

        # Learnable carrier weights
        if self.num_learned_modes > 0 and self.basis_type in ("hybrid", "pure_learned"):
            self.learned_weight = nn.Parameter(
                torch.empty(self.num_learned_modes, self.dimension)
            )
            nn.init.orthogonal_(self.learned_weight, gain=1.0)
        else:
            self.register_parameter("learned_weight", None)

        # Buffer for fixed/graph basis
        self.register_buffer("graph_basis", torch.zeros(self.num_graph_modes, self.dimension))
        self.basis_version = 0

    def set_graph_basis(self, basis_matrix: np.ndarray, version: int = 0) -> None:
        """Sets the connectome-derived graph eigenbasis (num_graph_modes, dimension)."""
        K, D = basis_matrix.shape
        use_K = min(K, self.num_graph_modes)
        use_D = min(D, self.dimension)
        t = torch.zeros(self.num_graph_modes, self.dimension, dtype=torch.float32)
        t[:use_K, :use_D] = torch.from_numpy(basis_matrix[:use_K, :use_D])
        self.graph_basis.copy_(t)
        self.basis_version = int(version)

    def get_effective_basis(self) -> torch.Tensor:
        """Constructs and returns the active orthonormal basis matrix (total_modes, dimension)."""
        D = self.dimension
        device = self.graph_basis.device
        dtype = self.graph_basis.dtype

        if self.basis_type == "identity":
            return torch.eye(min(self.total_modes, D), D, device=device, dtype=dtype)

        if self.basis_type in ("dct", "hadamard", "random_orthogonal"):
            key = (self.basis_type, self.total_modes, D)
            cached = getattr(self, "_fixed_basis", None)
            if cached is None or cached[0] != key:
                construct = {"dct": construct_dct_basis, "hadamard": construct_hadamard_basis,
                             "random_orthogonal": construct_random_orthogonal_basis}[self.basis_type]
                cached = (key, construct(self.total_modes, D))
                self._fixed_basis = cached
            # Callers own the returned tensor even when device and dtype match.
            return torch.tensor(cached[1], device=device, dtype=dtype)

        if self.basis_type == "graph" or self.learned_weight is None:
            # Normalize graph rows if nonzero
            norms = torch.norm(self.graph_basis, dim=1, keepdim=True).clamp_min(1e-8)
            return self.graph_basis / norms

        if self.basis_type == "pure_learned":
            Q, _ = torch.linalg.qr(self.learned_weight.T)
            return Q[:, :self.total_modes].T

        # Default: Hybrid Basis
        norms = torch.norm(self.graph_basis, dim=1, keepdim=True).clamp_min(1e-8)
        normed_graph = self.graph_basis / norms
        return orthogonalize_stacked_bases(normed_graph, self.learned_weight)

    def compute_basis_diagnostics(self) -> Dict[str, float]:
        """Calculates basis rank, condition number, orthogonality error, and energy spread."""
        B = self.get_effective_basis().detach()
        # Singular values for rank & condition number
        s = torch.linalg.svdvals(B)
        cond = float((s[0] / s[-1].clamp_min(1e-12)).item()) if s.numel() > 0 else 1.0
        rank = float(torch.sum(s > 1e-4).item())
        # Orthogonality error: ||B @ B^T - I||_F
        Gram = torch.matmul(B, B.T)
        I = torch.eye(Gram.shape[0], device=B.device, dtype=B.dtype)
        ortho_err = float(torch.norm(Gram - I, p="fro").item())
        # Energy concentration (ratio of top 2 singular values to total)
        energy_top2 = float((torch.sum(s[:2] ** 2) / torch.sum(s ** 2).clamp_min(1e-8)).item())

        return {
            "basis_rank": rank,
            "condition_number": cond,
            "orthogonality_error": ortho_err,
            "energy_top2_ratio": energy_top2,
            "basis_version": float(self.basis_version),
        }
