"""Graph-Laplacian Spectral Basis & Alignment Engine for Sonapse NeuroWave-X2.

Computes the normalized Laplacian eigenbasis from regional connectivity graphs,
maintains basis versioning, computes subspace drift, and aligns eigenvectors
across topology mutations via orthogonal Procrustes and deterministic sign convention.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn


def compute_normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Computes the normalized graph Laplacian L = I - D^(-1/2) A D^(-1/2).
    
    Args:
        adjacency: (N, N) non-negative adjacency matrix
    Returns:
        (N, N) symmetric normalized Laplacian
    """
    N = adjacency.shape[0]
    A = np.maximum(adjacency, 0.0)
    # Symmetrize for stability
    A = 0.5 * (A + A.T)
    degrees = np.sum(A, axis=1)
    # Avoid division by zero
    inv_sqrt_d = np.zeros(N, dtype=np.float64)
    nonzero = degrees > 1e-12
    inv_sqrt_d[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
    D_inv = np.diag(inv_sqrt_d)
    L = np.eye(N, dtype=np.float64) - D_inv @ A @ D_inv
    # Enforce symmetry
    return 0.5 * (L + L.T)


def compute_eigenbasis(
    adjacency: np.ndarray,
    num_modes: int = 8,
    carrier_strategy: str = "lowest_frequency",
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves L U = U Lambda and selects carrier modes.
    
    Returns:
        (eigenvalues, eigenvectors) where eigenvectors has shape (num_modes, N)
    """
    N = adjacency.shape[0]
    num_modes = min(num_modes, N)
    L = compute_normalized_laplacian(adjacency)

    evals, evecs = scipy.linalg.eigh(L)
    # Sort in ascending eigenvalue order (lowest frequency / smoothest graph signals first)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Deterministic sign alignment: ensure maximum absolute element is positive
    for col in range(N):
        max_idx = np.argmax(np.abs(evecs[:, col]))
        if evecs[max_idx, col] < 0:
            evecs[:, col] *= -1.0

    if carrier_strategy == "lowest_frequency":
        selected_idx = np.arange(num_modes)
    elif carrier_strategy == "energy_weighted":
        # Skip DC mode (index 0) if multiple modes available
        if N > num_modes and num_modes > 1:
            selected_idx = np.arange(1, num_modes + 1)
        else:
            selected_idx = np.arange(num_modes)
    else:
        selected_idx = np.arange(num_modes)

    selected_evals = evals[selected_idx]
    # Each row is a mode carrier: (num_modes, N)
    selected_modes = evecs[:, selected_idx].T.astype(np.float32)
    return selected_evals.astype(np.float32), selected_modes


def compute_subspace_drift(basis_old: np.ndarray, basis_new: np.ndarray) -> float:
    """Computes Frobenius norm subspace drift D_t = ||P_old - P_new||_F
    where P = U^T U is the projector onto the subspace.
    """
    # basis has shape (K, N)
    # Projector P = U^T U where U has shape (K, N)
    P_old = basis_old.T @ basis_old
    P_new = basis_new.T @ basis_new
    return float(np.linalg.norm(P_old - P_new, ord="fro"))


def procrustes_align_basis(basis_old: np.ndarray, basis_new: np.ndarray) -> np.ndarray:
    """Aligns basis_new to basis_old via Orthogonal Procrustes to resolve rotation/subspace swap.
    
    Minimizes ||basis_old - R @ basis_new||_F subject to R^T R = I.
    """
    if basis_old.shape != basis_new.shape:
        return basis_new
    # M = basis_old @ basis_new.T
    M = basis_old @ basis_new.T
    U, _, Vt = scipy.linalg.svd(M)
    R = U @ Vt
    aligned = R @ basis_new
    return aligned.astype(np.float32)


class SpectralBasisManager:
    """Manages cached graph eigenbases, versioning, alignment, and cache invalidation."""

    def __init__(self, num_modes: int = 8, carrier_strategy: str = "lowest_frequency") -> None:
        self.num_modes = int(num_modes)
        self.carrier_strategy = str(carrier_strategy)
        self.basis_version = 0
        self.topology_hash = ""
        self.cached_evals: Optional[np.ndarray] = None
        self.cached_basis: Optional[np.ndarray] = None  # (num_modes, N)
        self.last_drift = 0.0

    def get_or_compute_basis(self, adjacency: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """Returns (basis_version, evals, basis_matrix) with incremental caching and drift monitoring."""
        hasher = hashlib.sha256()
        hasher.update(adjacency.tobytes())
        hasher.update(str(self.num_modes).encode())
        hasher.update(self.carrier_strategy.encode())
        new_hash = hasher.hexdigest()

        if self.cached_basis is not None and new_hash == self.topology_hash:
            return self.basis_version, self.cached_evals, self.cached_basis

        # Topology mutated or first invocation: compute new basis
        evals, raw_basis = compute_eigenbasis(
            adjacency, num_modes=self.num_modes, carrier_strategy=self.carrier_strategy
        )

        if self.cached_basis is not None and self.cached_basis.shape == raw_basis.shape:
            # Align new basis to previous basis to minimize operational shock
            aligned_basis = procrustes_align_basis(self.cached_basis, raw_basis)
            self.last_drift = compute_subspace_drift(self.cached_basis, aligned_basis)
            self.cached_basis = aligned_basis
        else:
            self.last_drift = 0.0
            self.cached_basis = raw_basis

        self.cached_evals = evals
        self.topology_hash = new_hash
        self.basis_version += 1
        return self.basis_version, self.cached_evals, self.cached_basis
