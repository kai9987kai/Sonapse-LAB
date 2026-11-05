"""Spectral connectomics and connectome harmonic decomposition.
Decomposes recurrent connectome activity and weight updates into spatial harmonic eigenmodes of the graph Laplacian.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

def build_connectome_adjacency(graph, include_grafts: bool = True) -> np.ndarray:
    """Construct signed or absolute dense adjacency matrix from base and graft edges."""
    N = graph.n
    A = np.zeros((N, N), dtype=np.float64)

    # Base intra-neuropil connections
    for pre, post, w in zip(graph.base_pre, graph.base_post, graph.base_w):
        A[pre, post] += abs(float(w))

    # Inter-neuropil graft connections
    if include_grafts and hasattr(graph, 'graft_pre'):
        for pre, post in zip(graph.graft_pre, graph.graft_post):
            A[pre, post] += 1.0

    # Ensure symmetry for undirected structural harmonic analysis: A_sym = (A + A.T) / 2
    A_sym = 0.5 * (A + A.T)
    return A_sym


def compute_connectome_harmonics(graph, num_modes: int = 16) -> Dict[str, Any]:
    """Compute normalized graph Laplacian and extract lowest spatial harmonic eigenmodes.
    
    L_sym = I - D^(-1/2) * A * D^(-1/2)
    L_sym * psi_k = lambda_k * psi_k
    """
    N = graph.n
    k = min(num_modes, N - 1)
    A = build_connectome_adjacency(graph, include_grafts=True)

    degrees = np.sum(A, axis=1)
    # Avoid division by zero for isolated nodes
    d_inv_sqrt = np.zeros_like(degrees)
    non_zero = degrees > 1e-8
    d_inv_sqrt[non_zero] = 1.0 / np.sqrt(degrees[non_zero])

    D_mat = np.diag(d_inv_sqrt)
    # Normalized Laplacian
    L_sym = np.eye(N) - D_mat @ A @ D_mat

    # Symmetric eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    # Sort in ascending order (lowest frequency / most global first)
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx][:k]
    eigenvectors = eigenvectors[:, sort_idx][:, :k]

    return {
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors,
        'num_modes': k,
        'num_neurons': N,
        'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if k > 1 else 0.0,
    }


def project_spectral_activity(
    rates: np.ndarray | torch.Tensor,
    harmonics: Dict[str, Any],
) -> Dict[str, Any]:
    """Project live neural firing rates onto the connectome harmonic basis.
    
    Coefficients c = Psi^T * r
    Power spectrum P_k = c_k^2
    """
    if isinstance(rates, torch.Tensor):
        r = rates.detach().cpu().numpy().astype(np.float64)
    else:
        r = np.asarray(rates, dtype=np.float64)

    Psi = harmonics['eigenvectors'] # [N, k]
    k = harmonics['num_modes']

    # Project
    coefficients = Psi.T @ r # [k]
    power = coefficients ** 2
    total_power = float(np.sum(power)) + 1e-9

    # Energy in lowest 25% (global modes) vs highest 25% (localized modular modes)
    split = max(1, k // 4)
    low_freq_power = float(np.sum(power[:split])) / total_power
    high_freq_power = float(np.sum(power[-split:])) / total_power

    return {
        'status': 'ok',
        'coefficients': coefficients.tolist(),
        'power_spectrum': power.tolist(),
        'total_spectral_power': total_power,
        'global_mode_fraction': low_freq_power,
        'local_mode_fraction': high_freq_power,
        'dominant_mode': int(np.argmax(power)),
        'scope': 'connectome harmonic spectral activity projection',
    }


def spectral_graph_filter(
    weights: np.ndarray | torch.Tensor,
    harmonics: Dict[str, Any],
    low_pass_cutoff: int = 8,
) -> np.ndarray:
    """Filter weight perturbations through low-pass harmonic projection to eliminate
    high-frequency instabilities and preserve smooth global coordination.
    """
    is_torch = isinstance(weights, torch.Tensor)
    if is_torch:
        W = weights.detach().cpu().numpy().copy()
    else:
        W = np.asarray(weights).copy()

    Psi = harmonics['eigenvectors'][:, :low_pass_cutoff] # [N, cutoff]
    # Smooth operator: S = Psi * Psi^T
    S = Psi @ Psi.T

    # If matrix [N, M] or [N, N], filter rows
    if W.ndim == 2 and W.shape[0] == Psi.shape[0]:
        W_filtered = S @ W
    elif W.ndim == 1 and W.shape[0] == Psi.shape[0]:
        W_filtered = S @ W
    else:
        W_filtered = W

    return torch.from_numpy(W_filtered).float() if is_torch else W_filtered
