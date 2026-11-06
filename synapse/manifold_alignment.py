"""Gromov-Wasserstein Non-Euclidean Manifold Alignment for Connectome Circuits.

Computes diffeomorphism-invariant, coordinate-free optimal transport between
disparate neural metric spaces without requiring paired sensory-motor observations.
Uses entropic Gromov-Wasserstein optimization to find the coupling matrix T*
that minimizes structural geodesic distortion across donor and host graphs.
"""
from __future__ import annotations
import math
import numpy as np
import torch

def compute_graph_distance_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Compute all-pairs shortest path distance matrix via Floyd-Warshall."""
    n = len(adjacency)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    for i in range(n):
        for j in range(n):
            if adjacency[i, j] > 0:
                dist[i, j] = 1.0 / (abs(float(adjacency[i, j])) + 1e-4)

    # Floyd-Warshall shortest paths
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])

    # Replace remaining infinities with maximum diameter
    max_finite = np.max(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 1.0
    dist[np.isinf(dist)] = max_finite * 2.0
    # Normalize to [0, 1]
    if np.max(dist) > 0:
        dist /= np.max(dist)
    return dist

def sinkhorn_knopp(
    cost: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-4
) -> np.ndarray:
    """Entropic regularized optimal transport via Sinkhorn-Knopp."""
    K = np.exp(-cost / epsilon)
    u = np.ones_like(p) / len(p)
    v = np.ones_like(q) / len(q)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = p / (np.dot(K, v) + 1e-12)
        v = q / (np.dot(K.T, u) + 1e-12)
        if np.max(np.abs(u - u_prev)) < tol:
            break

    T = np.outer(u, v) * K
    return T

def gromov_wasserstein_discrepancy(
    D1: np.ndarray,
    D2: np.ndarray,
    p: np.ndarray | None = None,
    q: np.ndarray | None = None,
    epsilon: float = 0.05,
    max_iter: int = 20
) -> tuple[np.ndarray, float]:
    """Compute Gromov-Wasserstein optimal coupling matrix and discrepancy.

    Args:
        D1: Metric distance matrix of donor space [n1, n1].
        D2: Metric distance matrix of recipient space [n2, n2].
        p: Probability distribution over donor points (uniform if None).
        q: Probability distribution over recipient points (uniform if None).
        epsilon: Entropic regularization factor.
        max_iter: Frank-Wolfe iterations.

    Returns:
        T: Optimal transport coupling matrix [n1, n2].
        gw_cost: Gromov-Wasserstein structural distortion cost.
    """
    n1, n2 = len(D1), len(D2)
    if p is None:
        p = np.full(n1, 1.0 / n1, dtype=np.float64)
    if q is None:
        q = np.full(n2, 1.0 / n2, dtype=np.float64)

    # Initial coupling: outer product of marginals
    T = np.outer(p, q)

    # Precompute quadratic terms
    D1_sq = D1 ** 2
    D2_sq = D2 ** 2

    for _ in range(max_iter):
        # Gradient of GW objective: G = D1_sq @ p @ 1^T + 1 @ q^T @ D2_sq^T - 2 * D1 @ T @ D2^T
        term1 = np.outer(np.dot(D1_sq, p), np.ones(n2))
        term2 = np.outer(np.ones(n1), np.dot(D2_sq, q))
        term3 = -2.0 * np.dot(np.dot(D1, T), D2.T)
        cost = term1 + term2 + term3

        # Solve intermediate Sinkhorn step
        T_new = sinkhorn_knopp(cost, p, q, epsilon=epsilon, max_iter=30)
        # Bounded line-search update
        T = 0.5 * T + 0.5 * T_new

    # Compute final GW distortion
    term3 = -2.0 * np.dot(np.dot(D1, T), D2.T)
    gw_cost = float(np.sum((term1 + term2 + term3) * T))
    return T, max(0.0, gw_cost)

def align_subcircuit_gromov_wasserstein(
    donor_weights: np.ndarray,
    host_weights: np.ndarray,
    epsilon: float = 0.05
) -> dict:
    """Compute unsupervised non-Euclidean manifold alignment between donor and host subcircuits.

    Args:
        donor_weights: Adjacency or weight matrix of donor subcircuit [n1, n1].
        host_weights: Adjacency or weight matrix of host subcircuit [n2, n2].

    Returns:
        Dictionary with coupling matrix, distortion cost, and pushforward projection.
    """
    D_donor = compute_graph_distance_matrix(np.abs(donor_weights))
    D_host = compute_graph_distance_matrix(np.abs(host_weights))

    T, distortion = gromov_wasserstein_discrepancy(D_donor, D_host, epsilon=epsilon)

    # Row-normalize coupling matrix to obtain conditional pushforward mapping
    row_sums = T.sum(axis=1, keepdims=True) + 1e-12
    conditional_pushforward = T / row_sums

    return {
        'status': 'ok',
        'donor_nodes': len(donor_weights),
        'host_nodes': len(host_weights),
        'gromov_wasserstein_distortion': float(distortion),
        'coupling_matrix_shape': list(T.shape),
        'max_coupling_density': float(np.max(T)),
        'entropy': float(-np.sum(T * np.log(T + 1e-12))),
        'coupling': T.tolist(),
        'conditional_pushforward': conditional_pushforward.tolist(),
        'scope': 'Gromov-Wasserstein non-Euclidean manifold alignment'
    }
