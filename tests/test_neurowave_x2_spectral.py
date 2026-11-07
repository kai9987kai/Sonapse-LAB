"""Unit tests verifying Graph-Laplacian Spectral Basis & Alignment Engine."""
import math
import numpy as np
import pytest
from synapse.neurowave_x2.spectral_basis import (
    SpectralBasisManager,
    compute_eigenbasis,
    compute_normalized_laplacian,
    compute_subspace_drift,
    procrustes_align_basis,
)


def test_normalized_laplacian_symmetry_and_spectrum():
    # 4-node cycle graph
    A = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=np.float64)
    L = compute_normalized_laplacian(A)

    # 1. Symmetry
    np.testing.assert_allclose(L, L.T, atol=1e-10)

    # 2. Smallest eigenvalue is 0.0 (constant eigenvector)
    evals, evecs = compute_eigenbasis(A, num_modes=4)
    assert math.isclose(evals[0], 0.0, abs_tol=1e-6)
    # Eigenvalues must lie in [0, 2]
    assert np.all(evals >= -1e-6)
    assert np.all(evals <= 2.0 + 1e-6)


def test_spectral_alignment_and_subspace_drift():
    A1 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
    _, basis1 = compute_eigenbasis(A1, num_modes=2)

    # Subspace drift with identical basis must be 0.0
    drift_self = compute_subspace_drift(basis1, basis1)
    assert math.isclose(drift_self, 0.0, abs_tol=1e-6)

    # Minor perturbation to graph
    A2 = A1 + 0.1 * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float64)
    _, basis2 = compute_eigenbasis(A2, num_modes=2)

    aligned = procrustes_align_basis(basis1, basis2)
    drift = compute_subspace_drift(basis1, aligned)
    assert drift >= 0.0
    assert drift <= 2.0


def test_spectral_basis_manager_caching_and_versioning():
    mgr = SpectralBasisManager(num_modes=3)
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)

    ver1, ev1, b1 = mgr.get_or_compute_basis(A)
    assert ver1 == 1

    # Identical adjacency -> cache hit -> version unchanged
    ver2, ev2, b2 = mgr.get_or_compute_basis(A)
    assert ver2 == 1
    assert b1 is b2

    # Mutated adjacency -> cache invalidation -> version incremented
    A_mut = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    ver3, ev3, b3 = mgr.get_or_compute_basis(A_mut)
    assert ver3 == 2
