import math
import torch
import numpy as np
import pytest

from synapse.neurowave_x2.hybrid_basis import (
    HybridBasis,
    construct_dct_basis,
    construct_hadamard_basis,
    construct_random_orthogonal_basis,
    orthogonalize_stacked_bases,
)


def test_construct_dct_basis():
    dct = construct_dct_basis(num_modes=4, dimension=8)
    assert dct.shape == (4, 8)
    # Rows should be orthonormal: B @ B^T ~ I
    gram = dct @ dct.T
    assert np.allclose(gram, np.eye(4), atol=1e-5)


def test_construct_hadamard_basis():
    had = construct_hadamard_basis(num_modes=4, dimension=8)
    assert had.shape == (4, 8)
    gram = had @ had.T
    assert np.allclose(gram, np.eye(4), atol=1e-5)


def test_construct_random_orthogonal_basis():
    rand_o = construct_random_orthogonal_basis(num_modes=4, dimension=8, seed=123)
    assert rand_o.shape == (4, 8)
    gram = rand_o @ rand_o.T
    assert np.allclose(gram, np.eye(4), atol=1e-5)


def test_orthogonalize_stacked_bases():
    # Construct 2 orthogonal graph modes and 2 arbitrary learned vectors
    graph = torch.eye(2, 6)
    learned = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0, 1.0, 0.0, 0.0]])
    hybrid = orthogonalize_stacked_bases(graph, learned)
    assert hybrid.shape == (4, 6)
    gram = torch.matmul(hybrid, hybrid.T)
    # Check that rows are orthogonal
    eye = torch.eye(4)
    assert torch.allclose(gram, eye, atol=1e-5)
    # Check that the first two rows are still the graph rows
    assert torch.allclose(hybrid[:2], graph, atol=1e-5)


def test_hybrid_basis_module():
    basis = HybridBasis(num_graph_modes=3, num_learned_modes=3, dimension=10, basis_type="hybrid")
    # Initialize graph basis with random orthonormal vectors
    rng_basis = construct_random_orthogonal_basis(3, 10, seed=42)
    basis.set_graph_basis(rng_basis, version=1)

    eff = basis.get_effective_basis()
    assert eff.shape == (6, 10)

    diag = basis.compute_basis_diagnostics()
    assert diag["basis_rank"] == 6.0
    assert diag["orthogonality_error"] < 1e-4
    assert diag["basis_version"] == 1.0


def test_baseline_basis_types():
    for b_type in ["identity", "dct", "hadamard", "random_orthogonal"]:
        basis = HybridBasis(num_graph_modes=4, num_learned_modes=0, dimension=8, basis_type=b_type)
        eff = basis.get_effective_basis()
        assert eff.shape == (4, 8)
        diag = basis.compute_basis_diagnostics()
        assert diag["orthogonality_error"] < 1e-4
