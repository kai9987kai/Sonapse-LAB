"""Tests for Biological Connectome Compiler and Degree-Preserving Nulls."""
import pytest
import numpy as np
import torch
from synapse.connectome_compiler import BiologicalConnectomeCompiler, BiologicalConnectomeModule


def test_biological_connectome_compiler_and_null():
    compiler = BiologicalConnectomeCompiler()
    # Compile a Drosophila motor subcircuit
    bio_module = compiler.compile_subcircuit(query='motor', max_neurons=40)
    assert isinstance(bio_module, BiologicalConnectomeModule)
    assert bio_module.num_nodes <= 40
    assert bio_module.num_nodes > 0
    assert bio_module.weight_matrix.shape == (bio_module.num_nodes, bio_module.num_nodes)

    # Test forward simulation
    bio_module.reset_state()
    x = torch.ones(len(bio_module.sensory_indices))
    out = bio_module(x)
    assert out.shape == (bio_module.num_nodes,)
    assert torch.isfinite(out).all()

    # Generate degree-preserving null model
    null_module = compiler.generate_degree_preserving_null(bio_module, swaps=100, seed=42)
    assert null_module.num_nodes == bio_module.num_nodes

    # Verify that in-degree and out-degree are preserved
    w_bio = np.abs(bio_module.weight_matrix.cpu().numpy()) > 1e-6
    w_null = np.abs(null_module.weight_matrix.cpu().numpy()) > 1e-6
    assert np.count_nonzero(w_bio) == np.count_nonzero(w_null)

    # Run benchmark comparison
    res = compiler.benchmark_biological_vs_null(bio_module, null_module, steps=20)
    assert 'biological' in res
    assert 'degree_preserving_null' in res
    assert 'biological_signal_advantage' in res
    assert isinstance(res['biological']['spectral_radius'], float)


def test_sparse_biological_connectome_module():
    from synapse.connectome_compiler import BiologicalConnectomeCompiler, SparseBiologicalConnectomeModule
    compiler = BiologicalConnectomeCompiler()
    sparse_module = compiler.compile_sparse_subcircuit(query='visual', max_neurons=48)
    assert isinstance(sparse_module, SparseBiologicalConnectomeModule)
    assert sparse_module.num_nodes <= 48
    assert sparse_module.num_nodes > 0
    assert sparse_module.sparse_weight.is_sparse

    sparse_module.reset_state()
    x = torch.ones(len(sparse_module.sensory_indices))
    out = sparse_module(x)
    assert out.shape == (sparse_module.num_nodes,)
    assert torch.isfinite(out).all()
    assert isinstance(sparse_module.spectral_radius, float)
