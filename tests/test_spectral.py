import numpy as np
import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.spectral import (
    build_connectome_adjacency,
    compute_connectome_harmonics,
    project_spectral_activity,
    spectral_graph_filter,
)

def test_spectral_connectomics():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)

    A = build_connectome_adjacency(agent.graph)
    assert A.shape == (agent.graph.n, agent.graph.n)
    assert np.allclose(A, A.T)  # Symmetric

    harmonics = compute_connectome_harmonics(agent.graph, num_modes=8)
    assert harmonics['num_modes'] == 8
    assert len(harmonics['eigenvalues']) == 8
    assert harmonics['eigenvectors'].shape == (agent.graph.n, 8)
    # Eigenvalues must be non-decreasing
    assert all(harmonics['eigenvalues'][i] <= harmonics['eigenvalues'][i+1] for i in range(7))

    proj = project_spectral_activity(agent.brain.r, harmonics)
    assert proj['status'] == 'ok'
    assert len(proj['coefficients']) == 8
    assert proj['total_spectral_power'] > 0
    assert 0.0 <= proj['global_mode_fraction'] <= 1.0

    # Filter weights
    filtered = spectral_graph_filter(agent.brain.graft_w, harmonics, low_pass_cutoff=4)
    assert filtered.shape == agent.brain.graft_w.shape
