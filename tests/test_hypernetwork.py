import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.hypernetwork import WeightSpaceHypernetwork, compile_hypernetwork_patch

def test_hypernetwork_patch_synthesis():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    hypernet = WeightSpaceHypernetwork(in_dim=32, rank=2)

    diag = {
        'mean_return': 0.15,
        'total_steps': 30,
        'mean_graph_activity': 0.08,
        'mean_approach_valence': 0.2,
        'mean_avoid_valence': -0.1,
        'action_mean': [0.1, -0.05, 0.0, 0.0, 0.0],
        'task': 'forage',
    }

    patch = compile_hypernetwork_patch(agent, hypernet, diag, target_surface='motor.weight', max_edits=16)
    assert patch['schema'] == 'synapse-weight-program-v1'
    assert patch['writes'] > 0
    assert patch['delta_l2'] >= 0.0

    # Test targeting graft weights
    graft_patch = compile_hypernetwork_patch(agent, hypernet, diag, target_surface='brain.graft_w', max_edits=16)
    assert graft_patch['schema'] == 'synapse-weight-program-v1'
    assert graft_patch['writes'] > 0
