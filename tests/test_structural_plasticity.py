import numpy as np
import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.structural_plasticity import (
    audit_structural_topology,
    prune_dormant_grafts,
    sprout_correlated_synapses,
)

def test_audit_structural_topology():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    audit = audit_structural_topology(agent)
    assert audit['status'] == 'ok'
    assert audit['neurons'] == agent.graph.n
    assert audit['inter_neuropil_bridges'] > 0
    assert audit['synaptic_density'] > 0

def test_prune_dormant_grafts():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    # Set a few graft weights to near zero
    with torch.no_grad():
        agent.brain.graft_w[0] = 0.005
    res = prune_dormant_grafts(agent, prune_threshold=0.01)
    assert res['status'] == 'ok'
    assert res['pruned_synapses'] >= 1
    assert res['patch_packet']['schema'] == 'synapse-weight-program-v1'

def test_sprout_correlated_synapses():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    n = agent.graph.n
    coact = np.zeros((n, n), dtype=np.float32)
    # Simulate high cross-neuropil correlation between neuron 0 (flywire) and neuron 20 (manc)
    coact[0, 20] = 0.85
    coact[20, 0] = 0.85
    res = sprout_correlated_synapses(agent, coact, sprout_threshold=0.70, max_sprouts=4)
    assert res['status'] == 'ok'
    assert res['sprouted_count'] >= 1
    assert res['sprouted_edges'][0]['correlation'] >= 0.70
