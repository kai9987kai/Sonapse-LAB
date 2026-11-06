"""Tests for 3D Neural Observatory Embeddings & Live Telemetry."""
import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.observatory import NeuralObservatory


def test_neural_observatory():
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))
    obs = NeuralObservatory(agent, seed=42)

    struct = obs.get_structure_payload()
    assert 'nodes' in struct
    assert 'edges' in struct
    assert 'neuropils' in struct
    assert struct['total_nodes'] == agent.brain.n
    assert len(struct['neuropils']) == 3

    node0 = struct['nodes'][0]
    assert 'x' in node0 and 'y' in node0 and 'z' in node0
    assert 'neuropil' in node0

    act = obs.get_live_activity_payload()
    assert 'mean_rate' in act
    assert 'free_energy' in act
    assert 'neuromodulators' in act
    assert len(act['node_activations']) == agent.brain.n
