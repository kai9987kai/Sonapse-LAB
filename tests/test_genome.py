"""Tests for Developmental Neural Genome & Metabolic Neurogenesis."""
import pytest
import numpy as np
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.genome import DevelopmentalGenome, MetabolicNeurogenesisEngine


def test_developmental_genome_morphogenesis():
    g = DevelopmentalGenome()
    morph = g.morphogenesis(num_neurons=32, seed=42)

    assert morph['num_neurons'] == 32
    assert morph['positions'].shape == (32, 3)
    assert len(morph['cell_types']) == 32
    assert morph['adjacency_matrix'].shape == (32, 32)
    # Autapses (self-connections) should be zero
    assert np.all(np.diagonal(morph['adjacency_matrix']) == 0.0)


def test_metabolic_neurogenesis():
    g = DevelopmentalGenome()
    engine = MetabolicNeurogenesisEngine(g, max_atp=1.0)
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))

    # Stepping metabolism
    atp = engine.step_metabolism(mean_activity=0.5, total_neurons=agent.brain.n)
    assert 0.0 <= atp <= 1.0

    # Trigger activity-dependent neurogenesis under high prediction error
    event = engine.evaluate_neurogenesis(agent, local_free_energy=0.9, hotspot_neuropil=0)
    assert event is not None
    assert event['neuropil'] == 0
    assert event['atp_remaining'] < 1.0
    assert event['resulting_digest'] != event['parent_digest']
