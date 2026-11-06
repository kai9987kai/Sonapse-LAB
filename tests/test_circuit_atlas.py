"""Tests for Causal Circuit Atlas Mapping."""
import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.circuit_atlas import CausalCircuitAtlas


def test_causal_circuit_atlas():
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))
    atlas = CausalCircuitAtlas(agent)

    profile = atlas.map_circuit('FlyWire_Central_Sample', list(range(10)), episodes=2)
    assert profile.subcircuit_name == 'FlyWire_Central_Sample'
    assert 'thrust' in profile.behavioral_effects
    assert 'steering' in profile.behavioral_effects
    assert isinstance(profile.causal_necessity, float)
    assert isinstance(profile.causal_sufficiency, float)

    full = atlas.build_full_atlas()
    assert full['total_circuits_mapped'] >= 3
    assert 'FlyWire_Central_Brain' in full['circuits']
