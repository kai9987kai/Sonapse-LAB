import numpy as np
import pytest
from synapse.neuromodulation import NeuromodulatoryEngine, NeuromodulatorState

def test_neuromodulatory_state():
    state = NeuromodulatorState(dopamine=0.5, serotonin=0.3, octopamine=0.4, acetylcholine=0.6)
    vec = state.to_vector()
    assert len(vec) == 4
    assert vec[0] == 0.5
    d = state.to_dict()
    assert 'dopamine' in d and 'serotonin' in d and 'octopamine' in d and 'acetylcholine' in d

def test_neuromodulatory_engine_transitions():
    engine = NeuromodulatoryEngine()
    # Step 1: positive reward -> dopamine should increase
    s1 = engine.step(reward=1.0, sensory_contrast=0.8, threat_level=0.0, energy_depletion=0.1)
    assert s1.dopamine > 0.0
    assert s1.acetylcholine > 0.4

    # Step 2: high threat -> serotonin should rise
    s2 = engine.step(reward=-0.5, sensory_contrast=0.2, threat_level=0.9, energy_depletion=0.8)
    assert s2.serotonin > s1.serotonin
    assert s2.octopamine > s1.octopamine

    gains = engine.compute_gains()
    assert 'connectome_excitability' in gains
    assert 'plasticity_multiplier' in gains
    assert 'sensory_attention_gate' in gains
    assert 'motor_vigor' in gains
    for k, v in gains.items():
        assert v > 0.0
