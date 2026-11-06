import torch
import numpy as np
from synapse.predictive_coding import PredictiveCodingEngine

def test_predictive_coding_step():
    engine = PredictiveCodingEngine(obs_dim=155, latent_dim=128, pred_dim=33)
    obs = torch.rand(155)
    memory = torch.randn(128)
    res = engine.step(obs, memory, acetylcholine=0.6)
    assert 'free_energy' in res
    assert 'sensory_surprisal' in res
    assert 'steer_bias' in res
    assert isinstance(res['free_energy'], float)
    assert -0.5 <= res['steer_bias'] <= 0.5

def test_predictive_coding_state_roundtrip():
    engine = PredictiveCodingEngine()
    obs = torch.rand(155)
    memory = torch.randn(128)
    engine.step(obs, memory)
    state = engine.export_state()
    engine2 = PredictiveCodingEngine()
    engine2.import_state(state)
    assert engine.state.free_energy == engine2.state.free_energy
    assert engine.state.sensory_surprisal == engine2.state.sensory_surprisal
