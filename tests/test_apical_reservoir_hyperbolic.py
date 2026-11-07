"""Unit tests for Apical Amplification, Spectral Reservoir, and Hyperbolic Manifold."""

import pytest
import torch
import numpy as np
from synapse.apical_amplification import ApicalAmplificationEngine
from synapse.spectral_reservoir import SpectralReservoirEngine
from synapse.hyperbolic_manifold import HyperbolicManifoldEngine


def test_apical_amplification_dynamics_and_coincidence():
    engine = ApicalAmplificationEngine(num_units=16, ca_threshold=0.4, max_gain=2.5, seed=123)
    soma_in = torch.ones(16) * 0.8
    context_in = torch.ones(32) * 1.5

    # Step without context -> normal baseline gain
    out1, telem1 = engine(soma_in, apical_context=None)
    assert out1.shape == (16,)
    assert telem1.mean_amplification_gain >= 1.0

    # Step across multiple timesteps with strong top-down context -> triggers Ca2+ plateau and coincidence amplification
    gains = []
    for _ in range(12):
        out_step, telem_step = engine(soma_in, apical_context=context_in)
        gains.append(telem_step.mean_amplification_gain)

    assert max(gains) > 1.0
    assert len(telem_step.recent_somatic_trace) > 0
    assert len(telem_step.recent_apical_trace) > 0

    # Test state export and import
    exported = engine.export_state()
    assert 'v_somatic' in exported
    assert 'v_apical' in exported
    assert 'ca_timer' in exported

    engine_cloned = ApicalAmplificationEngine(num_units=16, seed=123)
    engine_cloned.import_state(exported)
    assert torch.allclose(engine.v_somatic, engine_cloned.v_somatic, atol=1e-5)
    assert torch.allclose(engine.v_apical, engine_cloned.v_apical, atol=1e-5)


def test_spectral_reservoir_echo_state_and_rls():
    engine = SpectralReservoirEngine(reservoir_size=32, input_dim=8, output_dim=3, spectral_radius=0.95, seed=456)

    inp = torch.tensor([0.5, -0.2, 0.8, -0.5, 0.1, 0.3, -0.7, 0.4])
    target = torch.tensor([1.0, 0.0, -1.0])

    # Run initial step without learning
    readout1, telem1 = engine.step(inp, learn=False)
    assert readout1.shape == (3,)
    assert telem1.spectral_radius == 0.95
    assert len(telem1.eigenvalues_real) > 0

    # Run steps with online RLS learning
    initial_err = 0.0
    for step in range(10):
        readout_step, telem_step = engine.step(inp, learn=True, target=target)
        if step == 0:
            initial_err = telem_step.rls_prediction_error
    final_err = telem_step.rls_prediction_error
    # RLS should adapt readout towards target
    assert final_err < initial_err or final_err < 0.5

    # Test state export and import
    exported = engine.export_state()
    assert 'state' in exported
    assert 'w_out' in exported

    engine_cloned = SpectralReservoirEngine(reservoir_size=32, input_dim=8, output_dim=3, seed=456)
    engine_cloned.import_state(exported)
    assert np.allclose(engine.state.cpu().numpy(), engine_cloned.state.cpu().numpy(), atol=1e-5)


def test_hyperbolic_manifold_metric_and_geodesics():
    engine = HyperbolicManifoldEngine(seed=789)

    # Test metric: d_H(u, u) == 0
    u = np.array([0.2, 0.3])
    d_zero = engine.hyperbolic_distance(u, u)
    assert abs(d_zero) < 1e-6

    # Test metric symmetry: d_H(u, v) == d_H(v, u)
    v = np.array([-0.4, 0.5])
    d_uv = engine.hyperbolic_distance(u, v)
    d_vu = engine.hyperbolic_distance(v, u)
    assert abs(d_uv - d_vu) < 1e-6
    assert d_uv > 0.0

    # Test geodesic arc points
    arc = engine.compute_geodesic_arc(u, v, num_points=8)
    assert len(arc) == 8
    # Start and end points must match within tolerance
    assert np.allclose(arc[0], u, atol=1e-2)
    assert np.allclose(arc[-1], v, atol=1e-2)
    # All arc points must lie strictly inside the unit disk ||p|| < 1
    for pt in arc:
        assert np.hypot(pt[0], pt[1]) < 1.0

    # Test telemetry payload
    telem = engine.step([0, 1])
    assert telem.curvature == -1.0
    assert len(telem.poincare_nodes) == 8
    assert len(telem.active_geodesics) > 0
    telem_dict = telem.to_dict()
    assert 'poincare_nodes' in telem_dict
    assert 'active_geodesics' in telem_dict
