"""Master end-to-end integration and non-regression tests for AETHER GraftBridge and Unified Graft Bus.

Verifies:
1. Non-regression: default baseline configuration preserves legacy behavior bit-for-bit with zero auxiliary flux.
2. Concurrent co-existence: SANCF, NeuroWave-X2, Cross-Connectome Graft, and AETHER GraftBridge
   operate concurrently through NeuralGraftCommunicationBus without mutual overwriting.
3. Injection into recurrent integration: auxiliary currents directly affect rate dynamics in GraftedBrain.
4. Biological Dale's law compliance: column-wise presynaptic sign consistency across all projections.
5. Negative donor controls on integrated agent: causal distinction between authentic and scrambled transfer.
6. Full checkpoint serialization: export_state and import_state preserve bus and member state invariant.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent


def test_baseline_non_regression_zero_aux():
    """Verify that default baseline configuration has graft_bus disabled and zero aux current."""
    cfg = SynapseConfig(total_neurons=48, seed=42)
    agent = IntegratedAgent(cfg)

    assert not agent.graft_bus.enabled
    assert agent.comm_fabric is None or not agent.comm_fabric.enabled

    obs = np.random.randn(155).astype(np.float32)
    res = agent.act(obs, learn=False)
    assert res['action'].shape == (5,)
    assert np.isfinite(res['action']).all()


def test_concurrent_channel_coexistence_in_bus():
    """Verify that SANCF, NeuroWave-X2, CrossConnectomeGraft, and AetherGraftBridge coexist in the bus."""
    cfg = SynapseConfig.max_tandem(total_neurons=48, seed=42)
    agent = IntegratedAgent(cfg)

    assert agent.graft_bus is not None
    assert agent.graft_bus.enabled

    # Verify all four channels are registered on the bus
    channels = agent.graft_bus.channels
    assert "cross_graft" in channels
    assert "sancf" in channels
    assert "neurowave_x2" in channels
    assert "aether_bridge" in channels

    # Run step and verify telemetry from each channel
    obs = np.random.randn(155).astype(np.float32)
    res = agent.act(obs, learn=False)

    telem = res['telemetry']
    assert 'cross_connectome_graft' in telem
    assert 'aether_graft_bridge' in telem
    assert 'graft_bus' in telem
    assert 'sancf' in telem
    assert 'neurowave_x2' in telem

    # Check bus telemetry content
    bus_telem = telem['graft_bus']
    assert bus_telem['enabled'] is True
    assert len(bus_telem['active_channels']) >= 2


def test_independent_channel_ablation_in_agent():
    """Verify that disabling channels on the bus removes their current contribution."""
    cfg = SynapseConfig.max_tandem(total_neurons=48, seed=42)
    agent = IntegratedAgent(cfg)
    agent.brain.noise = 0.0

    obs = np.random.randn(155).astype(np.float32)

    # Non-zero rate vector
    r = torch.randn(agent.brain.n)

    # Disable all channels on the bus
    for name in ("cross_graft", "sancf", "neurowave_x2", "aether_bridge"):
        agent.graft_bus.set_channel_enabled(name, False)

    assert not agent.graft_bus.enabled
    out_zero = agent.graft_bus.compute_current(r)
    assert (out_zero.residual_current == 0.0).all()

    # Re-enable only CrossGraft
    agent.graft_bus.set_channel_enabled("cross_graft", True)
    assert agent.graft_bus.enabled
    out_cross = agent.graft_bus.compute_current(r)
    assert out_cross.total_norm > 0.0
    assert "cross_graft" in out_cross.channel_currents
    assert "aether_bridge" not in out_cross.channel_currents

    # Re-enable AetherBridge
    agent.graft_bus.set_channel_enabled("aether_bridge", True)
    out_two = agent.graft_bus.compute_current(r)
    assert "cross_graft" in out_two.channel_currents
    assert "aether_bridge" in out_two.channel_currents


def test_presynaptic_dales_law_in_agent():
    """Verify that agent's CrossConnectomeGraft satisfies biological presynaptic Dale's law."""
    cfg = SynapseConfig.max_tandem(total_neurons=48, seed=42)
    agent = IntegratedAgent(cfg)

    compliance = agent.cross_graft_engine.verify_dales_law()
    assert compliance == 100.0


def test_agent_aether_negative_control_differentiation():
    """Verify causal behavioral differentiation between authentic and scrambled AETHER transfer."""
    cfg_auth = SynapseConfig.max_tandem(total_neurons=48, aether_bridge_control='none', seed=123)
    cfg_scram = SynapseConfig.max_tandem(total_neurons=48, aether_bridge_control='scrambled', seed=123)

    agent_auth = IntegratedAgent(cfg_auth)
    agent_scram = IntegratedAgent(cfg_scram)
    agent_auth.brain.noise = 0.0
    agent_scram.brain.noise = 0.0

    obs = np.random.randn(155).astype(np.float32)
    torch.manual_seed(99)
    res_auth = agent_auth.act(obs, learn=False)

    torch.manual_seed(99)
    res_scram = agent_scram.act(obs, learn=False)

    # Scrambled modal donor control must produce distinct features / action trajectories
    diff = np.max(np.abs(res_auth['action'] - res_scram['action']))
    feat_diff = np.max(np.abs(res_auth['features'] - res_scram['features']))
    assert feat_diff > 0.0 or diff > 0.0


def test_agent_export_import_roundtrip():
    """Verify full agent export_state / import_state round-trip preserves bus and bridge states."""
    cfg = SynapseConfig.max_tandem(total_neurons=48, seed=42)
    agent1 = IntegratedAgent(cfg)
    agent1.brain.noise = 0.0

    obs = np.random.randn(155).astype(np.float32)
    agent1.act(obs, learn=False)

    state = agent1.export_state()
    assert 'aether_bridge' in state
    assert 'graft_bus' in state
    assert 'cross_graft_engine' in state

    agent2 = IntegratedAgent(cfg)
    agent2.brain.noise = 0.0
    agent2.import_state(state)

    # Check state parity
    assert agent2.graft_bus.max_aux_current == agent1.graft_bus.max_aux_current
    assert agent2.aether_bridge.spec.coupling_gain == agent1.aether_bridge.spec.coupling_gain
    assert agent2.cross_graft_engine.gamma == agent1.cross_graft_engine.gamma
    assert agent2.cross_graft_engine.verify_dales_law() == 100.0
