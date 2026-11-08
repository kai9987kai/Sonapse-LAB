"""Comprehensive Test Suite for 9,000 Neuron Scaling & Zero-Gain Backward Compatibility.

Verifies:
1. Default neuron scaling to 9,000 across SynapseConfig, frontier_apex, and max_tandem.
2. matches_saved backward compatibility: legacy checkpoint dictionaries load without error.
3. Mid-episode state export / import round-trip preservation.
4. Non-negotiable governing compatibility invariant:
   F_extended(x, s, disabled) == F_legacy(x, s) bit-for-bit (Delta = 0.0).
"""
import copy
import pytest
import numpy as np
import torch

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent


def test_9000_neuron_scaling_defaults():
    # 1. Direct configuration
    cfg = SynapseConfig(total_neurons=9000)
    assert cfg.total_neuron_count() == 9000
    assert cfg.source_node_counts() == (3000, 3000, 3000)

    # 2. Frontier apex preset
    apex = SynapseConfig.frontier_apex()
    assert apex.total_neuron_count() == 9000

    # 3. Max tandem preset
    tandem = SynapseConfig.max_tandem()
    assert tandem.total_neuron_count() == 9000
    assert tandem.neuroendocrine is True
    assert tandem.synthetic_glands is True
    assert tandem.affective_core is True
    assert tandem.neural_pulses is True
    assert tandem.sleep_replay is True
    assert tandem.shared_knowledge is True
    assert tandem.virtual_ecology_v2 is True
    assert tandem.live_microphone is False  # Must remain permission-gated


def test_matches_saved_legacy_checkpoint_compatibility():
    # Simulate a legacy checkpoint dictionary that omits all modern flags
    legacy_saved = {
        'seed': 7,
        'task': 'forage',
        'nodes_per_source': 100,
        'connectome_sources': ['flywire', 'manc', 'optic-lobe'],
        'readout': 'absolute',
    }

    current_cfg = SynapseConfig(seed=7, task='forage', nodes_per_source=100)
    assert current_cfg.matches_saved(legacy_saved) is True


def test_zero_gain_governing_compatibility_invariant():
    """F_extended(x, s, disabled) == F_legacy(x, s) bit-for-bit."""
    # Build matched seed configurations
    cfg_legacy = SynapseConfig(seed=42, nodes_per_source=25, connectome_mode='proxy')
    cfg_extended = SynapseConfig(
        seed=42,
        nodes_per_source=25,
        connectome_mode='proxy',
        neuroendocrine=False,
        synthetic_glands=False,
        affective_core=False,
        neural_pulses=False,
        sleep_replay=False,
        shared_knowledge=False,
    )

    agent_legacy = IntegratedAgent(cfg_legacy)
    agent_extended = IntegratedAgent(cfg_extended)

    # Step both agents through identical observation sequences
    np.random.seed(123)
    for step in range(5):
        obs = np.random.randn(155).astype(np.float32)
        out_legacy = agent_legacy.act(obs, learn=False)
        out_extended = agent_extended.act(obs, learn=False)

        # Action vectors must match bit-for-bit (exact equality)
        delta_action = np.max(np.abs(out_legacy['action'] - out_extended['action']))
        assert delta_action == 0.0, f"Action divergence at step {step}: max delta = {delta_action}"

        # Features must match bit-for-bit
        delta_feat = np.max(np.abs(out_legacy['features'] - out_extended['features']))
        assert delta_feat == 0.0, f"Feature divergence at step {step}: max delta = {delta_feat}"


def test_extended_agent_state_export_import_roundtrip():
    cfg = SynapseConfig(
        seed=42,
        nodes_per_source=25,
        connectome_mode='proxy',
        neuroendocrine=True,
        synthetic_glands=True,
        affective_core=True,
        neural_pulses=True,
        sleep_replay=True,
        shared_knowledge=True,
    )
    agent = IntegratedAgent(cfg)

    # Advance several steps
    for _ in range(4):
        obs = np.random.randn(155).astype(np.float32)
        agent.act(obs, learn=True)

    exported = agent.export_state()
    assert 'wpac' in exported
    assert 'mixed_affect' in exported
    assert 'synthetic_glands' in exported
    assert 'wave_solver' in exported
    assert 'fatigue_controller' in exported

    # Import into a fresh agent instance
    agent_cloned = IntegratedAgent(cfg)
    agent_cloned.import_state(exported)

    # Verify state was restored
    assert float(agent_cloned.fatigue_controller.sleep_pressure.item()) == float(agent.fatigue_controller.sleep_pressure.item())
    assert torch.equal(agent_cloned.wpac.h_state, agent.wpac.h_state)
