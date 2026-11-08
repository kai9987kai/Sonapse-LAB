"""Tests for Multi-Organism Fusion Connectome (Scaling to 12,000+ Neurons)."""
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.multi_connectome_fusion import (
    MultiConnectomeFusionEngine,
    ConnectomeSubstrateInfo,
    FusionConnectomeTelemetry,
)
from synapse.agent import IntegratedAgent
from synapse.runtime import LabRuntime


def test_fusion_engine_neuron_counts_and_substrates():
    engine = MultiConnectomeFusionEngine(seed=42)
    assert engine.total_dim == 12082
    assert len(engine.substrates) == 6

    names = [s.name for s in engine.substrates]
    assert "c_elegans_somatic" in names
    assert "drosophila_malecns_optic" in names
    assert "poseidon_tidal_donor" in names
    assert "wpac_affective_core" in names
    assert "auditory_cochlear_column" in names
    assert "visual_retinotopic_column" in names

    # Total sum of substrate neurons matches 12,082
    total_neurons = sum(s.num_neurons for s in engine.substrates)
    assert total_neurons == 12082
    assert sum(s.num_synapses for s in engine.substrates) > 300000


def test_fusion_engine_chunked_ode_step():
    engine = MultiConnectomeFusionEngine(seed=42)
    engine.reset_state()
    assert float(engine.firing_rates.sum().item()) == 0.0

    # Execute step with mock sensory input
    sensory = torch.linspace(-0.5, 0.5, 155)
    r, telem = engine.step(sensory_input=sensory, neuromod_gain=1.2, dt_ms=2.0)

    assert r.shape == (12082,)
    assert (r >= 0.0).all() and (r <= 1.0).all()  # Rates bounded in [0, 1]
    assert telem.total_neurons == 12082
    assert telem.mean_firing_rate > 0.0
    assert telem.dales_law_compliance == 1.0
    assert telem.integration_latency_ms >= 0.0

    summary = engine.get_summary()
    assert summary["total_neurons"] == 12082
    assert len(summary["substrates"]) == 6
    assert summary["telemetry"] is not None


def test_config_scale_12000_and_frontier_12k():
    cfg = SynapseConfig.scale_12000(seed=19, task="continental_biosphere")
    assert cfg.total_neurons == 12082
    assert cfg.task == "continental_biosphere"
    assert cfg.virtual_ecology_v2 is True
    assert cfg.evo_compat is True
    assert cfg.neuroendocrine is True
    assert cfg.sleep_replay is True

    cfg_alias = SynapseConfig.frontier_12k(seed=19, task="continental_biosphere")
    assert cfg_alias.total_neurons == 12082
    assert cfg_alias == cfg


def test_primary_invariant_zero_gain_identity():
    """Verify governing compatibility invariant: F_extended(s, a; 0) = F_legacy(s, a) bit-for-bit."""
    cfg_base = SynapseConfig(seed=42, nodes_per_source=16)
    cfg_ext = SynapseConfig(seed=42, nodes_per_source=16, virtual_ecology_v2=False, evo_compat=False)

    torch.manual_seed(42)
    agent_base = IntegratedAgent(cfg_base)
    torch.manual_seed(42)
    agent_ext = IntegratedAgent(cfg_ext)

    obs = np.linspace(-0.5, 0.5, 155, dtype=np.float32)
    out_base = agent_base.act(obs, learn=False)
    out_ext = agent_ext.act(obs, learn=False)

    delta_act = float(np.max(np.abs(out_base['action'] - out_ext['action'])))
    delta_feat = float(np.max(np.abs(out_base['features'] - out_ext['features'])))
    assert delta_act == 0.0
    assert delta_feat == 0.0
