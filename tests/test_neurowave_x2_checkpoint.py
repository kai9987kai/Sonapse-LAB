import copy
import numpy as np
import torch
import pytest

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.neurowave_x2.config import NeuroWaveX2Config
from synapse.neurowave_x2.fabric import NeuroWaveX2Fabric


def test_neurowave_fabric_state_export_import():
    cfg = NeuroWaveX2Config(enabled=True, residual_gain=0.05)
    fabric = NeuroWaveX2Fabric(config=cfg, num_regions=4, neurons_per_region=8, total_neurons=32, seed=42)
    # Perform some steps to mutate internal state
    activations = torch.randn(32)
    fabric.compute_current(activations)
    fabric.compute_current(activations)

    state = fabric.export_state()
    assert "basis_version" in state
    assert "topology_routes" in state
    assert "step_count" in state
    assert state["step_count"] == 2

    # Create new fabric and import
    fabric2 = NeuroWaveX2Fabric(config=cfg, num_regions=4, neurons_per_region=8, total_neurons=32, seed=99)
    assert fabric2.step_count == 0
    fabric2.import_state(state)
    assert fabric2.step_count == 2
    assert fabric2.topology_ctrl.list_routes() == fabric.topology_ctrl.list_routes()
    assert fabric2.spectral_mgr.basis_version == fabric.spectral_mgr.basis_version


def test_agent_checkpoint_roundtrip_with_neurowave():
    cfg = SynapseConfig(
        seed=42,
        neurowave_x2_enabled=True,
        neurowave_x2_gamma=0.0,
    )
    ag = IntegratedAgent(cfg)
    obs = np.zeros(155, dtype=np.float32)
    ag.act(obs)

    state = ag.export_state()
    assert "neurowave_x2" in state
    assert state["neurowave_x2"] is not None

    # Restore into fresh agent
    ag2 = IntegratedAgent(cfg)
    ag2.import_state(state)
    assert ag2.neurowave_x2.step_count == ag.neurowave_x2.step_count


def test_agent_checkpoint_legacy_migration_without_neurowave():
    """Verify that importing a legacy checkpoint (without neurowave_x2 key) succeeds without error."""
    cfg = SynapseConfig(seed=42, neurowave_x2_enabled=False)
    ag = IntegratedAgent(cfg)
    state = ag.export_state()
    # Simulate legacy checkpoint by removing neurowave_x2 key
    if "neurowave_x2" in state:
        del state["neurowave_x2"]

    ag.import_state(state)
    # Successfully imported without raising
    assert True
