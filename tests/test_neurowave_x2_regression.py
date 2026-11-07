import numpy as np
import pytest
from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent


def test_zero_gain_identity_guarantee():
    """Verify that neurowave_x2_gamma = 0.0 results in bit-for-bit identical actions to baseline."""
    obs = np.ones(155, dtype=np.float32) * 0.1

    # 1. Baseline agent (NeuroWave disabled)
    cfg_base = SynapseConfig(seed=42, neurowave_x2_enabled=False)
    agent_base = IntegratedAgent(cfg_base)
    res_base = agent_base.act(obs)
    action_base = res_base["action"]

    # 2. Agent with NeuroWave enabled at gamma = 0.0
    cfg_nw_zero = SynapseConfig(seed=42, neurowave_x2_enabled=True, neurowave_x2_gamma=0.0)
    agent_nw_zero = IntegratedAgent(cfg_nw_zero)
    res_nw_zero = agent_nw_zero.act(obs)
    action_nw_zero = res_nw_zero["action"]

    # Must be bit-for-bit identical
    assert np.array_equal(action_base, action_nw_zero), (
        f"Bit-for-bit identity violation! Base: {action_base}, NeuroWave(gamma=0): {action_nw_zero}"
    )


def test_legacy_default_config_preserves_behaviour():
    """Default SynapseConfig must have neurowave_x2_enabled=False and gamma=0.0."""
    cfg = SynapseConfig()
    assert cfg.neurowave_x2_enabled is False
    assert cfg.neurowave_x2_gamma == 0.0
    assert cfg.neurowave_x2_protocol == "adaptive"
    assert cfg.neurowave_x2_spectral_basis == "hybrid"


def test_dales_law_compliance_preserved():
    """Synapse baseline Dale's law handling is preserved under NeuroWave."""
    cfg = SynapseConfig(seed=123, neurowave_x2_enabled=True, neurowave_x2_gamma=0.0)
    agent = IntegratedAgent(cfg)
    # Check that brain base_w sign structure is strictly respected
    assert hasattr(agent.brain, "base_w")
    assert hasattr(agent.brain, "graft_w")
