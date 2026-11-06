"""Verification suite for dormant expert lifecycle, two-channel evoked readout,
value-guided planning, continual backprop, fast-weight memory, and latent deliberation.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.dormant_lifecycle import DormantLifecycleManager, DormantState
from synapse.experience import ExperienceWorldModel, TDValueModel
from synapse.experience_planner import ExperienceMPC, HybridExperiencePlanner
from synapse.stateio import digest
from synapse.runtime import LabRuntime


def test_dormant_lifecycle_transitions_and_consolidation():
    mgr = DormantLifecycleManager(num_slots=2, feature_dim=32, seed=42)
    assert mgr.records[0].state == DormantState.DORMANT.value
    assert mgr.records[0].activation_gain == 0.0

    # Step with low novelty -> remains DORMANT
    recruited, slot, gain = mgr.step_novelty(novelty=0.005, preferred_slot=0)
    assert not recruited
    assert mgr.records[0].state == DormantState.DORMANT.value

    # Novelty trigger 1 -> INITIALIZING
    recruited, slot, gain = mgr.step_novelty(novelty=0.03, preferred_slot=0)
    assert recruited
    assert slot == 0
    assert mgr.records[0].state == DormantState.INITIALIZING.value
    assert mgr.records[0].activation_gain > 0.0

    # Subsequent novelty triggers ramp activation -> ACTIVATING -> ACTIVE
    for _ in range(8):
        mgr.step_novelty(novelty=0.03, preferred_slot=0)

    assert mgr.records[0].state == DormantState.ACTIVE.value
    assert mgr.records[0].activation_gain == pytest.approx(1.0)

    # Evaluate functional contribution
    pooled = torch.randn(32)
    score = mgr.evaluate_contribution(pooled, slot_idx=0)
    assert score > 0.0

    # Trigger consolidation into native module
    native = nn.Linear(32, 32, bias=False)
    nn.init.zeros_(native.weight)
    native_before = native.weight.clone()

    # Step consolidation
    for _ in range(60):
        res = mgr.step_consolidation(native, recruited_slot=0, rate=0.05)

    assert mgr.consolidations_completed >= 1
    # Native weights must have absorbed expert knowledge
    assert not torch.equal(native.weight, native_before)
    assert float(native.weight.detach().norm()) > 0.0
    # Slot 0 must have retired and recycled back to DORMANT
    assert mgr.records[0].state == DormantState.DORMANT.value


def test_agent_dormant_lifecycle_integration():
    cfg = SynapseConfig(nodes_per_source=16, seed=42, dormant_lifecycle=True)
    agent = IntegratedAgent(cfg)
    assert hasattr(agent, 'dormant_lifecycle')
    assert len(agent.dormant) == 2

    obs = np.ones(155, dtype=np.float32) * 0.1
    # Run a few steps
    for _ in range(5):
        res = agent.act(obs, learn=True)
        assert 'dormant_lifecycle' in res['telemetry']
        assert 'slots' in res['telemetry']['dormant_lifecycle']


def test_two_channel_persistent_evoked_readout():
    cfg = SynapseConfig(nodes_per_source=16, seed=42, readout='persistent_evoked')
    agent = IntegratedAgent(cfg)
    obs = np.ones(155, dtype=np.float32) * 0.1
    res = agent.act(obs, learn=False)

    channels = res['telemetry']['evoked_channels']
    assert 'channel_m_norm' in channels
    assert 'channel_d_norm' in channels
    assert 'gate_alpha' in channels
    assert 'persistent_norm' in channels
    assert 0.0 <= channels['gate_alpha'] <= 1.0


def test_td_value_model_and_value_guided_mpc():
    val_model = TDValueModel(state_dim=3, gamma=0.9, lr=0.1, seed=17)
    s = np.array([0.5, -0.2, 0.1])
    s_next = np.array([0.6, -0.1, 0.2])

    before = val_model.predict(s)
    for _ in range(20):
        val_model.update(s, reward=1.0, next_state=s_next, done=False)
    after = val_model.predict(s)
    # Value prediction increases towards discounted return
    assert after > before

    # Test ValueGuided MPC in ExperienceMPC
    wm = ExperienceWorldModel(state_dim=3, action_dim=1, context_dim=1, seed=17)
    rng = np.random.default_rng(42)
    for _ in range(100):
        st = rng.uniform(-1, 1, 3)
        act = rng.uniform(-1, 1, 1)
        nst = st + np.array([0.1 * act[0], 0.05, 0.0])
        wm.update(st, act, nst, reward=st[0] + act[0])

    mpc_normal = ExperienceMPC(wm, horizon=3, value_guided=False)
    mpc_val = ExperienceMPC(wm, horizon=3, value_guided=True)

    assert mpc_val.value_guided is True
    assert mpc_normal.value_guided is False

    baseline = np.array([0.0])
    _, diag_val = mpc_val.plan(s, baseline)
    assert diag_val['value_guided'] is True


def test_continual_backprop_in_integrated_agent():
    cfg = SynapseConfig(nodes_per_source=16, seed=42, continual_backprop=True)
    agent = IntegratedAgent(cfg)
    assert agent.continual_backprop is not None

    obs = np.ones(155, dtype=np.float32) * 0.1
    next_obs = np.ones(155, dtype=np.float32) * 0.12
    agent.act(obs, learn=True)
    diag = agent.observe_transition(next_obs, reward=0.5, done=False)

    assert agent.continual_backprop.step_count >= 1
    assert 'continual_backprop' in agent.last_telemetry
    assert agent.last_telemetry['continual_backprop']['enabled'] is True


def test_fast_weight_memory_in_integrated_agent():
    cfg = SynapseConfig(nodes_per_source=16, seed=42, fast_weight_memory=True)
    agent = IntegratedAgent(cfg)
    assert agent.fast_weight_memory is not None

    obs = np.ones(155, dtype=np.float32) * 0.1
    next_obs = np.ones(155, dtype=np.float32) * 0.12
    out = agent.act(obs, learn=True)
    agent.observe_transition(next_obs, reward=0.5, done=False)

    assert agent.fast_weight_memory.inner_step_count >= 1
    assert 'fast_weight_memory' in out['telemetry']
    assert out['telemetry']['fast_weight_memory']['enabled'] is True


def test_coconut_latent_deliberation_in_integrated_agent():
    cfg = SynapseConfig(nodes_per_source=16, seed=42, latent_reasoning=True)
    agent = IntegratedAgent(cfg)
    assert agent.latent_reasoner is not None

    obs = np.ones(155, dtype=np.float32) * 0.1
    # Trigger threat / free energy to engage thinking
    agent.homeostasis.state.threat = 0.8
    out = agent.act(obs, learn=False)

    assert 'latent_reasoner' in out['telemetry']
    assert out['telemetry']['latent_reasoner']['engaged'] is True
    assert out['telemetry']['latent_reasoner']['recurrent_depth'] >= 1


def test_complete_state_serialization_with_all_advancements(tmp_path):
    cfg = SynapseConfig(
        nodes_per_source=16,
        seed=42,
        dormant_lifecycle=True,
        continual_backprop=True,
        fast_weight_memory=True,
        latent_reasoning=True,
        value_guided_planning=True,
        readout='persistent_evoked'
    )
    lab = LabRuntime(cfg, tmp_path)
    lab.step(4, learn=True)

    saved = lab.save('all_advancements')
    assert saved['steps'] == 4

    fresh = LabRuntime(cfg, tmp_path)
    fresh.load('all_advancements')

    assert fresh.total_steps == 4
    assert np.array_equal(fresh.obs, lab.obs)
    assert digest(fresh.agent.export_state()) == digest(lab.agent.export_state())
