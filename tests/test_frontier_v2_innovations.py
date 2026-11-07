"""Comprehensive unit test suite for Frontier V2 Innovations:
1. Non-Equilibrium Neural Thermodynamics & Arrow-of-Time Dissipation
2. Homeostatic Metaplasticity & BCM Sliding Threshold
3. Dynamic Transfer Entropy & Partial Information Decomposition (PID)
4. 8,000-Neuron Connectome Scaling & Replay Invariance
5. 3D Causal & Transfer-Entropy Streamlines in Observatory
6. Tandem Max Section 03 Assay & Server Integration
"""

import math
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.thermodynamics import NeuralThermodynamicsEngine, run_thermodynamics_assay
from synapse.metaplasticity_bcm import BcmMetaplasticityEngine, run_bcm_metaplasticity_assay
from synapse.transfer_entropy import TransferEntropyEngine, run_transfer_entropy_assay
from synapse.observatory import NeuralObservatory
from synapse.tandem_max import run_frontier_v2_innovations_assay


def test_thermodynamics_engine_broken_detailed_balance():
    engine = NeuralThermodynamicsEngine(num_states=8)
    rng = np.random.RandomState(42)

    # Drive cyclic non-equilibrium state transitions: 0 -> 1 -> 2 -> ... -> 7 -> 0
    prev_state = 0
    for step in range(32):
        act = np.zeros(64, dtype=np.float32)
        # Shift activation block
        k = (step % 8) * 8
        act[k : k + 8] = 0.8
        st = engine.step(act, task_utility=0.8, learn=True)

    assert st.total_transitions == 32
    assert st.entropy_production_rate >= 0.0
    assert st.thermodynamic_efficiency > 0.0

    # Serialization test
    exported = engine.export_state()
    engine2 = NeuralThermodynamicsEngine(num_states=8)
    engine2.import_state(exported)
    assert engine2.state.entropy_production_rate == engine.state.entropy_production_rate
    assert engine2.state.total_transitions == engine.state.total_transitions


def test_thermodynamics_assay():
    res = run_thermodynamics_assay(None, steps=20, seed=42)
    assert res['status'] == 'completed'
    assert res['steps_evaluated'] == 20
    assert res['positive_entropy_production'] is True
    assert 'disclaimer' in res


def test_bcm_metaplasticity_sliding_threshold():
    n_nodes = 64
    engine = BcmMetaplasticityEngine(num_neurons=n_nodes, target_activity=0.25)
    rng = np.random.RandomState(42)

    # Test learn=False invariant: zero modifications
    r_zero = np.full(n_nodes, 0.5, dtype=np.float32)
    w_delta, st = engine.step(r_zero, learn=False)
    assert w_delta is None
    assert st.updates_count == 0

    # Run learning bursts
    for _ in range(15):
        r_high = np.clip(0.6 + 0.1 * rng.randn(n_nodes), 0.0, 1.0).astype(np.float32)
        engine.step(r_high, learn=True)

    # Threshold must have elevated
    assert engine.state.mean_theta_m > 0.25 ** 2

    # Serialization test
    exported = engine.export_state()
    engine2 = BcmMetaplasticityEngine(num_neurons=n_nodes)
    engine2.import_state(exported)
    assert math.isclose(engine2.state.mean_theta_m, engine.state.mean_theta_m, rel_tol=1e-5)


def test_bcm_assay():
    res = run_bcm_metaplasticity_assay(None, steps=24, seed=42)
    assert res['status'] == 'completed'
    assert res['sliding_threshold_adapted'] is True
    assert 0.0 <= res['homeostatic_scaling_factor'] <= 2.0


def test_transfer_entropy_engine_and_assay():
    engine = TransferEntropyEngine(buffer_size=32)
    origins = np.array([2] * 32 + [0] * 32 + [1] * 32, dtype=np.int64)

    for step in range(25):
        r = np.zeros(96, dtype=np.float32)
        # Driven sequence: sensory (2) triggers central (0) triggers motor (1)
        r[:32] = float(0.5 + 0.4 * math.sin(step * 0.5))
        r[32:64] = float(0.5 + 0.3 * math.sin((step - 1) * 0.5))
        r[64:96] = float(0.5 + 0.3 * math.sin((step - 2) * 0.5))
        engine.step(r, origins, learn=True)

    st = engine.state
    assert st.history_steps >= 20
    assert st.te_sensory_to_central >= 0.0

    # Serialization test
    exported = engine.export_state()
    engine2 = TransferEntropyEngine(buffer_size=32)
    engine2.import_state(exported)
    assert engine2.state.te_sensory_to_central == engine.state.te_sensory_to_central


def test_transfer_entropy_assay():
    res = run_transfer_entropy_assay(None, steps=24, seed=42)
    assert res['status'] == 'completed'
    assert res['steps_evaluated'] == 24
    assert 'causal_flow_confirmed' in res


def test_connectome_8000_neuron_allocation():
    cfg = SynapseConfig.frontier_apex(total_neurons=8000)
    assert cfg.total_neurons == 8000
    counts = cfg.source_node_counts()
    assert counts == (2667, 2667, 2666)
    assert sum(counts) == 8000


def test_observatory_streamlines_and_lod():
    cfg = SynapseConfig(total_neurons=48, seed=42)
    agent = IntegratedAgent(cfg)
    obs = NeuralObservatory(agent, seed=42)
    payload = obs.get_structure_payload()

    assert 'lod_clusters' in payload
    assert payload['num_lod_clusters'] > 0
    assert 'streamlines' in payload
    assert isinstance(payload['streamlines'], list)
    if payload['streamlines']:
        sl = payload['streamlines'][0]
        assert 'source_cluster' in sl
        assert 'target_cluster' in sl
        assert 'flow_type' in sl
        assert 'p0' in sl and 'p1' in sl


def test_frontier_v2_mid_episode_checkpoint_replay_invariance():
    """Verify bit-for-bit identical actions after mid-episode state restore."""
    cfg = SynapseConfig(
        total_neurons=192,
        seed=42,
        thermodynamics=True,
        metaplasticity_bcm=True,
        transfer_entropy=True,
    )
    agent = IntegratedAgent(cfg)
    obs = torch.randn(155)

    # Run steps to populate state
    for _ in range(8):
        agent.act(obs, learn=False)

    saved_state = agent.export_state()

    # Step baseline for 8 steps
    baseline = []
    for _ in range(8):
        res = agent.act(obs, learn=False)
        baseline.append(res['action'])

    # Restore in new agent
    restored_agent = IntegratedAgent(cfg)
    restored_agent.import_state(saved_state)

    restored = []
    for _ in range(8):
        res = restored_agent.act(obs, learn=False)
        restored.append(res['action'])

    diffs = [np.max(np.abs(a - b)) for a, b in zip(baseline, restored)]
    assert max(diffs) == 0.0, f"Replay divergence detected: {max(diffs)}"


def test_frontier_v2_tandem_assay():
    from synapse.runtime import LabRuntime
    cfg = SynapseConfig(total_neurons=192, seed=42)
    rt = LabRuntime(cfg)
    res = run_frontier_v2_innovations_assay(rt, seed=42)
    assert res['status'] == 'completed'
    assert 'thermodynamics' in res
    assert 'metaplasticity_bcm' in res
    assert 'transfer_entropy' in res
