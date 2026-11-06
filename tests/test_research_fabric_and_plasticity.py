"""Unit tests for Research Fabric Contracts and Plasticity Monitor."""
import pytest
import numpy as np
from synapse.research_fabric import (
    FabricObservation,
    FabricAction,
    FabricTransition,
    EvaluationReceipt,
    PoseidonBridgeAdapter,
    NeuroGraftAdapter,
    FaultlineStressAdapter
)
from synapse.plasticity_monitor import PlasticityMonitor, PlasticityReport


def test_fabric_data_contracts():
    obs_arr = np.random.randn(155).astype(np.float32)
    obs = FabricObservation.from_numpy(obs_arr, step=10)
    assert obs.dim == 155
    assert np.allclose(obs.to_numpy(), obs_arr)

    act_arr = np.array([2.5, -0.5, 0.0, 1.2, -3.0], dtype=np.float32)
    act = FabricAction.from_numpy(act_arr)
    # Action should be clipped to [-1, 1]
    assert np.all(np.abs(act.to_numpy()) <= 1.0)

    trans = FabricTransition(
        obs=obs,
        action=act,
        reward=1.0,
        next_obs=obs,
        done=False
    )
    d = trans.to_dict()
    assert d['reward'] == 1.0
    assert not d['done']


def test_evaluation_receipt_tamper_detection():
    receipt = EvaluationReceipt.create(
        benchmark_name="flagship_continual",
        task_sequence=["forage", "maze"],
        score_matrix=[[1.0, 0.0], [0.8, 1.2]],
        retention=0.95,
        forgetting=0.05,
        adaptation_speed=1.0,
        base_seed=42
    )
    assert receipt.verify()

    # Tamper with retention rate
    receipt.final_retention_rate = 0.99
    assert not receipt.verify()


def test_neurograft_dales_law_audit():
    # Compliant: rows are either all non-negative or all non-positive
    w_compliant = np.array([
        [0.5, 0.2, 0.1],   # excitatory
        [-0.4, -0.1, 0.0], # inhibitory
        [0.0, 0.0, 0.0]    # neutral
    ])
    audit1 = NeuroGraftAdapter.audit_dales_law(w_compliant, ["ach", "gaba", "glut"])
    assert audit1['compliant']

    # Non-compliant: row with mixed signs
    w_violation = np.array([
        [0.5, -0.2, 0.1]
    ])
    audit2 = NeuroGraftAdapter.audit_dales_law(w_violation, ["mixed"])
    assert not audit2['compliant']
    assert audit2['violating_count'] == 1


def test_faultline_stress_adapter():
    adapter = FaultlineStressAdapter(perturbation_type="sensor_dropout", severity=0.5, seed=42)
    obs = FabricObservation.from_numpy(np.ones(100, dtype=np.float32), step=1)
    p_obs = adapter.perturb_observation(obs)
    arr = p_obs.to_numpy()
    # About half should be zeroed
    zeros = np.sum(arr == 0.0)
    assert 30 <= zeros <= 70


def test_plasticity_monitor_erank_and_collapse_detection():
    monitor = PlasticityMonitor(num_features=32, buffer_size=60, collapse_threshold=0.20)
    rng = np.random.default_rng(42)

    # 1. Healthy isotropic representations
    for _ in range(40):
        feat = rng.normal(0.0, 1.0, size=32).astype(np.float32)
        monitor.record_activity(feat)

    rep1 = monitor.evaluate_plasticity()
    assert isinstance(rep1, PlasticityReport)
    assert not rep1.is_collapsed
    assert rep1.normalized_erank > 0.40
    assert rep1.recommended_action == 'HEALTHY'

    # 2. Representational collapse: all features collapse to rank-1 along 1st dimension
    for _ in range(70):
        feat = np.zeros(32, dtype=np.float32)
        feat[0] = float(rng.normal(1.0, 0.1))
        monitor.record_activity(feat)

    rep2 = monitor.evaluate_plasticity()
    assert rep2.is_collapsed
    assert rep2.normalized_erank < 0.20
    assert rep2.dead_neuron_ratio > 0.50
    assert rep2.recommended_action in ('SCHEDULE_CBP_RESET', 'TRIGGER_NEUROGENESIS')
    assert monitor.total_collapse_alarms > 0
