"""PROJECT NEXUS-Ω: Test Suite for Core Interfaces, Causal Memory, and Auditor.
"""
import numpy as np
import pytest

from poseidon.nexus_omega.contracts import (
    StructuredObservation,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
    HypothesisSpec,
    AuditReceipt,
)
from poseidon.nexus_omega.memory import CausalMemoryFabric
from poseidon.nexus_omega.world_model import EnsembleWorldModel
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.auditor import IndependentScientificAuditor


def test_contracts_and_hashing():
    obs = StructuredObservation(
        source_id="sensor_tidepool",
        step=1,
        data=np.array([1.0, 0.5, 0.2], dtype=np.float32),
        uncertainty=np.array([0.01, 0.01, 0.01], dtype=np.float32),
    )
    assert len(obs.provenance_hash) == 16

    receipt = AuditReceipt(
        audit_id="audit_test_01",
        hypothesis_id="hyp_01",
        timestamp="2026-10-11T14:00:00",
        seeds_evaluated=[101, 102],
        baseline_metrics={"reward": 2.5},
        candidate_metrics={"reward": 3.0},
        p_value=0.01,
        effect_size=1.2,
        retained_in_library=True,
        verdict="ACCEPTED",
    )
    digest = receipt.seal()
    assert len(digest) == 64
    assert receipt.receipt_hash == digest


def test_causal_memory_fabric_interventions():
    mem = CausalMemoryFabric(d_working=16, d_key=8, d_value=8)

    # 1. Test working write and episodic store
    mem.write_working(np.ones(16, dtype=np.float32) * 0.5)
    assert np.linalg.norm(mem.state.working_state) > 0.0

    k = np.random.randn(8).astype(np.float32)
    v = np.random.randn(8).astype(np.float32)
    mem.store_episode(k, v)
    assert len(mem.state.episodic_keys) == 1

    retrieved = mem.retrieve_episodic(k)
    assert retrieved.shape == (8,)

    # 2. Test Ablation Intervention
    mem.apply_intervention(InterventionSpec(InterventionType.ABLATE_CARRIER, MemoryCarrierType.WORKING))
    assert np.all(mem.state.working_state == 0.0)

    # 3. Test Reconstruction Intervention
    mem.apply_intervention(InterventionSpec(InterventionType.RECONSTRUCT_CARRIER, MemoryCarrierType.WORKING))
    assert np.linalg.norm(mem.state.working_state) > 0.0  # Successfully reconstructed from structural prior!

    # 4. Test Transplant Intervention
    donor = np.ones(16, dtype=np.float32) * 0.99
    mem.apply_intervention(InterventionSpec(
        InterventionType.TRANSPLANT_CARRIER,
        MemoryCarrierType.WORKING,
        donor_state=donor,
    ))
    assert np.allclose(mem.state.working_state, donor)


def test_ensemble_world_model_uncertainty():
    wm = EnsembleWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3)
    obs = np.random.randn(8).astype(np.float32)
    mem = np.random.randn(18).astype(np.float32)

    pred = wm.predict(obs, mem, action=1)
    assert pred.predicted_obs.shape == (8,)
    assert pred.epistemic_uncertainty >= 0.0
    assert pred.ensemble_variance.shape == (8,)

    # Comparator residual calculation
    real_next = obs + 0.1
    res = wm.compute_comparator_residual(real_next)
    assert res.shape == (8,)
    assert np.allclose(res, real_next - pred.predicted_obs)


def test_nexus_omega_agent_decision_loop():
    agent = NexusOmegaAgent(d_obs=8, d_working=16, n_actions=4)
    obs = np.ones(8, dtype=np.float32) * 0.5

    action = agent.act(obs)
    assert 0 <= action < 4

    next_obs = obs + 0.05
    e_t = agent.observe_transition(next_obs, reward=0.5, done=False)
    assert e_t.shape == (8,)
    # Working memory should have updated from comparator error
    assert np.linalg.norm(agent.memory.state.working_state) > 0.0


def test_four_fold_memory_distinction():
    auditor = IndependentScientificAuditor(max_steps=16)
    results = auditor.run_four_fold_memory_test(seed=42)

    assert "retention_reward" in results
    assert "ablation_reward" in results
    assert "reconstruction_reward" in results
    assert "mimicry_reactive_reward" in results
    assert results["is_distinguishable"] is True


def test_independent_scientific_auditor_falsification():
    auditor = IndependentScientificAuditor(seeds=[2001, 2002, 2003, 2004], max_steps=16)
    hypothesis = HypothesisSpec(
        hypothesis_id="hyp_causal_memory_01",
        statement="Intact causal memory significantly improves POMDP survival over ablated memory",
        null_hypothesis="There is no difference in reward between intact and ablated memory",
        independent_variable="Working memory presence (intact vs ablated)",
        dependent_variable="Episode total reward",
        alpha=0.05,
        minimum_effect_size=0.01,
    )

    receipt = auditor.run_causal_memory_assay(hypothesis)
    assert receipt.audit_id.startswith("audit_")
    assert receipt.receipt_hash != ""
    assert receipt.verdict in ["ACCEPTED", "FALSIFIED"]
    assert len(receipt.seeds_evaluated) == 4
