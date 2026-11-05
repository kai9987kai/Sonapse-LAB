import json
from pathlib import Path
import pytest
import torch

from poseidon.adapt import (
    AdaptConfig,
    AnchorDataset,
    ContinualAdapter,
    ReplayBuffer,
    ResidualPolicyAdapter,
    Transition,
)
from poseidon.core import CoreRuntime, active_core_path
from poseidon.information_seeking import (
    CandidateAdapterSession,
    EpistemicInformationProber,
    ProbeConfig,
    RetentionAndAdaptationAuditor,
)
from poseidon.world import N_ACTIONS, OBS_SIZE, TidePool


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_epistemic_probe_execution(core):
    prober = EpistemicInformationProber(core, config=ProbeConfig(budget_steps=8))
    env = TidePool(seed=777, max_steps=8)

    outcome_targeted = prober.run_probe(env, probe_type="targeted")
    assert outcome_targeted.steps_executed > 0
    assert outcome_targeted.probe_type == "targeted"
    assert outcome_targeted.state_coverage_volume >= 0.0
    assert outcome_targeted.action_entropy >= 0.0
    assert len(outcome_targeted.executed_actions) == outcome_targeted.steps_executed

    env2 = TidePool(seed=777, max_steps=8)
    outcome_random = prober.run_probe(env2, probe_type="random")
    assert outcome_random.steps_executed > 0
    assert outcome_random.probe_type == "random"


def test_compare_probes_paired_benchmark(core):
    prober = EpistemicInformationProber(core, config=ProbeConfig(budget_steps=8, eval_seeds=(801, 802)))
    res = prober.compare_probes()

    assert res["num_seeds"] == 2
    assert "targeted" in res
    assert "random" in res
    assert "comparison" in res
    assert "delta_reward" in res["comparison"]
    assert "delta_uncertainty_reduction" in res["comparison"]
    assert "targeted_advantage_supported" in res["comparison"]


def test_candidate_adapter_session_isolation_and_rollback(core):
    cadapter = ContinualAdapter(core, config=AdaptConfig(anchor_capacity=20))
    initial_weights = copy_weights = {
        name: param.clone() for name, param in cadapter.adapter.named_parameters()
    }

    session = CandidateAdapterSession(cadapter)

    # Synthetic transitions
    transitions = [
        Transition(
            obs=[0.5] * OBS_SIZE,
            action=i % N_ACTIONS,
            reward=1.0,
            next_obs=[0.5] * OBS_SIZE,
        )
        for i in range(8)
    ]

    # Train batch in candidate session
    train_res = session.train_batch(transitions, steps=3, lr=1e-2)
    assert train_res["steps"] == 3

    # Incumbent adapter in parent must be completely unchanged
    for name, param in cadapter.adapter.named_parameters():
        assert torch.allclose(param, initial_weights[name], atol=1e-7)

    # Cancel session
    session.cancel()
    assert session.cancelled is True
    # Committing cancelled session must return False and leave incumbent untouched
    admitted = session.commit_if_audited({"status": "ADMITTED"})
    assert admitted is False
    for name, param in cadapter.adapter.named_parameters():
        assert torch.allclose(param, initial_weights[name], atol=1e-7)


def test_retention_and_adaptation_auditor(core, tmp_path):
    anchors = AnchorDataset(capacity=20)
    anchors.populate_from_core(core, count=10)

    auditor = RetentionAndAdaptationAuditor(
        core=core,
        anchors=anchors,
        eval_seeds=(901, 902),
        min_return_delta=0.0,
        max_anchor_regression=0.05,
    )

    candidate = ResidualPolicyAdapter()
    incumbent = ResidualPolicyAdapter()

    # Zero-init adapter has 100% retention and 0 delta
    receipt = auditor.audit_candidate(
        candidate_adapter=candidate,
        incumbent_adapter=incumbent,
        receipt_dir=tmp_path,
    )

    assert receipt["status"] == "ADMITTED"
    assert receipt["anchor_retention"] == 1.0
    assert receipt["anchor_regression"] == 0.0
    assert "receipt_digest" in receipt
    assert "candidate_sha256" in receipt

    # Saved receipt file check
    saved_files = list(tmp_path.glob("audit_*.json"))
    assert len(saved_files) == 1
    content = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert content["receipt_digest"] == receipt["receipt_digest"]


def test_retention_auditor_rejects_regression(core):
    anchors = AnchorDataset(capacity=20)
    anchors.populate_from_core(core, count=10)

    auditor = RetentionAndAdaptationAuditor(
        core=core,
        anchors=anchors,
        eval_seeds=(901, 902),
        min_return_delta=0.5, # Impossible return delta to force rejection
        max_anchor_regression=0.01,
    )

    candidate = ResidualPolicyAdapter()
    incumbent = ResidualPolicyAdapter()

    receipt = auditor.audit_candidate(candidate_adapter=candidate, incumbent_adapter=incumbent)
    assert receipt["status"] == "REJECTED"
    assert any("insufficient_return_delta" in r for r in receipt["reasons"])
