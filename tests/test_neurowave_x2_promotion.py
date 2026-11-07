import pytest

from synapse.neurowave_x2.promotion import FailClosedPromotionGate, PromotionDecision


def test_promotion_gate_insufficient_seeds_fail_closed():
    gate = FailClosedPromotionGate(min_independent_seeds=3)
    dec, reason, audit = gate.evaluate_candidate(
        candidate_returns=[10.0, 12.0],  # only 2 seeds
        baseline_returns=[5.0, 6.0],
        retention_baseline_return=5.0,
        retention_candidate_return=5.0,
        replay_verified=True,
        measured_wire_bits=1000,
        measured_flops=10000,
    )
    assert dec == PromotionDecision.INCONCLUSIVE
    assert "Insufficient independent seeds" in reason


def test_promotion_gate_replay_failed():
    gate = FailClosedPromotionGate(min_independent_seeds=2)
    dec, reason, audit = gate.evaluate_candidate(
        candidate_returns=[10.0, 12.0],
        baseline_returns=[5.0, 6.0],
        retention_baseline_return=5.0,
        retention_candidate_return=5.0,
        replay_verified=False,  # Replay failure
        measured_wire_bits=1000,
        measured_flops=10000,
    )
    assert dec == PromotionDecision.REJECTED
    assert "replay verification failed" in reason.lower()


def test_promotion_gate_budget_exceeded():
    gate = FailClosedPromotionGate(min_independent_seeds=2, max_wire_bits=500)
    dec, reason, audit = gate.evaluate_candidate(
        candidate_returns=[10.0, 12.0],
        baseline_returns=[5.0, 6.0],
        retention_baseline_return=5.0,
        retention_candidate_return=5.0,
        replay_verified=True,
        measured_wire_bits=1000,  # Exceeds 500
        measured_flops=10000,
    )
    assert dec == PromotionDecision.REJECTED
    assert "bandwidth budget exceeded" in reason.lower()


def test_promotion_gate_retention_loss():
    gate = FailClosedPromotionGate(min_independent_seeds=2, max_allowed_retention_loss=0.2)
    dec, reason, audit = gate.evaluate_candidate(
        candidate_returns=[10.0, 12.0],
        baseline_returns=[5.0, 6.0],
        retention_baseline_return=5.0,
        retention_candidate_return=3.0,  # 2.0 loss > 0.2
        replay_verified=True,
        measured_wire_bits=100,
        measured_flops=10000,
    )
    assert dec == PromotionDecision.REJECTED
    assert "catastrophic forgetting" in reason.lower()


def test_promotion_gate_success():
    gate = FailClosedPromotionGate(
        min_independent_seeds=2,
        min_return_improvement=2.0,
        max_allowed_retention_loss=0.5,
    )
    dec, reason, audit = gate.evaluate_candidate(
        candidate_returns=[12.0, 14.0],  # mean 13.0
        baseline_returns=[8.0, 8.0],     # mean 8.0 (delta +5.0)
        retention_baseline_return=5.0,
        retention_candidate_return=5.0,
        replay_verified=True,
        measured_wire_bits=100,
        measured_flops=10000,
    )
    assert dec == PromotionDecision.PROMOTED
    assert audit["delta_return"] == 5.0
