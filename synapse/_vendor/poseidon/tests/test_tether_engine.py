"""Tests for Project TETHER theoretical and computational engines."""

import pytest
from poseidon.tether_engine import (
    TetherContinuousReliabilityEngine,
    CounterfactualCovariationInvarianceEngine,
    CausalAttributionSaliencyEngine,
    ZeroShotCorruptionDetector,
    EvidenceCalibratedFallbackGate,
    TetherEngine,
)


def test_tether_continuous_reliability_engine():
    engine = TetherContinuousReliabilityEngine(temperature=1.0)
    # 5 ensemble members predicting 4-dimensional state delta
    preds = [
        [0.10, 0.20, 0.05, -0.10],
        [0.11, 0.19, 0.06, -0.09],
        [0.09, 0.21, 0.04, -0.11],
        [0.10, 0.20, 0.05, -0.10],
        [0.85, -0.60, 0.90, 0.70],  # Outlier / corrupted member
    ]
    rewards = [1.0, 1.1, 0.9, 1.0, -2.5]
    res = engine.evaluate_reliability(preds, rewards=rewards)
    assert res["verified"] is True
    assert len(res["weights"]) == 5
    assert abs(sum(res["weights"]) - 1.0) < 1e-6
    # Outlier member (index 4) should have lowest weight
    assert res["weights"][4] < res["weights"][0]
    assert res["weights"][4] < res["weights"][1]
    assert 0.0 <= res["epistemic_trust"] <= 1.0


def test_counterfactual_covariation_invariance_engine():
    engine = CounterfactualCovariationInvarianceEngine(effect_weight=1.0, invariance_weight=1.0)
    fact_pred = [0.1, 0.2]
    fact_tgt = [0.1, 0.2]
    rel_pred = [0.6, 0.7]
    rel_tgt = [0.6, 0.7]
    irrel_pred = [0.101, 0.201]
    dmg_pred = [0.102, 0.202]

    # Test all 7 arms
    for arm in CounterfactualCovariationInvarianceEngine.ARMS:
        res = engine.evaluate_four_views(
            factual_pred=fact_pred,
            factual_target=fact_tgt,
            relevant_pred=rel_pred,
            relevant_target=rel_tgt,
            irrelevant_pred=irrel_pred,
            damaged_pred=dmg_pred,
            corrupted_weights=[0.01],
            arm=arm,
        )
        assert res["verified"] is True
        assert res["loss"] >= 0.0


def test_causal_attribution_saliency_engine():
    engine = CausalAttributionSaliencyEngine()
    preds = [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.5, 0.8],
    ]
    weights = [0.45, 0.45, 0.10]
    res = engine.compute_member_saliency(preds, weights)
    assert res["verified"] is True
    assert len(res["member_percentages"]) == 3
    assert abs(sum(res["member_percentages"]) - 100.0) < 1e-5


def test_zero_shot_corruption_detector():
    detector = ZeroShotCorruptionDetector(z_score_threshold=1.5)
    # 4 normal members close to [0.1, 0.2], 1 outlier far away
    preds = [
        [0.10, 0.20],
        [0.11, 0.21],
        [0.09, 0.19],
        [0.10, 0.20],
        [2.50, 3.80],  # Outlier
    ]
    res = detector.detect_corrupted_members(preds)
    assert res["verified"] is True
    assert 4 in res["corrupted_indices"]
    assert len(res["corrupted_indices"]) == 1
    assert res["clean_consensus_preserved"] is True


def test_evidence_calibrated_fallback_gate():
    gate = EvidenceCalibratedFallbackGate(
        epistemic_threshold=0.35,
        margin_threshold=0.15,
        max_admissible_risk=0.40,
    )
    # Case 1: Low trust -> epistemic abstention
    res1 = gate.evaluate_override(
        base_action=0,
        candidate_action=1,
        candidate_margin=0.30,
        epistemic_trust=0.20,
        calibrated_risk=0.10,
    )
    assert res1["admit_override"] is False
    assert res1["selected_action"] == 0
    assert res1["decision_type"] == "epistemic_abstain_fallback"

    # Case 2: High trust, sufficient margin, low risk -> override admitted
    res2 = gate.evaluate_override(
        base_action=0,
        candidate_action=1,
        candidate_margin=0.30,
        epistemic_trust=0.85,
        calibrated_risk=0.10,
    )
    assert res2["admit_override"] is True
    assert res2["selected_action"] == 1
    assert res2["decision_type"] == "qualified_override_admitted"


def test_tether_unified_synthesis():
    engine = TetherEngine()
    preds = [
        [0.1, 0.2, -0.1],
        [0.12, 0.18, -0.09],
        [0.08, 0.22, -0.11],
        [0.11, 0.20, -0.10],
        [0.10, 0.19, -0.10],
    ]
    rewards = [1.0, 1.1, 0.9, 1.05, 0.95]
    res = engine.synthesize(
        member_predictions=preds,
        member_rewards=rewards,
        base_action=0,
        candidate_action=1,
        candidate_margin=0.20,
    )
    assert res["schema"] == TetherEngine.SCHEMA
    assert res["status"] == "completed"
    assert res["verified"] is True
    assert "receipt_sha256" in res
    assert len(res["receipt_sha256"]) == 64
