"""Tests for Project FAULTLINE Engine."""

from fractions import Fraction
import pytest

from poseidon.faultline_engine import (
    ContinuousJointErrorEngine,
    FaultlineDiscreteEngine,
    FaultlineEngine,
    ParityBenchmark,
    POLICIES,
    analyze,
)


def test_faultline_dataset_validation():
    # Valid canonical dataset
    data = FaultlineEngine._create_canonical_dataset(seed=42)
    validated = FaultlineDiscreteEngine.validate_dataset(data)
    assert validated["schema"] == "faultline-dataset-v1"
    assert len(validated["components"]) == 5
    assert len(validated["calibration"]) > 0
    assert len(validated["test"]) > 0

    # Missing schema raises
    bad_data = dict(data)
    bad_data["schema"] = "wrong-schema"
    with pytest.raises(ValueError, match="Unknown or missing schema"):
        FaultlineDiscreteEngine.validate_dataset(bad_data)


def test_faultline_discrete_panel_selection_and_analysis():
    data = FaultlineEngine._create_canonical_dataset(seed=42)
    res = analyze(data, size=3)

    assert res["dataset"] == "poseidon-oracle-5m-tidepool-benchmark"
    assert res["panel_size"] == 3
    assert res["candidate_panels"] == 10  # 5 choose 3

    policies = {p["id"]: p for p in res["policies"]}
    assert set(policies.keys()) == set(POLICIES)

    for pid in POLICIES:
        p = policies[pid]
        assert len(p["panel"]) == 3
        assert len(p["panel_indices"]) == 3
        assert 0.0 <= p["calibration"]["error"] <= 1.0
        assert 0.0 <= p["test"]["worst_group_error"] <= 1.0

    # In canonical dataset, joint policy should achieve lower or equal worst_group_error on test
    joint_worst = policies["joint"]["test"]["worst_group_error"]
    acc_worst = policies["accuracy"]["test"]["worst_group_error"]
    assert joint_worst <= acc_worst


def test_continuous_joint_error_covariance():
    engine = ContinuousJointErrorEngine()
    # 5 models, 64 error samples
    errors = FaultlineEngine._simulate_tidepool_model_errors(N=64, K=5, seed=123)
    cov_res = engine.compute_joint_error_covariance(errors)

    assert cov_res["num_models"] == 5
    assert cov_res["num_samples"] == 64
    assert len(cov_res["means"]) == 5
    assert len(cov_res["covariance_matrix"]) == 5
    assert cov_res["ensemble_variance"] > 0.0
    assert cov_res["diversity_gain"] >= 0.0
    assert cov_res["verified"] is True


def test_continuous_minimax_panel_selection():
    engine = ContinuousJointErrorEngine()
    cond_errors = FaultlineEngine._simulate_conditioned_model_errors(N_per_cond=32, M=5, seed=42)
    res = engine.select_minimax_continuous_panel(cond_errors, size=3)

    assert len(res["selected_panel"]) == 3
    assert res["candidate_panels_evaluated"] == 10
    assert res["worst_condition_error"] > 0.0
    assert res["pooled_error"] > 0.0
    assert res["verified"] is True


def test_parity_benchmark_higher_order_hyperedges():
    # Even parity: unanimous or 2-of-3 co-failures
    even_case = ParityBenchmark.generate_parity_case(parity_type="even")
    assert even_case["pairwise_cofailure"] == 0.25
    assert even_case["majority_error_rate"] == 0.75
    assert even_case["parity_type"] == "even"

    # Odd parity: pairwise co-failure is identical (0.25) but majority error rate is 0.25!
    odd_case = ParityBenchmark.generate_parity_case(parity_type="odd")
    assert odd_case["pairwise_cofailure"] == 0.25
    assert odd_case["majority_error_rate"] == 0.25
    assert odd_case["parity_type"] == "odd"

    # Verifies the mathematical theorem: pairwise bounds fail to distinguish 25% vs 75% majority error
    assert even_case["pairwise_cofailure"] == odd_case["pairwise_cofailure"]
    assert even_case["majority_error_rate"] != odd_case["majority_error_rate"]


def test_faultline_engine_synthesize_and_receipt():
    engine = FaultlineEngine()
    res = engine.synthesize(seed=42, panel_size=3)

    assert res["schema"] == "poseidon-faultline-synthesis-v1"
    assert res["project"] == "FAULTLINE"
    assert res["verified"] is True
    assert "sha256" in res
    assert len(res["sha256"]) == 64
    assert "discrete_analysis" in res
    assert "continuous_error_covariance" in res
    assert "continuous_minimax_panel" in res
    assert "higher_order_parity_proof" in res
    assert res["joint_advantage"] >= 0.0
