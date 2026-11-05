import copy
import hashlib
import json

import pytest
import torch

from poseidon.research import prediction_gate
from poseidon.research.candidate_evaluation import EVALUATION_SCHEMA, evaluate_oracle
from poseidon.research.contracts import canonical_bytes
from poseidon.research.oracle_training import FIT_RECEIPT_SCHEMA, fit_oracle


def _write(path, receipt):
    content = copy.deepcopy(receipt)
    content.pop("receipt_sha256", None)
    content["receipt_sha256"] = hashlib.sha256(canonical_bytes(content)).hexdigest()
    path.write_text(json.dumps(content), encoding="utf-8")
    return content


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    artifact = tmp_path / "oracle.pt"
    artifact.write_bytes(b"unit-test artifact; verifier is explicitly mocked")
    metrics = {
        "candidate_state_delta_mse": 0.1,
        "linear_action_delta_mse": 0.3,
        "paired_state_mse_gain_over_linear": 0.2,
        "paired_state_mse_gain_over_linear_95pct_bootstrap_interval": [0.1, 0.3],
        "candidate_reward_mae": 0.1,
        "action_mean_reward_mae": 0.2,
        "paired_reward_mae_gain_over_action_mean": 0.1,
        "paired_reward_mae_gain_95pct_bootstrap_interval": [0.0, 0.2],
    }
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model_sha256": "a" * 64,
        "promoted": False,
        "seed_disjointness_verified": True,
        "results": {condition: {"episodes": 16, "metrics": copy.deepcopy(metrics)}
                    for condition in ("id_test", "ood_scarcity_test")},
    }
    path = tmp_path / "evaluation_receipt.json"
    receipt = _write(path, receipt)
    _write(tmp_path / "fit_receipt.json", {
        "schema": FIT_RECEIPT_SCHEMA, "artifact_sha256": receipt["artifact_sha256"],
        "model_sha256": receipt["model_sha256"], "config": {}, "training": {},
    })
    calls = []

    def verified(receipt_path, *, replay):
        calls.append((receipt_path, replay))
        return {"valid": True, "replayed": replay, "artifact_sha256": receipt["artifact_sha256"],
                "model_sha256": receipt["model_sha256"], "conditions": list(receipt["results"])}

    monkeypatch.setattr(prediction_gate, "verify_evaluation_receipt", verified)
    def fit_verified(receipt_path, *, replay_data):
        assert receipt_path == tmp_path / "fit_receipt.json"
        assert replay_data is True
        return {"valid": True, "replayed_data": True,
                "artifact_sha256": receipt["artifact_sha256"], "model_sha256": receipt["model_sha256"]}

    monkeypatch.setattr(prediction_gate, "verify_fit_receipt", fit_verified)
    return path, receipt, artifact, calls


def _assert_inactive(report):
    assert report["promoted"] is False
    assert report["control_evidence_required"] is True
    assert report["control_evidence_provided"] is False
    encoded = copy.deepcopy(report)
    digest = encoded.pop("report_sha256")
    assert hashlib.sha256(canonical_bytes(encoded)).hexdigest() == digest


def test_qualified_screen_replays_and_binds_exact_evidence_without_promotion(evidence):
    path, receipt, artifact, calls = evidence
    before = {file: file.read_bytes() for file in (path, artifact)}
    report = prediction_gate.assess_prediction_gate(path)
    assert calls == [(path.resolve(), True)]
    assert report["state"] == "prediction_qualified"
    assert report["verification"] == {"valid": True, "replayed": True}
    assert report["fit_verification"] == {"valid": True, "replayed_data": True}
    assert report["reasons"] == []
    assert report["evidence"]["evaluation_file_sha256"] == hashlib.sha256(before[path]).hexdigest()
    assert report["evidence"]["evaluation_receipt_sha256"] == receipt["receipt_sha256"]
    assert report["evidence"]["artifact_sha256"] == receipt["artifact_sha256"]
    assert report["evidence"]["model_sha256"] == receipt["model_sha256"]
    assert report["evidence"]["fit_file_sha256"] == hashlib.sha256((path.parent / "fit_receipt.json").read_bytes()).hexdigest()
    assert {file: file.read_bytes() for file in (path, artifact)} == before
    _assert_inactive(report)


@pytest.mark.parametrize("condition", ["id_test", "ood_scarcity_test"])
@pytest.mark.parametrize("field,lower,qualified", [
    ("paired_state_mse_gain_over_linear_95pct_bootstrap_interval", 0.0, False),
    ("paired_state_mse_gain_over_linear_95pct_bootstrap_interval", -1e-300, False),
    ("paired_state_mse_gain_over_linear_95pct_bootstrap_interval", 1e-300, True),
    ("paired_reward_mae_gain_95pct_bootstrap_interval", 0.0, True),
    ("paired_reward_mae_gain_95pct_bootstrap_interval", -1e-300, False),
])
def test_each_condition_requires_exact_lower_bound_with_no_tolerance(evidence, condition, field, lower, qualified):
    path, receipt, _, _ = evidence
    receipt["results"][condition]["metrics"][field][0] = lower
    _write(path, receipt)
    report = prediction_gate.assess_prediction_gate(path)
    assert (report["state"] == "prediction_qualified") is qualified
    assert report["conditions"][condition]["passed"] is qualified
    if not qualified:
        assert any(condition in reason for reason in report["reasons"])
    _assert_inactive(report)


@pytest.mark.parametrize("mutation", [
    "missing_condition", "extra_condition", "missing_metric", "missing_metrics", "missing_linear_baseline",
    "boolean_metric", "string_metric", "negative_error", "empty_interval", "long_interval",
    "reversed_interval", "boolean_interval", "missing_episodes", "boolean_episodes", "zero_episodes",
    "promoted", "not_disjoint", "invalid_artifact_hash", "invalid_model_hash", "wrong_schema",
])
def test_malformed_or_missing_evidence_fails_closed(evidence, mutation):
    path, receipt, _, _ = evidence
    result = receipt["results"]["id_test"]
    metrics = result["metrics"]
    interval_key = "paired_state_mse_gain_over_linear_95pct_bootstrap_interval"
    if mutation == "missing_condition":
        del receipt["results"]["ood_scarcity_test"]
    elif mutation == "extra_condition":
        receipt["results"]["other"] = copy.deepcopy(result)
    elif mutation == "missing_metric":
        del metrics[interval_key]
    elif mutation == "missing_metrics":
        del result["metrics"]
    elif mutation == "missing_linear_baseline":
        del metrics["linear_action_delta_mse"]
    elif mutation == "boolean_metric":
        metrics["candidate_state_delta_mse"] = True
    elif mutation == "string_metric":
        metrics["candidate_state_delta_mse"] = "0.1"
    elif mutation == "negative_error":
        metrics["candidate_state_delta_mse"] = -0.1
    elif mutation == "empty_interval":
        metrics[interval_key] = []
    elif mutation == "long_interval":
        metrics[interval_key] = [0.1, 0.2, 0.3]
    elif mutation == "reversed_interval":
        metrics[interval_key] = [0.3, 0.1]
    elif mutation == "boolean_interval":
        metrics[interval_key] = [True, 1.0]
    elif mutation == "missing_episodes":
        del result["episodes"]
    elif mutation == "boolean_episodes":
        result["episodes"] = True
    elif mutation == "zero_episodes":
        result["episodes"] = 0
    elif mutation == "promoted":
        receipt["promoted"] = True
    elif mutation == "not_disjoint":
        receipt["seed_disjointness_verified"] = False
    elif mutation == "invalid_artifact_hash":
        receipt["artifact_sha256"] = "bad"
    elif mutation == "invalid_model_hash":
        receipt["model_sha256"] = "bad"
    elif mutation == "wrong_schema":
        receipt["schema"] = "unverified-schema"
    _write(path, receipt)
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert report["reasons"]
    _assert_inactive(report)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_nonfinite_json_rejected_without_sanitization(evidence, value):
    path, _, _, calls = evidence
    text = path.read_text(encoding="utf-8").replace('"candidate_state_delta_mse": 0.1',
                                                    f'"candidate_state_delta_mse": {value}')
    path.write_text(text, encoding="utf-8")
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert calls == []
    _assert_inactive(report)


def test_duplicate_json_fields_rejected(evidence):
    path, _, _, _ = evidence
    path.write_text(path.read_text(encoding="utf-8").replace('"promoted": false', '"promoted": false, "promoted": true'),
                    encoding="utf-8")
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert "duplicate JSON field" in report["reasons"][0]


@pytest.mark.parametrize("failure", ["raises", "invalid", "no_replay", "artifact_identity", "model_identity", "changed_receipt", "changed_artifact"])
def test_failed_or_unbound_replay_cannot_qualify(evidence, monkeypatch, failure):
    path, receipt, artifact, _ = evidence

    def verifier(receipt_path, *, replay):
        assert replay is True
        if failure == "raises":
            raise RuntimeError("replay mismatch")
        if failure == "changed_receipt":
            receipt["results"]["id_test"]["metrics"]["candidate_reward_mae"] = 0.0
            _write(path, receipt)
        if failure == "changed_artifact":
            artifact.write_bytes(b"different artifact")
        return {"valid": failure != "invalid", "replayed": failure != "no_replay",
                "artifact_sha256": "b" * 64 if failure == "artifact_identity" else receipt["artifact_sha256"],
                "model_sha256": "b" * 64 if failure == "model_identity" else receipt["model_sha256"],
                "conditions": list(receipt["results"])}

    monkeypatch.setattr(prediction_gate, "verify_evaluation_receipt", verifier)
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert report["verification"]["valid"] is False
    assert report["reasons"]
    _assert_inactive(report)


def test_missing_receipt_fails_closed(tmp_path):
    report = prediction_gate.assess_prediction_gate(tmp_path / "absent.json")
    assert report["state"] == "rejected"
    _assert_inactive(report)


@pytest.mark.parametrize("failure", ["missing", "raises", "invalid", "no_replay", "identity", "changed"])
def test_fit_provenance_requires_bound_stable_replay(evidence, monkeypatch, failure):
    path, receipt, _, _ = evidence
    fit_path = path.parent / "fit_receipt.json"
    if failure == "missing":
        fit_path.unlink()

    def fit_verifier(receipt_path, *, replay_data):
        assert replay_data is True
        if failure == "raises":
            raise ValueError("replayed linear baseline mismatch")
        if failure == "changed":
            fit = json.loads(fit_path.read_text(encoding="utf-8"))
            fit["training"]["changed"] = True
            _write(fit_path, fit)
        return {"valid": failure != "invalid", "replayed_data": failure != "no_replay",
                "artifact_sha256": "b" * 64 if failure == "identity" else receipt["artifact_sha256"],
                "model_sha256": receipt["model_sha256"]}

    monkeypatch.setattr(prediction_gate, "verify_fit_receipt", fit_verifier)
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert report["fit_verification"]["valid"] is False
    assert report["reasons"]
    _assert_inactive(report)


def test_zero_residual_scale_cannot_qualify_through_rounding_gain(evidence):
    path, _, _, _ = evidence
    fit_path = path.parent / "fit_receipt.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["config"]["architecture"] = "linear-residual-v1"
    fit["training"]["residual_selection"] = {"selected_scale": 0.0}
    _write(fit_path, fit)
    report = prediction_gate.assess_prediction_gate(path)
    assert all(condition["passed"] for condition in report["conditions"].values())
    assert report["state"] == "rejected"
    assert report["residual_selection"] == {"selected_scale": 0.0}
    assert any("linear_anchor_retained" in reason for reason in report["reasons"])


def test_real_tiny_candidate_evaluation_is_replayed_and_mutation_rejected(tmp_path):
    fit = fit_oracle(tmp_path / "fit", base_seed=685_000_000, train_episodes=4,
                     selection_episodes=2, calibration_episodes=4, max_steps=8,
                     training_updates=1, batch_size=8, hidden_size=16, ensemble_size=2,
                     coverage=0.8)
    receipt = evaluate_oracle(fit["artifact_path"], tmp_path / "evaluation", base_seed=685_000_000,
                              episodes_per_condition=2, bootstrap_reps=100)
    path = tmp_path / "evaluation" / "evaluation_receipt.json"
    report = prediction_gate.assess_prediction_gate(path)
    assert report["verification"] == {"valid": True, "replayed": True}
    expected = all(
        result["metrics"]["paired_state_mse_gain_over_linear_95pct_bootstrap_interval"][0] > 0
        and result["metrics"]["paired_reward_mae_gain_95pct_bootstrap_interval"][0] >= 0
        for result in receipt["results"].values()
    )
    assert report["state"] == ("prediction_qualified" if expected else "rejected")
    _assert_inactive(report)
    receipt["results"]["id_test"]["metrics"]["paired_state_mse_gain_over_linear_95pct_bootstrap_interval"] = [10.0, 20.0]
    _write(path, receipt)
    forged = prediction_gate.assess_prediction_gate(path)
    assert forged["state"] == "rejected"
    assert forged["verification"]["valid"] is False
    assert "do not replay" in forged["reasons"][0]


def test_resealed_easier_linear_baseline_is_rejected_by_training_replay(tmp_path):
    fit = fit_oracle(tmp_path / "fit", base_seed=686_000_000, train_episodes=4,
                     selection_episodes=2, calibration_episodes=4, max_steps=8,
                     training_updates=1, batch_size=8, hidden_size=16, ensemble_size=2,
                     coverage=0.8)
    artifact_path = tmp_path / "fit" / "oracle.pt"
    artifact = torch.load(artifact_path, weights_only=True)
    # MLP predictions stay exactly the same, while the forged baseline is made
    # arbitrarily poor. Evaluation replay alone accepts the resealed benchmark.
    for row in artifact["training"]["linear_delta_baseline"]["coefficients"]:
        for index in range(len(row)):
            row[index] = 100.0
    torch.save(artifact, artifact_path)
    fit["training"] = artifact["training"]
    fit["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _write(tmp_path / "fit" / "fit_receipt.json", fit)
    evaluation = evaluate_oracle(artifact_path, tmp_path / "evaluation", base_seed=686_000_000,
                                 episodes_per_condition=2, bootstrap_reps=100)
    assert evaluation["results"]["id_test"]["metrics"]["paired_state_mse_gain_over_linear_95pct_bootstrap_interval"][0] > 0
    path = tmp_path / "evaluation" / "evaluation_receipt.json"
    assert prediction_gate.verify_evaluation_receipt(path, replay=True)["valid"]
    report = prediction_gate.assess_prediction_gate(path)
    assert report["state"] == "rejected"
    assert report["fit_verification"]["valid"] is False
    assert "replayed linear baseline" in report["reasons"][0]
