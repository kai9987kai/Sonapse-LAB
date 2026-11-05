"""Replay-backed screening of one-step predictions, never model promotion.

The four required intervals are descriptive, marginal 95% episode-bootstrap
intervals. They are not a simultaneous confidence region or a correction for
repeated candidate search. Use a frozen candidate and previously untouched
evaluation seeds; qualifying here still requires independent control evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .candidate_evaluation import EVALUATION_SCHEMA, verify_evaluation_receipt
from .contracts import canonical_bytes
from .oracle_model import LINEAR_RESIDUAL_ARCHITECTURE
from .oracle_training import FIT_RECEIPT_SCHEMA, verify_fit_receipt


PREDICTION_GATE_SCHEMA = "poseidon-prediction-gate-v1"
_CONDITIONS = ("id_test", "ood_scarcity_test")
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_CHECKS = {
    "state_gain_over_linear": {
        "candidate_metric": "candidate_state_delta_mse",
        "baseline_metric": "linear_action_delta_mse",
        "gain_metric": "paired_state_mse_gain_over_linear",
        "interval_metric": "paired_state_mse_gain_over_linear_95pct_bootstrap_interval",
        "requirement": "lower_bound > 0",
    },
    "reward_gain_over_action_mean": {
        "candidate_metric": "candidate_reward_mae",
        "baseline_metric": "action_mean_reward_mae",
        "gain_metric": "paired_reward_mae_gain_over_action_mean",
        "interval_metric": "paired_reward_mae_gain_95pct_bootstrap_interval",
        "requirement": "lower_bound >= 0",
    },
}


def _read_bounded(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(_MAX_RECEIPT_BYTES + 1)
    if len(data) > _MAX_RECEIPT_BYTES:
        raise ValueError("evaluation receipt exceeds 64 MiB")
    return data


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > 512 * 1024 * 1024:
                raise ValueError("ORACLE artifact exceeds 512 MiB")
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("metric must be a finite number, not a boolean or string")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("metric must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError("metric must be a finite number")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {label}")
    return value


def _decode_receipt(raw: bytes, schema: str, label: str) -> dict[str, Any]:
    receipt = json.loads(raw, parse_constant=_reject_constant,
                         parse_float=lambda value: _number(float(value)), object_pairs_hook=_object)
    if not isinstance(receipt, dict) or receipt.get("schema") != schema:
        raise ValueError(f"unsupported {label} schema")
    digest = _sha256(receipt.get("receipt_sha256"), f"{label} receipt SHA256")
    if hashlib.sha256(canonical_bytes({key: value for key, value in receipt.items() if key != "receipt_sha256"})).hexdigest() != digest:
        raise ValueError(f"{label} receipt digest mismatch")
    return receipt


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    report["report_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def _condition_checks(result: Any) -> dict[str, Any]:
    checks = {}
    for name, spec in _CHECKS.items():
        check = {"passed": False, "requirement": spec["requirement"],
                 "candidate_error": None, "baseline_error": None, "paired_gain": None,
                 "interval_95pct": None, "reason": None}
        try:
            if not isinstance(result, dict) or type(result.get("episodes")) is not int or result["episodes"] < 1:
                raise ValueError("missing or invalid episode evidence")
            metrics = result["metrics"]
            if not isinstance(metrics, dict):
                raise ValueError("metrics must be an object")
            candidate = _number(metrics[spec["candidate_metric"]])
            baseline = _number(metrics[spec["baseline_metric"]])
            gain = _number(metrics[spec["gain_metric"]])
            if min(candidate, baseline) < 0:
                raise ValueError("error metrics must be nonnegative")
            interval = metrics[spec["interval_metric"]]
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("paired 95% interval must have exactly two bounds")
            lower, upper = (_number(bound) for bound in interval)
            if lower > upper:
                raise ValueError("paired 95% interval bounds are reversed")
            passed = lower > 0 if name == "state_gain_over_linear" else lower >= 0
            check.update(candidate_error=candidate, baseline_error=baseline, paired_gain=gain,
                         interval_95pct=[lower, upper], passed=passed,
                         reason="requirement_met" if passed else "lower_bound_does_not_meet_requirement")
        except (KeyError, TypeError, ValueError) as exc:
            check["reason"] = f"invalid_evidence: {exc}"
        checks[name] = check
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def assess_prediction_gate(evaluation_receipt_path: str | Path) -> dict[str, Any]:
    """Return a fail-closed, hash-bound prediction screen for a replayed receipt.

    The function does not save, modify, activate or promote any artifact. The
    underlying verifier performs its replay in its own temporary directory.
    Missing, malformed, changed or non-replayable evidence returns ``rejected``.
    No threshold tolerance, survival score or calibration claim is added.
    """
    path = Path(evaluation_receipt_path).resolve()
    report: dict[str, Any] = {
        "schema": PREDICTION_GATE_SCHEMA,
        "state": "rejected",
        "promoted": False,
        "control_evidence_required": True,
        "control_evidence_provided": False,
        "evidence": {"evaluation_receipt_path": str(path), "evaluation_file_sha256": None,
                     "evaluation_receipt_sha256": None, "fit_receipt_path": None,
                     "fit_file_sha256": None, "fit_receipt_sha256": None,
                     "artifact_sha256": None, "model_sha256": None},
        "verification": {"valid": False, "replayed": False},
        "fit_verification": {"valid": False, "replayed_data": False},
        "residual_selection": None,
        "conditions": {name: _condition_checks(None) for name in _CONDITIONS},
        "reasons": [],
        "scope": "One-step prediction in TidePool ID and scarcity-shift episodes only.",
        "limits": [
            "Prediction qualification is not activation or promotion; separate paired control evidence and attribution checks are required.",
            "No survival improvement, death-probability calibration, planning benefit or structural transfer is established.",
            "The linear state and action-mean reward baselines are recomputed from training episodes by fit-receipt replay.",
            "These are four marginal 95% episode-bootstrap intervals, not a simultaneous confidence region; no multiple-comparison correction is applied.",
            "Repeated candidate search or reuse of test seeds can invalidate an acceptance interpretation; freeze the candidate and reserve untouched evaluation seeds.",
            "Intervals describe this finite episode sample and can be degenerate or unreliable at small sample sizes.",
        ],
    }
    try:
        before = _read_bounded(path)
        report["evidence"]["evaluation_file_sha256"] = hashlib.sha256(before).hexdigest()
        # Read identities before replay; derive no metric checks until replay
        # and stability checks confirm these exact receipt/artifact contents.
        receipt = _decode_receipt(before, EVALUATION_SCHEMA, "evaluation")
        digest = receipt["receipt_sha256"]
        artifact_hash = _sha256(receipt.get("artifact_sha256"), "artifact SHA256")
        model_hash = _sha256(receipt.get("model_sha256"), "model SHA256")
        if not isinstance(receipt.get("artifact_path"), str) or not receipt["artifact_path"]:
            raise ValueError("missing artifact path")
        artifact_path = Path(receipt["artifact_path"]).resolve()
        if _artifact_sha256(artifact_path) != artifact_hash:
            raise ValueError("artifact does not match evaluation receipt")
        fit_path = artifact_path.parent / "fit_receipt.json"
        fit_before = _read_bounded(fit_path)
        fit = _decode_receipt(fit_before, FIT_RECEIPT_SCHEMA, "fit")
        fit_verified = verify_fit_receipt(fit_path, replay_data=True)
        if (not isinstance(fit_verified, dict) or fit_verified.get("valid") is not True
                or fit_verified.get("replayed_data") is not True):
            raise ValueError("fit verifier did not confirm full data and baseline replay")
        if (fit_verified.get("artifact_sha256") != artifact_hash or fit.get("artifact_sha256") != artifact_hash
                or fit_verified.get("model_sha256") != model_hash or fit.get("model_sha256") != model_hash):
            raise ValueError("fit and evaluation artifact identities differ")
        verified = verify_evaluation_receipt(path, replay=True)
        if not isinstance(verified, dict) or verified.get("valid") is not True or verified.get("replayed") is not True:
            raise ValueError("evaluation verifier did not confirm full replay")
        if (_read_bounded(path) != before or _read_bounded(fit_path) != fit_before
                or _artifact_sha256(artifact_path) != artifact_hash):
            raise ValueError("fit/evaluation receipt or artifact changed during verification")
        if verified.get("artifact_sha256") != artifact_hash or verified.get("model_sha256") != model_hash:
            raise ValueError("verifier and receipt artifact identities differ")
        if receipt.get("promoted") is not False or receipt.get("seed_disjointness_verified") is not True:
            raise ValueError("evaluation must be inactive and verified seed-disjoint")
        if not isinstance(receipt.get("results"), dict):
            raise ValueError("missing evaluation results")
        if set(verified.get("conditions", [])) != set(_CONDITIONS) or set(receipt["results"]) != set(_CONDITIONS):
            raise ValueError("evaluation must contain exactly ID and scarcity-shift conditions")
        report["evidence"].update(evaluation_receipt_sha256=digest, artifact_sha256=artifact_hash, model_sha256=model_hash,
                                 fit_receipt_path=str(fit_path), fit_file_sha256=hashlib.sha256(fit_before).hexdigest(),
                                 fit_receipt_sha256=fit["receipt_sha256"])
        report["verification"] = {"valid": True, "replayed": True}
        report["fit_verification"] = {"valid": True, "replayed_data": True}
        report["conditions"] = {name: _condition_checks(receipt["results"][name]) for name in _CONDITIONS}
        for condition, result in report["conditions"].items():
            for name, check in result["checks"].items():
                if not check["passed"]:
                    report["reasons"].append(f"{condition}.{name}: {check['reason']}")
        if fit.get("config", {}).get("architecture") == LINEAR_RESIDUAL_ARCHITECTURE:
            selected_scale = _number(fit["training"]["residual_selection"]["selected_scale"])
            report["residual_selection"] = {"selected_scale": selected_scale}
            if selected_scale == 0:
                report["reasons"].append("linear_anchor_retained: zero selected residual scale cannot establish residual improvement")
        if not report["reasons"] and all(result["passed"] for result in report["conditions"].values()):
            report["state"] = "prediction_qualified"
    except Exception as exc:
        # Replay can fail through its artifact decoder or model implementation,
        # not just JSON parsing. Every such failure leaves the screen rejected.
        report["reasons"].append(f"evidence_verification_failed: {type(exc).__name__}: {exc}")
    return _finish(report)
