"""Offline screening of a frozen policy using complete-episode binary outcomes.

The one-sided Clopper--Pearson bounds apply to independent, identically
distributed Bernoulli episodes within each prespecified condition. Bonferroni
allocates the declared failure probability across all conditions. Episode IDs
establish structural disjointness only: supplied hashes and outcomes cannot
authenticate a study, prove independence, or prove it was frozen beforehand.
An episode-level bound is never a conditional per-action risk estimate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from scipy.stats import beta

from ..provenance import assert_source_current
from .contracts import canonical_bytes


SCHEMA = "poseidon-episode-risk-report-v1"
PROTOCOL_SCHEMA = "poseidon-episode-risk-protocol-v1"
MAX_EPISODES = 100_000
MAX_JSON_BYTES = 32 * 1024 * 1024
_SOURCE_PATHS = ("research/episode_risk.py", "research/contracts.py", "provenance.py")
_DECLARATION = "Caller declares identities, thresholds, conditions and episode IDs were frozen before collecting audit outcomes."


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sources() -> dict[str, str]:
    current = assert_source_current()
    return {f"poseidon/{name}": current[name] for name in _SOURCE_PATHS}


def _integer(value: Any, label: str, low: int = 0, high: int = 2**63 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ValueError(f"{label} must be finite with magnitude <= 1e100")
    return result


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{label} must be a nonempty string of at most 128 characters")
    return value


def _sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _ids(values: Any, label: str, *, nonempty: bool = False) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) > MAX_EPISODES or (nonempty and not values):
        raise ValueError(f"{label} must be a bounded sequence of episode IDs")
    ids = [_integer(value, label) for value in values]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate episode IDs")
    return sorted(ids)


def binomial_risk_upper_bound(events: int, episodes: int, *, failure_probability: float) -> float:
    """Exact one-sided binomial upper confidence bound, including edge cases."""
    n = _integer(episodes, "episodes", 1, MAX_EPISODES)
    k = _integer(events, "events", 0, n)
    tail = _number(failure_probability, "failure_probability")
    if not 0 < tail < 1:
        raise ValueError("failure_probability must lie strictly between zero and one")
    if k == n:
        return 1.0
    # isf avoids rounding 1 - tail to 1 for a very small allocated tail.
    bound = float(beta.isf(tail, k + 1, n - k))
    if not math.isfinite(bound) or not 0 <= bound <= 1:
        raise FloatingPointError("nonfinite or invalid binomial confidence bound")
    return bound


def make_episode_risk_protocol(
    *, policy_sha256: str, predictor_sha256: str, panel_sha256: str,
    thresholds: Mapping[str, float], required_conditions: Sequence[str],
    fit_episode_ids: Sequence[int], selection_episode_ids: Sequence[int],
    calibration_episode_ids: Sequence[int], max_episode_risk: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Declare a fixed episode audit; this function cannot prove a precommitment.

    ``calibration_episode_ids`` are the complete held-out audit population, not
    individual transitions or only episodes on which an override was selected.
    Fit and selection IDs may be empty for an entirely fixed, unfitted policy.
    Every supplied outcome must belong to exactly one declared condition.
    """
    source = _sources()
    policy = _sha256(policy_sha256, "policy_sha256")
    predictor = _sha256(predictor_sha256, "predictor_sha256")
    panel = _sha256(panel_sha256, "panel_sha256")
    if not isinstance(thresholds, Mapping) or not 1 <= len(thresholds) <= 64:
        raise ValueError("thresholds must contain 1..64 fixed numeric settings")
    fixed_thresholds = {_name(key, "threshold name"): _number(value, "threshold")
                        for key, value in thresholds.items()}
    if not isinstance(required_conditions, (list, tuple)) or not 1 <= len(required_conditions) <= 64:
        raise ValueError("required_conditions must contain 1..64 condition names")
    conditions = [_name(value, "condition") for value in required_conditions]
    if len(set(conditions)) != len(conditions):
        raise ValueError("required_conditions contains duplicate conditions")
    fit_ids = _ids(fit_episode_ids, "fit_episode_ids")
    selection_ids = _ids(selection_episode_ids, "selection_episode_ids")
    calibration_ids = _ids(calibration_episode_ids, "calibration_episode_ids", nonempty=True)
    all_ids = fit_ids + selection_ids + calibration_ids
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("fit, selection and calibration episode IDs overlap")
    if len(all_ids) > MAX_EPISODES:
        raise ValueError("combined episode partitions exceed workload bounds")
    risk = _number(max_episode_risk, "max_episode_risk")
    conf = _number(confidence, "confidence")
    if not 0 <= risk < 1 or not 0.5 <= conf < 1:
        raise ValueError("max_episode_risk must be in [0, 1); confidence must be in [0.5, 1)")
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "policy_sha256": policy, "predictor_sha256": predictor, "panel_sha256": panel,
        "thresholds": dict(sorted(fixed_thresholds.items())),
        "required_conditions": sorted(conditions),
        "fit_episode_ids": fit_ids, "selection_episode_ids": selection_ids,
        "calibration_episode_ids": calibration_ids,
        "max_episode_risk": risk, "confidence": conf,
        "event_definition": "At least one adverse override during one complete episode, including episodes with no overrides.",
        "freeze_before_collection": _DECLARATION,
        "freeze_verified": False,
        "source_sha256": source,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    return protocol


def _validated_protocol(protocol: Any) -> dict[str, Any]:
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported episode-risk protocol")
    expected = _sha256(protocol.get("protocol_sha256"), "protocol_sha256")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if _digest(unsigned) != expected:
        raise ValueError("episode-risk protocol digest mismatch")
    inputs = ("policy_sha256", "predictor_sha256", "panel_sha256", "thresholds",
              "required_conditions", "fit_episode_ids", "selection_episode_ids",
              "calibration_episode_ids", "max_episode_risk", "confidence")
    try:
        rebuilt = make_episode_risk_protocol(**{key: protocol[key] for key in inputs})
    except KeyError as exc:
        raise ValueError("incomplete episode-risk protocol") from exc
    if canonical_bytes(protocol) != canonical_bytes(rebuilt):
        raise ValueError("episode-risk protocol semantics or sources do not match current analysis")
    return rebuilt


def audit_episode_risk(protocol: Mapping[str, Any], outcomes_by_condition: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute simultaneous bounds from all prespecified held-out episode rows.

    Eligibility requires every declared condition's upper bound to be at most
    the fixed tolerance. The report does not activate a policy or certify an
    individual action. IID sampling within each condition is an assumption,
    and no assumption is inferred from integer IDs or a valid content digest.
    """
    fixed = _validated_protocol(protocol)
    required = fixed["required_conditions"]
    if not isinstance(outcomes_by_condition, Mapping) or set(outcomes_by_condition) != set(required):
        raise ValueError("outcomes must contain exactly every prespecified condition")
    rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    seen: set[int] = set()
    for condition in required:
        values = outcomes_by_condition[condition]
        if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_EPISODES:
            raise ValueError(f"condition {condition!r} requires complete episode outcomes")
        rows = []
        for value in values:
            if not isinstance(value, Mapping) or set(value) != {"episode_id", "adverse_event"}:
                raise ValueError("episode outcome requires exactly episode_id and adverse_event")
            episode_id = _integer(value["episode_id"], "episode_id")
            if episode_id in seen:
                raise ValueError("duplicate episode ID within or across conditions")
            if type(value["adverse_event"]) is not bool:
                raise ValueError("adverse_event must be a binary boolean complete-episode outcome")
            seen.add(episode_id)
            if len(seen) > MAX_EPISODES:
                raise ValueError("episode outcome workload exceeds bounds")
            rows.append({"episode_id": episode_id, "adverse_event": value["adverse_event"]})
        rows_by_condition[condition] = sorted(rows, key=lambda row: row["episode_id"])
    if seen != set(fixed["calibration_episode_ids"]):
        raise ValueError("outcome episode IDs must exactly cover the held-out calibration partition")
    tail = (1 - fixed["confidence"]) / len(required)
    condition_results = {}
    reasons = []
    for condition, rows in rows_by_condition.items():
        events = sum(row["adverse_event"] for row in rows)
        bound = binomial_risk_upper_bound(events, len(rows), failure_probability=tail)
        passed = bound <= fixed["max_episode_risk"]
        condition_results[condition] = {
            "episodes": len(rows), "adverse_episodes": events,
            "observed_risk": events / len(rows), "risk_upper_bound": bound,
            "passed": passed,
        }
        if not passed:
            reasons.append(f"{condition}: episode risk upper bound exceeds fixed tolerance")
    report = {
        "schema": SCHEMA, "state": "episode_risk_qualified" if not reasons else "rejected",
        "eligible": not reasons, "promoted": False,
        "method": "one_sided_clopper_pearson_bonferroni",
        "evidence_scope": "caller_supplied_complete_episode_outcomes",
        "risk_unit": "probability of at least one adverse override in a complete episode",
        "conditional_action_risk_established": False,
        "disjointness_verified": True, "independence_verified": False,
        "freeze_verified": False, "outcome_authenticity_verified": False,
        "per_condition_failure_probability": tail,
        "protocol": fixed, "outcomes_by_condition": rows_by_condition,
        "conditions": condition_results, "reasons": reasons,
        "source_sha256": _sources(),
        "assumptions": [
            "Complete episode outcomes are independent and identically distributed Bernoulli observations within each prespecified condition.",
            "The policy, predictor, panel, thresholds, conditions and audit population were fixed before collecting or viewing audit outcomes.",
            "Every prespecified audit episode is reported, including episodes without an override; no outcome-dependent inclusion or threshold selection occurred.",
            "Future episodes follow the same within-condition distribution; no distribution-shift guarantee is supplied.",
        ],
        "limits": [
            "Content recomputation and disjoint integer IDs do not authenticate outcomes, prove independence, or establish freeze-before-collection.",
            "The bound concerns complete episodes, not risk conditional on an override, a particular state, a selected action, or an individual prediction.",
            "Bonferroni covers only this declared condition family, not repeated protocols, candidate search, or reuse of audit outcomes.",
            "The event labels and policy/predictor/panel identities are caller supplied; environment replay and external attestation are not performed.",
            "Eligibility is an offline evidence screen and does not promote or activate a model or establish real-world safety.",
        ],
    }
    report["report_sha256"] = _digest(report)
    return report


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_episode_risk_json(path: str | Path) -> dict[str, Any]:
    """Read one bounded stable JSON object, rejecting ambiguous numeric data."""
    target = Path(path)
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(MAX_JSON_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("episode-risk JSON exceeds 32 MiB")
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("episode-risk JSON changed during reading")
    def finite_float(value: str) -> float:
        return _number(float(value), "JSON number")
    value = json.loads(raw, parse_constant=_reject_constant, parse_float=finite_float,
                       object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("episode-risk JSON must be an object")
    return value


def write_episode_risk_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Atomically save a valid recomputed report without weakening verification."""
    snapshot = deepcopy(dict(report))
    verification = verify_episode_risk_report(snapshot)
    if not verification["valid"]:
        raise ValueError(f"cannot save invalid episode-risk report: {verification['reason']}")
    data = json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("episode-risk report exceeds 32 MiB")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, delete=False) as stream:
            name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def verify_episode_risk_report(report_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Fail closed unless identities, sources, rows and every derived field replay.

    ``recomputed`` means statistical recomputation of supplied binary rows; it
    does not mean replay of an environment or authentication of those outcomes.
    """
    try:
        _sources()
        supplied = (read_episode_risk_json(report_or_path) if isinstance(report_or_path, (str, Path))
                    else deepcopy(report_or_path))
        if not isinstance(supplied, Mapping) or supplied.get("schema") != SCHEMA:
            raise ValueError("unsupported episode-risk report")
        expected = _sha256(supplied.get("report_sha256"), "report_sha256")
        if _digest({key: value for key, value in supplied.items() if key != "report_sha256"}) != expected:
            raise ValueError("episode-risk report digest mismatch")
        rebuilt = audit_episode_risk(supplied["protocol"], supplied["outcomes_by_condition"])
        if canonical_bytes(supplied) != canonical_bytes(rebuilt):
            raise ValueError("episode-risk report statistics or semantics do not recompute")
        return {"valid": True, "recomputed": True, "eligible": rebuilt["eligible"],
                "state": rebuilt["state"], "promoted": False,
                "report_sha256": expected, "protocol_sha256": rebuilt["protocol"]["protocol_sha256"],
                "outcome_authenticity_verified": False, "reason": "report_recomputed"}
    except Exception as exc:
        return {"valid": False, "recomputed": False, "eligible": False,
                "state": "rejected", "promoted": False, "outcome_authenticity_verified": False,
                "reason": f"{type(exc).__name__}: {exc}"}
