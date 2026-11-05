"""Held-out episode screening must not turn empirical success into zero risk."""
from copy import deepcopy
import hashlib
import math

import pytest

from poseidon.research import episode_risk
from poseidon.research.contracts import canonical_bytes


def protocol(*, conditions=("normal",), n=100, **changes):
    values = dict(policy_sha256="a" * 64, predictor_sha256="b" * 64,
                  panel_sha256="c" * 64, thresholds={"margin": 0.15},
                  required_conditions=list(conditions), fit_episode_ids=[1, 2],
                  selection_episode_ids=[3, 4], calibration_episode_ids=list(range(100, 100 + n)),
                  max_episode_risk=0.1, confidence=0.95)
    values.update(changes)
    return episode_risk.make_episode_risk_protocol(**values)


def outcomes(n=100, events=0, condition="normal", first_id=100):
    return {condition: [{"episode_id": first_id + i, "adverse_event": i < events}
                        for i in range(n)]}


def reseal(value, key):
    value.pop(key, None)
    value[key] = hashlib.sha256(canonical_bytes(value)).hexdigest()


@pytest.mark.parametrize("events,n,tail,expected", [
    (0, 1, 0.05, 0.95), (0, 10, 0.05, 0.2588655508930523),
    (1, 10, 0.05, 0.3941633024365047), (10, 10, 0.05, 1.0),
    (0, 100, 0.025, 0.0362166926451764),
])
def test_exact_one_sided_bounds_match_independent_known_values(events, n, tail, expected):
    assert episode_risk.binomial_risk_upper_bound(events, n, failure_probability=tail) == pytest.approx(expected)


def test_more_events_and_stronger_confidence_cannot_lower_bound():
    bounds = [episode_risk.binomial_risk_upper_bound(k, 20, failure_probability=0.05) for k in range(21)]
    assert bounds == sorted(bounds)
    assert episode_risk.binomial_risk_upper_bound(2, 20, failure_probability=0.01) > bounds[2]


def test_zero_adverse_events_requires_enough_complete_episodes():
    small = episode_risk.audit_episode_risk(protocol(n=5), outcomes(n=5))
    assert small["eligible"] is False
    assert small["conditions"]["normal"]["risk_upper_bound"] == pytest.approx(0.450719728346941)
    large = episode_risk.audit_episode_risk(protocol(), outcomes())
    assert large["eligible"] is True
    assert large["conditions"]["normal"]["observed_risk"] == 0
    assert 0 < large["conditions"]["normal"]["risk_upper_bound"] < 0.1
    assert large["promoted"] is False
    assert large["independence_verified"] is False
    assert large["disjointness_verified"] is True


def test_bonferroni_uses_all_prespecified_conditions_and_every_condition_must_pass():
    p = protocol(conditions=("normal", "storm"), n=200)
    rows = outcomes()
    rows.update(outcomes(condition="storm", first_id=200))
    report = episode_risk.audit_episode_risk(p, rows)
    assert report["per_condition_failure_probability"] == pytest.approx(0.025)
    assert report["conditions"]["normal"]["risk_upper_bound"] == pytest.approx(0.0362166926451764)
    assert report["eligible"] is True
    rows["storm"] = outcomes(events=50, condition="storm", first_id=200)["storm"]
    assert episode_risk.audit_episode_risk(p, rows)["eligible"] is False


def test_every_failure_always_rejects_for_risk_tolerance_below_one():
    report = episode_risk.audit_episode_risk(protocol(n=10), outcomes(n=10, events=10))
    assert report["conditions"]["normal"]["risk_upper_bound"] == 1
    assert report["eligible"] is False


def test_episodes_cannot_be_reused_across_conditions():
    p = protocol(conditions=("normal", "storm"), n=200)
    rows = outcomes()
    rows.update(outcomes(condition="storm", first_id=200))
    rows["storm"][0]["episode_id"] = 100
    with pytest.raises(ValueError, match="duplicate episode"):
        episode_risk.audit_episode_risk(p, rows)


def test_stale_imported_source_blocks_creation_and_verification(monkeypatch):
    from poseidon.provenance import StaleSourceError
    report = episode_risk.audit_episode_risk(protocol(), outcomes())

    def stale():
        raise StaleSourceError("changed source")

    monkeypatch.setattr(episode_risk, "assert_source_current", stale)
    with pytest.raises(StaleSourceError):
        protocol()
    assert episode_risk.verify_episode_risk_report(report)["valid"] is False


@pytest.mark.parametrize("changes", [
    {"fit_episode_ids": [1, 1]}, {"selection_episode_ids": [2]},
    {"calibration_episode_ids": [1, 100]}, {"calibration_episode_ids": [3, 100]},
    {"calibration_episode_ids": [True]}, {"fit_episode_ids": ["1"]},
    {"selection_episode_ids": [-1]}, {"calibration_episode_ids": []},
    {"required_conditions": []}, {"required_conditions": ["normal", "normal"]},
    {"required_conditions": [True]}, {"policy_sha256": "a" * 63},
    {"predictor_sha256": "B" * 64}, {"panel_sha256": True},
    {"thresholds": {"margin": True}}, {"thresholds": {"margin": math.nan}},
    {"max_episode_risk": True}, {"max_episode_risk": 1.0},
    {"max_episode_risk": -0.1}, {"confidence": math.nan}, {"confidence": True},
])
def test_protocol_rejects_overlap_duplicates_and_malformed_fixed_identities(changes):
    with pytest.raises(ValueError):
        protocol(**changes)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate_id", "missing_id", "overlap", "event_int", "event_nan", "id_bool", "extra_row_key", "empty"])
def test_raw_rows_cannot_change_conditions_episode_population_or_binary_event_contract(mutation):
    rows = outcomes()
    if mutation == "missing": rows = {}
    elif mutation == "extra": rows["unplanned"] = []
    elif mutation == "duplicate_id": rows["normal"][1]["episode_id"] = 100
    elif mutation == "missing_id": rows["normal"].pop()
    elif mutation == "overlap": rows["normal"][0]["episode_id"] = 1
    elif mutation == "event_int": rows["normal"][0]["adverse_event"] = 0
    elif mutation == "event_nan": rows["normal"][0]["adverse_event"] = math.nan
    elif mutation == "id_bool": rows["normal"][0]["episode_id"] = True
    elif mutation == "extra_row_key": rows["normal"][0]["weight"] = 1
    elif mutation == "empty": rows["normal"] = []
    with pytest.raises(ValueError):
        episode_risk.audit_episode_risk(protocol(), rows)


def test_verifier_recomputes_resealed_summaries_and_binds_source_and_protocol(tmp_path):
    report = episode_risk.audit_episode_risk(protocol(), outcomes())
    path = tmp_path / "risk.json"
    episode_risk.write_episode_risk_report(path, report)
    verified = episode_risk.verify_episode_risk_report(path)
    assert verified["valid"] is True
    assert verified["recomputed"] is True
    assert verified["eligible"] is True
    for section, key, value in [
        (None, "eligible", False), ("conditions", "normal", {"risk_upper_bound": 0}),
        ("source_sha256", "poseidon/research/episode_risk.py", "0" * 64),
    ]:
        altered = deepcopy(report)
        (altered if section is None else altered[section])[key] = value
        reseal(altered, "report_sha256")
        assert episode_risk.verify_episode_risk_report(altered)["valid"] is False
    altered = deepcopy(report)
    altered["protocol"]["thresholds"]["margin"] = 0.5
    reseal(altered, "report_sha256")
    assert episode_risk.verify_episode_risk_report(altered)["valid"] is False


def test_equally_valid_resealed_raw_evidence_is_not_authenticated():
    report = episode_risk.audit_episode_risk(protocol(), outcomes(events=50))
    assert report["eligible"] is False
    # Recomputing a different complete record remains possible: content hashes
    # cannot authenticate outcomes, independence, or freeze-before-collection.
    report["outcomes_by_condition"] = outcomes()
    recomputed = episode_risk.audit_episode_risk(report["protocol"], report["outcomes_by_condition"])
    assert episode_risk.verify_episode_risk_report(recomputed)["valid"] is True
    assert recomputed["evidence_scope"] == "caller_supplied_complete_episode_outcomes"


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": 1e9999}', '[]'])
def test_json_reader_rejects_duplicate_nonfinite_and_non_object_documents(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        episode_risk.read_episode_risk_json(path)


@pytest.mark.parametrize("k,n,tail", [(True, 10, 0.05), (0, False, 0.05), (-1, 10, 0.05), (11, 10, 0.05), (0, 0, 0.05), (0, 10, True), (0, 10, math.nan), (0, 10, 0)])
def test_binomial_inputs_require_valid_counts_and_finite_probability(k, n, tail):
    with pytest.raises(ValueError):
        episode_risk.binomial_risk_upper_bound(k, n, failure_probability=tail)
