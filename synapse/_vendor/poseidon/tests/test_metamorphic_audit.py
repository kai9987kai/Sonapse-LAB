"""Related inputs need independent correctness as well as consistency checks."""
import copy
import importlib

import pytest


def _audit():
    return importlib.import_module("poseidon.research.metamorphic_audit")


def test_generation_is_seeded_bounded_and_exercises_five_relations():
    audit = _audit()
    groups = audit.generate_groups(101, 2)
    assert groups == audit.generate_groups(101, 2)
    assert groups != audit.generate_groups(102, 2)
    assert len(groups) == 10
    assert len({group["family"] for group in groups}) == 5
    assert all(len(group["cases"]) == 2 for group in groups)
    assert any(".00000000000000001" in case["prompt"]
               for group in groups for case in group["cases"])


def test_constant_wrong_answers_can_be_consistent_but_never_jointly_correct():
    audit = _audit()
    report = audit.run_audit(101, 1, router=lambda prompt: (
        "constant", {"answer": "999999", "verified": True}))
    assert report["metrics"]["relation_passed"] > 0
    assert report["metrics"]["joint_correct_groups"] == 0
    assert report["metrics"]["correct"] == 0
    assert report["metrics"]["wrong"] == 10
    assert report["metrics"]["selective_risk"] == 1
    assert report["router_id"] == "custom_unreplayable"
    assert audit.verify_audit(report, replay=False)["integrity_verified"]
    with pytest.raises(ValueError, match="custom"):
        audit.verify_audit(report, replay=True)


@pytest.mark.parametrize("raw", ["1e1000000000", "1e-1000000000", "NaN", "1/0"])
def test_metamorphic_numeric_answers_reject_resource_exponents_before_conversion(monkeypatch, raw):
    audit = _audit()
    def forbidden_conversion(*args, **kwargs):
        pytest.fail("unbounded answer reached Fraction conversion")
    monkeypatch.setattr(audit, "Fraction", forbidden_conversion)
    assert audit._numeric_answer({"family": "arithmetic"}, {"answer": raw}) is None


def test_refusals_are_unavailable_relations_and_have_no_accepted_risk():
    report = _audit().run_audit(101, 1, router=lambda prompt: (None, None))
    assert report["metrics"]["refusals"] == 10
    assert report["metrics"]["relation_unavailable"] == 5
    assert report["metrics"]["relation_violations"] == 0
    assert report["metrics"]["selective_risk"] is None


def test_unexpected_solver_crashes_are_errors_separate_from_refusals():
    def broken(prompt):
        raise RuntimeError("broken")
    report = _audit().run_audit(101, 1, router=broken)
    assert report["metrics"]["errors"] == 10
    assert report["metrics"]["refusals"] == 0
    assert report["metrics"]["relation_unavailable"] == 5


def test_changed_second_answer_exposes_relation_violations():
    counter = 0
    def drifting(prompt):
        nonlocal counter
        counter += 1
        return "drift", {"answer": str(counter), "verified": True}
    report = _audit().run_audit(101, 1, router=drifting)
    assert report["metrics"]["relation_violations"] == 4
    assert report["metrics"]["relation_unavailable"] == 1  # edit answer has no valid display format
    assert report["metrics"]["relation_passed"] == 0


def test_verifier_recomputes_rows_and_metrics_after_refingerprinting():
    audit = _audit()
    report = audit.run_audit(101, 1, router=lambda prompt: (
        "constant", {"answer": "999999", "verified": True}))
    altered = copy.deepcopy(report)
    altered["groups"][0]["rows"][0]["correct"] = True
    altered.pop("report_sha256")
    altered["report_sha256"] = audit._hash(altered)
    with pytest.raises(ValueError, match="row"):
        audit.verify_audit(altered, replay=False)


def test_verifier_rejects_case_manifest_mutations():
    audit = _audit()
    report = audit.run_audit(101, 1, router=lambda prompt: (None, None))
    report["groups"][0]["cases"][0]["expected"] = "999999"
    report.pop("report_sha256")
    report["report_sha256"] = audit._hash(report)
    with pytest.raises(ValueError, match="manifest"):
        audit.verify_audit(report, replay=False)


def test_verifier_rejects_integer_flags_after_refingerprinting():
    audit = _audit()
    report = audit.run_audit(101, 1, router=lambda prompt: (
        "constant", {"answer": "999999", "verified": True}))
    report["groups"][0]["rows"][0]["result"]["verified"] = 1
    report.pop("report_sha256")
    report["report_sha256"] = audit._hash(report)
    with pytest.raises(ValueError, match="row"):
        audit.verify_audit(report, replay=False)


@pytest.mark.parametrize("mutation", ["acceptance_policy", "limits", "sources_missing", "sources_wrong",
                                     "source_name", "source_digest", "top_extra", "group_extra", "row_extra"])
def test_verifier_rejects_resealed_schema_and_metadata_mutations(mutation):
    audit = _audit()
    report = audit.run_audit(101, 1, router=lambda prompt: (None, None))
    if mutation == "acceptance_policy":
        report["acceptance_policy"] = "unverified_answers_are_safe"
    elif mutation == "limits":
        report["limits"] = ["This proves general accuracy."]
    elif mutation == "sources_missing":
        del report["source_hashes"]
    elif mutation == "sources_wrong":
        report["source_hashes"] = {}
    elif mutation == "source_name":
        report["source_hashes"]["invented.py"] = "0" * 64
    elif mutation == "source_digest":
        report["source_hashes"]["poseidon/reasoning.py"] = "invalid"
    elif mutation == "top_extra":
        report["claimed_certified"] = True
    elif mutation == "group_extra":
        report["groups"][0]["claimed_certified"] = True
    else:
        report["groups"][0]["rows"][0]["claimed_certified"] = True
    report.pop("report_sha256")
    report["report_sha256"] = audit._hash(report)
    with pytest.raises(ValueError):
        audit.verify_audit(report, replay=False)


def test_verifier_requires_current_source_for_replay():
    audit = _audit()
    report = audit.run_audit(101, 1)
    report["source_hashes"]["poseidon/reasoning.py"] = "0" * 64
    report.pop("report_sha256")
    report["report_sha256"] = audit._hash(report)
    assert audit.verify_audit(report, replay=False)["source_matches"] is False
    with pytest.raises(ValueError, match="source changed"):
        audit.verify_audit(report, replay=True)


@pytest.mark.parametrize("seed,samples", [(True, 1), (1, True), (1, 0), (1, 33), ("1", 1)])
def test_strict_seed_and_budget_types(seed, samples):
    with pytest.raises(ValueError):
        _audit().generate_groups(seed, samples)


def test_stale_source_is_rejected(monkeypatch):
    audit = _audit()
    from poseidon.provenance import StaleSourceError
    def stale():
        raise StaleSourceError("restart required")
    monkeypatch.setattr(audit, "assert_source_current", stale)
    with pytest.raises(StaleSourceError):
        audit.run_audit(101, 1)


def test_real_tools_pass_correctness_and_relations_and_replay():
    audit = _audit()
    report = audit.run_audit(101, 2)
    assert report["metrics"]["correct"] == 20
    assert report["metrics"]["wrong"] == 0
    assert report["metrics"]["relation_passed"] == 10
    assert report["metrics"]["joint_correct_groups"] == 10
    checked = audit.verify_audit(report, replay=True)
    assert checked["replay_verified"] is True
    assert checked["scope"] == "finite_generated_metamorphic_suite"
