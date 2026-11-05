"""Independent generated checks must reject plausible-looking wrong answers."""
import copy

import pytest

from poseidon.research import reasoning_audit as audit


def test_seeded_cases_are_reproducible_and_inputs_change():
    cases = audit.generate_cases(seed=101, samples=3)
    assert cases == audit.generate_cases(seed=101, samples=3)
    assert cases != audit.generate_cases(seed=102, samples=3)
    assert len({case['id'] for case in cases}) == len(cases)
    assert len({case['family'] for case in cases}) == 8


def test_exact_answer_checker_ignores_answer_in_explanatory_prose():
    case = next(c for c in audit.generate_cases(101, 1) if c['family'] == 'arithmetic')
    wrong = {'answer': '999999', 'text': f"The expected value was {case['expected']}", 'verified': True}
    assert audit.check_answer(case, wrong) is False
    assert audit.check_answer(case, {'answer': case['expected'], 'verified': True}) is True
    assert audit.check_answer(case, {'answer': 'NaN', 'verified': True}) is False


@pytest.mark.parametrize("raw", ["1e1000000000", "1e-1000000000", "0e1000000000",
                                 "NaN", "Infinity", "1/0", "9" * 81])
def test_numeric_answer_budget_is_checked_before_fraction_conversion(monkeypatch, raw):
    def forbidden_conversion(*args, **kwargs):
        pytest.fail("oversized or invalid answer reached Fraction conversion")
    monkeypatch.setattr(audit, "Fraction", forbidden_conversion)
    case = {"answerable": True, "family": "arithmetic", "expected": "1"}
    assert audit.check_answer(case, {"answer": raw, "verified": True}) is False


@pytest.mark.parametrize("raw,expected", [("1.25e2", "125"), ("-5e-3", "-1/200"),
                                          (".5", "1/2"), ("3/12", "1/4"),
                                          ("0.00000000000000001", "1/100000000000000000")])
def test_bounded_fraction_helper_preserves_small_exact_formats(raw, expected):
    assert str(audit._bounded_answer_fraction(raw)) == expected


def test_hanoi_checker_replays_moves_and_checks_requested_disk_count():
    case = next(c for c in audit.generate_cases(101, 1) if c['family'] == 'hanoi')
    from poseidon.problem_solver import solve_problem
    result = solve_problem(case['prompt'])
    assert audit.check_answer(case, result)
    result['moves'][0] = '9:A->C'
    assert not audit.check_answer(case, result)


def test_edit_checker_rejects_wrong_displayed_answer_with_correct_structured_fields():
    case = next(c for c in audit.generate_cases(101, 1) if c['family'] == 'edit')
    from poseidon.problem_solver import solve_problem
    result = solve_problem(case['prompt'])
    assert audit.check_answer(case, result)
    result['answer'] = 'edit_distance = 999999'
    assert not audit.check_answer(case, result)


def test_all_refused_has_undefined_risk_and_correct_negative_refusals():
    report = audit.run_audit(seed=101, samples=1, router=lambda prompt: (None, None))
    assert report['metrics']['accepted'] == 0
    assert report['metrics']['selective_risk'] is None
    assert report['metrics']['coverage'] == 0
    assert report['metrics']['correct_refusals'] > 0
    assert report['metrics']['missed_answerable'] == 6


def test_false_self_verification_is_measured_and_not_trusted():
    def router(prompt):
        return 'bad_solver', {'answer': '999999', 'verified': True}
    report = audit.run_audit(seed=101, samples=1, router=router)
    assert report['metrics']['false_verifications'] == report['metrics']['total']
    assert report['metrics']['selective_risk'] == 1
    assert report['metrics']['false_acceptances'] > 0


def test_unexpected_solver_crash_is_not_a_correct_refusal():
    def broken(prompt):
        raise RuntimeError('solver crashed')
    report = audit.run_audit(seed=101, samples=1, router=broken)
    assert report['metrics']['errors'] == report['metrics']['total']
    assert report['metrics']['correct_refusals'] == 0


def test_audit_rejects_cached_code_after_source_changes(monkeypatch):
    from poseidon.provenance import StaleSourceError
    def stale():
        raise StaleSourceError('restart required')
    monkeypatch.setattr(audit, 'assert_source_current', stale, raising=False)
    with pytest.raises(StaleSourceError):
        audit.run_audit(seed=101, samples=1)


def test_real_generated_suite_replays_without_model_weights():
    report = audit.run_audit(seed=101, samples=2)
    assert report['metrics']['wrong'] == 0
    assert report['metrics']['missed_answerable'] == 0
    assert report['metrics']['correct_refusals'] > 0
    checked = audit.verify_audit(report, replay=True)
    assert checked['integrity_verified'] is True
    assert checked['replay_verified'] is True
    assert checked['scope'] == 'finite_generated_suite'
    mutated = copy.deepcopy(report)
    mutated['metrics']['wrong'] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        audit.verify_audit(mutated)


@pytest.mark.parametrize('seed,samples', [(True, 1), (1, True), (1, 0), (1, 33), ('1', 1)])
def test_generation_budget_and_types_are_strict(seed, samples):
    with pytest.raises(ValueError):
        audit.generate_cases(seed, samples)
