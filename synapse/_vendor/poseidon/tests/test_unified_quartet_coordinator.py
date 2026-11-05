"""Evidence and action contracts for the opt-in research coordinator."""
from copy import deepcopy
import math

import pytest

from poseidon.crosswire_engine import CandidateMapping
from poseidon.research.unified_quartet_coordinator import UnifiedQuartetCoordinator


OBS = [0.8] * 16
PREDICTIONS = [[0.1] * 16, [0.1] * 16, [0.9] * 16]
ERRORS = {"normal": [[0.1, 0.2, 8.0], [0.2, 0.1, 9.0]],
          "storm": [[0.2, 0.2, 7.0]]}


def coordinator(**kwargs):
    config = dict(num_actions=4, num_oracle_members=3, panel_size=2,
                  min_trust_threshold=0.2,
                  hypotheses=[CandidateMapping("identity", (0, 1, 2, 3))])
    config.update(kwargs)
    result = UnifiedQuartetCoordinator(**config)
    result.fit_panel(ERRORS)
    return result


def evaluate(engine, **kwargs):
    evidence = dict(current_state=0, member_predictions=PREDICTIONS,
                    candidate_margin=0.3, calibrated_risk=0.05,
                    risk_calibration_id="fixture-calibration-not-safety-proof")
    evidence.update(kwargs)
    return engine.evaluate_step(OBS, base_action=0, intended_action=1, **evidence)


def test_missing_evidence_preserves_baseline_and_unknown_quantities():
    engine = UnifiedQuartetCoordinator()
    result = engine.evaluate_step(OBS, base_action=2, intended_action=1)
    assert result.action == 2
    assert result.override_admitted is False
    assert result.epistemic_trust is None
    assert result.successor_value is None
    assert result.lyapunov_stable is None
    assert "missing_member_predictions" in result.telemetry["abstention_reasons"]
    assert "panel_not_fitted" in result.telemetry["abstention_reasons"]


def test_empirical_panel_is_fitted_separately_and_frozen_across_decisions():
    engine = coordinator()
    assert engine.active_panel == [0, 1]
    first = evaluate(engine)
    second = evaluate(engine, member_predictions=[[0.9] * 16] * 3)
    assert first.active_panel == second.active_panel == [0, 1]
    assert first.telemetry["panel_fit_sha256"] == second.telemetry["panel_fit_sha256"]
    assert first.action == 1 and first.override_admitted is True


def test_same_input_has_same_receipt_despite_repeated_evaluations():
    engine = coordinator()
    assert evaluate(engine).to_dict() == evaluate(engine).to_dict()
    assert evaluate(coordinator()).to_dict() == evaluate(engine).to_dict()


@pytest.mark.parametrize("kwargs, reason", [
    ({"member_predictions": None}, "missing_member_predictions"),
    ({"current_state": None}, "missing_current_state"),
    ({"candidate_margin": None}, "missing_candidate_margin"),
    ({"calibrated_risk": None}, "missing_calibrated_risk"),
    ({"risk_calibration_id": None}, "missing_risk_calibration_id"),
    ({"candidate_margin": 0.01}, "insufficient_candidate_margin"),
    ({"calibrated_risk": 0.9}, "risk_threshold_exceeded"),
])
def test_partial_or_rejected_evidence_cannot_override(kwargs, reason):
    decision = evaluate(coordinator(), **kwargs)
    assert decision.action == 0
    assert decision.override_admitted is False
    assert reason in decision.telemetry["abstention_reasons"]


def test_member_disagreement_abstains_instead_of_fabricating_agreement():
    result = evaluate(coordinator(), member_predictions=[[-10.0] * 16, [10.0] * 16, [0.0] * 16])
    assert result.action == 0
    assert result.epistemic_trust < 0.2
    assert "insufficient_epistemic_trust" in result.telemetry["abstention_reasons"]


def test_shield_checks_actual_state_and_cannot_be_overruled_by_gate():
    result = evaluate(coordinator(), current_state=2)
    assert result.shield_verdict == "shield_intercept_fallback"
    assert result.action == 0
    assert result.override_admitted is False
    assert "shield_rejected_candidate" in result.telemetry["abstention_reasons"]


def test_all_hypotheses_checked_including_adversary_after_first_six():
    hypotheses = [CandidateMapping(f"identity-{i}", (0, 1, 2, 3)) for i in range(6)]
    hypotheses.append(CandidateMapping("adversary", (0, 3, 2, 1)))
    result = evaluate(coordinator(hypotheses=hypotheses))
    assert result.action == 0
    assert result.override_admitted is False
    assert result.telemetry["shield"]["surviving_hypotheses_count"] == 7


def test_default_model_rejects_unsupported_action_instead_of_modulo_remapping():
    engine = coordinator(num_actions=6)
    result = engine.evaluate_step(OBS, 0, 5, current_state=0,
                                 member_predictions=PREDICTIONS, candidate_margin=0.3,
                                 calibrated_risk=0.05, risk_calibration_id="fixture")
    assert result.intended_action == 5
    assert result.action == 0
    assert "unsupported_interface_action" in result.telemetry["abstention_reasons"]


def test_incomplete_transition_model_abstains_instead_of_assuming_self_loop():
    result = evaluate(coordinator(transition_model={(0, 0): 0}))
    assert result.action == 0
    assert "incomplete_transition_model" in result.telemetry["abstention_reasons"]


def test_internal_action_and_external_dispatch_are_distinct():
    hypotheses = [CandidateMapping("swap", (0, 2, 1, 3))]
    result = evaluate(coordinator(hypotheses=hypotheses))
    assert result.action == result.intended_action == 1
    assert result.override_admitted is True
    assert result.telemetry["dispatched_action_ext"] == 2


def test_empty_version_space_preserves_baseline():
    engine = coordinator()
    engine.tracker.surviving = []
    result = evaluate(engine)
    assert result.action == 0
    assert result.override_admitted is False
    assert "empty_version_space" in result.telemetry["abstention_reasons"]


def test_candidate_equal_to_baseline_is_never_reported_as_override():
    engine = coordinator()
    decision = engine.evaluate_step(OBS, 1, 1, current_state=0,
        member_predictions=PREDICTIONS, candidate_margin=0.3,
        calibrated_risk=0.05, risk_calibration_id="fixture")
    assert decision.action == 1
    assert decision.override_admitted is False
    assert decision.telemetry["dispatched_action_ext"] is None


def test_custom_complete_six_action_model_supports_action_five():
    model = {(8, action): 8 for action in range(6)}
    engine = coordinator(num_actions=6, transition_model=model, hazard_states=[],
        hypotheses=[CandidateMapping("identity-6", (0, 1, 2, 3, 4, 5))])
    result = engine.evaluate_step(OBS, 0, 5, current_state=8,
        member_predictions=PREDICTIONS, candidate_margin=0.3,
        calibrated_risk=0.05, risk_calibration_id="fixture")
    assert result.action == 5
    assert result.override_admitted is True
    assert result.telemetry["dispatched_action_ext"] == 5


@pytest.mark.parametrize("observation, base_action, intended_action", [
    ([0.8] * 15, 0, 1), ([math.nan] * 16, 0, 1), ([math.inf] * 16, 0, 1),
    ([True] * 16, 0, 1), ([1.1] * 16, 0, 1), (OBS, True, 1), (OBS, 0, -1),
])
def test_malformed_observation_and_policy_actions_rejected(observation, base_action, intended_action):
    with pytest.raises(ValueError):
        coordinator().evaluate_step(observation, base_action, intended_action)


@pytest.mark.parametrize("kwargs", [
    {"hypotheses": ["not-a-hypothesis"]},
    {"hypotheses": [CandidateMapping("bad", (0, 1, 1, 3))]},
    {"transition_model": {(0, True): 1}},
    {"hazard_states": [True]},
    {"num_oracle_members": 1}, {"min_trust_threshold": math.nan},
])
def test_malformed_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        UnifiedQuartetCoordinator(**kwargs)


@pytest.mark.parametrize("predictions", [[], [[0.1] * 16], [[0.1] * 15] * 3,
    [[math.nan] * 16] * 3, [[math.inf] * 16] * 3, [[True] * 16] * 3])
def test_malformed_prediction_evidence_is_rejected(predictions):
    with pytest.raises(ValueError):
        evaluate(coordinator(), member_predictions=predictions)


@pytest.mark.parametrize("kwargs", [
    {"candidate_margin": math.nan}, {"calibrated_risk": math.inf},
    {"calibrated_risk": -0.1}, {"calibrated_risk": 1.1},
    {"current_state": True}, {"current_state": -1},
    {"risk_calibration_id": ""}, {"member_errors": [0.1, 0.2, 0.3]},
])
def test_malformed_scalar_evidence_and_per_decision_fit_are_rejected(kwargs):
    with pytest.raises(ValueError):
        evaluate(coordinator(), **kwargs)


@pytest.mark.parametrize("errors", [{}, {"normal": []}, {"normal": [[0.1]]},
    {"normal": [[-0.1, 0.2, 0.3]]}, {"normal": [[math.nan, 0.2, 0.3]]}])
def test_invalid_fit_cannot_replace_frozen_panel(errors):
    engine = coordinator()
    before = evaluate(engine).to_dict()
    with pytest.raises(ValueError):
        engine.fit_panel(errors)
    assert evaluate(engine).to_dict() == before


def test_receipt_fingerprint_binds_inputs_panel_model_and_decision():
    result = evaluate(coordinator()).to_dict()
    assert UnifiedQuartetCoordinator.verify_decision_integrity(result) is True
    for section, key, value in [
        (None, "action", 3),
        ("telemetry", "calibrated_risk", 0.01),
        ("telemetry", "member_predictions", [[0.2] * 16] * 3),
        ("telemetry", "panel_fit_sha256", "0" * 64),
    ]:
        changed = deepcopy(result)
        (changed if section is None else changed[section])[key] = value
        assert UnifiedQuartetCoordinator.verify_decision_integrity(changed) is False
    assert result["telemetry"]["receipt_scope"] == "fingerprint_only"
