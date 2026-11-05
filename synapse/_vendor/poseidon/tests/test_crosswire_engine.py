"""Tests for Project CROSSWIRE Engine."""

import pytest
import math

from poseidon.crosswire_engine import (
    ActiveDiagnosticProber,
    CandidateMapping,
    CandidateSpace,
    ClosedLoopRuntimeShield,
    CrosswireEngine,
    InterfaceContractSpec,
    InterfaceContractVerifier,
    TransitionTrace,
    VersionSpaceTracker,
    SCHEMA_ID,
)


def test_candidate_space_generation():
    perms3 = CandidateSpace.generate_permutations(3)
    assert len(perms3) == 6
    assert all(isinstance(h, CandidateMapping) for h in perms3)
    assert len(set(h.permutation for h in perms3)) == 6

    perms4 = CandidateSpace.generate_permutations(4)
    assert len(perms4) == 24

    with pytest.raises(ValueError):
        CandidateSpace.generate_permutations(8)


def test_candidate_mapping_action_and_invert():
    h = CandidateMapping("h_test", (2, 0, 1))
    assert h.map_action(0) == 2
    assert h.map_action(1) == 0
    assert h.map_action(2) == 1
    assert h.invert(2) == 0
    assert h.invert(0) == 1
    assert h.invert(1) == 2

    with pytest.raises(IndexError):
        h.map_action(5)


def test_version_space_tracker_entropy_and_pruning():
    perms = CandidateSpace.generate_permutations(3)
    # Model: state 0 + action 1 -> state 1; state 0 + action 2 -> state 2
    transition_model = {(0, 1): 1, (0, 2): 2, (0, 0): 0}
    tracker = VersionSpaceTracker(perms, transition_model)

    assert tracker.version_space_size == 6
    assert tracker.initial_size == 6
    assert math.isclose(tracker.compute_entropy(), math.log2(6), rel_tol=1e-5)

    # Observe: action_ext 0 leads to state 1 from state 0
    # Hypotheses where h(0) == 1 survive!
    trace1 = TransitionTrace(step=0, action_ext=0, state_before=0, state_after=1)
    p_info = tracker.prune_with_trace(trace1)

    assert tracker.version_space_size == 2
    assert p_info["eliminated_count"] == 4
    for h in tracker.surviving:
        assert h.map_action(0) == 1


def test_active_diagnostic_prober_greedy_selection():
    k = 4
    engine = CrosswireEngine(num_actions=k)
    all_hypotheses = CandidateSpace.generate_permutations(k)
    prober = ActiveDiagnosticProber(k, engine.transition_model)

    gains = prober.evaluate_action_information_gain(current_state=0, surviving_hypotheses=all_hypotheses)
    assert len(gains) == k
    assert any(g > 0 for g in gains.values())

    best_action, best_gain = prober.select_best_probe(current_state=0, surviving_hypotheses=all_hypotheses)
    assert 0 <= best_action < k
    assert best_gain >= 0.0


def test_active_probing_convergence_to_true_hypothesis():
    k = 3
    engine = CrosswireEngine(num_actions=k)
    all_hypotheses = CandidateSpace.generate_permutations(k)
    true_h = CandidateMapping("h_true", (1, 2, 0))
    prober = ActiveDiagnosticProber(k, engine.transition_model)

    res = prober.generate_probing_sequence(
        start_state=0,
        candidate_hypotheses=all_hypotheses,
        true_hypothesis=true_h,
        budget=8,
    )

    assert res["fully_disambiguated"] is True
    assert res["is_correct"] is True
    assert res["final_surviving"] == 1
    assert res["identified_mapping"]["permutation"] == [1, 2, 0]


def test_interface_contract_all_hypothesis_verification():
    engine = CrosswireEngine(num_actions=4)
    all_hypotheses = CandidateSpace.generate_permutations(4)
    verifier = InterfaceContractVerifier(engine.transition_model)
    contract = InterfaceContractSpec(
        hazard_states={4},
        goal_states={3},
        max_steps=8,
    )

    # Safe plan under identity mapping: [FORWARD, FORWARD] -> [1, 1] goes 0 -> 1 -> 3 (GOAL)
    identity_h = CandidateMapping("h_id", (0, 1, 2, 3))
    res_safe = verifier.verify_plan_all_hypotheses(
        start_state=0,
        plan_ext=[1, 1],
        surviving_hypotheses=[identity_h],
        contract=contract,
    )
    assert res_safe["admitted"] is True
    assert res_safe["violating_count"] == 0

    # Unsafe plan: action 3 leads directly to hazard from state 0
    res_unsafe = verifier.verify_plan_all_hypotheses(
        start_state=0,
        plan_ext=[3],
        surviving_hypotheses=[identity_h],
        contract=contract,
    )
    assert res_unsafe["admitted"] is False
    assert res_unsafe["violating_count"] == 1

    # Across ambiguous version space (size 24), [1, 1] might violate hazard under some permutations
    res_ambiguous = verifier.verify_plan_all_hypotheses(
        start_state=0,
        plan_ext=[1, 1],
        surviving_hypotheses=all_hypotheses,
        contract=contract,
    )
    assert res_ambiguous["admitted"] is False
    assert res_ambiguous["violating_count"] > 0


def test_closed_loop_runtime_shield():
    engine = CrosswireEngine(num_actions=4)
    all_hypotheses = CandidateSpace.generate_permutations(4)
    shield = ClosedLoopRuntimeShield(
        fallback_action_int=0,
        hazard_states={4},
        transition_model=engine.transition_model,
    )

    # Under ambiguous version space, intended action 1 is intercepted because mappings disagree
    eval_ambiguous = shield.evaluate_and_filter(
        current_state=0,
        intended_action_int=1,
        version_space=all_hypotheses,
    )
    assert eval_ambiguous["decision"] == "shield_intercept_fallback"
    assert eval_ambiguous["is_fallback"] is True
    assert shield.intercept_count == 1

    # Under resolved singleton version space
    singleton = [CandidateMapping("h_resolved", (0, 1, 2, 3))]
    eval_resolved = shield.evaluate_and_filter(
        current_state=0,
        intended_action_int=1,
        version_space=singleton,
    )
    assert eval_resolved["decision"] == "certified_admit"
    assert eval_resolved["is_fallback"] is False
    assert eval_resolved["dispatched_action_ext"] == 1
    assert shield.admitted_count == 1


def test_crosswire_engine_synthesize_and_receipt():
    engine = CrosswireEngine(num_actions=4)
    res = engine.synthesize(seed=42, budget=16, num_actions=4)

    assert res["schema"] == SCHEMA_ID
    assert "receipt_sha256" in res
    assert "sha256" in res
    assert res["initial_hypothesis_count"] == 24
    assert res["probing_results"]["fully_disambiguated"] is True
    assert res["probing_results"]["is_correct"] is True
    assert res["telemetry_summary"]["interoperability_certified"] is True
    assert res["telemetry_summary"]["zero_hazard_violations"] is True
    assert res["verified"] is True
