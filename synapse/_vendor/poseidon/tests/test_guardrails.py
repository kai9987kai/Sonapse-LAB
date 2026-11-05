"""Tests for formal safety guardrails module."""
import pytest
from poseidon.guardrails import FormalSafetyGuard, GuardrailAudit


def test_guardrails_nominal():
    guard = FormalSafetyGuard()
    # Nominal observation: high health, energy, hydration, moderate storm
    obs = [1.0, 0.8, 0.8, 0.8, 0.1, 0.0, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.3, 0.5, 0.0, 0.0]
    audit = guard.audit_and_constrain(proposed_action=4, observation=obs)
    assert not audit.vetoed
    assert audit.action == 4
    assert audit.action_name == "explore"
    assert len(audit.active_invariants) == 0


def test_guardrails_dehydration_veto():
    guard = FormalSafetyGuard()
    # Dehydration: hydration = 0.08, local water = 0.60
    obs = [0.9, 0.8, 0.08, 0.8, 0.1, 0.0, 0.5, 0.60, 0.5, 0.2, 0.5, 0.5, 0.3, 0.5, 0.0, 0.0]
    audit = guard.audit_and_constrain(proposed_action=4, observation=obs)
    assert audit.vetoed
    assert audit.original_action == 4
    assert audit.action == 2  # drink
    assert audit.action_name == "drink"
    assert "HydrationStarvationGuard" in audit.active_invariants


def test_guardrails_storm_veto():
    guard = FormalSafetyGuard()
    # Severe storm: storm_severity = 0.85, shelter = 0.70, health = 0.40
    obs = [0.40, 0.8, 0.8, 0.8, 0.6, 0.0, 0.5, 0.5, 0.70, 0.85, 0.5, 0.5, 0.3, 0.5, 0.0, 0.0]
    audit = guard.audit_and_constrain(proposed_action=4, observation=obs)
    assert audit.vetoed
    assert audit.action == 3  # shelter
    assert audit.action_name == "shelter"
    assert "SevereStormShelterGuard" in audit.active_invariants


def test_guardrails_cliff_veto():
    guard = FormalSafetyGuard()
    obs = [0.50, 0.8, 0.8, 0.8, 0.1, 0.0, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.3, 0.5, 0.0, 0.0]
    # Predicted healths: action 4 leads to 0.0 (death), action 0 leads to 0.55
    pred_h = {0: 0.55, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.0, 5: 0.20}
    audit = guard.audit_and_constrain(proposed_action=4, observation=obs, predicted_healths=pred_h)
    assert audit.vetoed
    assert audit.action == 0  # best surviving alternative
    assert "CatastrophicCliffVeto" in audit.active_invariants
