"""Tests for Tessera Formal Safety Contract Prover and Invariant Shield."""
import pytest
from poseidon.tessera_prover import (
    SafetyContractSpec,
    SafetyProofReceipt,
    TesseraProver,
)


def test_tessera_prover_nominal_safety():
    prover = TesseraProver()
    # Nominal healthy state
    obs_nominal = [1.0, 0.9, 0.9, 0.9, 0.1, 0.05, 0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
    action_plan = [4, 4, 0]  # explore, explore, rest
    receipt = prover.verify_action_trajectory(obs_nominal, action_plan)

    assert receipt.is_safe is True
    assert receipt.shield_triggered is False
    assert receipt.original_action == 4
    assert receipt.certified_action == 4
    assert len(receipt.contract_violations) == 0
    assert len(receipt.certificate_sha256) == 64


def test_tessera_prover_unsafe_plan_triggers_shield():
    prover = TesseraProver()
    # Critical dehydration state near mortality boundary
    obs_dehydrated = [0.6, 0.8, 0.09, 0.5, 0.1, 0.05, 0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
    # Dangerous plan: flee into the desert
    dangerous_plan = [5, 5, 5]
    receipt = prover.verify_action_trajectory(obs_dehydrated, dangerous_plan)

    assert receipt.is_safe is False
    assert receipt.shield_triggered is True
    assert receipt.original_action == 5
    # Should be shielded to drink (action 2)
    assert receipt.certified_action == 2
    assert len(receipt.contract_violations) > 0
    assert receipt.counterexample_step is not None
    assert len(receipt.certificate_sha256) == 64


def test_tessera_prover_shield_action():
    prover = TesseraProver()
    obs_starving = [0.5, 0.09, 0.8, 0.5, 0.1, 0.05, 0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
    certified_action, receipt = prover.shield_action(obs_starving, candidate_action=4)

    assert receipt.shield_triggered is True
    assert certified_action == 1  # forage
