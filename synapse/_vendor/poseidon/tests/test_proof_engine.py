"""Unit tests for Project ARCHIMEDES: Formal Proof Engine."""

import pytest
from poseidon.proof_engine import (
    ProofEngine,
    ProofStep,
    verify_proof_step,
    verify_algebraic_identity,
)


def test_modus_ponens_proof():
    engine = ProofEngine()
    # P, P -> Q |- Q
    res = engine.prove_propositional(premises=["P", "P -> Q"], goal="Q")
    assert res["proved"] is True
    assert res["goal"] == "Q"
    assert res["certificate"]["is_valid"] is True
    assert len(res["steps"]) >= 3


def test_modus_tollens_proof():
    engine = ProofEngine()
    # P -> Q, ~Q |- ~P
    res = engine.prove_propositional(premises=["P -> Q", "~Q"], goal="~P")
    assert res["proved"] is True
    assert res["goal"] == "~P"
    assert res["certificate"]["is_valid"] is True


def test_algebraic_identity_proving():
    engine = ProofEngine()
    # Difference of squares: (a - b) * (a + b) == a^2 - b^2
    res = engine.prove_algebraic_identity(lhs="(a - b) * (a + b)", rhs="a^2 - b^2")
    assert res["proved"] is True
    assert res["max_residual"] < 1e-6
    assert res["certificate"]["is_valid"] is True

    # Binomial expansion: (x + 1)^2 == x^2 + 2*x + 1
    res2 = engine.prove_algebraic_identity(lhs="(x + 1)^2", rhs="x^2 + 2*x + 1")
    assert res2["proved"] is True
    assert res2["max_residual"] < 1e-6

    # False identity should be refuted
    res_false = engine.prove_algebraic_identity(lhs="x^2 + 1", rhs="x^2 + 2")
    assert res_false["proved"] is False


def test_mathematical_induction_sum_integers():
    engine = ProofEngine()
    res = engine.prove_by_induction("sum_integers")
    assert res["proved"] is True
    assert "sum(i, 1..n)" in res["theorem"]
    assert len(res["steps"]) == 5
    assert res["certificate"]["is_valid"] is True


def test_mathematical_induction_sum_squares():
    engine = ProofEngine()
    res = engine.prove_by_induction("sum_squares")
    assert res["proved"] is True
    assert res["certificate"]["is_valid"] is True


def test_proof_verifier_soundness():
    engine = ProofEngine()
    valid_steps = [
        {"step_id": 1, "formula": "P", "rule": "Premise", "antecedents": []},
        {"step_id": 2, "formula": "P -> Q", "rule": "Premise", "antecedents": []},
        {"step_id": 3, "formula": "Q", "rule": "ModusPonens", "antecedents": [1, 2]},
    ]
    check = engine.verify_proof(valid_steps)
    assert check["is_valid"] is True
    assert check["conclusion"] == "Q"

    # Cyclic or bogus dependency
    invalid_steps = [
        {"step_id": 1, "formula": "P", "rule": "Premise", "antecedents": []},
        {"step_id": 2, "formula": "Q", "rule": "ModusPonens", "antecedents": [1, 3]},  # Forward dependency on 3
        {"step_id": 3, "formula": "P -> Q", "rule": "Premise", "antecedents": []},
    ]
    check_bad = engine.verify_proof(invalid_steps)
    assert check_bad["is_valid"] is False
    assert check_bad["failed_step"] == 2
