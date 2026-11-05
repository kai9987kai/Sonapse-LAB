"""Unit tests for Project ATHENAEUM: E-Graph Equality Saturation, PRM-MCTS & SINDy."""

import math
from poseidon.athenaeum import (
    AthenaeumEngine,
    EGraph,
    EqualitySaturationEngine,
    ProcessRewardModel,
    SymbolicRegressionSINDy,
    TestTimeComputeEngine,
)


def test_egraph_equality_saturation_simplification():
    engine = EqualitySaturationEngine(max_iterations=6)
    # ((x + y) * (x - y)) + (y ^ 2)  ->  ((x ^ 2) - (y ^ 2)) + (y ^ 2)  ->  x ^ 2
    res = engine.saturate_and_simplify("((x + y) * (x - y)) + (y ^ 2)", target_expr="x ^ 2")
    assert res["eclass_equivalent"] is True
    assert res["simplified_expression"] in ("x ^ 2", "x * x")
    assert res["optimal_mdl_cost"] < res["initial_mdl_cost"]


def test_egraph_distributivity_and_identity():
    engine = EqualitySaturationEngine(max_iterations=5)
    res = engine.saturate_and_simplify("(a * 1) + (b - b)", target_expr="a")
    assert res["eclass_equivalent"] is True
    assert res["simplified_expression"] == "a"


def test_prm_and_puct_mcts_search():
    ttc = TestTimeComputeEngine()
    res = ttc.search("((a - b) * (a + b)) + (b ^ 2)", goal_expr="a ^ 2", budget_rollouts=16)
    assert res["total_nodes_explored"] >= 2
    assert res["best_prm_score"] >= 0.7
    assert res["optimal_expression"] in ("a ^ 2", "a * a")
    assert len(res["trajectory"]) >= 1


def test_sindy_sparse_equation_discovery():
    sindy = SymbolicRegressionSINDy()
    # True underlying law: y = 2*x^2 - 3*x + 5
    samples = [(float(x), 2.0 * (x ** 2) - 3.0 * x + 5.0) for x in (-3, -2, -1, 0, 1, 2, 3, 4)]
    res = sindy.discover(samples)
    assert res["verified"] is True
    assert res["mse"] < 1e-6
    assert res["active_terms"] == 3
    assert res["coefficients"].get("x^2") == 2.0
    assert res["coefficients"].get("x") == -3.0
    assert res["coefficients"].get("1") == 5.0


def test_athenaeum_unified_synthesis():
    ath = AthenaeumEngine()
    status = ath.status()
    assert status["ready"] is True
    assert status["egraph_equality_saturation"] is True

    synth = ath.synthesize("((x + y) * (x - y)) + (y ^ 2)", target_expression="x ^ 2")
    assert synth["verified"] is True
    assert len(synth["receipt_sha256"]) == 64
    assert synth["egraph_saturation"]["eclass_equivalent"] is True
    assert synth["sindy_discovery"]["discovered_equation"] == "1 - 2*x + 3*x^2"
