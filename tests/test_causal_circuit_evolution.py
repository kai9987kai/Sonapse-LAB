"""Tests for SONAPSE-EVO Causal Neural Circuit Evolution."""
import pytest
import torch

from synapse.evo_compat.circuit_evolution import (
    CausalCircuitEvolutionEngine,
    CircuitMutationProposal,
)
from synapse.evo_compat.growth_transactions import GrowthTransactionReceipt


def test_circuit_evolution_engine_initialization():
    engine = CausalCircuitEvolutionEngine(
        lambda_param=0.002,
        lambda_compute=0.001,
        lambda_uncertainty=0.02,
        promotion_threshold=0.03,
        seed=55,
    )
    assert engine.lambda_p == 0.002
    assert engine.lambda_c == 0.001
    assert engine.lambda_u == 0.02
    assert engine.threshold == 0.03
    assert len(engine.history) == 0


def test_motif_duplication_dales_law_compliance():
    engine = CausalCircuitEvolutionEngine(seed=77)
    prop = engine.generate_motif_duplication(
        source_subcircuit="pacemaker_cluster",
        num_nodes=10,
        excitatory_ratio=0.8,
    )
    assert isinstance(prop, CircuitMutationProposal)
    assert prop.mutation_type == "motif_duplication"
    assert prop.delta_nodes == 10
    assert prop.weights is not None
    assert prop.weights.shape == (10, 10)

    # Check Dale's Law signs: 80% excitatory columns, 20% inhibitory
    w = prop.weights
    # In each non-zero column of inhibitory neurons, signs must be negative
    num_inh = max(1, round(10 * (1.0 - 0.8)))
    inh_part = w[:, -num_inh:]
    # All non-zero entries in inhibitory part must be <= 0
    assert (inh_part[inh_part != 0.0] <= 0.0).all()
    # All non-zero entries in excitatory part must be >= 0
    exc_part = w[:, :-num_inh]
    assert (exc_part[exc_part != 0.0] >= 0.0).all()


def test_utility_evaluation_and_two_phase_transaction():
    engine = CausalCircuitEvolutionEngine(
        lambda_param=0.01,
        lambda_compute=0.01,
        lambda_uncertainty=0.05,
        promotion_threshold=0.05,
        seed=99,
    )
    prop = engine.generate_topological_rewire(num_nodes=8, rewire_fraction=0.1)

    # 1. High delta reward -> Net utility > threshold -> PROMOTED
    util_high, promoted, receipt = engine.evaluate_utility(prop, measured_delta_reward=0.25)
    assert util_high > engine.threshold
    assert promoted is True
    assert receipt.promoted is True
    assert receipt.rolled_back is False
    assert receipt.evidence is not None

    # 2. Low/negative delta reward -> Net utility <= threshold -> ROLLED BACK
    util_low, promoted_low, receipt_low = engine.evaluate_utility(prop, measured_delta_reward=0.01)
    assert util_low <= engine.threshold
    assert promoted_low is False
    assert receipt_low.promoted is False
    assert receipt_low.rolled_back is True


def test_sandbox_evolution_cycle_and_summary():
    engine = CausalCircuitEvolutionEngine(seed=123)

    def dummy_eval(cand):
        # Reward candidates with more nodes higher
        return 0.35 if cand.delta_nodes >= 6 else 0.02

    cycle_res = engine.run_sandbox_evolution_cycle(dummy_eval, trials=3)
    assert cycle_res["status"] == "completed"
    assert cycle_res["evaluated_candidates"] == 3
    assert cycle_res["promoted_candidates"] >= 1

    summary = engine.get_evolution_summary()
    assert summary["total_transactions"] == 3
    assert summary["promoted_count"] >= 1
    assert 0.0 <= summary["promotion_rate"] <= 1.0
    assert len(summary["recent_receipts"]) == 3
