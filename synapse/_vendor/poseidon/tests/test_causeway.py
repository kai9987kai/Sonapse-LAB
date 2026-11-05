"""Tests for Causeway Quantum-Inspired Superposition Engine."""
import math
from poseidon.causeway import CausewaySuperpositionEngine


def test_causeway_initialization_and_normalization():
    engine = CausewaySuperpositionEngine(num_actions=6)
    probs = engine.probabilities()
    assert len(probs) == 6
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    for p in probs:
        assert math.isclose(p, 1.0 / 6.0, rel_tol=1e-5)
    entropy = engine.von_neumann_entropy()
    assert math.isclose(entropy, math.log(6.0), rel_tol=1e-4)


def test_causeway_evolution_and_entropy():
    engine = CausewaySuperpositionEngine(num_actions=6)
    # Give strong potential to action 2
    potentials = [0.1, 0.2, 5.0, 0.1, 0.05, 0.1]
    engine.evolve(potentials)
    probs = engine.probabilities()
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-5)
    # The state should be modified and entropy should change
    entropy = engine.von_neumann_entropy()
    assert 0.0 <= entropy <= math.log(6.0) + 1e-4


def test_causeway_destructive_interference():
    engine = CausewaySuperpositionEngine(num_actions=6)
    # Opposing phase shifts
    shifts = [0.0, math.pi, 0.0, math.pi, 0.0, math.pi]
    potentials = [1.0] * 6
    engine.evolve(potentials, phase_shifts=shifts)
    matrix = engine.interference_matrix()
    assert len(matrix) == 6
    assert len(matrix[0]) == 6
    conflicts = engine.detect_destructive_conflicts(threshold=0.0)
    assert isinstance(conflicts, list)


def test_causeway_collapse_and_telemetry():
    engine = CausewaySuperpositionEngine(num_actions=6)
    potentials = [0.5, 0.2, 2.5, 0.1, 0.1, 0.1]
    engine.evolve(potentials)
    action, telemetry = engine.collapse(seed=42)
    assert 0 <= action <= 5
    assert "chosen_action" in telemetry
    assert "probabilities" in telemetry
    assert "von_neumann_entropy" in telemetry
    assert "coherence_length" in telemetry
    assert "destructive_conflicts_count" in telemetry
    assert len(telemetry["probabilities"]) == 6
