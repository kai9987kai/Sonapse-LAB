import pytest
import math

from poseidon.quantum_planner import QuantumSuperpositionPlanner
from poseidon.world import N_ACTIONS


def test_quantum_amplitude_encoding():
    qp = QuantumSuperpositionPlanner()
    # High health, zero uncertainty -> should produce purely real amplitude (phase ~ 0)
    amp_safe = qp.encode_amplitude(predicted_health=0.95, predicted_energy=0.9, predicted_hydration=0.9, uncertainty=0.0)
    assert abs(amp_safe.imag) < 1e-4
    assert amp_safe.real > 0.5

    # High uncertainty -> phase should rotate significantly
    amp_risky = qp.encode_amplitude(predicted_health=0.95, predicted_energy=0.9, predicted_hydration=0.9, uncertainty=0.5)
    assert abs(amp_risky.imag) > 0.1


def test_quantum_superposition_interference():
    qp = QuantumSuperpositionPlanner()
    candidates = [
        {"predicted_health": 0.95, "predicted_energy": 0.9, "predicted_hydration": 0.9},
        {"predicted_health": 0.92, "predicted_energy": 0.8, "predicted_hydration": 0.8},
        {"predicted_health": 0.80, "predicted_energy": 0.7, "predicted_hydration": 0.7},
        {"predicted_health": 0.70, "predicted_energy": 0.6, "predicted_hydration": 0.6},
        {"predicted_health": 0.60, "predicted_energy": 0.5, "predicted_hydration": 0.5},
        {"predicted_health": 0.50, "predicted_energy": 0.4, "predicted_hydration": 0.4},
    ]
    uncs = [0.02, 0.05, 0.15, 0.30, 0.45, 0.60]

    plan = qp.plan_superposition(candidates, uncs, base_action=0)
    assert 0 <= plan["selected_action"] < N_ACTIONS
    assert len(plan["action_probabilities"]) == N_ACTIONS
    assert abs(sum(plan["action_probabilities"]) - 1.0) < 1e-3
    assert len(plan["branches"]) == N_ACTIONS
