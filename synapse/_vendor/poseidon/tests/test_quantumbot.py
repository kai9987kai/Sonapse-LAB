"""Tests for QuantumBot Bell State Entanglement & Quantum Teleportation Engine."""
import math
from poseidon.quantumbot import QuantumBotCircuitEngine


def test_quantumbot_gates_and_normalization():
    circuit = QuantumBotCircuitEngine()
    circuit.reset()

    # Initial state |0000> norm = 1.0
    initial_norm = sum(abs(c) ** 2 for c in circuit.state)
    assert abs(initial_norm - 1.0) < 1e-6

    # Hadamard on qubit 0
    circuit.hadamard(0)
    h_norm = sum(abs(c) ** 2 for c in circuit.state)
    assert abs(h_norm - 1.0) < 1e-6
    # Amplitudes of |0000> and |1000> should be 1/sqrt(2)
    assert abs(abs(circuit.state[0]) - 1.0 / math.sqrt(2.0)) < 1e-6
    assert abs(abs(circuit.state[8]) - 1.0 / math.sqrt(2.0)) < 1e-6


def test_quantumbot_bell_pair_preparation():
    circuit = QuantumBotCircuitEngine()
    circuit.reset()

    # Prepare Bell pair on qubits 2 and 3
    fidelity = circuit.prepare_bell_pair(q_a=2, q_b=3)
    assert abs(fidelity - 1.0) < 1e-6

    # In |Phi+>, only |0000> and |0011> should have amplitude 1/sqrt(2)
    assert abs(abs(circuit.state[0]) - 1.0 / math.sqrt(2.0)) < 1e-6
    assert abs(abs(circuit.state[3]) - 1.0 / math.sqrt(2.0)) < 1e-6


def test_quantumbot_teleportation_protocol():
    circuit = QuantumBotCircuitEngine()
    circuit.reset()

    circuit.prepare_bell_pair(q_a=2, q_b=3)
    alpha, beta = circuit.prepare_cognitive_state(vital_urgency=0.7, threat=0.3)

    teleport_info = circuit.teleport_cognitive_state(source_q=1, entangled_a=2, entangled_b=3)
    assert teleport_info["teleportation_success"] is True
    assert teleport_info["measurement_m1"] in (0, 1)
    assert teleport_info["measurement_m2"] in (0, 1)
    assert teleport_info["entanglement_entropy"] >= 0.0


def test_quantumbot_step_telemetry():
    circuit = QuantumBotCircuitEngine()
    dummy_obs = [0.9, 0.8, 0.7, 0.6, 0.1, 0.2]
    telemetry = circuit.step(dummy_obs, action=1)

    assert "bell_fidelity" in telemetry
    assert "cognitive_alpha_mag" in telemetry
    assert "teleport_m1" in telemetry
    assert telemetry["total_teleportations"] == 1
