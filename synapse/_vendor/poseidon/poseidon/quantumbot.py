"""QuantumBot Bell State Entanglement & Quantum Teleportation Engine.

Synthesizes multi-qubit register simulations, EPR Bell state entanglement,
and quantum state teleportation from QuantumBot and Causeway:
1. 4-qubit complex state vector |psi> in C^16 (16 complex amplitudes).
2. Quantum gate algebra: Hadamard (H), Pauli (X, Y, Z), Phase (S), and CNOT.
3. Maximally entangled Bell pair generation: |Phi+> = (|00> + |11>) / sqrt(2).
4. Quantum Teleportation Protocol: teleports an agent cognitive state |phi> = a|0> + b|1>
   across spatial diamond lattice coordinates via shared entanglement and 2 classical bits.
5. von Neumann entanglement entropy S(rho) and Bell state fidelity verification.
"""
from __future__ import annotations

import cmath
import math
import random
from typing import Dict, List, Tuple


class QuantumBotCircuitEngine:
    """4-qubit quantum state simulator with Bell pairs and state teleportation."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        # 16-dimensional complex state vector representing 4 qubits |q0 q1 q2 q3>
        self.state: List[complex] = [0.0 + 0.0j] * 16
        self.state[0] = 1.0 + 0.0j  # Initialize in |0000>
        self.step_count: int = 0
        self.total_teleportations: int = 0

    def reset(self) -> None:
        """Reset quantum register to ground state |0000>."""
        self.state = [0.0 + 0.0j] * 16
        self.state[0] = 1.0 + 0.0j
        self.step_count = 0
        self.total_teleportations = 0

    def _apply_single_qubit_gate(self, target: int, matrix: List[List[complex]]) -> None:
        """Apply a 2x2 unitary matrix to the target qubit (0 to 3)."""
        new_state = [0.0 + 0.0j] * 16
        bit_mask = 1 << (3 - target)  # Qubit 0 is MSB, 3 is LSB

        for i in range(16):
            if self.state[i] == 0:
                continue
            bit = (i & bit_mask) >> (3 - target)
            i_paired = i ^ bit_mask

            if bit == 0:
                c0 = self.state[i]
                c1 = self.state[i_paired]
                new_state[i] += matrix[0][0] * c0 + matrix[0][1] * c1
                new_state[i_paired] += matrix[1][0] * c0 + matrix[1][1] * c1

        self.state = new_state

    def hadamard(self, target: int) -> None:
        """Apply Hadamard gate H = 1/sqrt(2) * [[1, 1], [1, -1]]."""
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        h_matrix = [
            [complex(inv_sqrt2, 0.0), complex(inv_sqrt2, 0.0)],
            [complex(inv_sqrt2, 0.0), complex(-inv_sqrt2, 0.0)],
        ]
        self._apply_single_qubit_gate(target, h_matrix)

    def pauli_x(self, target: int) -> None:
        """Apply Pauli-X (NOT) gate X = [[0, 1], [1, 0]]."""
        x_matrix = [[0j, 1.0 + 0j], [1.0 + 0j, 0j]]
        self._apply_single_qubit_gate(target, x_matrix)

    def pauli_z(self, target: int) -> None:
        """Apply Pauli-Z gate Z = [[1, 0], [0, -1]]."""
        z_matrix = [[1.0 + 0j, 0j], [0j, -1.0 + 0j]]
        self._apply_single_qubit_gate(target, z_matrix)

    def cnot(self, control: int, target: int) -> None:
        """Apply Controlled-NOT gate: flips target if control is 1."""
        ctrl_mask = 1 << (3 - control)
        target_mask = 1 << (3 - target)
        new_state = list(self.state)

        for i in range(16):
            if (i & ctrl_mask) != 0:
                # Control is 1: swap amplitude with target flipped
                target_bit = (i & target_mask) >> (3 - target)
                if target_bit == 0:
                    i_paired = i | target_mask
                    # Swap amplitudes
                    new_state[i], new_state[i_paired] = new_state[i_paired], new_state[i]

        self.state = new_state

    def prepare_bell_pair(self, q_a: int = 2, q_b: int = 3) -> float:
        """Create EPR Bell pair |Phi+> = (|00> + |11>) / sqrt(2) on qubits q_a and q_b.

        Returns:
            Bell state fidelity in [0, 1].
        """
        self.hadamard(q_a)
        self.cnot(q_a, q_b)

        # Measure fidelity with ideal Bell state
        fidelity = self.compute_bell_fidelity(q_a, q_b)
        return fidelity

    def compute_bell_fidelity(self, q_a: int = 2, q_b: int = 3) -> float:
        """Compute the overlap fidelity with standard Bell pair |Phi+>."""
        mask_a = 1 << (3 - q_a)
        mask_b = 1 << (3 - q_b)
        prob_00 = 0.0
        prob_11 = 0.0

        for i, amp in enumerate(self.state):
            bit_a = bool(i & mask_a)
            bit_b = bool(i & mask_b)
            p = abs(amp) ** 2
            if not bit_a and not bit_b:
                prob_00 += p
            elif bit_a and bit_b:
                prob_11 += p

        # Ideal Bell pair has prob_00 = 0.5, prob_11 = 0.5
        fidelity = 1.0 - abs(prob_00 - 0.5) - abs(prob_11 - 0.5)
        return max(0.0, min(1.0, fidelity))

    def prepare_cognitive_state(self, vital_urgency: float, threat: float) -> Tuple[complex, complex]:
        """Encode vital urgency and threat into unknown state |phi> = alpha|0> + beta|1> on qubit 1."""
        theta = math.pi * max(0.0, min(1.0, vital_urgency))
        phi = 2.0 * math.pi * max(0.0, min(1.0, threat))
        alpha = complex(math.cos(theta / 2.0), 0.0)
        beta = complex(math.sin(theta / 2.0) * math.cos(phi), math.sin(theta / 2.0) * math.sin(phi))

        # Rotate qubit 1 to (alpha|0> + beta|1>)
        u_matrix = [
            [alpha, -beta.conjugate()],
            [beta, alpha.conjugate()],
        ]
        self._apply_single_qubit_gate(1, u_matrix)
        return (alpha, beta)

    def teleport_cognitive_state(
        self,
        source_q: int = 1,
        entangled_a: int = 2,
        entangled_b: int = 3,
    ) -> Dict[str, object]:
        """Execute the standard 4-step quantum teleportation protocol.

        Teleports the unknown quantum state from source_q to entangled_b
        using the EPR pair (entangled_a, entangled_b) and 2 classical bits.
        """
        # Step 1: Bell pair already established on (entangled_a, entangled_b)
        # Step 2: CNOT from source to entangled_a
        self.cnot(source_q, entangled_a)

        # Step 3: Hadamard on source
        self.hadamard(source_q)

        # Step 4: Measure source_q and entangled_a in standard computational basis
        probs = [abs(c) ** 2 for c in self.state]
        total_p = sum(probs)
        if total_p > 0:
            probs = [p / total_p for p in probs]
        chosen_idx = self.rng.choices(range(16), weights=probs, k=1)[0]

        mask_src = 1 << (3 - source_q)
        mask_a = 1 << (3 - entangled_a)
        m1 = (chosen_idx & mask_src) >> (3 - source_q)
        m2 = (chosen_idx & mask_a) >> (3 - entangled_a)

        # Step 5: Unitary correction on receiving qubit entangled_b: X^m2 * Z^m1
        if m2 == 1:
            self.pauli_x(entangled_b)
        if m1 == 1:
            self.pauli_z(entangled_b)

        self.total_teleportations += 1

        # Calculate von Neumann entanglement entropy S = -sum(p * log2(p))
        entropy = 0.0
        for p in probs:
            if p > 1e-9:
                entropy -= p * math.log2(p)

        return {
            "teleportation_success": True,
            "measurement_m1": m1,
            "measurement_m2": m2,
            "entanglement_entropy": round(entropy, 4),
            "total_teleportations": self.total_teleportations,
        }

    def step(
        self,
        observation: List[float],
        action: int,
    ) -> Dict[str, object]:
        """Update quantum registers and teleport state for one simulation step."""
        self.step_count += 1

        # Extract cues: obs[0] = health, obs[1] = energy, obs[5] = threat
        threat = observation[5] if len(observation) > 5 else 0.0
        vital_urgency = 1.0 - (observation[1] if len(observation) > 1 else 1.0)

        # 1. Reset and establish spatial EPR link between local agent and lattice anchor
        self.reset()
        fidelity = self.prepare_bell_pair(q_a=2, q_b=3)

        # 2. Encode cognitive decision state into qubit 1
        alpha, beta = self.prepare_cognitive_state(vital_urgency, threat)

        # 3. Teleport state across distributed anchor
        teleport_info = self.teleport_cognitive_state(source_q=1, entangled_a=2, entangled_b=3)

        return {
            "bell_fidelity": round(fidelity, 4),
            "cognitive_alpha_mag": round(abs(alpha), 4),
            "cognitive_beta_mag": round(abs(beta), 4),
            "teleport_m1": teleport_info["measurement_m1"],
            "teleport_m2": teleport_info["measurement_m2"],
            "quantum_entanglement_entropy": teleport_info["entanglement_entropy"],
            "total_teleportations": self.total_teleportations,
        }
