"""Causeway: Quantum-Inspired Superposition Decision Engine for Poseidon.

Maintains a 6-dimensional complex amplitude state vector |psi> over actions,
governed by unitary Hamiltonian phase evolution, pairwise interference
matrices, von Neumann entropy decoherence tracking, and Born rule collapse.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class CausewaySuperpositionEngine:
    """Quantum-inspired superposition action manifold."""

    def __init__(self, num_actions: int = 6, coupling_strength: float = 0.25, dt: float = 0.1):
        self.num_actions = num_actions
        self.coupling_strength = coupling_strength
        self.dt = dt
        # Real and imaginary components of state vector |psi> = sum alpha_a |a>
        self.real = [1.0 / math.sqrt(num_actions)] * num_actions
        self.imag = [0.0] * num_actions
        self._normalize()

    @property
    def phases(self) -> List[float]:
        """Complex phase angles theta_a = atan2(imag, real) for each action."""
        return [math.atan2(i, r) for r, i in zip(self.real, self.imag)]

    def _normalize(self) -> None:
        norm_sq = sum(r * r + i * i for r, i in zip(self.real, self.imag))
        if norm_sq < 1e-12:
            self.real = [1.0 / math.sqrt(self.num_actions)] * self.num_actions
            self.imag = [0.0] * self.num_actions
            return
        norm = math.sqrt(norm_sq)
        self.real = [r / norm for r in self.real]
        self.imag = [i / norm for i in self.imag]

    def reset(self) -> None:
        """Reset state vector to uniform superposition with zero phase."""
        self.real = [1.0 / math.sqrt(self.num_actions)] * self.num_actions
        self.imag = [0.0] * self.num_actions

    def evolve(self, action_potentials: List[float], phase_shifts: List[float] | None = None) -> None:
        """Advance the wavefunction under Hamiltonian potential and directional phase shifts.

        H_aa = -action_potentials[a]
        H_ab = -coupling_strength for directional neighbors
        """
        if len(action_potentials) != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} action potentials, got {len(action_potentials)}")

        shifts = phase_shifts or [0.0] * self.num_actions
        # Apply external potential phase rotation and non-Hermitian amplitude gain:
        # alpha_a <- alpha_a * exp(gain * (V_a - V_mean)) * exp(-i * (V_a + shift_a) * dt)
        mean_pot = sum(action_potentials) / max(1, len(action_potentials))
        new_real = [0.0] * self.num_actions
        new_imag = [0.0] * self.num_actions

        for a in range(self.num_actions):
            gain = math.exp(0.8 * (action_potentials[a] - mean_pot))
            theta = -(action_potentials[a] + shifts[a]) * self.dt
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            # Complex multiplication with potential gain:
            r = (self.real[a] * cos_t - self.imag[a] * sin_t) * gain
            i = (self.real[a] * sin_t + self.imag[a] * cos_t) * gain
            new_real[a] = r
            new_imag[a] = i

        # Apply neighbor coupling tunneling operator (tridiagonal circular coupling)
        coupled_real = list(new_real)
        coupled_imag = list(new_imag)
        for a in range(self.num_actions):
            prev_a = (a - 1) % self.num_actions
            next_a = (a + 1) % self.num_actions
            # Off-diagonal Hamiltonian tunneling -i * kappa * dt
            tunnel_r = -(new_imag[prev_a] + new_imag[next_a]) * self.coupling_strength * self.dt
            tunnel_i = (new_real[prev_a] + new_real[next_a]) * self.coupling_strength * self.dt
            coupled_real[a] += tunnel_r
            coupled_imag[a] += tunnel_i

        self.real = coupled_real
        self.imag = coupled_imag
        self._normalize()

    def probabilities(self) -> List[float]:
        """Born rule: p_a = |alpha_a|^2."""
        probs = [r * r + i * i for r, i in zip(self.real, self.imag)]
        total = sum(probs)
        if total <= 1e-12:
            return [1.0 / self.num_actions] * self.num_actions
        return [p / total for p in probs]

    def von_neumann_entropy(self) -> float:
        """S = -sum p_a * ln(p_a)."""
        probs = self.probabilities()
        entropy = 0.0
        for p in probs:
            if p > 1e-12:
                entropy -= p * math.log(p)
        return float(entropy)

    def interference_matrix(self) -> List[List[float]]:
        """Compute pairwise coherent superposition interference I(a, b) = 2 * Re(alpha_a* * alpha_b)."""
        matrix = [[0.0] * self.num_actions for _ in range(self.num_actions)]
        for a in range(self.num_actions):
            for b in range(self.num_actions):
                # alpha_a* * alpha_b = (r_a - i*i_a) * (r_b + i*i_b)
                # Re = r_a * r_b + i_a * i_b
                matrix[a][b] = 2.0 * (self.real[a] * self.real[b] + self.imag[a] * self.imag[b])
        return matrix

    def detect_destructive_conflicts(self, threshold: float = -0.15) -> List[Tuple[int, int, float]]:
        """Identify action pairs suffering from destructive interference below threshold."""
        conflicts = []
        matrix = self.interference_matrix()
        for a in range(self.num_actions):
            for b in range(a + 1, self.num_actions):
                if matrix[a][b] < threshold:
                    conflicts.append((a, b, matrix[a][b]))
        return conflicts

    def collapse(self, seed: int | None = None) -> Tuple[int, Dict[str, Any]]:
        """Collapse wavefunction via Born rule measurement and generate telemetry."""
        probs = self.probabilities()
        entropy = self.von_neumann_entropy()
        conflicts = self.detect_destructive_conflicts()

        # Deterministic or seeded pseudo-random selection
        if seed is not None:
            # Deterministic selection based on seed hashing
            h = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            u = (h % 1000000) / 1000000.0
            cumulative = 0.0
            chosen = self.num_actions - 1
            for a, p in enumerate(probs):
                cumulative += p
                if u <= cumulative:
                    chosen = a
                    break
        else:
            # Maximum-likelihood amplitude collapse
            chosen = int(max(range(self.num_actions), key=lambda a: probs[a]))

        telemetry = {
            "chosen_action": chosen,
            "probabilities": [round(p, 4) for p in probs],
            "von_neumann_entropy": round(entropy, 4),
            "coherence_length": round(math.exp(-entropy), 4),
            "destructive_conflicts_count": len(conflicts),
            "destructive_conflicts": [
                {"action_a": c[0], "action_b": c[1], "interference": round(c[2], 4)}
                for c in conflicts
            ],
            "amplitudes_real": [round(r, 4) for r in self.real],
            "amplitudes_imag": [round(i, 4) for i in self.imag],
        }

        # Partial post-measurement decoherence: concentrate 75% on chosen action, retain 25% superposition
        for a in range(self.num_actions):
            if a == chosen:
                self.real[a] = math.sqrt(0.75)
                self.imag[a] = 0.0
            else:
                self.real[a] = math.sqrt(0.25 / (self.num_actions - 1))
                self.imag[a] = 0.0
        self._normalize()

        return chosen, telemetry
