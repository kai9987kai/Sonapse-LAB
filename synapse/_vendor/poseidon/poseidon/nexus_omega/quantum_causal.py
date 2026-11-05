"""Stage 11 Quantum Holographic Superposition Kernel.

Represents macro-intent cognitive options not as a collapsed classical choice,
but as a holonomic quantum state vector in Hilbert space C^4 with unitary
Hamiltonian Free Energy evolution and geometric Berry phase holonomy.

Inspired by QuantumBot (interferometer phase shifts), Causeway (wave function collapse),
and FLY-DIAMOND-NEXUS (holographic topological phase lattice).
"""
from __future__ import annotations

import cmath
import math
from typing import List, Sequence, Tuple, Optional
import numpy as np

from .contracts import MacroIntent, QuantumCausalState


class QuantumCausalKernel:
    """Holonomic quantum superposition engine for macro-intent arbitration."""

    INTENT_ORDER = [
        MacroIntent.FORAGE_EXPEDITION,
        MacroIntent.DIAPAUSE_HIBERNATION,
        MacroIntent.HABITAT_RELOCATION,
        MacroIntent.EMERGENCY_FORTIFICATION,
    ]

    def __init__(
        self,
        tunneling_rate: float = 0.15,
        dephasing_rate: float = 0.02,
        collapse_threshold_purity: float = 0.85,
    ) -> None:
        self.dim = 4
        self.tunneling = tunneling_rate
        self.dephasing = dephasing_rate
        self.collapse_threshold = collapse_threshold_purity

        # Initialize in balanced superposition with zero phase
        init_amp = 1.0 / math.sqrt(self.dim)
        self.psi = np.array([complex(init_amp, 0.0) for _ in range(self.dim)], dtype=complex)
        self.prev_psi = self.psi.copy()

        # Accumulated Berry Phase (holonomic geometric phase)
        self.accumulated_berry_phase: float = 0.0
        self.time_step: int = 0

    def reset(self) -> None:
        init_amp = 1.0 / math.sqrt(self.dim)
        self.psi = np.array([complex(init_amp, 0.0) for _ in range(self.dim)], dtype=complex)
        self.prev_psi = self.psi.copy()
        self.accumulated_berry_phase = 0.0
        self.time_step = 0

    def construct_hamiltonian(self, free_energies: Sequence[float]) -> np.ndarray:
        """Constructs 4x4 Hermitian Hamiltonian from Expected Free Energies & tunneling couplings."""
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # Diagonal entries are Expected Free Energies (bounded to avoid numerical instability)
        for i in range(self.dim):
            fe = float(free_energies[i]) if i < len(free_energies) else 1.0
            H[i, i] = complex(max(-10.0, min(10.0, fe)), 0.0)

        # Off-diagonal tunneling couplings between complementary intent options:
        # Forage (0) <-> Relocate (2)
        H[0, 2] = complex(self.tunneling, 0.0)
        H[2, 0] = complex(self.tunneling, 0.0)

        # Diapause (1) <-> Fortify (3)
        H[1, 3] = complex(self.tunneling * 0.75, 0.0)
        H[3, 1] = complex(self.tunneling * 0.75, 0.0)

        # Emergency cross-coupling under high threat: Forage (0) <-> Fortify (3)
        H[0, 3] = complex(self.tunneling * 0.25, 0.0)
        H[3, 0] = complex(self.tunneling * 0.25, 0.0)

        return H

    def evolve_step(
        self,
        macro_free_energies: Sequence[float],
        dt: float = 0.20,
    ) -> QuantumCausalState:
        """Performs unitary time evolution |psi(t+dt)> = exp(-i * H * dt) |psi(t)> and dephasing."""
        self.time_step += 1
        self.prev_psi = self.psi.copy()

        H = self.construct_hamiltonian(macro_free_energies)

        # Matrix exponential via eigendecomposition: exp(-i * H * dt)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * dt)) @ eigenvectors.conj().T

        # Unitary propagation
        new_psi = U @ self.psi

        # Apply mild environmental dephasing towards lowest free-energy state
        min_fe_idx = int(np.argmin(macro_free_energies))
        for i in range(self.dim):
            if i != min_fe_idx:
                new_psi[i] *= math.exp(-self.dephasing)

        # Re-normalize state vector: ||psi|| = 1
        norm = np.linalg.norm(new_psi)
        if norm > 1e-9:
            self.psi = new_psi / norm
        else:
            self.psi = np.array([complex(1.0 / math.sqrt(self.dim), 0.0)] * self.dim, dtype=complex)

        # Compute Berry Phase step: Im <psi(t) | psi(t+dt) - psi(t)>
        overlap = np.vdot(self.prev_psi, self.psi - self.prev_psi)
        berry_delta = float(overlap.imag)
        self.accumulated_berry_phase = (self.accumulated_berry_phase + berry_delta) % (2.0 * math.pi)
        if self.accumulated_berry_phase > math.pi:
            self.accumulated_berry_phase -= 2.0 * math.pi

        # Compute Born rule probabilities: P_m = |c_m|^2
        probabilities = [float(abs(c) ** 2) for c in self.psi]
        total_p = sum(probabilities)
        probabilities = [p / total_p for p in probabilities]

        # Compute phase angles
        phases = [float(cmath.phase(c)) for c in self.psi]

        # Density matrix rho = |psi><psi|
        # Purity Tr(rho^2) = sum |c_i|^4 for pure state (here with partial decoherence)
        purity = float(sum(p ** 2 for p in probabilities))

        # Von Neumann Entropy: S_vN = - sum P_i log P_i
        vn_entropy = -float(sum(p * math.log(max(1e-12, p)) for p in probabilities))

        return QuantumCausalState(
            amplitudes=list(self.psi),
            probabilities=probabilities,
            phases=phases,
            berry_phase=self.accumulated_berry_phase,
            purity=purity,
            von_neumann_entropy=vn_entropy,
            tunneling_rate=self.tunneling,
        )

    def collapse_state(
        self,
        rng: Optional[np.random.RandomState] = None,
        force_index: Optional[int] = None,
    ) -> Tuple[MacroIntent, int]:
        """Applies quantum measurement / Born rule collapse operator."""
        if force_index is not None and 0 <= force_index < self.dim:
            chosen_idx = force_index
        else:
            probs = [float(abs(c) ** 2) for c in self.psi]
            total = sum(probs)
            probs = [p / total for p in probs]
            r = rng.rand() if rng else float(np.random.rand())
            cumulative = 0.0
            chosen_idx = 0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen_idx = i
                    break

        # Post-measurement projection: |psi> = |chosen_idx>
        self.psi = np.zeros(self.dim, dtype=complex)
        self.psi[chosen_idx] = complex(1.0, 0.0)

        return self.INTENT_ORDER[chosen_idx], chosen_idx

    def get_highest_probability_intent(self) -> Tuple[MacroIntent, float]:
        """Returns intent with maximum Born probability without forcing collapse."""
        probs = [float(abs(c) ** 2) for c in self.psi]
        best_idx = int(np.argmax(probs))
        return self.INTENT_ORDER[best_idx], probs[best_idx]
