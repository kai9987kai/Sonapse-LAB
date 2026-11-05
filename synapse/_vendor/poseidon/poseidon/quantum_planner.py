"""POSEIDON QUANTUM-PLANNER: Quantum-Inspired Superposition Amplitude Tree Search.

Inspired by QuantumBot (probabilistic decision trees, amplitude superposition):
1. Quantum Amplitude Superposition:
   Represents future candidate pathways not as single scalar values, but as
   complex probability amplitudes:
     \\psi(s, a) = r(s, a) \\cdot e^{i \\theta(s, a)}
   where:
     - Modulus r(s, a) \\in [0, 1] captures pragmatic expected survival return
     - Phase \\theta(s, a) \\in [-\\pi, \\pi] captures epistemic uncertainty and risk phase

2. Wave-Function Interference of Futures:
   When rolling out multi-step trajectory ensembles:
     \\Psi_{total}(a) = \\sum_{\\tau} w_\\tau \\cdot \\psi(s_\\tau, a_\\tau)
   - Constructive Interference: Convergent, robustly surviving pathways align in phase,
     amplifying the probability of execution (|\\sum c_i|^2 >> \\sum |c_i|^2).
   - Destructive Interference: Ambiguous, high-variance hazard pathways have opposing
     phases, naturally canceling each other out without ad-hoc risk heuristics.

3. Amplitude Amplification (Grover-Inspired Rotation):
   Rotates the composite superposition towards high-coherence, low-entropy actions.
"""
from __future__ import annotations

import cmath
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch

from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass
class QuantumActionBranch:
    action: int
    action_name: str
    amplitude: complex
    modulus: float
    phase_rad: float
    interference_gain: float
    survival_coherence: float


class QuantumSuperpositionPlanner:
    """Quantum-inspired multi-horizon amplitude planner with wave-function interference."""

    def __init__(
        self,
        phase_uncertainty_scale: float = math.pi,
        coherence_temperature: float = 1.0,
        grover_amplification_steps: int = 1,
        enabled: bool = True,
    ):
        self.phase_uncertainty_scale = phase_uncertainty_scale
        self.coherence_temperature = coherence_temperature
        self.grover_amplification_steps = grover_amplification_steps
        self.enabled = enabled

    def encode_amplitude(
        self,
        predicted_health: float,
        predicted_energy: float,
        predicted_hydration: float,
        uncertainty: float,
    ) -> complex:
        """Encodes state predictions into a complex probability amplitude psi = r * e^(i * theta)."""
        # Vital survival magnitude r in [0.01, 1.0]
        vitals = max(0.01, min(1.0, predicted_health * math.sqrt(max(0.01, min(1.0, predicted_energy * predicted_hydration)))))
        r = vitals

        # Epistemic phase theta in [-pi, pi]: zero uncertainty -> theta=0 (pure real constructive phase)
        # High uncertainty -> phase rotates towards pi (anti-phase cancellation)
        theta = min(math.pi, max(-math.pi, uncertainty * self.phase_uncertainty_scale))
        return cmath.rect(r, theta)

    def plan_superposition(
        self,
        candidate_evaluations: Sequence[dict[str, Any]],
        calibrated_uncertainties: Sequence[float],
        base_action: int,
    ) -> dict[str, Any]:
        """Performs quantum wave-function interference across candidate actions."""
        branches: list[QuantumActionBranch] = []
        raw_amplitudes: list[complex] = []

        for act_idx in range(len(candidate_evaluations)):
            c_eval = candidate_evaluations[act_idx]
            unc = calibrated_uncertainties[act_idx] if act_idx < len(calibrated_uncertainties) else 0.05
            p_health = c_eval.get("predicted_health", 0.9)
            p_energy = c_eval.get("predicted_energy", 0.8)
            p_hydration = c_eval.get("predicted_hydration", 0.8)

            amp = self.encode_amplitude(p_health, p_energy, p_hydration, unc)
            raw_amplitudes.append(amp)

        # Compute mean superposition amplitude |Psi_mean> for Grover reflection
        mean_amp = sum(raw_amplitudes) / len(raw_amplitudes)

        # Amplitude amplification: 2 * |Psi_mean> - psi_i (reflection around mean)
        amplified = []
        for amp in raw_amplitudes:
            if self.grover_amplification_steps > 0:
                amp_rot = 2.0 * mean_amp - amp
            else:
                amp_rot = amp
            amplified.append(amp_rot)

        # Calculate interference gains
        for act_idx, amp in enumerate(amplified):
            modulus = abs(amp)
            phase = cmath.phase(amp)
            # Constructive coherence: cos(phase) > 0 aligns with real axis
            coherence = max(0.0, math.cos(phase)) if self.enabled else 0.0
            interference_gain = ((modulus ** 2) * coherence) if self.enabled else 0.0

            branches.append(
                QuantumActionBranch(
                    action=act_idx,
                    action_name=ACTIONS[act_idx],
                    amplitude=amp,
                    modulus=float(modulus),
                    phase_rad=float(phase),
                    interference_gain=float(interference_gain),
                    survival_coherence=float(coherence),
                )
            )

        # Normalize execution probability distribution
        total_gain = sum(b.interference_gain for b in branches)
        probs = [b.interference_gain / total_gain if total_gain > 0 else 1.0 / len(branches) for b in branches]

        best_branch = max(branches, key=lambda b: b.interference_gain)
        base_branch = branches[base_action]

        return {
            "selected_action": best_branch.action,
            "selected_action_name": best_branch.action_name,
            "best_interference_gain": best_branch.interference_gain,
            "base_interference_gain": base_branch.interference_gain,
            "coherence_margin": best_branch.interference_gain - base_branch.interference_gain,
            "action_probabilities": [round(p, 4) for p in probs],
            "branches": [
                {
                    "action": b.action,
                    "action_name": b.action_name,
                    "modulus": round(b.modulus, 4),
                    "phase_deg": round(math.degrees(b.phase_rad), 1),
                    "coherence": round(b.survival_coherence, 4),
                    "interference_gain": round(b.interference_gain, 4),
                }
                for b in branches
            ],
        }
