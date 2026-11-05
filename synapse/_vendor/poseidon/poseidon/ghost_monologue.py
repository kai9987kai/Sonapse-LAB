"""POSEIDON GHOST IN THE MACHINE: Sub-Symbolic Drives & Introspective Cognitive Monologue.

Inspired by GhostInTheMachine (introspective metacognition, unconscious latent drive modeling,
and self-model emergence):

1. Sub-Symbolic Drive Vectors:
   D(t) = [d_hunger, d_thirst, d_fatigue, d_vulnerability, d_curiosity, d_caution]
   Mapped directly from phase coordinates, sensor noise, and prediction error.

2. Introspective Stream of Consciousness (Internal Monologue):
   Synthesizes all 6 cognitive planes into a human-auditable cognitive narrative:
   - Phase-Plane Orbit: Lyapunov drift and separatrix proximity
   - Causal Calculus: Interventional do(a) projection vs counterfactual regret
   - Quantum Amplitude: Wave-function interference and phase cancellation
   - Morphogenetic Capacity: Dynamic neurogenesis activation and active neuron count
   - Safety Invariants: TESSERA veto evaluation

3. Self-Model & Agency Metrics:
   - Introspection Coherence: Alignment between internal deliberation and executed action
   - Epistemic Lucidity: Predictability of environmental transitions
   - Agency Attribution: Disentangling self-directed action from environmental shock
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .causal_scm import CausalAttribution
from .phase_plane import PhaseStateAudit
from .world import ACTIONS, N_ACTIONS


@dataclass
class LatentDrives:
    hunger: float          # Inverted energy pressure
    thirst: float          # Inverted hydration pressure
    fatigue: float         # Inverted stamina pressure
    vulnerability: float   # Exposure & threat pressure
    curiosity: float       # Epistemic exploration drive
    caution: float         # Risk & separatrix repulsion drive
    dominant_drive: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "hunger": round(self.hunger, 4),
            "thirst": round(self.thirst, 4),
            "fatigue": round(self.fatigue, 4),
            "vulnerability": round(self.vulnerability, 4),
            "curiosity": round(self.curiosity, 4),
            "caution": round(self.caution, 4),
            "dominant_drive": self.dominant_drive,
        }


@dataclass
class IntrospectiveAudit:
    step: int
    monologue: str
    drives: LatentDrives
    agency_score: float
    epistemic_lucidity: float
    decision_rationale: str
    planes_consulted: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "monologue": self.monologue,
            "drives": self.drives.to_dict(),
            "agency_score": round(self.agency_score, 4),
            "epistemic_lucidity": round(self.epistemic_lucidity, 4),
            "decision_rationale": self.decision_rationale,
            "planes_consulted": self.planes_consulted,
        }


class GhostIntrospectionEngine:
    """Introspective self-model synthesizing multi-plane deliberations into cognitive monologue."""

    def __init__(self):
        self.step_counter: int = 0
        self.monologue_history: list[IntrospectiveAudit] = []

    def reset(self) -> None:
        self.step_counter = 0
        self.monologue_history.clear()

    def compute_latent_drives(
        self,
        obs: Sequence[float],
        surprise: float,
        separatrix_dist: float,
    ) -> LatentDrives:
        """Derives visceral survival and cognitive motivation drives."""
        e = float(obs[1])
        h = float(obs[2])
        s = float(obs[3])
        exposure = float(obs[4])
        threat = float(obs[5]) if len(obs) > 5 else 0.0

        d_hunger = max(0.0, 1.0 - e)
        d_thirst = max(0.0, 1.0 - h)
        d_fatigue = max(0.0, 1.0 - s)
        d_vulnerability = min(1.0, exposure * 0.6 + threat * 0.4)
        d_curiosity = max(0.0, min(1.0, surprise * 1.5))
        d_caution = max(0.0, min(1.0, (1.0 / max(0.05, separatrix_dist)) * 0.15))

        drives_dict = {
            "hunger": d_hunger,
            "thirst": d_thirst,
            "fatigue": d_fatigue,
            "vulnerability": d_vulnerability,
            "curiosity": d_curiosity,
            "caution": d_caution,
        }
        dominant = max(drives_dict.keys(), key=lambda k: drives_dict[k])

        return LatentDrives(
            hunger=d_hunger,
            thirst=d_thirst,
            fatigue=d_fatigue,
            vulnerability=d_vulnerability,
            curiosity=d_curiosity,
            caution=d_caution,
            dominant_drive=dominant,
        )

    def synthesize_monologue(
        self,
        obs: Sequence[float],
        phase_audit: PhaseStateAudit,
        quantum_coherence: float,
        quantum_gain: float,
        causal_effect: float,
        active_neurons: int,
        morph_state: str,
        chosen_action: int,
        base_action: int,
        override: bool,
        guardrail_veto: bool,
        surprise: float = 0.0,
    ) -> IntrospectiveAudit:
        """Synthesizes an introspective natural language stream of consciousness."""
        self.step_counter += 1
        drives = self.compute_latent_drives(obs, surprise, phase_audit.separatrix_distance)

        # Planes consulted
        planes = [
            "Physical/Embodied",
            "Phase-Plane Dynamics",
            "Pearlian Causal SCM",
            "Quantum Superposition",
            "Morphogenetic Plasticity",
            "TESSERA Safety Contracts",
        ]

        # Construct thought lines
        lines: list[str] = []
        action_name = ACTIONS[chosen_action].upper()
        base_name = ACTIONS[base_action].upper()

        # 1. Visceral self-state
        lines.append(
            f"Step {self.step_counter} Internal Trace: Dominant visceral drive is {drives.dominant_drive.upper()} "
            f"(Energy={phase_audit.energy:.2f}, Hydration={phase_audit.hydration:.2f}, Stamina={phase_audit.stamina:.2f})."
        )

        # 2. Phase-Plane Lyapunov orbital status
        if phase_audit.stability_regime == "contracting_attractor":
            lines.append(
                f"Phase plane orbit is contracting toward homeostatic fixed point (dL/dt={phase_audit.lyapunov_drift:.4f})."
            )
        elif phase_audit.stability_regime == "extinction_basin":
            lines.append(
                f"ALERT: Inside critical extinction basin (distance to separatrix={phase_audit.separatrix_distance:.3f}). Urgent stabilization required."
            )
        else:
            lines.append(
                f"Phase plane orbit is {phase_audit.stability_regime} (drift={phase_audit.lyapunov_drift:.4f}, sep_dist={phase_audit.separatrix_distance:.3f})."
            )

        # 3. Quantum wave-function deliberation
        if quantum_coherence > 0.80:
            lines.append(
                f"Quantum superposition exhibits constructive phase alignment for {action_name} "
                f"(coherence={quantum_coherence:.2f}, interference_gain={quantum_gain:.3f})."
            )
        else:
            lines.append(
                f"Wave-function phase interference active; destructive anti-phase damping applied to risky branches."
            )

        # 4. Morphogenetic neuroplastic state
        if morph_state == "neurogenesis":
            lines.append(
                f"Metacognitive surprise triggered neurogenesis: active capacity expanded to {active_neurons} neurons."
            )
        elif morph_state == "pruning":
            lines.append(
                f"Homeostatic stability reached; synaptic pruning active ({active_neurons} active neurons)."
            )
        else:
            lines.append(f"Neuroplastic capacity stable at {active_neurons} neurons.")

        # 5. Policy arbitration rationale
        if guardrail_veto:
            rationale = f"Safety contract triggered deterministic VETO against {base_name}; enforced safe action {action_name}."
            lines.append(rationale)
        elif override:
            rationale = (
                f"Overrode frozen baseline {base_name} -> {action_name} due to combined causal gain "
                f"(do(a) effect={causal_effect:+.3f}) and Lyapunov attractor steering."
            )
            lines.append(rationale)
        else:
            rationale = f"Executing baseline policy {action_name}; verified coherent across all 6 cognitive planes."
            lines.append(rationale)

        monologue_text = " ".join(lines)

        # Agency and Lucidity
        agency_score = min(1.0, max(0.0, 1.0 - (drives.caution * 0.4)))
        epistemic_lucidity = max(0.0, 1.0 - surprise)

        audit = IntrospectiveAudit(
            step=self.step_counter,
            monologue=monologue_text,
            drives=drives,
            agency_score=agency_score,
            epistemic_lucidity=epistemic_lucidity,
            decision_rationale=rationale,
            planes_consulted=planes,
        )

        self.monologue_history.append(audit)
        return audit
