"""POSEIDON AUTOPOIESIS: Autopoietic Self-Maintenance, Affective Appraisal & Intrinsic Curiosity.

Synthesizes GhostInTheMachine, Morpheus, and Prometheus-Alpha:
1. Autopoietic Living System Theory (Maturana & Varela):
   Maintains organizational continuity and negentropy against environmental dissipation:
     dS_org / dt <= 0 through active metabolic ingestion and oneiric consolidation.
2. PAD Affective Appraisal Model:
   - Valence / Pleasure V in [-1, +1]: Rate of homeostatic convergence d(Phi_vital)/dt.
   - Arousal A in [0, 1]: Magnitude of environmental sensory surprise / epistemic uncertainty.
   - Dominance D in [-1, +1]: Perceived causal agency and distance to mortality separatrix.
3. Intrinsic Epistemic Curiosity Drive:
   Generates intrinsic reward for actions that maximize expected information gain:
     G_epistemic(a) = E_q[ D_KL( q(theta | s', a) || p(theta) ) ]
4. Spontaneous Micro-Sleep & Synaptic Self-Repair:
   Automatically downscales synaptic fatigue and resets cognitive entropy when cognitive drift exceeds threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence


@dataclass
class PADAffectiveState:
    """Psychophysiological PAD (Pleasure, Arousal, Dominance) affective core."""
    valence: float = 0.0      # [-1, +1] (Negative distress to positive flourishing)
    arousal: float = 0.2      # [0, 1]   (Calm / vegetative to heightened alertness)
    dominance: float = 0.5    # [-1, +1] (Helplessness to complete sovereign control)
    mood_label: str = "serene"

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "dominance": round(self.dominance, 4),
            "mood_label": self.mood_label,
        }


@dataclass
class AutopoieticAudit:
    """State audit of the organism's autopoietic organizational health."""
    negentropy_flux: float
    cumulative_cognitive_entropy: float
    curiosity_drive: float
    micro_sleep_active: bool
    affect: PADAffectiveState
    hypotheses_generated: list[str]


class AutopoieticEngine:
    """Engine managing autopoietic homeostasis, PAD affect, curiosity, and self-repair."""

    def __init__(self, entropy_threshold_sleep: float = 1.8):
        self.entropy_threshold_sleep = entropy_threshold_sleep
        self.prev_vitals: list[float] = [1.0, 1.0, 1.0, 1.0]  # [h, e, w, s]
        self.cumulative_entropy: float = 0.0
        self.total_sleep_resets: int = 0
        self.epistemic_history: list[float] = []

    def update(
        self,
        current_obs: list[float],
        predicted_obs: list[float] | None = None,
        separatrix_distance: float = 0.8,
    ) -> AutopoieticAudit:
        """Computes PAD affective update, epistemic curiosity, and autopoietic entropy budget."""
        h = current_obs[0] if len(current_obs) > 0 else 1.0
        e = current_obs[1] if len(current_obs) > 1 else 1.0
        w = current_obs[2] if len(current_obs) > 2 else 1.0
        s = current_obs[3] if len(current_obs) > 3 else 1.0
        curr_v = [h, e, w, s]

        # 1. Valence: Homeostatic vital trajectory delta
        vital_diffs = sum((c - p) for c, p in zip(curr_v, self.prev_vitals))
        valence = math.tanh(vital_diffs * 4.0)

        # 2. Arousal: Environmental surprise
        surprise = 0.1
        if predicted_obs is not None and len(predicted_obs) >= 4:
            err = math.sqrt(sum((c - p)**2 for c, p in zip(current_obs[:4], predicted_obs[:4])))
            surprise = min(1.0, err * 3.0)
        arousal = min(1.0, max(0.0, surprise + (0.5 * (1.0 - min(e, w)))))

        # 3. Dominance: Sovereignty / Clearance from death boundary
        dominance = max(-1.0, min(1.0, 2.0 * separatrix_distance - 1.0))

        # Mood taxonomy derivation
        mood = self._derive_mood(valence, arousal, dominance)
        affect = PADAffectiveState(valence=valence, arousal=arousal, dominance=dominance, mood_label=mood)

        # 4. Negentropy flux & Cognitive Entropy Accumulation
        # Organism generates entropy from high arousal or low vitals
        negentropy_flux = (valence * 0.5) - (arousal * 0.3)
        step_entropy = arousal * 0.08 + (0.05 if min(e, w) < 0.2 else 0.01)
        self.cumulative_entropy += step_entropy

        # 5. Spontaneous Micro-Sleep Self-Repair Trigger
        micro_sleep = False
        if self.cumulative_entropy >= self.entropy_threshold_sleep:
            micro_sleep = True
            self.cumulative_entropy *= 0.15  # Downscale synaptic fatigue by 85%
            self.total_sleep_resets += 1

        # 6. Intrinsic Curiosity Drive (Information Gain)
        # Higher curiosity when arousal is moderate and dominance is high
        curiosity = max(0.0, (1.0 - abs(arousal - 0.45)) * max(0.1, dominance))
        self.epistemic_history.append(curiosity)

        # Hypotheses formulation
        hypotheses = []
        if e < 0.3:
            hypotheses.append("Hypothesis: Algal patch foraging will restore energetic valence.")
        if w < 0.3:
            hypotheses.append("Hypothesis: Tidal spring hydration is required to avoid separatrix collapse.")
        if arousal > 0.6:
            hypotheses.append("Hypothesis: Environmental surprise indicates unmapped predator territory.")
        if not hypotheses:
            hypotheses.append("Hypothesis: Stationary basking will preserve autopoietic negentropy.")

        self.prev_vitals = curr_v

        return AutopoieticAudit(
            negentropy_flux=round(negentropy_flux, 4),
            cumulative_cognitive_entropy=round(self.cumulative_entropy, 4),
            curiosity_drive=round(curiosity, 4),
            micro_sleep_active=micro_sleep,
            affect=affect,
            hypotheses_generated=hypotheses,
        )

    def _derive_mood(self, v: float, a: float, d: float) -> str:
        """Classifies emotional quadrant in the 3D PAD space."""
        if v > 0.3:
            if a > 0.4:
                return "exuberant_play" if d > 0.0 else "elated_receptivity"
            else:
                return "serene_contentment" if d > 0.0 else "tranquil_rest"
        elif v < -0.3:
            if a > 0.4:
                return "defensive_rage" if d > 0.0 else "acute_terror"
            else:
                return "morose_frustration" if d > 0.0 else "catatonic_despair"
        else:
            return "vigilant_curiosity" if a > 0.4 else "homeostatic_equilibrium"
