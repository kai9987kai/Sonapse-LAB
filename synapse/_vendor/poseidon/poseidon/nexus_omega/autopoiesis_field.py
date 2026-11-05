"""Stage 11 Thermodynamic Autopoiesis Field and PAD Affect Valence Engine.

Computes continuous nonequilibrium thermodynamic entropy dissipation rates
and 3-Axis Pleasure-Arousal-Dominance (PAD) emotional affect vectors to dynamically
modulate active inference curiosity temperatures and conformal risk boundaries.

Inspired by prometheus-alpha (synthetic entropy & drive heuristics) and
GhostInTheMachine (autopoiesis & homeostatic PAD affect).
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional
import numpy as np

from .contracts import ThermodynamicState


class AutopoiesisThermodynamicField:
    """Nonequilibrium dissipation and PAD affective valence engine."""

    def __init__(
        self,
        base_curiosity_temp: float = 0.80,
        base_risk_aversion: float = 1.0,
    ) -> None:
        self.beta_0 = base_curiosity_temp
        self.lambda_0 = base_risk_aversion

        # Running history for temporal derivative dS/dt
        self.prev_entropy: float = 0.5
        self.prev_vitals: float = 0.8

    def reset(self) -> None:
        self.prev_entropy = 0.5
        self.prev_vitals = 0.8

    def update(self, *args, **kwargs) -> ThermodynamicState:
        obs = kwargs.get("obs") if "obs" in kwargs else (args[0] if args else [1.0, 0.9, 0.9, 0.8])
        epistemic = kwargs.get("epistemic_uncertainty", 0.05)
        return self.evaluate_state(obs, epistemic_surprise=epistemic)

    def evaluate_state(
        self,
        observation: np.ndarray | Sequence[float],
        epistemic_surprise: float = 0.05,
        empowerment: float = 0.50,
    ) -> ThermodynamicState:
        """Evaluates thermodynamic flux, PAD valence, and dynamic active inference scales."""
        obs = np.asarray(observation, dtype=np.float32)

        health = float(obs[0]) if len(obs) > 0 else 1.0
        energy = float(obs[1]) if len(obs) > 1 else 1.0
        hydration = float(obs[2]) if len(obs) > 2 else 1.0
        exposure = float(obs[4]) if len(obs) > 4 else 0.0

        mean_vitals = (health + energy + hydration) / 3.0
        min_vital = min(health, energy, hydration)

        # 1. Autopoietic Viability Distance: distance from lethal extinction boundary
        autopoietic_integrity = max(0.0, min(1.0, min_vital * (1.0 - exposure * 0.5)))

        # 2. Thermodynamic Entropy Dissipation
        # Internal entropy increases as vitals degrade or exposure grows
        current_internal_entropy = float(
            1.0 - mean_vitals + 0.5 * exposure + 0.3 * epistemic_surprise
        )
        entropy_rate = float(current_internal_entropy - self.prev_entropy)
        self.prev_entropy = current_internal_entropy

        # Heat dissipation: negative rate of vital depletion
        heat_dissipation = float(max(0.0, (self.prev_vitals - mean_vitals) * 2.0 + 0.05))
        self.prev_vitals = mean_vitals

        # 3. 3-Axis PAD Affect Valence
        # Pleasure: surplus above basic subsistence threshold (0.60)
        pleasure = float(math.tanh(2.5 * (mean_vitals - 0.65)))

        # Arousal: surprise intensity and environmental hazard
        arousal_drive = epistemic_surprise * 3.0 + exposure * 1.5 - 0.4
        arousal = float(math.tanh(arousal_drive))

        # Dominance: perceived agency / empowerment and vital robustness
        dominance_drive = (empowerment - 0.45) * 2.5 + (health - 0.5) * 1.5
        dominance = float(math.tanh(dominance_drive))

        # 4. Modulate Active Inference Scales via PAD
        # High Pleasure + Dominance -> curiosity blooms (secure exploration)
        # Low Pleasure + High Arousal (panic) -> curiosity extinguished, risk aversion spikes
        p_factor = 1.0 + 0.5 * pleasure
        d_factor = 1.0 + 0.3 * dominance
        a_suppression = 1.0 - 0.6 * max(0.0, arousal - 0.2) if pleasure < 0.0 else 1.0

        effective_beta = float(max(0.02, self.beta_0 * p_factor * d_factor * a_suppression))

        # Risk aversion escalates under threat (negative pleasure, high arousal)
        risk_escalation = 1.0 + 1.5 * max(0.0, -pleasure) + 1.0 * max(0.0, arousal) - 0.3 * max(0.0, dominance)
        effective_lambda = float(max(0.20, self.lambda_0 * risk_escalation))

        return ThermodynamicState(
            entropy_production_rate=entropy_rate,
            heat_dissipation_rate=heat_dissipation,
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            effective_curiosity_temp=effective_beta,
            effective_risk_aversion=effective_lambda,
            autopoietic_integrity=autopoietic_integrity,
        )
