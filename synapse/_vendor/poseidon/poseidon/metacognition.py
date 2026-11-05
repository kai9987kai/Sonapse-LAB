"""POSEIDON METACOGNITION: Second-Order Introspection, Epistemic Humility & Dynamic Compute.

Inspired by introspective metacognitive monitoring (GhostInTheMachine):
1. Epistemic Volatility & Alert Status:
   Tracks a sliding temporal window of prediction surprises e_t and model disagreements U_t:
     \\sigma_e = \\sqrt{ \\frac{1}{W} \\sum_{\\tau=1}^W (e_\\tau - \\bar{e})^2 }
   When volatility spikes (e.g., during sudden weather storms or entering unmapped biomes),
   metacognition elevates the Epistemic Alert Level, widening safety margins and suppressing
   overconfident world model extrapolations.

2. Thermodynamic Compute Deliberation Budget:
   Biologically, deep cognitive imagination consumes metabolic resources (glucose/energy).
   Dynamically scales planning horizon depth based on physiological energy reserves:
     - Abundant Energy (energy > 0.5): Deep strategic imagination (H in {8, 16, 32}).
     - Depleted Energy (energy < 0.2): Fast tactical reflexes (H in {1, 4}), conserving stamina.

3. Second-Order Confidence Score:
   Audits the predictive reliability of the internal world model in real time:
     C_{meta} = \\exp(- \\alpha \\cdot \\text{ECE}_{window})
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class MetacognitionConfig:
    window_size: int = 16
    volatility_threshold_elevated: float = 0.15
    volatility_threshold_critical: float = 0.30
    energy_deep_horizon_threshold: float = 0.50
    stamina_shallow_horizon_threshold: float = 0.20


@dataclass
class MetacognitiveState:
    alert_level: str  # "nominal", "elevated", "critical"
    epistemic_volatility: float
    metacognitive_confidence: float
    recommended_horizon: int
    epistemic_caution_active: bool


class MetacognitiveMonitor:
    """Second-order introspective monitor tracking model validity and allocating deliberation compute."""

    def __init__(self, config: MetacognitionConfig | None = None):
        self.config = config or MetacognitionConfig()
        self.surprise_history: collections.deque[float] = collections.deque(maxlen=self.config.window_size)
        self.uncertainty_history: collections.deque[float] = collections.deque(maxlen=self.config.window_size)
        self.calibration_errors: collections.deque[float] = collections.deque(maxlen=self.config.window_size)
        self.step_counter: int = 0

    def register_transition(
        self,
        predicted_obs: Sequence[float] | None,
        realized_obs: Sequence[float],
        predicted_uncertainty: float,
    ) -> None:
        """Registers realized outcome to update introspective error tracking."""
        if predicted_obs is not None:
            # L2 Euclidean prediction surprise
            error = math.sqrt(sum((p - r) ** 2 for p, r in zip(predicted_obs, realized_obs)))
            self.surprise_history.append(error)
            self.uncertainty_history.append(predicted_uncertainty)

            # Calibration mismatch: |error - predicted_uncertainty|
            ece_step = abs(error - predicted_uncertainty)
            self.calibration_errors.append(ece_step)

        self.step_counter += 1

    def evaluate_state(self, current_obs: Sequence[float]) -> MetacognitiveState:
        """Evaluates volatility, sets alert status, and determines deliberation horizon."""
        energy = max(0.0, min(1.0, float(current_obs[1])))
        stamina = max(0.0, min(1.0, float(current_obs[3])))

        # 1. Epistemic Volatility
        if len(self.surprise_history) >= 3:
            mean_err = sum(self.surprise_history) / len(self.surprise_history)
            variance = sum((e - mean_err) ** 2 for e in self.surprise_history) / len(self.surprise_history)
            volatility = math.sqrt(variance)
        else:
            volatility = 0.0

        # 2. Alert Level Determination
        if volatility >= self.config.volatility_threshold_critical:
            alert = "critical"
            caution = True
        elif volatility >= self.config.volatility_threshold_elevated:
            alert = "elevated"
            caution = True
        else:
            alert = "nominal"
            caution = False

        # 3. Metacognitive Confidence
        if self.calibration_errors:
            mean_ece = sum(self.calibration_errors) / len(self.calibration_errors)
            confidence = math.exp(-3.0 * mean_ece)
        else:
            confidence = 1.0

        # 4. Thermodynamic Deliberation Horizon Allocation
        if energy > self.config.energy_deep_horizon_threshold and stamina > 0.30:
            recommended_h = 16  # Deep strategic planning
        elif energy < self.config.stamina_shallow_horizon_threshold or stamina < self.config.stamina_shallow_horizon_threshold:
            recommended_h = 4   # Shallow emergency reflex
        else:
            recommended_h = 8   # Standard operational horizon

        return MetacognitiveState(
            alert_level=alert,
            epistemic_volatility=float(volatility),
            metacognitive_confidence=float(confidence),
            recommended_horizon=recommended_h,
            epistemic_caution_active=caution,
        )
