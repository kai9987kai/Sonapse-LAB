"""PROMETHEUS: Meta-Plasticity & Homeostatic Drive Self-Regulation.

Synthesizes online non-gradient meta-parameter adaptation and physiological allostasis
for the TidePool agent, dynamically tuning Causeway quantum gains and holographic memory
retention based on rolling scarcity entropy.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Deque, Dict, List, Optional
import numpy as np


class PrometheusPlasticityEngine:
    """Online meta-plastic adaptation and homeostatic drive balancing engine."""

    def __init__(
        self,
        base_causeway_gain: float = 0.50,
        base_hologram_decay: float = 0.95,
        window_size: int = 16,
        entropy_gain_sensitivity: float = 0.35,
    ):
        self.base_gain = float(base_causeway_gain)
        self.base_decay = float(base_hologram_decay)
        self.window_size = int(window_size)
        self.sensitivity = float(entropy_gain_sensitivity)

        self.observation_window: Deque[List[float]] = deque(maxlen=self.window_size)
        self.recent_entropies: Deque[float] = deque(maxlen=self.window_size)
        self.step_count = 0

    def reset(self) -> None:
        """Reset Prometheus buffers and internal states."""
        self.observation_window.clear()
        self.recent_entropies.clear()
        self.step_count = 0

    def compute_scarcity_entropy(self) -> float:
        """Compute Shannon entropy across recent observation resource indicators."""
        if len(self.observation_window) < 3:
            return 0.50

        # Extract food, water, stamina features from 16-d observation
        # Indices in TidePool obs: [0: food, 1: water, 2: health, 3: stamina, ...]
        indicators = []
        for obs in self.observation_window:
            if len(obs) >= 4:
                val = float(obs[0] + obs[1] + obs[3]) / 3.0
                indicators.append(max(0.01, min(0.99, val)))
            else:
                indicators.append(0.5)

        hist, _ = np.histogram(indicators, bins=4, range=(0.0, 1.0), density=True)
        # Normalize to probability distribution
        p = hist / (np.sum(hist) + 1e-9)
        entropy = -float(np.sum([pi * math.log2(pi + 1e-9) for pi in p if pi > 1e-6]))
        return max(0.0, min(2.0, entropy))

    def evaluate(self, obs: List[float]) -> Dict[str, Any]:
        """Update homeostatic drives and return meta-plastically adapted parameters."""
        self.step_count += 1
        self.observation_window.append(list(obs))

        # 1. Compute rolling scarcity entropy
        entropy = self.compute_scarcity_entropy()
        self.recent_entropies.append(entropy)

        # 2. Compute adapted Causeway non-Hermitian gain
        # When environment is scarce/volatile, increase gain to punch through local minima
        adapted_gain = self.base_gain * (1.0 + self.sensitivity * entropy)
        adapted_gain = max(0.10, min(1.50, adapted_gain))

        # 3. Compute adapted holographic retention decay
        # During high entropy/volatility, accelerate decay to prevent stale associations
        volatility = float(np.std(self.recent_entropies)) if len(self.recent_entropies) > 1 else 0.0
        adapted_decay = self.base_decay * (1.0 - 0.15 * min(1.0, volatility))
        adapted_decay = max(0.70, min(0.99, adapted_decay))

        # 4. Compute homeostatic physiological drives
        food = float(obs[0]) if len(obs) > 0 else 0.5
        water = float(obs[1]) if len(obs) > 1 else 0.5
        health = float(obs[2]) if len(obs) > 2 else 1.0
        stamina = float(obs[3]) if len(obs) > 3 else 1.0
        storm_threat = float(obs[13]) if len(obs) > 13 else 0.0

        hunger_drive = max(0.0, 1.0 - food)
        thirst_drive = max(0.0, 1.0 - water)
        fatigue_drive = max(0.0, 1.0 - stamina)
        exposure_drive = storm_threat if health < 0.8 else storm_threat * 0.5

        drives = {
            "hunger": round(hunger_drive, 4),
            "thirst": round(thirst_drive, 4),
            "fatigue": round(fatigue_drive, 4),
            "exposure": round(exposure_drive, 4),
        }
        dominant_drive = max(drives.items(), key=lambda item: item[1])[0]
        homeostatic_urgency = float(max(drives.values()))

        return {
            "scarcity_entropy": round(entropy, 4),
            "adapted_gain": round(adapted_gain, 4),
            "adapted_decay": round(adapted_decay, 4),
            "homeostatic_urgency": round(homeostatic_urgency, 4),
            "dominant_drive": dominant_drive,
            "drives": drives,
        }
