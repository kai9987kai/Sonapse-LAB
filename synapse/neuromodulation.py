"""Multi-factor biological neuromodulatory system for connectome intelligence.
Models four interacting neurochemical fields grounded in insect neurobiology:
1. Dopamine (DA): reward prediction error & positive reinforcement (LTP).
2. Serotonin (5-HT): harm anticipation, patience & risk mitigation.
3. Octopamine (OA): arousal, stress response & actuator gain amplification.
4. Acetylcholine (ACh): expected uncertainty & sensory attention gating.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

@dataclass
class NeuromodulatorState:
    dopamine: float = 0.0      # [-1, 1] RPE and valence
    serotonin: float = 0.2     # [0, 1] harm avoidance / patience
    octopamine: float = 0.2    # [0, 1] arousal / actuator vigor
    acetylcholine: float = 0.5 # [0, 1] sensory attention gate

    def to_vector(self) -> np.ndarray:
        return np.array([self.dopamine, self.serotonin, self.octopamine, self.acetylcholine], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {
            'dopamine': float(self.dopamine),
            'serotonin': float(self.serotonin),
            'octopamine': float(self.octopamine),
            'acetylcholine': float(self.acetylcholine),
        }


class NeuromodulatoryEngine(nn.Module):
    """Integrates multi-timescale neuromodulatory signals to modulate connectome dynamics,
    plasticity rates, and motor execution.
    """
    def __init__(
        self,
        decay_da: float = 0.70,
        decay_5ht: float = 0.90,
        decay_oa: float = 0.85,
        decay_ach: float = 0.80,
    ):
        super().__init__()
        self.decay_da = decay_da
        self.decay_5ht = decay_5ht
        self.decay_oa = decay_oa
        self.decay_ach = decay_ach

        self.state = NeuromodulatorState()
        self.v_predicted = 0.0
        self.history: List[Dict[str, float]] = []

    def reset(self):
        self.state = NeuromodulatorState()
        self.v_predicted = 0.0
        self.history.clear()

    def step(
        self,
        reward: float,
        sensory_contrast: float,
        threat_level: float,
        energy_depletion: float,
        gamma: float = 0.95,
    ) -> NeuromodulatorState:
        """Update neuromodulatory concentrations based on sensory telemetry and reward feedback."""
        # 1. Dopamine: Temporal Difference Reward Prediction Error
        # δ = r + γ * V(s') - V(s)
        # Approximate V(s') based on reward and resource preservation
        v_next = 0.8 * self.v_predicted + 0.2 * float(reward)
        rpe = float(reward) + gamma * v_next - self.v_predicted
        self.v_predicted = v_next
        new_da = self.decay_da * self.state.dopamine + (1.0 - self.decay_da) * np.clip(rpe, -1.0, 1.0)

        # 2. Serotonin: Harm anticipation and behavioral restraint
        new_5ht = self.decay_5ht * self.state.serotonin + (1.0 - self.decay_5ht) * np.clip(threat_level, 0.0, 1.0)

        # 3. Octopamine: Arousal, urgent metabolic stress, and fight-or-flight drive
        stress = float(energy_depletion + 0.5 * threat_level)
        new_oa = self.decay_oa * self.state.octopamine + (1.0 - self.decay_oa) * np.clip(stress, 0.0, 1.0)

        # 4. Acetylcholine: Expected uncertainty and sensory contrast gating
        new_ach = self.decay_ach * self.state.acetylcholine + (1.0 - self.decay_ach) * np.clip(sensory_contrast, 0.0, 1.0)

        self.state = NeuromodulatorState(
            dopamine=float(new_da),
            serotonin=float(new_5ht),
            octopamine=float(new_oa),
            acetylcholine=float(new_ach),
        )

        record = self.state.to_dict()
        record['rpe'] = float(rpe)
        self.history.append(record)
        if len(self.history) > 256:
            self.history.pop(0)

        return self.state

    def compute_gains(self) -> Dict[str, float]:
        """Compute modulation coefficients for downstream connectome and motor systems."""
        s = self.state
        # Octopamine boosts overall connectome excitability; serotonin exerts calming inhibition
        connectome_excitability = float(np.clip(1.0 + 0.6 * s.octopamine - 0.3 * s.serotonin, 0.5, 2.0))
        # Dopamine strongly scales synaptic plasticity rate
        plasticity_rate_multiplier = float(np.clip(1.0 + 1.5 * max(0.0, s.dopamine), 0.2, 3.0))
        # Acetylcholine gates top-down sensory projection weight
        sensory_attention_gate = float(np.clip(0.4 + 0.6 * s.acetylcholine, 0.1, 1.0))
        # Motor gain boosted by octopamine (arousal) and dampened by high serotonin (caution)
        motor_vigor = float(np.clip(1.0 + 0.5 * s.octopamine - 0.4 * s.serotonin, 0.3, 1.8))

        return {
            'connectome_excitability': connectome_excitability,
            'plasticity_multiplier': plasticity_rate_multiplier,
            'sensory_attention_gate': sensory_attention_gate,
            'motor_vigor': motor_vigor,
        }
