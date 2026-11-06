"""Synapse-Specific Metaplasticity & 4-Factor Neurochemical Regulation.

1. Synaptic Metaplasticity: Assigns each individual connection an adaptive learning rate
   eta_{ij}(t), a sliding modification threshold theta_{ij}(t), and a consolidation tag c_{ij}(t).
2. Neurochemical Learning Regulation: Extends DA, 5-HT, OA, ACh into independent functional roles:
   - Dopamine (DA): Synaptic consolidation gate & LTP eligibility threshold.
   - Serotonin (5-HT): Behavioral persistence & impulsive switching suppression.
   - Octopamine (OA): Motor vigor & exploratory noise injection.
   - Acetylcholine (ACh): Sensory attention sharpness & top-down predictive precision.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

@dataclass
class SynapticTagState:
    adaptive_lr: np.ndarray
    sliding_threshold: np.ndarray
    consolidation_tags: np.ndarray  # 0.0 (labile) to 1.0 (consolidated/protected)

class MetaplasticityEngine:
    """Manages connection-specific adaptive learning rates, thresholds, and tags."""

    def __init__(self, num_synapses: int, base_lr: float = 0.01):
        self.num_synapses = num_synapses
        self.base_lr = base_lr
        self.state = SynapticTagState(
            adaptive_lr=np.full(num_synapses, base_lr, dtype=np.float32),
            sliding_threshold=np.zeros(num_synapses, dtype=np.float32),
            consolidation_tags=np.zeros(num_synapses, dtype=np.float32)
        )

    def reset(self):
        self.state.adaptive_lr.fill(self.base_lr)
        self.state.sliding_threshold.fill(0.0)
        self.state.consolidation_tags.fill(0.0)

    def update_metaplasticity(
        self,
        synapse_weights: np.ndarray,
        coactivations: np.ndarray,
        dopamine: float = 0.0,
        serotonin: float = 0.2,
        acetylcholine: float = 0.5
    ) -> np.ndarray:
        """Update adaptive learning rates and return effective synaptic weight adjustments.

        Consolidated synapses (high tag c) resist modification, preventing catastrophic forgetting.
        """
        n = min(len(synapse_weights), self.num_synapses)
        coact = coactivations[:n]
        weights = synapse_weights[:n]

        # 1. Sliding BCM-style threshold per connection
        self.state.sliding_threshold[:n] = 0.95 * self.state.sliding_threshold[:n] + 0.05 * (coact ** 2)

        # 2. Adaptive learning rate modulated by acetylcholine attention
        ach_boost = 1.0 + 0.5 * (acetylcholine - 0.5)
        # Learning rate slows down when weights grow large (homeostatic protection)
        saturation = 1.0 / (1.0 + np.abs(weights))
        self.state.adaptive_lr[:n] = np.clip(self.base_lr * ach_boost * saturation, 1e-4, 0.1)

        # 3. Synaptic tagging and capture modulated by dopamine
        # High dopamine consolidates active synapses into protected long-term storage
        da_signal = max(0.0, dopamine)
        self.state.consolidation_tags[:n] = np.clip(
            self.state.consolidation_tags[:n] * 0.99 + 0.1 * da_signal * np.abs(coact),
            0.0, 1.0
        )

        # 4. Plasticity effective scale: labile synapses adapt freely, consolidated ones are preserved
        plasticity_scale = (1.0 - 0.8 * self.state.consolidation_tags[:n])
        effective_lr = self.state.adaptive_lr[:n] * plasticity_scale

        return effective_lr

def test_neurochemical_isolation(agent, neurotransmitter: str) -> Dict[str, Any]:
    """Test the isolated causal effect of a single neurotransmitter mechanism."""
    valid = ['dopamine', 'serotonin', 'octopamine', 'acetylcholine']
    if neurotransmitter not in valid:
        raise ValueError(f'Invalid neurotransmitter {neurotransmitter}; must be in {valid}')

    # Establish baseline
    obs = np.linspace(0.1, 0.9, 155, dtype=np.float32)
    with torch.no_grad():
        base_res = agent.act(obs, learn=False)
        base_action = base_res['action']

    # Perturb target neurotransmitter in isolation
    state_before = getattr(agent.neuromod.state, neurotransmitter)
    perturbed_val = 1.0 if state_before < 0.5 else 0.0
    setattr(agent.neuromod.state, neurotransmitter, perturbed_val)

    with torch.no_grad():
        perturbed_res = agent.act(obs, learn=False)
        perturbed_action = perturbed_res['action']

    # Restore original state
    setattr(agent.neuromod.state, neurotransmitter, state_before)

    action_delta = float(np.linalg.norm(base_action - perturbed_action))
    mechanisms = {
        'dopamine': 'reward-prediction error gating and synaptic consolidation tagging',
        'serotonin': 'behavioral persistence, threat mitigation, and temporal patience',
        'octopamine': 'locomotor vigor, active exploration, and energetic mobilization',
        'acetylcholine': 'sensory contrast enhancement and top-down precision weighting'
    }

    return {
        'status': 'ok',
        'neurotransmitter': neurotransmitter,
        'baseline_value': float(state_before),
        'perturbed_value': float(perturbed_val),
        'isolated_action_delta': action_delta,
        'causally_distinct': action_delta > 1e-4,
        'specific_mechanism': mechanisms[neurotransmitter],
        'scope': 'factorial isolated neurochemical regulation test'
    }
