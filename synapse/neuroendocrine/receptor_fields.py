"""Synthetic Receptor Bank.

Maps 12-channel synthetic gland deltas Delta g_{k, t} = g_{k, t} - g_{k, 0}
into target recipient gains through biologically-grounded affinity matrix R:
gamma_{j, t} = gamma_{j, 0} + sum_k R_{jk} Delta g_{k, t}

Recipient target properties:
1. excitability (membrane firing threshold multiplier)
2. plasticity (synaptic learning rate / STDP multiplier)
3. sensory_gating (perceptual filtering / attention gain)
4. decision_urgency (action temperature / urgency multiplier)
5. fatigue_sensitivity (metabolic cost / effort penalty)
6. social_openness (prosocial affiliation / cooperation threshold)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.neuroendocrine.synthetic_glands import GLAND_NAMES
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)

TARGET_PROPERTIES: List[str] = [
    "excitability",
    "plasticity",
    "sensory_gating",
    "decision_urgency",
    "fatigue_sensitivity",
    "social_openness",
]


class SyntheticReceptorBank(nn.Module):
    """Receptor sensitivity and affinity bank mapping gland dynamics to target gains."""

    def __init__(
        self,
        num_glands: int = len(GLAND_NAMES),
        num_targets: int = len(TARGET_PROPERTIES),
        min_gain: float = 0.1,
        max_gain: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_glands = num_glands
        self.num_targets = num_targets
        self.min_gain = min_gain
        self.max_gain = max_gain

        # Biologically motivated affinity matrix R (6 targets x 12 glands)
        r_mat = torch.zeros(num_targets, num_glands)
        g_idx = {name: i for i, name in enumerate(GLAND_NAMES)}
        t_idx = {name: i for i, name in enumerate(TARGET_PROPERTIES)}

        # 1. Excitability
        r_mat[t_idx["excitability"], g_idx["dopamine"]] = 0.40
        r_mat[t_idx["excitability"], g_idx["adrenaline"]] = 0.80
        r_mat[t_idx["excitability"], g_idx["noradrenaline"]] = 0.60
        r_mat[t_idx["excitability"], g_idx["orexin"]] = 0.50
        r_mat[t_idx["excitability"], g_idx["serotonin"]] = -0.30
        r_mat[t_idx["excitability"], g_idx["adenosine"]] = -0.70
        r_mat[t_idx["excitability"], g_idx["endogenous_opioid"]] = -0.40
        r_mat[t_idx["excitability"], g_idx["melatonin"]] = -0.50

        # 2. Plasticity
        r_mat[t_idx["plasticity"], g_idx["dopamine"]] = 0.70  # D1-mediated LTP gating
        r_mat[t_idx["plasticity"], g_idx["acetylcholine"]] = 0.80  # Muscarinic/nicotinic plasticity
        r_mat[t_idx["plasticity"], g_idx["noradrenaline"]] = 0.40
        r_mat[t_idx["plasticity"], g_idx["cortisol"]] = -0.50  # Stress-induced LTP impairment
        r_mat[t_idx["plasticity"], g_idx["adenosine"]] = -0.40

        # 3. Sensory Gating
        r_mat[t_idx["sensory_gating"], g_idx["acetylcholine"]] = 0.90  # Focus / signal-to-noise
        r_mat[t_idx["sensory_gating"], g_idx["noradrenaline"]] = 0.50
        r_mat[t_idx["sensory_gating"], g_idx["adenosine"]] = -0.60
        r_mat[t_idx["sensory_gating"], g_idx["melatonin"]] = -0.60
        r_mat[t_idx["sensory_gating"], g_idx["cortisol"]] = -0.30

        # 4. Decision Urgency
        r_mat[t_idx["decision_urgency"], g_idx["adrenaline"]] = 0.90
        r_mat[t_idx["decision_urgency"], g_idx["noradrenaline"]] = 0.60
        r_mat[t_idx["decision_urgency"], g_idx["octopamine"]] = 0.50
        r_mat[t_idx["decision_urgency"], g_idx["dopamine"]] = 0.40
        r_mat[t_idx["decision_urgency"], g_idx["serotonin"]] = -0.60  # Promotes patience
        r_mat[t_idx["decision_urgency"], g_idx["endogenous_opioid"]] = -0.40

        # 5. Fatigue Sensitivity
        r_mat[t_idx["fatigue_sensitivity"], g_idx["adenosine"]] = 1.00
        r_mat[t_idx["fatigue_sensitivity"], g_idx["melatonin"]] = 0.70
        r_mat[t_idx["fatigue_sensitivity"], g_idx["cortisol"]] = 0.50
        r_mat[t_idx["fatigue_sensitivity"], g_idx["orexin"]] = -0.80
        r_mat[t_idx["fatigue_sensitivity"], g_idx["dopamine"]] = -0.50
        r_mat[t_idx["fatigue_sensitivity"], g_idx["adrenaline"]] = -0.60

        # 6. Social Openness
        r_mat[t_idx["social_openness"], g_idx["oxytocin"]] = 1.00
        r_mat[t_idx["social_openness"], g_idx["serotonin"]] = 0.50
        r_mat[t_idx["social_openness"], g_idx["endogenous_opioid"]] = 0.40
        r_mat[t_idx["social_openness"], g_idx["cortisol"]] = -0.50
        r_mat[t_idx["social_openness"], g_idx["adrenaline"]] = -0.40
        r_mat[t_idx["social_openness"], g_idx["noradrenaline"]] = -0.30

        self.register_buffer("affinity_matrix", r_mat)
        self.register_buffer("base_gains", torch.ones(num_targets))  # gamma_{j, 0} = 1.0

    def compute_gains(self, gland_deltas: torch.Tensor) -> Dict[str, float]:
        """Computes recipient gains gamma_j = clip(1.0 + R_j * Delta g, min_gain, max_gain).

        Args:
            gland_deltas: (12,) tensor representing deviations from resting baseline.
        """
        device = self.affinity_matrix.device
        deltas = gland_deltas.to(device).flatten()

        shifts = torch.matmul(self.affinity_matrix, deltas)
        raw_gains = self.base_gains + shifts
        clamped_gains = torch.clamp(raw_gains, self.min_gain, self.max_gain)

        gain_dict = {
            prop: float(clamped_gains[i].item())
            for i, prop in enumerate(TARGET_PROPERTIES)
        }
        return gain_dict

    def get_affinity_summary(self) -> Dict[str, Dict[str, float]]:
        """Returns readable table of affinities."""
        summary = {}
        for t_i, target in enumerate(TARGET_PROPERTIES):
            summary[target] = {}
            for g_i, gland in enumerate(GLAND_NAMES):
                val = float(self.affinity_matrix[t_i, g_i].item())
                if abs(val) > 1e-4:
                    summary[target][gland] = val
        return summary
