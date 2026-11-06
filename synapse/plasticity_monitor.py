"""Plasticity-Aware Structural Learning Monitor.

Grounding & Review Requirements:
Continual learning systems frequently suffer from "loss of plasticity" / representational collapse
(Dohare et al. Nature 2024, Lyle et al. ICML 2023, Kumar et al. ICML 2024), where neurons
saturate, representation rank collapses, and the agent becomes incapable of acquiring new skills.

This module provides:
1. Effective Rank (erank):
       erank(X) = exp(-sum_i p_i * log(p_i)) where p_i = sigma_i / sum_j sigma_j
   derived from the singular values of the latent activation matrix.
2. Stable Rank (srank):
       srank(X) = ||X||_F^2 / ||X||_2^2
3. Dead / Dormant Neuron Ratio:
   Fraction of neurons with near-zero variance across active episodes.
4. Plasticity Collapse Alarm:
   Early warning trigger recommending structural intervention (Continual Backprop replacement
   or neurogenesis) before catastrophic performance degradation occurs.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class PlasticityReport:
    """Quantitative report on representation rank and plasticity status."""
    step: int
    effective_rank: float     # erank in [1, D]
    normalized_erank: float   # erank / D in [0, 1]
    stable_rank: float        # srank >= 1.0
    dead_neuron_ratio: float  # fraction in [0, 1]
    spectral_entropy: float   # Shannon entropy of normalized singular values
    is_collapsed: bool        # True if normalized erank < collapse_threshold
    recommended_action: str   # 'HEALTHY', 'SCHEDULE_CBP_RESET', 'TRIGGER_NEUROGENESIS'
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PlasticityMonitor:
    """Monitors representation rank and detects spectral collapse."""

    def __init__(
        self,
        num_features: int,
        buffer_size: int = 128,
        collapse_threshold: float = 0.20,
        dormant_threshold: float = 1e-4
    ):
        self.num_features = num_features
        self.buffer_size = buffer_size
        self.collapse_threshold = collapse_threshold
        self.dormant_threshold = dormant_threshold

        self.activity_buffer: List[np.ndarray] = []
        self.step_counter: int = 0
        self.total_collapse_alarms: int = 0
        self.history: List[PlasticityReport] = []

    def record_activity(self, features: np.ndarray, step: Optional[int] = None):
        """Record a feature vector (e.g. graph activations or latent embeddings)."""
        self.step_counter = step if step is not None else (self.step_counter + 1)
        flat = np.asarray(features, dtype=np.float32).flatten()
        if len(flat) > self.num_features:
            flat = flat[:self.num_features]
        elif len(flat) < self.num_features:
            flat = np.pad(flat, (0, self.num_features - len(flat)))

        self.activity_buffer.append(flat)
        if len(self.activity_buffer) > self.buffer_size:
            self.activity_buffer.pop(0)

    def evaluate_plasticity(self) -> PlasticityReport:
        """Compute effective rank, stable rank, and check for representation collapse."""
        if len(self.activity_buffer) < 10:
            return PlasticityReport(
                step=self.step_counter,
                effective_rank=float(self.num_features),
                normalized_erank=1.0,
                stable_rank=1.0,
                dead_neuron_ratio=0.0,
                spectral_entropy=1.0,
                is_collapsed=False,
                recommended_action='WARMUP'
            )

        X = np.stack(self.activity_buffer, axis=0)  # [N, D]
        N, D = X.shape

        # Center data
        X_centered = X - np.mean(X, axis=0, keepdims=True)

        # Singular Value Decomposition
        try:
            _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            S = np.ones(min(N, D), dtype=np.float32)

        # 1. Effective Rank (erank) via Shannon entropy of singular values
        S_sum = np.sum(S)
        if S_sum > 1e-8:
            p = S / S_sum
            # Filter non-positive entries
            p_pos = p[p > 1e-12]
            entropy = -float(np.sum(p_pos * np.log(p_pos)))
            erank = float(np.exp(entropy))
        else:
            entropy = 0.0
            erank = 1.0

        normalized_erank = float(erank / D)

        # 2. Stable Rank: ||X||_F^2 / ||X||_2^2
        fro_sq = float(np.sum(S ** 2))
        top_s_sq = float(S[0] ** 2) if len(S) > 0 and S[0] > 1e-8 else 1.0
        srank = float(fro_sq / (top_s_sq + 1e-8))

        # 3. Dead / Dormant Neurons: variance across samples < threshold
        variances = np.var(X, axis=0)
        dead_count = int(np.sum(variances < self.dormant_threshold))
        dead_ratio = float(dead_count / D)

        # 4. Collapse Detection and Action Recommendation
        is_collapsed = bool(normalized_erank < self.collapse_threshold or dead_ratio > 0.40)
        if is_collapsed:
            self.total_collapse_alarms += 1
            if dead_ratio > 0.40:
                action = 'SCHEDULE_CBP_RESET'
            else:
                action = 'TRIGGER_NEUROGENESIS'
        else:
            action = 'HEALTHY'

        report = PlasticityReport(
            step=self.step_counter,
            effective_rank=round(erank, 3),
            normalized_erank=round(normalized_erank, 4),
            stable_rank=round(srank, 3),
            dead_neuron_ratio=round(dead_ratio, 4),
            spectral_entropy=round(entropy, 4),
            is_collapsed=is_collapsed,
            recommended_action=action,
            details={
                'singular_values_top5': [round(float(s), 4) for s in S[:5]],
                'dead_neuron_count': dead_count,
                'total_alarms': self.total_collapse_alarms
            }
        )

        self.history.append(report)
        if len(self.history) > 100:
            self.history.pop(0)

        return report

    def summary(self) -> Dict[str, Any]:
        """Summary metrics of plasticity monitoring."""
        if not self.history:
            return {'status': 'uninitialized', 'step': self.step_counter}
        latest = self.history[-1]
        return {
            'step': latest.step,
            'effective_rank': latest.effective_rank,
            'normalized_erank': latest.normalized_erank,
            'stable_rank': latest.stable_rank,
            'dead_neuron_ratio': latest.dead_neuron_ratio,
            'is_collapsed': latest.is_collapsed,
            'recommended_action': latest.recommended_action,
            'total_collapse_alarms': self.total_collapse_alarms
        }
