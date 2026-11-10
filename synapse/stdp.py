"""Spike-Timing-Dependent Plasticity (STDP) Engine with Dale's Law Compliance.

Scientific Foundations:
1. Bi & Poo (J. Neurosci. 1998): Asymmetric Hebbian STDP window in cortical and
   hippocampal synapses:
   - Pre-before-post (Delta t > 0): Long-Term Potentiation (LTP) with exponential decay tau_+
   - Post-before-pre (Delta t < 0): Long-Term Depression (LTD) with exponential decay tau_-
2. Song, Miller & Abbott (Nature Neurosci. 2000): Competitive Hebbian learning
   via STDP with net LTD bias (A_- > A_+) for self-stabilizing synaptic distribution.
3. Dale's Principle (Eccles 1954; Janelia Drosophila Connectomics):
   Synapses preserve physiological neurotransmitter signs under plastic modification:
   Excitatory (ACh) weights remain in [0, +inf), Inhibitory (GABA/Glu) weights in (-inf, 0].

Governing Invariant:
When learn=False or stdp_gain == 0.0:
F_extended(w; 0) = F_legacy(w) bit-for-bit identity.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class STDPConfig:
    tau_plus_ms: float = 20.0
    tau_minus_ms: float = 20.0
    a_plus: float = 0.010
    a_minus: float = 0.0105  # Slight depression dominance for stability
    learning_rate: float = 0.005
    max_weight_mag: float = 1.0
    min_weight_mag: float = 0.0
    dt_ms: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_plus_ms": float(self.tau_plus_ms),
            "tau_minus_ms": float(self.tau_minus_ms),
            "a_plus": float(self.a_plus),
            "a_minus": float(self.a_minus),
            "learning_rate": float(self.learning_rate),
            "max_weight_mag": float(self.max_weight_mag),
            "min_weight_mag": float(self.min_weight_mag),
            "dt_ms": float(self.dt_ms),
        }


@dataclasses.dataclass
class STDPTelemetry:
    mean_dw: float = 0.0
    dw_ltp_sum: float = 0.0
    dw_ltd_sum: float = 0.0
    mean_weight: float = 0.0
    weight_stability: float = 1.0
    active_pre_spikes: int = 0
    active_post_spikes: int = 0
    compliance_rate: float = 1.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_dw": round(float(self.mean_dw), 6),
            "dw_ltp_sum": round(float(self.dw_ltp_sum), 6),
            "dw_ltd_sum": round(float(self.dw_ltd_sum), 6),
            "mean_weight": round(float(self.mean_weight), 4),
            "weight_stability": round(float(self.weight_stability), 4),
            "active_pre_spikes": int(self.active_pre_spikes),
            "active_post_spikes": int(self.active_post_spikes),
            "compliance_rate": round(float(self.compliance_rate), 4),
            "timestamp": round(float(self.timestamp), 3),
        }


class SpikeTimingPlasticityEngine(nn.Module):
    """Pair-based Spike-Timing-Dependent Plasticity (STDP) with online eligibility traces."""

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        config: Optional[STDPConfig] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.config = config or STDPConfig()

        # Online exponential eligibility traces
        self.register_buffer("trace_pre", torch.zeros(self.n_pre, dtype=torch.float32))
        self.register_buffer("trace_post", torch.zeros(self.n_post, dtype=torch.float32))

        self.decay_pre = math.exp(-self.config.dt_ms / self.config.tau_plus_ms)
        self.decay_post = math.exp(-self.config.dt_ms / self.config.tau_minus_ms)

        self._last_telemetry: Optional[STDPTelemetry] = None
        self._step_count = 0

    def reset_state(self) -> None:
        """Resets synaptic eligibility traces."""
        self.trace_pre.zero_()
        self.trace_post.zero_()
        self._last_telemetry = None
        self._step_count = 0

    def update_traces(
        self,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decays eligibility traces and increments on new spikes."""
        with torch.no_grad():
            self.trace_pre.mul_(self.decay_pre).add_(s_pre)
            self.trace_post.mul_(self.decay_post).add_(s_post)
            return self.trace_pre, self.trace_post

    def compute_dw(
        self,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
        stdp_gain: float = 1.0,
    ) -> Tuple[torch.Tensor, float, float]:
        """Calculates weight delta matrix [n_post, n_pre] using online traces.

        LTP is triggered when a post-synaptic spike arrives (s_post > 0)
        sensing the pre-synaptic trace (trace_pre).
        LTD is triggered when a pre-synaptic spike arrives (s_pre > 0)
        sensing the post-synaptic trace (trace_post).
        """
        # LTP: outer product of s_post and trace_pre
        ltp = torch.outer(s_post, self.trace_pre) * self.config.a_plus
        # LTD: outer product of trace_post and s_pre
        ltd = torch.outer(self.trace_post, s_pre) * self.config.a_minus

        dw = (ltp - ltd) * (self.config.learning_rate * stdp_gain)
        ltp_sum = float(ltp.sum().item())
        ltd_sum = float(ltd.sum().item())
        return dw, ltp_sum, ltd_sum

    def step(
        self,
        s_pre: torch.Tensor,
        s_post: torch.Tensor,
        weights: torch.Tensor,
        presynaptic_signs: Optional[torch.Tensor] = None,
        learn: bool = False,
        stdp_gain: float = 0.0,
    ) -> Tuple[torch.Tensor, STDPTelemetry]:
        """Advances STDP eligibility traces and updates synaptic weights.

        Args:
            s_pre: Pre-synaptic spike vector [n_pre] (binary or rate in [0, 1]).
            s_post: Post-synaptic spike vector [n_post] (binary or rate in [0, 1]).
            weights: Current weight matrix [n_post, n_pre].
            presynaptic_signs: Optional vector [n_pre] indicating Dale sign (+1 or -1).
            learn: Whether plasticity updates are active.
            stdp_gain: Scaling gain for plastic updates (0.0 = zero update).

        Returns:
            Tuple of (updated_weights, telemetry).
        """
        self._step_count += 1
        t0 = time.time()

        s_pre_f = s_pre.flatten()[:self.n_pre].float()
        s_post_f = s_post.flatten()[:self.n_post].float()

        # Update traces even when learning is passive to maintain online history
        self.update_traces(s_pre_f, s_post_f)

        if not learn or stdp_gain <= 0.0:
            # Primary Invariant: Exactly zero modification
            telemetry = STDPTelemetry(
                mean_dw=0.0,
                dw_ltp_sum=0.0,
                dw_ltd_sum=0.0,
                mean_weight=float(weights.abs().mean().item()),
                weight_stability=1.0,
                active_pre_spikes=int((s_pre_f > 0.1).sum().item()),
                active_post_spikes=int((s_post_f > 0.1).sum().item()),
                compliance_rate=1.0,
                timestamp=t0,
            )
            self._last_telemetry = telemetry
            return weights, telemetry

        # Compute delta weights
        dw, ltp_sum, ltd_sum = self.compute_dw(s_pre_f, s_post_f, stdp_gain=stdp_gain)

        with torch.no_grad():
            new_w = weights + dw

            # Project onto Dale's Law convex cone if signs provided
            if presynaptic_signs is not None:
                signs = presynaptic_signs.view(1, -1)  # broadcast over n_post
                pos_mask = signs > 0
                neg_mask = signs < 0

                # Clamping bounds: Excitatory in [0, max], Inhibitory in [-max, 0]
                new_w = torch.where(
                    pos_mask,
                    torch.clamp(new_w, min=self.config.min_weight_mag, max=self.config.max_weight_mag),
                    new_w,
                )
                new_w = torch.where(
                    neg_mask,
                    torch.clamp(new_w, min=-self.config.max_weight_mag, max=-self.config.min_weight_mag),
                    new_w,
                )

                # Measure Dale's Law compliance
                valid_pos = (new_w[:, pos_mask.squeeze(0)] >= 0.0).all()
                valid_neg = (new_w[:, neg_mask.squeeze(0)] <= 0.0).all()
                compliance = 1.0 if (valid_pos and valid_neg) else 0.0
            else:
                new_w.clamp_(-self.config.max_weight_mag, self.config.max_weight_mag)
                compliance = 1.0

            # Weight stability index (cosine similarity between w and new_w)
            w_flat = weights.flatten()
            new_w_flat = new_w.flatten()
            denom = (w_flat.norm() * new_w_flat.norm()).clamp_min(1e-8)
            cos_sim = float(torch.dot(w_flat, new_w_flat) / denom)

            telemetry = STDPTelemetry(
                mean_dw=float(dw.abs().mean().item()),
                dw_ltp_sum=ltp_sum,
                dw_ltd_sum=ltd_sum,
                mean_weight=float(new_w.abs().mean().item()),
                weight_stability=float(cos_sim),
                active_pre_spikes=int((s_pre_f > 0.1).sum().item()),
                active_post_spikes=int((s_post_f > 0.1).sum().item()),
                compliance_rate=compliance,
                timestamp=t0,
            )
            self._last_telemetry = telemetry
            return new_w, telemetry

    def get_summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary."""
        if self._last_telemetry is None:
            return {
                "mean_dw": 0.0,
                "dw_ltp_sum": 0.0,
                "dw_ltd_sum": 0.0,
                "mean_weight": 0.0,
                "weight_stability": 1.0,
                "active_pre_spikes": 0,
                "active_post_spikes": 0,
                "compliance_rate": 1.0,
                "config": self.config.to_dict(),
                "step_count": self._step_count,
            }
        d = self._last_telemetry.to_dict()
        d["config"] = self.config.to_dict()
        d["step_count"] = self._step_count
        return d
