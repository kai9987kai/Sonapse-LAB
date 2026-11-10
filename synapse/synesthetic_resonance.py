"""Synesthetic Cross-Modal Binding and Phase Coherence Engine.

Based on cognitive neuroscience of multisensory integration:
- Lachaux et al. (Hum Brain Mapp 1999): Measuring phase synchronization in brain signals (Phase-Locking Value, PLV).
- Varela et al. (Nature Rev Neurosci 2001): The brainweb: Phase synchronization and large-scale integration.
- Cytowic (2002): Synesthesia: A Union of the Senses.

Governing Invariant:
When resonance_gain == 0.0 or disabled:
Cross-modal state computation is read-only sidecar telemetry and introduces zero side effects.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class SynestheticTelemetry:
    cross_modal_binding_index: float
    visual_auditory_plv: float
    visual_endocrine_plv: float
    auditory_endocrine_plv: float
    synesthetic_burst_active: bool
    binding_confidence: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cross_modal_binding_index": round(float(self.cross_modal_binding_index), 4),
            "visual_auditory_plv": round(float(self.visual_auditory_plv), 4),
            "visual_endocrine_plv": round(float(self.visual_endocrine_plv), 4),
            "auditory_endocrine_plv": round(float(self.auditory_endocrine_plv), 4),
            "synesthetic_burst_active": self.synesthetic_burst_active,
            "binding_confidence": round(float(self.binding_confidence), 4),
            "timestamp": self.timestamp,
        }


class SynestheticResonanceEngine(nn.Module):
    """Calculates pairwise Phase-Locking Value (PLV) across Visual, Auditory, and Endocrine modalities."""

    def __init__(
        self,
        window_size: int = 64,
        binding_threshold: float = 0.65,
    ):
        super().__init__()
        self.window_size = window_size
        self.binding_threshold = binding_threshold

        # Phase trajectory history buffers
        self.register_buffer("vis_phase_hist", torch.zeros(window_size))
        self.register_buffer("aud_phase_hist", torch.zeros(window_size))
        self.register_buffer("endo_phase_hist", torch.zeros(window_size))
        self.register_buffer("buf_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buf_count", torch.zeros(1, dtype=torch.long))

        self.resonance_gain = 0.0
        self._last_telemetry: Optional[SynestheticTelemetry] = None

    def reset_state(self) -> None:
        self.vis_phase_hist.zero_()
        self.aud_phase_hist.zero_()
        self.endo_phase_hist.zero_()
        self.buf_ptr.zero_()
        self.buf_count.zero_()
        self._last_telemetry = None

    def _instantaneous_phase(self, signal: torch.Tensor) -> float:
        """Approximates instantaneous phase from mean and variance via Hilbert quadrature proxy."""
        if signal.numel() == 0:
            return 0.0
        m = signal.mean().item()
        s = signal.std().item() if signal.numel() > 1 else 0.0
        return math.atan2(s, m + 1e-6)

    def record_modalities(
        self,
        vis_features: torch.Tensor,
        aud_features: torch.Tensor,
        endo_features: torch.Tensor,
    ) -> None:
        """Extracts instantaneous phase from visual, auditory, and endocrine state tensors."""
        idx = int(self.buf_ptr.item())
        self.vis_phase_hist[idx] = self._instantaneous_phase(vis_features)
        self.aud_phase_hist[idx] = self._instantaneous_phase(aud_features)
        self.endo_phase_hist[idx] = self._instantaneous_phase(endo_features)

        self.buf_ptr[0] = (idx + 1) % self.window_size
        if self.buf_count[0] < self.window_size:
            self.buf_count[0] += 1

    def compute_plv(self, phi1: torch.Tensor, phi2: torch.Tensor, count: int) -> float:
        """Computes Lachaux et al. (1999) Phase-Locking Value: PLV = |1/N * sum(exp(i * (phi1 - phi2)))|."""
        if count < 8:
            return 0.0
        d_phi = phi1[:count] - phi2[:count]
        cos_mean = torch.cos(d_phi).mean().item()
        sin_mean = torch.sin(d_phi).mean().item()
        return float(math.hypot(cos_mean, sin_mean))

    def step(
        self,
        vis_features: Optional[torch.Tensor] = None,
        aud_features: Optional[torch.Tensor] = None,
        endo_features: Optional[torch.Tensor] = None,
        resonance_gain: Optional[float] = None,
    ) -> Tuple[float, SynestheticTelemetry]:
        """Calculates Cross-Modal Binding Index (CMBI) and returns neuromodulatory resonance drive."""
        t0 = time.time()
        eff_gain = self.resonance_gain if resonance_gain is None else float(resonance_gain)

        v_t = vis_features if vis_features is not None else torch.zeros(64)
        a_t = aud_features if aud_features is not None else torch.zeros(64)
        e_t = endo_features if endo_features is not None else torch.zeros(12)

        self.record_modalities(v_t, a_t, e_t)
        count = int(self.buf_count.item())

        plv_va = self.compute_plv(self.vis_phase_hist, self.aud_phase_hist, count)
        plv_ve = self.compute_plv(self.vis_phase_hist, self.endo_phase_hist, count)
        plv_ae = self.compute_plv(self.aud_phase_hist, self.endo_phase_hist, count)

        # Composite Cross-Modal Binding Index (weighted average of pairwise PLVs)
        cmbi = float(0.45 * plv_va + 0.30 * plv_ve + 0.25 * plv_ae)
        synesthetic_active = cmbi > self.binding_threshold
        confidence = float(min(1.0, count / float(self.window_size)))

        telemetry = SynestheticTelemetry(
            cross_modal_binding_index=cmbi,
            visual_auditory_plv=plv_va,
            visual_endocrine_plv=plv_ve,
            auditory_endocrine_plv=plv_ae,
            synesthetic_burst_active=synesthetic_active,
            binding_confidence=confidence,
            timestamp=t0,
        )
        self._last_telemetry = telemetry

        # Resonance modulation pulse: dopamine/oxytocin elevation boost
        resonance_pulse = (eff_gain * (cmbi - self.binding_threshold)) if (synesthetic_active and eff_gain > 0.0) else 0.0
        return resonance_pulse, telemetry

    def get_summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable synesthetic binding summary."""
        if self._last_telemetry is None:
            return {
                "cross_modal_binding_index": 0.0,
                "visual_auditory_plv": 0.0,
                "visual_endocrine_plv": 0.0,
                "auditory_endocrine_plv": 0.0,
                "synesthetic_burst_active": False,
                "binding_confidence": 0.0,
                "resonance_gain": self.resonance_gain,
                "timestamp": time.time(),
            }
        d = self._last_telemetry.to_dict()
        d["resonance_gain"] = self.resonance_gain
        return d
