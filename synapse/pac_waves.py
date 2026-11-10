"""Phase-Amplitude Coupling (PAC) and Spatiotemporal Traveling Wave Engine.

Based on foundational neurophysiology:
- Canolty et al. (Science 2006): High gamma power is phase-locked to theta oscillations.
- Tort et al. (J Neurophysiol 2010): Modulation Index (MI) based on normalized entropy / KL divergence.
- Muller et al. (Nature Rev Neurosci 2018): Cortical traveling waves and spatiotemporal phase gradients.

Governing Invariant:
When pac_gain == 0.0 or disabled:
F_extended(x, s; 0) = F_legacy(x, s) bit-for-bit identity.
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
class PACTelemetry:
    modulation_index: float
    preferred_phase_rad: float
    theta_freq_hz: float
    gamma_freq_hz: float
    wave_coherence: float
    wave_velocity: float
    num_phase_bins: int
    phase_distribution: List[float]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modulation_index": round(float(self.modulation_index), 5),
            "preferred_phase_rad": round(float(self.preferred_phase_rad), 3),
            "theta_freq_hz": round(float(self.theta_freq_hz), 1),
            "gamma_freq_hz": round(float(self.gamma_freq_hz), 1),
            "wave_coherence": round(float(self.wave_coherence), 4),
            "wave_velocity": round(float(self.wave_velocity), 2),
            "num_phase_bins": self.num_phase_bins,
            "phase_distribution": [round(float(p), 4) for p in self.phase_distribution],
            "timestamp": self.timestamp,
        }


class PhaseAmplitudeCouplingEngine(nn.Module):
    """Computes Tort Modulation Index and coordinates spatiotemporal traveling waves across connectome substrates."""

    def __init__(
        self,
        num_substrates: int = 6,
        theta_freq_hz: float = 6.0,
        gamma_freq_hz: float = 40.0,
        num_phase_bins: int = 18,
        dt_ms: float = 2.0,
        seed: int = 42,
    ):
        super().__init__()
        self.num_substrates = num_substrates
        self.theta_freq_hz = theta_freq_hz
        self.gamma_freq_hz = gamma_freq_hz
        self.num_phase_bins = num_phase_bins
        self.dt_ms = dt_ms

        # Phase bin boundaries in [-pi, pi]
        self.bin_edges = np.linspace(-np.pi, np.pi, num_phase_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Quadrature oscillator states for low-frequency theta pacemaker: [cos(theta), sin(theta)]
        omega_theta = 2.0 * math.pi * theta_freq_hz * (dt_ms / 1000.0)
        self.register_buffer("omega_theta", torch.tensor(omega_theta, dtype=torch.float32))
        self.register_buffer("theta_phase", torch.zeros(num_substrates, dtype=torch.float32))

        # Spatial conduction delays between substrates (ms)
        # 0: C. elegans, 1: Drosophila, 2: Poseidon, 3: WPAC, 4: Auditory, 5: Visual
        substrate_delays_ms = torch.tensor([0.0, 1.5, 3.0, 2.5, 0.5, 0.5], dtype=torch.float32)
        # Phase offsets derived from conduction delays: phi_k = omega * (tau_k / dt)
        phase_offsets = (substrate_delays_ms / dt_ms) * omega_theta
        self.register_buffer("substrate_phase_offsets", phase_offsets)

        # Running circular buffer of (phase, amplitude) samples for Tort MI calculation
        self.buffer_size = 360
        self.register_buffer("phase_hist", torch.zeros(self.buffer_size, dtype=torch.float32))
        self.register_buffer("amp_hist", torch.zeros(self.buffer_size, dtype=torch.float32))
        self.register_buffer("hist_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("hist_count", torch.zeros(1, dtype=torch.long))

        self._last_telemetry: Optional[PACTelemetry] = None

    def reset_state(self) -> None:
        """Resets oscillator phases and history buffers."""
        self.theta_phase.zero_()
        self.phase_hist.zero_()
        self.amp_hist.zero_()
        self.hist_ptr.zero_()
        self.hist_count.zero_()
        self._last_telemetry = None

    def advance_phase(self) -> torch.Tensor:
        """Advances the low-frequency theta traveling wave and returns instantaneous substrate phases."""
        with torch.no_grad():
            self.theta_phase.add_(self.omega_theta)
            # Wrap phase to [-pi, pi]
            self.theta_phase.remainder_(2.0 * math.pi)
            self.theta_phase.sub_(math.pi)
            # Add spatial delay offsets
            effective_phases = self.theta_phase + self.substrate_phase_offsets
            effective_phases.remainder_(2.0 * math.pi).sub_(math.pi)
            return effective_phases

    def calculate_gating(
        self,
        substrate_phases: torch.Tensor,
        pac_gain: float = 0.0,
    ) -> torch.Tensor:
        """Returns multiplicative excitability modulation factor for each substrate.

        g_k = 1.0 + pac_gain * cos(phi_k)
        When pac_gain == 0.0: exactly 1.0 for all substrates.
        """
        if pac_gain <= 0.0:
            return torch.ones_like(substrate_phases)
        return 1.0 + pac_gain * torch.cos(substrate_phases)

    def record_step(
        self,
        theta_ref_phase: float,
        gamma_amp: float,
    ) -> None:
        """Stores a (phase, amplitude) pair in the ring buffer for entropy estimation."""
        idx = int(self.hist_ptr.item())
        self.phase_hist[idx] = float(theta_ref_phase)
        self.amp_hist[idx] = max(1e-6, float(gamma_amp))
        self.hist_ptr[0] = (idx + 1) % self.buffer_size
        if self.hist_count[0] < self.buffer_size:
            self.hist_count[0] += 1

    def compute_tort_modulation_index(self) -> Tuple[float, float, List[float]]:
        """Calculates Tort (2010) Modulation Index (MI) via Kullback-Leibler divergence from uniform.

        MI = D_KL(P || U) / log(N_bins) = (log(N) - H(P)) / log(N)
        """
        count = int(self.hist_count.item())
        if count < 36:
            # Insufficient sample history; return baseline zero coupling
            uniform = [1.0 / self.num_phase_bins] * self.num_phase_bins
            return 0.0, 0.0, uniform

        phases = self.phase_hist[:count].cpu().numpy()
        amps = self.amp_hist[:count].cpu().numpy()

        bin_indices = np.digitize(phases, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_phase_bins - 1)

        bin_sums = np.zeros(self.num_phase_bins, dtype=np.float64)
        bin_counts = np.zeros(self.num_phase_bins, dtype=np.float64)
        np.add.at(bin_sums, bin_indices, amps)
        np.add.at(bin_counts, bin_indices, 1.0)

        # Mean amplitude per bin, with pseudo-count epsilon for empty bins
        mean_amps = np.where(bin_counts > 0, bin_sums / np.maximum(bin_counts, 1.0), 1e-6)
        total_amp = np.sum(mean_amps)
        if total_amp <= 0.0:
            p_dist = np.full(self.num_phase_bins, 1.0 / self.num_phase_bins)
        else:
            p_dist = mean_amps / total_amp

        # Shannon entropy H(P)
        h_p = -np.sum(p_dist * np.log(np.maximum(p_dist, 1e-12)))
        log_n = np.log(self.num_phase_bins)
        mi = float((log_n - h_p) / log_n)
        mi = max(0.0, min(1.0, mi))

        # Preferred phase: circular mean weighted by amplitude
        pref_phase = float(self.bin_centers[int(np.argmax(p_dist))])
        return mi, pref_phase, p_dist.tolist()

    def compute_wave_coherence(self, substrate_phases: torch.Tensor) -> float:
        """Computes Kuramoto order parameter across substrate phases: R = |1/M * sum(exp(i * phi_k))|."""
        cos_sum = torch.cos(substrate_phases).sum().item()
        sin_sum = torch.sin(substrate_phases).sum().item()
        return float(math.hypot(cos_sum, sin_sum) / len(substrate_phases))

    def step(
        self,
        ref_gamma_activity: float,
        pac_gain: float = 0.0,
    ) -> Tuple[torch.Tensor, PACTelemetry]:
        """Advances wave propagation and computes PAC modulation index."""
        t0 = time.time()
        sub_phases = self.advance_phase()
        gating = self.calculate_gating(sub_phases, pac_gain=pac_gain)

        # Record reference phase (WPAC substrate index = 3) and gamma activity
        wpac_phase = float(sub_phases[3].item())
        self.record_step(wpac_phase, ref_gamma_activity)

        mi, pref_phase, p_dist = self.compute_tort_modulation_index()
        wave_coherence = self.compute_wave_coherence(sub_phases)
        # Approximate propagation velocity across 12k connectome column (mm/s)
        wave_velocity = self.theta_freq_hz * 12.0 * (0.8 + 0.4 * wave_coherence)

        telemetry = PACTelemetry(
            modulation_index=mi,
            preferred_phase_rad=pref_phase,
            theta_freq_hz=self.theta_freq_hz,
            gamma_freq_hz=self.gamma_freq_hz,
            wave_coherence=wave_coherence,
            wave_velocity=wave_velocity,
            num_phase_bins=self.num_phase_bins,
            phase_distribution=p_dist,
            timestamp=t0,
        )
        self._last_telemetry = telemetry
        return gating, telemetry

    def get_summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable PAC telemetry summary."""
        if self._last_telemetry is None:
            return {
                "modulation_index": 0.0,
                "preferred_phase_rad": 0.0,
                "theta_freq_hz": self.theta_freq_hz,
                "gamma_freq_hz": self.gamma_freq_hz,
                "wave_coherence": 1.0,
                "wave_velocity": 0.0,
                "num_phase_bins": self.num_phase_bins,
                "phase_distribution": [1.0 / self.num_phase_bins] * self.num_phase_bins,
                "timestamp": time.time(),
            }
        return self._last_telemetry.to_dict()
