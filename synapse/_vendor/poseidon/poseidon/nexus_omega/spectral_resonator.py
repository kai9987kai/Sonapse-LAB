"""Stage 11 Bioharmonic Spectral Fourier Resonator.

Implements online Fourier wavelet decomposition and spectral phase estimation
to anticipate non-stationary cyclical environmental pulses (tidal, storm, and
scarcity harmonics) before instantaneous sensory cues manifest.

Inspired by periodic beacon broadcast (Flipper Zero NTAG), raster signal modulation
(quibble / svideo), and temporal flux fields (nexusflow).
"""
from __future__ import annotations

import cmath
import math
from typing import List, Sequence, Optional
import numpy as np

from .contracts import HarmonicSpectralProfile


class BioharmonicResonator:
    """Sliding-window Fourier wavelet phase-frequency estimator."""

    def __init__(
        self,
        window_size: int = 16,
        num_frequency_bins: int = 8,
        hazard_peak_phase: float = 0.0,
    ) -> None:
        self.window_size = window_size
        self.num_bins = num_frequency_bins
        self.hazard_peak_phase = hazard_peak_phase
        self.buffer: List[float] = [0.0] * window_size
        self.tick: int = 0

        # Frequency grid: omega_k = 2 * pi * (k + 1) / window_size
        self.frequencies = [
            2.0 * math.pi * (k + 1) / float(window_size)
            for k in range(self.num_bins)
        ]
        # Precomputed Hanning window
        self.window_weights = [
            0.5 * (1.0 - math.cos(2.0 * math.pi * n / max(1, window_size - 1)))
            for n in range(window_size)
        ]

    def update(
        self,
        cue_or_exposure: float,
    ) -> HarmonicSpectralProfile:
        """Appends a scalar environmental signal and evaluates Fourier decomposition."""
        self.buffer.pop(0)
        self.buffer.append(float(cue_or_exposure))
        self.tick += 1

        # Compute Discrete Fourier Transform across frequency bins
        powers: List[float] = []
        phases: List[float] = []

        for omega in self.frequencies:
            dft_complex = complex(0.0, 0.0)
            for n in range(self.window_size):
                sample = self.buffer[n] * self.window_weights[n]
                # exp(-i * omega * n)
                angle = -omega * float(n)
                dft_complex += sample * complex(math.cos(angle), math.sin(angle))

            power = abs(dft_complex) ** 2
            powers.append(power)
            phases.append(cmath.phase(dft_complex))

        total_power = sum(powers)
        if total_power > 1e-9:
            norm_powers = [p / total_power for p in powers]
            spectral_entropy = -sum(
                p * math.log(p + 1e-12) for p in norm_powers if p > 0.0
            ) / math.log(float(self.num_bins))
        else:
            norm_powers = [1.0 / float(self.num_bins)] * self.num_bins
            spectral_entropy = 1.0

        # Find dominant frequency
        dominant_idx = int(np.argmax(powers))
        dom_freq = self.frequencies[dominant_idx]
        dom_period = (2.0 * math.pi / dom_freq) if dom_freq > 1e-6 else float(self.window_size)
        curr_phase = phases[dominant_idx]

        # Phase advance projection over horizon H=4
        H_default = 4
        predicted_advance = (curr_phase + dom_freq * H_default) % (2.0 * math.pi)
        if predicted_advance > math.pi:
            predicted_advance -= 2.0 * math.pi

        # Storm resonance probability: alignment with hazard peak phase + spectral coherence
        phase_distance = abs(curr_phase - self.hazard_peak_phase)
        phase_alignment = 0.5 * (1.0 + math.cos(phase_distance))
        coherence = max(0.0, min(1.0, 1.0 - spectral_entropy))
        resonance_prob = float(phase_alignment * (0.3 + 0.7 * coherence))

        # Boost resonance if recent signal has high mean
        recent_mean = float(np.mean(self.buffer[-4:]))
        if recent_mean > 0.5:
            resonance_prob = min(1.0, resonance_prob + 0.3 * (recent_mean - 0.5))

        return HarmonicSpectralProfile(
            dominant_frequency=dom_freq,
            dominant_period=dom_period,
            spectral_power=norm_powers,
            current_phase=curr_phase,
            predicted_phase_advance=predicted_advance,
            spectral_entropy=spectral_entropy,
            storm_resonance_probability=resonance_prob,
        )

    def anticipate_hazard(self, horizon: int = 4) -> float:
        """Forecasts likelihood of a cyclical hazard surge within H steps."""
        if len(self.buffer) == 0:
            return 0.0
        # Check if projected phase will pass through the hazard peak [hazard_peak_phase - pi/4, hazard_peak_phase + pi/4]
        profile = self.update(self.buffer[-1])
        dom_freq = profile.dominant_frequency
        curr_phase = profile.current_phase

        # Scan intermediate ticks
        max_threat = profile.storm_resonance_probability
        for step in range(1, horizon + 1):
            future_phase = (curr_phase + dom_freq * step) % (2.0 * math.pi)
            if future_phase > math.pi:
                future_phase -= 2.0 * math.pi
            threat_at_step = 0.5 * (1.0 + math.cos(future_phase - self.hazard_peak_phase))
            if threat_at_step > max_threat:
                max_threat = threat_at_step

        return float(max(0.0, min(1.0, max_threat)))
