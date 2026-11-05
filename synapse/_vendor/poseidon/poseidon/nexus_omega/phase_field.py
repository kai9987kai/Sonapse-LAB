"""Stage 12 Nonstationary Phase-Field Resonator.

Unifies online Fourier wavelet decomposition with multi-scale nonstationary cycle
drift tracking, anticipating environmental period shifts (such as POMDPPhaseWorldV3
hidden schedule changes) and predicting hazard boundary proximity before sensory cues manifest.

Inspired by periodic beacon broadcast (Flipper Zero NTAG), streaming flux fields (nexusflow),
and temporal video generation (quibble / svideo).
"""
from __future__ import annotations

import cmath
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from .contracts import HarmonicSpectralProfile


@dataclass
class NonstationaryPhaseState:
    """Telemetry contract for the nonstationary phase field."""
    dominant_frequency: float
    dominant_period: float
    confirmed_interval: Optional[float]
    estimated_interval: float
    ticks_since_pulse: Optional[int]
    phase_boundary_latch_active: bool
    drift_velocity: float
    interval_change_streak: int
    resonance_hazard_probability: float
    harmonic_profile: HarmonicSpectralProfile
    timestamp: float = field(default_factory=time.time)

    @property
    def estimated_pulse_interval(self) -> float:
        return self.estimated_interval

    @property
    def confirmed_pulse_interval(self) -> Optional[float]:
        return self.confirmed_interval

    @property
    def frequency_drift_velocity(self) -> float:
        return self.drift_velocity

    @property
    def storm_resonance_probability(self) -> float:
        return self.resonance_hazard_probability

    @property
    def spectral_entropy(self) -> float:
        return self.harmonic_profile.spectral_entropy

    @property
    def fourier_amplitudes(self) -> List[float]:
        return self.harmonic_profile.fourier_amplitudes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_frequency": round(float(self.dominant_frequency), 4),
            "dominant_period": round(float(self.dominant_period), 2),
            "confirmed_interval": round(float(self.confirmed_interval), 2) if self.confirmed_interval is not None else None,
            "estimated_interval": round(float(self.estimated_interval), 2),
            "estimated_pulse_interval": round(float(self.estimated_interval), 2),
            "confirmed_pulse_interval": round(float(self.confirmed_interval), 2) if self.confirmed_interval is not None else None,
            "ticks_since_pulse": self.ticks_since_pulse,
            "phase_boundary_latch_active": bool(self.phase_boundary_latch_active),
            "drift_velocity": round(float(self.drift_velocity), 4),
            "frequency_drift_velocity": round(float(self.drift_velocity), 4),
            "interval_change_streak": int(self.interval_change_streak),
            "resonance_hazard_probability": round(float(self.resonance_hazard_probability), 4),
            "storm_resonance_probability": round(float(self.resonance_hazard_probability), 4),
            "harmonic_profile": self.harmonic_profile.to_dict(),
            "timestamp": round(float(self.timestamp), 4),
        }


class NonstationaryPhaseFieldResonator:
    """Dual-rate Fourier-wavelet nonstationary cycle drift tracker."""

    DEFAULT_INTERVAL = 8.0
    MIN_INTERVAL = 2.0
    MAX_INTERVAL = 64.0
    HISTORY_CAP = 12

    def __init__(
        self,
        window_size: int = 16,
        num_frequency_bins: int = 8,
        initial_interval: float = 8.0,
    ) -> None:
        self.window_size = window_size
        self.num_bins = num_frequency_bins
        self.buffer: List[float] = [0.0] * window_size
        self.tick: int = 0

        # Frequency grid: omega_k = 2 * pi * (k + 1) / window_size
        self.frequencies = [
            2.0 * math.pi * (k + 1) / float(window_size)
            for k in range(self.num_bins)
        ]
        self.window_weights = [
            0.5 * (1.0 - math.cos(2.0 * math.pi * n / max(1, window_size - 1)))
            for n in range(window_size)
        ]

        # Multi-scale nonstationary interval estimator
        self.estimated_interval: float = float(initial_interval)
        self.confirmed_interval: Optional[float] = None
        self.interval_candidate: Optional[float] = None
        self.interval_streak: int = 0
        self.interval_history: List[int] = []

        self.last_pulse_tick: Optional[int] = None
        self.ticks_since_pulse: Optional[int] = None
        self.phase_boundary_latch_active: bool = False
        self.drift_velocity: float = 0.0
        self.last_frequency: float = 2.0 * math.pi / max(1.0, float(initial_interval))

    def reset(self) -> None:
        self.buffer = [0.0] * self.window_size
        self.tick = 0
        self.estimated_interval = self.DEFAULT_INTERVAL
        self.confirmed_interval = None
        self.interval_candidate = None
        self.interval_streak = 0
        self.interval_history.clear()
        self.last_pulse_tick = None
        self.ticks_since_pulse = None
        self.phase_boundary_latch_active = False
        self.drift_velocity = 0.0
        self.last_frequency = 2.0 * math.pi / self.DEFAULT_INTERVAL

    def update_pulse_interval(self, observed_interval: int) -> None:
        """Applies dual-rate interval update rule from PhaseWorld v3.

        - Shortening intervals indicate hazard acceleration: updated immediately.
        - Lengthening intervals require a 2-observation confirmation streak.
        """
        observed = int(max(self.MIN_INTERVAL, min(self.MAX_INTERVAL, observed_interval)))
        self.interval_history.append(observed)
        self.interval_history = self.interval_history[-self.HISTORY_CAP:]

        current_est = float(self.estimated_interval)
        candidate = self.interval_candidate

        if observed < current_est:
            # Immediate response to cycle shortening (hazard frequency escalation)
            self.estimated_interval = float(observed)
            self.interval_candidate = float(observed)
            self.interval_streak = 1
        elif observed == current_est and candidate is None:
            self.interval_streak = 0
        elif candidate is not None and abs(observed - candidate) <= 1:
            self.interval_candidate = float(round((observed + candidate) / 2.0))
            self.interval_streak += 1
        else:
            self.interval_candidate = float(observed)
            self.interval_streak = 1

        if self.interval_streak >= 2 and self.interval_candidate is not None:
            confirmed = float(self.interval_candidate)
            self.estimated_interval = confirmed
            self.confirmed_interval = confirmed
            self.interval_candidate = None
            self.interval_streak = 0

    def update(
        self,
        cue_or_exposure: float,
        is_pulse_observed: bool = False,
        current_tick: Optional[int] = None,
    ) -> NonstationaryPhaseState:
        """Ergonomic update alias supporting explicit current_tick and cue-based pulse detection."""
        if current_tick is not None and current_tick > self.tick:
            self.tick = current_tick - 1
        is_pulse = is_pulse_observed or (float(cue_or_exposure) > 0.8)
        return self.step(cue_or_exposure, is_pulse_observed=is_pulse)

    def step(
        self,
        cue_or_exposure: float,
        is_pulse_observed: bool = False,
    ) -> NonstationaryPhaseState:
        """Steps resonator with observation, updating spectral wavelets and drift state."""
        self.buffer.pop(0)
        self.buffer.append(float(cue_or_exposure))
        self.tick += 1

        if is_pulse_observed:
            if self.last_pulse_tick is not None:
                interval = self.tick - self.last_pulse_tick
                if interval > 0:
                    self.update_pulse_interval(interval)
            self.last_pulse_tick = self.tick
            self.ticks_since_pulse = 0
            self.phase_boundary_latch_active = False
        elif self.ticks_since_pulse is not None:
            self.ticks_since_pulse += 1

        # Check phase boundary proximity
        expected_interval = (
            self.confirmed_interval
            if self.confirmed_interval is not None
            else self.estimated_interval
        )
        if (
            not is_pulse_observed
            and self.ticks_since_pulse is not None
            and self.ticks_since_pulse >= max(1.0, expected_interval - 1.0)
        ):
            self.phase_boundary_latch_active = True
        else:
            if is_pulse_observed:
                self.phase_boundary_latch_active = False

        # Compute Discrete Fourier Transform across frequency bins
        powers: List[float] = []
        phases: List[float] = []

        for omega in self.frequencies:
            dft_complex = complex(0.0, 0.0)
            for n in range(self.window_size):
                sample = self.buffer[n] * self.window_weights[n]
                angle = -omega * float(n)
                dft_complex += sample * complex(math.cos(angle), math.sin(angle))
            powers.append(abs(dft_complex) ** 2)
            phases.append(cmath.phase(dft_complex))

        total_power = sum(powers)
        if total_power > 1e-9:
            norm_powers = [p / total_power for p in powers]
            spectral_entropy = -sum(p * math.log(max(1e-9, p)) for p in norm_powers) / math.log(float(self.num_bins))
            dominant_idx = int(np.argmax(powers))
            dominant_freq = self.frequencies[dominant_idx]
            dominant_period = 2.0 * math.pi / max(1e-6, dominant_freq)
            current_phase = phases[dominant_idx]
        else:
            norm_powers = [1.0 / float(self.num_bins)] * self.num_bins
            spectral_entropy = 1.0
            dominant_freq = 2.0 * math.pi / max(1.0, expected_interval)
            dominant_period = expected_interval
            current_phase = 0.0

        # Calculate frequency drift velocity
        current_est_freq = 2.0 * math.pi / max(1.0, expected_interval)
        self.drift_velocity = current_est_freq - self.last_frequency
        self.last_frequency = current_est_freq

        # Hazard resonance probability
        phase_diff = abs(current_phase)
        phase_alignment = math.exp(-0.5 * (phase_diff / 0.5) ** 2)
        hazard_prob = float(min(1.0, max(0.0, 0.7 * phase_alignment + 0.3 * (1.0 - spectral_entropy))))
        if self.phase_boundary_latch_active:
            hazard_prob = max(hazard_prob, 0.85)

        harmonic_profile = HarmonicSpectralProfile(
            dominant_frequency=dominant_freq,
            dominant_period=dominant_period,
            spectral_power=norm_powers,
            current_phase=current_phase,
            predicted_phase_advance=dominant_freq,
            spectral_entropy=spectral_entropy,
            storm_resonance_probability=hazard_prob,
            timestamp=time.time(),
        )

        return NonstationaryPhaseState(
            dominant_frequency=dominant_freq,
            dominant_period=dominant_period,
            confirmed_interval=self.confirmed_interval,
            estimated_interval=self.estimated_interval,
            ticks_since_pulse=self.ticks_since_pulse,
            phase_boundary_latch_active=self.phase_boundary_latch_active,
            drift_velocity=self.drift_velocity,
            interval_change_streak=self.interval_streak,
            resonance_hazard_probability=hazard_prob,
            harmonic_profile=harmonic_profile,
            timestamp=time.time(),
        )
