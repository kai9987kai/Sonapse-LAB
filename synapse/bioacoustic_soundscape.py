"""Bioacoustic Soundscape & Acoustic Niche Partitioning Engine.

Theoretical Grounding:
Inspired by Bernie Krause's Acoustic Niche Hypothesis (ANH, 1987) and Soundscape Ecology
(Pijanowski et al., 2011; Sueur et al., 2008):
1. Biophony: Autonomous ecological organisms partition the acoustic frequency spectrum
   and temporal cadences to communicate with conspecifics without mutual acoustic masking.
2. Interspecific Acoustic Signatures:
   - Foragers emit harmonic food contact trills (200 - 450 Hz, bins 2-4).
   - Sentinels emit high-frequency alarm whistles (1.5 - 3.2 kHz, bins 7-9).
   - Artisans emit low-frequency structural resonance (80 - 180 Hz, bins 0-2).
   - Predators emit broadband stalking hunting chirps (600 - 950 Hz, bins 5-6).
3. Acoustic Niche Partitioning (ANP) Metric:
   Measures spectral niche segregation across species:
       ANP = 1.0 - (1 / (K * (K - 1) / 2)) * sum_{j < k} CosineSimilarity(S_j, S_k)
   where S_j is the long-term power spectral density of niche j.
4. Cochlear Transduction:
   Converts the spatial acoustic pressure field into 16-channel tonotopic firing rates
   directly coupled to the 256-neuron auditory column of the 12k MultiConnectomeFusionEngine.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclasses.dataclass
class VocalizationEvent:
    vocal_id: str
    agent_id: str
    niche: str
    x: float
    y: float
    amplitude: float
    fundamental_hz: float
    spectral_profile: List[float]  # 16-channel energy
    duration_steps: int
    remaining_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocal_id": self.vocal_id,
            "agent_id": self.agent_id,
            "niche": self.niche,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "amplitude": round(float(self.amplitude), 3),
            "fundamental_hz": round(float(self.fundamental_hz), 1),
            "remaining_steps": self.remaining_steps,
        }


@dataclasses.dataclass
class SoundscapeTelemetry:
    step_count: int
    active_vocalizations: int
    ambient_db_spl: float
    anp_index: float
    dominant_niche: str
    channel_energies: List[float]
    wallclock_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "active_vocalizations": self.active_vocalizations,
            "ambient_db_spl": round(float(self.ambient_db_spl), 2),
            "anp_index": round(float(self.anp_index), 4),
            "dominant_niche": self.dominant_niche,
            "channel_energies": [round(float(v), 4) for v in self.channel_energies],
            "wallclock_timestamp": self.wallclock_timestamp,
        }


class BioacousticSoundscapeEngine:
    """Simulates spatial acoustic wave propagation, spectral niche allocation, and cochlear coupling."""

    NUM_CHANNELS = 16
    FREQ_MIN = 60.0
    FREQ_MAX = 8000.0

    NICHE_ACOUSTICS = {
        "artisan": {
            "f0": 110.0,
            "f0_std": 15.0,
            "harmonic_falloff": 0.5,
            "center_bin": 1.2,
            "bin_spread": 1.2,
            "default_amp": 0.7,
        },
        "forager": {
            "f0": 280.0,
            "f0_std": 30.0,
            "harmonic_falloff": 0.6,
            "center_bin": 3.5,
            "bin_spread": 1.4,
            "default_amp": 0.8,
        },
        "predator": {
            "f0": 720.0,
            "f0_std": 50.0,
            "harmonic_falloff": 0.7,
            "center_bin": 5.8,
            "bin_spread": 1.5,
            "default_amp": 0.9,
        },
        "sentinel": {
            "f0": 2200.0,
            "f0_std": 150.0,
            "harmonic_falloff": 0.8,
            "center_bin": 8.5,
            "bin_spread": 1.3,
            "default_amp": 1.2,
        },
    }

    def __init__(self, biome_half: float = 10.0, seed: int = 42):
        self.biome_half = float(biome_half)
        self.rng = np.random.RandomState(seed)
        self.step_count = 0

        # Frequency channel center frequencies (log-spaced from 60 Hz to 8000 Hz)
        self.center_freqs = [
            self.FREQ_MIN * ((self.FREQ_MAX / self.FREQ_MIN) ** (k / (self.NUM_CHANNELS - 1)))
            for k in range(self.NUM_CHANNELS)
        ]

        # Current spatial-averaged ambient pressure spectrum across 16 channels
        self.channel_pressures = np.zeros(self.NUM_CHANNELS, dtype=np.float32)

        # Active acoustic emissions in transit
        self.active_vocalizations: List[VocalizationEvent] = []

        # Long-term power spectral density per niche for ANP calculation
        self.niche_psd = {
            "artisan": np.zeros(self.NUM_CHANNELS, dtype=np.float32),
            "forager": np.zeros(self.NUM_CHANNELS, dtype=np.float32),
            "predator": np.zeros(self.NUM_CHANNELS, dtype=np.float32),
            "sentinel": np.zeros(self.NUM_CHANNELS, dtype=np.float32),
        }
        self.niche_counts = {k: 0 for k in self.niche_psd}

        # Spectral history for waterfall spectrogram visualization (rolling buffer of 64 steps)
        self.spectrogram_history: List[List[float]] = []
        self.max_history = 64

        self._last_telemetry: Optional[SoundscapeTelemetry] = None

    def synthesize_spectral_profile(self, niche: str, amp: float) -> Tuple[float, List[float]]:
        """Generates a 16-channel spectral profile for an organism's vocalization."""
        cfg = self.NICHE_ACOUSTICS.get(niche, self.NICHE_ACOUSTICS["forager"])
        f0 = float(self.rng.normal(cfg["f0"], cfg["f0_std"]))
        f0 = max(40.0, min(7500.0, f0))

        # Gaussian spectral envelope in log-frequency space
        c_bin = cfg["center_bin"]
        spread = cfg["bin_spread"]
        bins = np.arange(self.NUM_CHANNELS, dtype=np.float32)
        envelope = np.exp(-0.5 * ((bins - c_bin) / max(0.5, spread)) ** 2)

        # Add harmonic overtones (2*f0, 3*f0)
        harmonic_decay = cfg["harmonic_falloff"]
        profile = envelope.copy()
        for h in [2, 3]:
            h_bin = c_bin + math.log2(h) * 2.2
            if h_bin < self.NUM_CHANNELS:
                h_env = np.exp(-0.5 * ((bins - h_bin) / max(0.4, spread * 0.8)) ** 2)
                profile += (harmonic_decay ** (h - 1)) * h_env

        # Normalize and scale by amplitude
        p_max = float(np.max(profile))
        if p_max > 1e-6:
            profile = (profile / p_max) * float(amp)
        return f0, profile.tolist()

    def emit_vocalization(
        self,
        agent_id: str,
        niche: str,
        x: float,
        y: float,
        amplitude: Optional[float] = None,
        duration_steps: int = 3,
    ) -> VocalizationEvent:
        """Registers a new acoustic vocalization in the virtual biome."""
        cfg = self.NICHE_ACOUSTICS.get(niche, self.NICHE_ACOUSTICS["forager"])
        amp = float(amplitude if amplitude is not None else cfg["default_amp"])
        f0, profile = self.synthesize_spectral_profile(niche, amp)

        vocal_id = f"voc_{self.step_count}_{agent_id}_{len(self.active_vocalizations)}"
        event = VocalizationEvent(
            vocal_id=vocal_id,
            agent_id=agent_id,
            niche=niche,
            x=float(x),
            y=float(y),
            amplitude=amp,
            fundamental_hz=f0,
            spectral_profile=profile,
            duration_steps=max(1, duration_steps),
            remaining_steps=max(1, duration_steps),
        )
        self.active_vocalizations.append(event)

        # Accumulate long-term PSD
        prof_arr = np.array(profile, dtype=np.float32)
        if niche in self.niche_psd:
            self.niche_psd[niche] += prof_arr
            self.niche_counts[niche] += 1

        return event

    def inject_external_audio(self, pcm_chunk: np.ndarray, sample_rate: int = 16000) -> None:
        """Injects live external microphone audio energy into the biome soundscape."""
        if len(pcm_chunk) < 64:
            return
        # Compute real FFT power spectrum
        fft_mags = np.abs(np.fft.rfft(pcm_chunk))
        fft_freqs = np.fft.rfftfreq(len(pcm_chunk), d=1.0 / sample_rate)

        # Map to 16 log-spaced channels
        ext_channels = np.zeros(self.NUM_CHANNELS, dtype=np.float32)
        for i in range(self.NUM_CHANNELS):
            f_low = self.center_freqs[max(0, i - 1)] if i > 0 else 20.0
            f_high = self.center_freqs[min(self.NUM_CHANNELS - 1, i + 1)]
            idx = np.where((fft_freqs >= f_low) & (fft_freqs <= f_high))[0]
            if len(idx) > 0:
                ext_channels[i] = float(np.mean(fft_mags[idx]))

        # Normalize and add to current pressures
        ext_max = float(np.max(ext_channels))
        if ext_max > 1e-4:
            norm_ext = (ext_channels / ext_max) * 0.8
            self.channel_pressures = np.clip(self.channel_pressures + norm_ext, 0.0, 3.0)

    def calculate_anp_index(self) -> float:
        """Computes the Acoustic Niche Partitioning (ANP) index via spectral orthogonality.

        ANP in [0, 1]:
          1.0 = completely orthogonal frequency allocations (zero cross-species masking).
          0.0 = completely identical frequency profiles (maximum competition/masking).
        """
        niches = [k for k, c in self.niche_counts.items() if c > 0]
        if len(niches) < 2:
            return 1.0  # Insufficient active species to measure competition

        pairwise_overlaps = []
        for i in range(len(niches)):
            for j in range(i + 1, len(niches)):
                n1, n2 = niches[i], niches[j]
                v1 = self.niche_psd[n1] / max(1, self.niche_counts[n1])
                v2 = self.niche_psd[n2] / max(1, self.niche_counts[n2])
                norm1 = float(np.linalg.norm(v1))
                norm2 = float(np.linalg.norm(v2))
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
                    pairwise_overlaps.append(cosine)
                else:
                    pairwise_overlaps.append(0.0)

        mean_overlap = float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0
        anp = max(0.0, min(1.0, 1.0 - mean_overlap))
        return anp

    def get_acoustic_at_point(self, x: float, y: float) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Calculates total acoustic pressure, 16ch spectrum, and niche components at (x, y)."""
        tot_amp = 0.0
        tot_spectrum = np.zeros(self.NUM_CHANNELS, dtype=np.float32)
        niche_amps = {"artisan": 0.0, "forager": 0.0, "predator": 0.0, "sentinel": 0.0}

        for voc in self.active_vocalizations:
            d2 = (x - voc.x) ** 2 + (y - voc.y) ** 2
            # Inverse-distance attenuation with spatial damping
            atten = 1.0 / (1.0 + 0.18 * d2)
            eff_amp = voc.amplitude * atten
            tot_amp += eff_amp
            tot_spectrum += np.array(voc.spectral_profile, dtype=np.float32) * atten
            if voc.niche in niche_amps:
                niche_amps[voc.niche] += eff_amp

        # Background ambient floor
        tot_spectrum += self.channel_pressures * 0.1
        tot_amp += float(np.sum(self.channel_pressures)) * 0.05
        return tot_amp, tot_spectrum, niche_amps

    def step(self) -> SoundscapeTelemetry:
        """Advances acoustic physics by one time step: decay, wave propagation, and telemetry."""
        self.step_count += 1

        # 1. Reverberant decay of ambient channel pressures (6 dB / step)
        decay_factor = 0.78
        self.channel_pressures *= decay_factor

        # 2. Add current vocalizations to ambient channel pressures
        active_remaining = []
        niche_energy = {"artisan": 0.0, "forager": 0.0, "predator": 0.0, "sentinel": 0.0}

        for voc in self.active_vocalizations:
            prof = np.array(voc.spectral_profile, dtype=np.float32)
            self.channel_pressures += prof * 0.25
            if voc.niche in niche_energy:
                niche_energy[voc.niche] += voc.amplitude

            voc.remaining_steps -= 1
            if voc.remaining_steps > 0:
                active_remaining.append(voc)
        self.active_vocalizations = active_remaining

        # 3. Ambient dB SPL (Sound Pressure Level) estimation
        rms_pressure = float(np.sqrt(np.mean(self.channel_pressures ** 2)))
        db_spl = 20.0 * math.log10(max(1e-4, rms_pressure) / 1e-4)
        db_spl = max(30.0, min(110.0, 30.0 + db_spl * 0.8))

        # 4. Dominant active niche
        dominant_niche = max(niche_energy.items(), key=lambda kv: kv[1])[0] if any(v > 0 for v in niche_energy.values()) else "ambient"

        # 5. Acoustic Niche Partitioning index
        anp = self.calculate_anp_index()

        # 6. Record rolling spectrogram history
        self.spectrogram_history.append([float(v) for v in self.channel_pressures])
        if len(self.spectrogram_history) > self.max_history:
            self.spectrogram_history.pop(0)

        telemetry = SoundscapeTelemetry(
            step_count=self.step_count,
            active_vocalizations=len(self.active_vocalizations),
            ambient_db_spl=db_spl,
            anp_index=anp,
            dominant_niche=dominant_niche,
            channel_energies=self.channel_pressures.tolist(),
            wallclock_timestamp=time.time(),
        )
        self._last_telemetry = telemetry
        return telemetry

    def to_cochlear_rates(self) -> np.ndarray:
        """Converts current 16-channel soundscape pressures into firing rates for the 256-neuron auditory column."""
        # 16 channels, each mapped to 16 tonotopic neurons = 256 total auditory neurons
        rates = np.zeros(256, dtype=np.float32)
        norm_channels = np.clip(self.channel_pressures / (np.max(self.channel_pressures) + 1e-5), 0.0, 1.0)
        for ch in range(self.NUM_CHANNELS):
            ch_val = float(norm_channels[ch])
            # Add minor tonotopic neuron variation within each frequency band
            neuron_start = ch * 16
            neuron_end = neuron_start + 16
            rates[neuron_start:neuron_end] = ch_val * 0.9 + 0.1 * np.random.uniform(0.8, 1.2, 16)
        return np.clip(rates, 0.0, 1.0)

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured JSON summary of the soundscape."""
        return {
            "step_count": self.step_count,
            "center_freqs": [round(f, 1) for f in self.center_freqs],
            "active_vocalizations": [v.to_dict() for v in self.active_vocalizations],
            "spectrogram_recent": self.spectrogram_history[-16:] if self.spectrogram_history else [],
            "telemetry": self._last_telemetry.to_dict() if self._last_telemetry else None,
        }
