"""Live Audio Gateway and Feature Ingress.

Provides real-time local microphone capture and audio feature extraction.

PRIVACY & SAFETY INVARIANT:
- Disabled by default (enabled=False). Requires explicit user activation.
- All audio processing is strictly local (no remote transmission).
- Extracts spectral features (RMS energy, spectral centroid, zero-crossing rate, mel-filterbanks)
  without persisting raw conversational audio to disk unless explicitly commanded.
"""
from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class AudioFeatures:
    timestamp: float
    rms_energy: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    mel_energies: List[float]  # 16-channel mel filterbank
    is_speech_active: bool
    evidence_kind: EvidenceKind = EvidenceKind.MEASURED_BIOLOGICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "rms_energy": self.rms_energy,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mel_energies": self.mel_energies,
            "is_speech_active": self.is_speech_active,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
        }


class LiveAudioGateway:
    """Safely manages audio stream input, feature extraction, and privacy gating."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        frame_size: int = 512,
        hop_size: int = 256,
        num_mel_bins: int = 16,
        enabled: bool = False,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.num_mel_bins = num_mel_bins
        self.enabled = enabled

        # Circular buffer for incoming PCM samples
        self.buffer = collections.deque(maxlen=sample_rate_hz * 5)  # 5-second buffer

        # Precompute simple triangular mel filterbank
        self.mel_filters = self._build_mel_filterbank(num_mel_bins, frame_size, sample_rate_hz)

        self.last_features: Optional[AudioFeatures] = None
        self._processed_frames = 0

    def _build_mel_filterbank(self, num_bins: int, frame_size: int, sr: int) -> np.ndarray:
        """Constructs simple triangular mel filterbank matrix (num_bins x (frame_size // 2 + 1))."""
        num_fft = frame_size // 2 + 1
        filterbank = np.zeros((num_bins, num_fft), dtype=np.float32)

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(100.0)
        max_mel = hz_to_mel(sr / 2.0)
        mel_points = np.linspace(min_mel, max_mel, num_bins + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((frame_size + 1) * hz_points / sr).astype(int)

        for m in range(1, num_bins + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if k < num_fft and (f_m - f_m_minus) > 0:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if k < num_fft and (f_m_plus - f_m) > 0:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return filterbank

    def ingest_samples(self, samples: np.ndarray) -> None:
        """Pushes raw mono float32 PCM samples in [-1.0, 1.0]."""
        if not self.enabled:
            return
        self.buffer.extend(samples.flatten().tolist())

    def process_latest_frame(self) -> Optional[AudioFeatures]:
        """Extracts features from the latest audio frame."""
        if not self.enabled or len(self.buffer) < self.frame_size:
            # If disabled or empty, return neutral resting features
            return AudioFeatures(
                timestamp=time.time(),
                rms_energy=0.0,
                spectral_centroid_hz=0.0,
                zero_crossing_rate=0.0,
                mel_energies=[0.0] * self.num_mel_bins,
                is_speech_active=False,
                evidence_kind=EvidenceKind.SYNTHETIC_SIMULATION if not self.enabled else EvidenceKind.MEASURED_BIOLOGICAL,
            )

        # Pull frame
        buf_list = list(self.buffer)
        frame = np.array(buf_list[-self.frame_size:], dtype=np.float32)

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-10))

        # 2. Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame + 1e-12)))) / 2.0)

        # 3. FFT Magnitude spectrum
        windowed = frame * np.hanning(self.frame_size)
        spec = np.abs(np.fft.rfft(windowed))

        # 4. Spectral Centroid
        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate_hz)
        spec_sum = np.sum(spec) + 1e-8
        centroid = float(np.sum(freqs * spec) / spec_sum)

        # 5. Mel energies
        mel_raw = np.dot(self.mel_filters, spec)
        mel_log = np.log10(np.maximum(mel_raw, 1e-6)).tolist()

        # Simple energy threshold for voice activity detection (VAD)
        is_speech = rms > 0.02 and 200.0 < centroid < 4000.0

        features = AudioFeatures(
            timestamp=time.time(),
            rms_energy=rms,
            spectral_centroid_hz=centroid,
            zero_crossing_rate=zcr,
            mel_energies=mel_log,
            is_speech_active=is_speech,
            evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
        )

        self.last_features = features
        self._processed_frames += 1
        return features

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.buffer.clear()
            self.last_features = None
