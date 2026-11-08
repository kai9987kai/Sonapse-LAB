"""Auditory Spike Encoder.

Biomimetic cochlear model converting acoustic energy and frequency bands
into simulated auditory nerve fiber spike trains.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from synapse.multimodal_live.audio_gateway import AudioFeatures
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class AuditorySpikeEncoder:
    """Translates audio spectral bands into Poisson/Bernoulli auditory nerve spike trains."""

    def __init__(
        self,
        num_channels: int = 16,
        max_firing_rate_hz: float = 250.0,
        spontaneous_rate_hz: float = 10.0,
        dt: float = 0.01,
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.max_rate = max_firing_rate_hz
        self.spontaneous_rate = spontaneous_rate_hz
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        # Baseline channel thresholds
        self.thresholds = np.linspace(-3.0, 0.0, num_channels)
        self.last_spikes = np.zeros(num_channels, dtype=np.int32)
        self.last_rates = np.full(num_channels, spontaneous_rate_hz)

    def encode(self, features: Optional[AudioFeatures]) -> Tuple[np.ndarray, np.ndarray]:
        """Encodes audio features into firing rates and binary spike trains.

        Returns:
            spikes: (num_channels,) binary array {0, 1}
            firing_rates: (num_channels,) firing rates in Hz
        """
        if features is None or not features.is_speech_active and features.rms_energy < 1e-4:
            # Spontaneous baseline firing
            rates = np.full(self.num_channels, self.spontaneous_rate)
            probs = rates * self.dt
            spikes = (self.rng.rand(self.num_channels) < probs).astype(np.int32)
            self.last_spikes = spikes
            self.last_rates = rates
            return spikes, rates

        mel = np.array(features.mel_energies[:self.num_channels], dtype=np.float32)
        if len(mel) < self.num_channels:
            mel = np.pad(mel, (0, self.num_channels - len(mel)), constant_values=-4.0)

        # Rate transduction: logistic curve mapping log-mel energy to firing rate
        # r = spontaneous + (max - spontaneous) / (1 + exp(-(mel - threshold) / slope))
        rates = self.spontaneous_rate + (self.max_rate - self.spontaneous_rate) / (
            1.0 + np.exp(-(mel - self.thresholds) / 0.8)
        )

        probs = np.clip(rates * self.dt, 0.0, 1.0)
        spikes = (self.rng.rand(self.num_channels) < probs).astype(np.int32)

        self.last_spikes = spikes
        self.last_rates = rates
        return spikes, rates

    def to_feature_vector(self) -> np.ndarray:
        """Returns normalized 32-d auditory feature vector (spikes + rates)."""
        norm_rates = self.last_rates / max(self.max_rate, 1.0)
        return np.concatenate([self.last_spikes.astype(np.float32), norm_rates], axis=0)
