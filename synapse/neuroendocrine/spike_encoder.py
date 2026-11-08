"""Neural Spike Encoder for Action Potential Spike Trains.

Provides biologically principled conversion of continuous sensory inputs:
- Rate Coding: Mean firing rate proportional to input activation lambda_k(t)
- Poisson / Bernoulli Spike Sampling: Probabilistic action potential generation
- Refractory Periods: Absolute refractory clamp preventing unphysical spike bursts
- Deterministic seeded RNG streams for exact scientific reproducibility
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple


class NeuralSpikeEncoder:
    """Encodes continuous input activations into discrete action potential spike trains."""

    def __init__(
        self,
        num_channels: int,
        max_rate_hz: float = 200.0,
        dt_sec: float = 0.001,  # 1 ms bin
        refractory_bins: int = 2,
        seed: int = 42,
        dt: Optional[float] = None,
        refractory_period_steps: Optional[int] = None,
        **kwargs,
    ):
        if dt is not None:
            dt_sec = dt
        if refractory_period_steps is not None:
            refractory_bins = refractory_period_steps
        self.num_channels = num_channels
        self.max_rate_hz = max_rate_hz
        self.dt_sec = dt_sec
        self.refractory_bins = refractory_bins
        self.rng = np.random.default_rng(seed)

        self.refractory_counters = np.zeros(num_channels, dtype=np.int32)
        self.total_spikes_emitted = np.zeros(num_channels, dtype=np.int64)
        self.last_spike_train = np.zeros(num_channels, dtype=np.float32)

    def encode(self, continuous_inputs: Any, deterministic: bool = False) -> torch.Tensor:
        """Alias accepting torch.Tensor or np.ndarray and returning torch.Tensor."""
        if hasattr(continuous_inputs, "detach"):
            continuous_inputs = continuous_inputs.detach().cpu().numpy()
        arr = self.encode_step(continuous_inputs, deterministic=deterministic)
        return torch.from_numpy(arr)

    def encode_step(self, continuous_inputs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Produce a binary spike vector for the current time step dt.
        
        Args:
            continuous_inputs: Normalized stimulus intensity in [0, 1]^num_channels
            deterministic: If True, uses thresholded rate rather than stochastic sampling
        
        Returns:
            spike_vector: Binary vector {0, 1}^num_channels
        """
        x = np.asarray(continuous_inputs, dtype=np.float32).flatten()
        if len(x) < self.num_channels:
            x = np.pad(x, (0, self.num_channels - len(x)))
        elif len(x) > self.num_channels:
            x = x[:self.num_channels]
        x = np.clip(x, 0.0, 1.0)

        # Firing rates in Hz
        rates = x * self.max_rate_hz
        # Spike probability per bin: P(spike) = 1 - exp(-rate * dt)
        p_spike = 1.0 - np.exp(-rates * self.dt_sec)

        spikes = np.zeros(self.num_channels, dtype=np.float32)

        for i in range(self.num_channels):
            if self.refractory_counters[i] > 0:
                self.refractory_counters[i] -= 1
                continue

            if deterministic:
                will_fire = (p_spike[i] >= 0.5)
            else:
                will_fire = (self.rng.random() < p_spike[i])

            if will_fire:
                spikes[i] = 1.0
                self.refractory_counters[i] = self.refractory_bins
                self.total_spikes_emitted[i] += 1

        self.last_spike_train = spikes
        return spikes

    def export_state(self) -> Dict[str, Any]:
        return {
            "num_channels": self.num_channels,
            "max_rate_hz": self.max_rate_hz,
            "dt_sec": self.dt_sec,
            "refractory_bins": self.refractory_bins,
            "refractory_counters": self.refractory_counters.tolist(),
            "total_spikes_emitted": self.total_spikes_emitted.tolist(),
            "last_spike_train": self.last_spike_train.tolist(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if "refractory_counters" in state:
            self.refractory_counters = np.array(state["refractory_counters"], dtype=np.int32)
        if "total_spikes_emitted" in state:
            self.total_spikes_emitted = np.array(state["total_spikes_emitted"], dtype=np.int64)
        if "last_spike_train" in state:
            self.last_spike_train = np.array(state["last_spike_train"], dtype=np.float32)
