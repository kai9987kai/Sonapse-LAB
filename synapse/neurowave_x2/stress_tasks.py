"""Communication Stress Environments & Synthetic Benchmarks for Sonapse NeuroWave-X2.

Provides strictly isolated communication testing environments:
1. Strict Source/Target Separation Task:
   Region A receives cue X; Region B determines motor action without direct access to X.
2. Compositional Information Benchmark:
   Receiver must reconstruct non-linear relational combinations (e.g. x1*x2, x3*x4)
   from transmitted vectors without sensory shortcuts.
3. Communication Stress Suite:
   Parametric noise, burst packet loss, extended latency, route lesions, and distractors.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StrictSeparationBenchmark:
    """Benchmark isolating communication capacity: Region A observes cue, Region B acts."""

    def __init__(self, cue_dim: int = 8, action_dim: int = 4, delay_steps: int = 2, seed: int = 42) -> None:
        self.cue_dim = int(cue_dim)
        self.action_dim = int(action_dim)
        self.delay_steps = int(delay_steps)
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.active_cue = np.zeros(self.cue_dim, dtype=np.float32)
        self.expected_action = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.step_count = 0
        # Discrete cue category 0..action_dim-1 encoded as one-hot or random vector
        self.expected_action = int(self.rng.randint(0, self.action_dim))
        self.active_cue = np.zeros(self.cue_dim, dtype=np.float32)
        self.active_cue[self.expected_action % self.cue_dim] = 1.0
        # Add random background distractor to non-active dimensions
        self.active_cue += self.rng.normal(0.0, 0.05, size=self.cue_dim).astype(np.float32)
        return self.active_cue.copy()

    def evaluate_receiver_action(self, receiver_action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluates whether the isolated receiver deduced the correct action exclusively via comm."""
        correct = (int(receiver_action) == self.expected_action)
        reward = 1.0 if correct else -0.5
        info = {
            "expected": self.expected_action,
            "chosen": int(receiver_action),
            "correct": correct,
            "step": self.step_count,
        }
        self.step_count += 1
        return reward, correct, info


class CompositionalInformationBenchmark:
    """Tests preservation of structured non-linear relational information across communication channels."""

    def __init__(self, vector_dim: int = 8, seed: int = 42) -> None:
        self.vector_dim = int(vector_dim)
        self.rng = np.random.RandomState(seed)

    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates input vector x and expected non-linear compositional target y.
        
        Input: x in [-1, 1]^8
        Target: [x1, x2, ..., x8, x1*x2, x3*x4, x5*x6, x7*x8, sin(x1*pi), cos(x3*pi)]
        """
        x = self.rng.uniform(-1.0, 1.0, size=self.vector_dim).astype(np.float32)
        relations = [
            float(x[0] * x[1]),
            float(x[2] * x[3]),
            float(x[4] * x[5]),
            float(x[6] * x[7]),
            float(math.sin(x[0] * math.pi)),
            float(math.cos(x[2] * math.pi)),
        ]
        target = np.concatenate([x, np.array(relations, dtype=np.float32)])
        return x, target

    def evaluate_reconstruction(self, target: np.ndarray, decoded_output: np.ndarray) -> Dict[str, float]:
        """Calculates linear vs non-linear relational reconstruction error."""
        linear_mse = float(np.mean((target[:self.vector_dim] - decoded_output[:self.vector_dim]) ** 2))
        non_linear_mse = float(np.mean((target[self.vector_dim:] - decoded_output[self.vector_dim:]) ** 2))
        total_mse = float(np.mean((target - decoded_output) ** 2))
        return {
            "total_mse": total_mse,
            "linear_component_mse": linear_mse,
            "non_linear_composition_mse": non_linear_mse,
            "compositional_accuracy": float(max(0.0, 1.0 - non_linear_mse)),
        }


class CommunicationStressHarness:
    """Generates controlled synthetic channel impairments: burst loss, noise, latency jitter."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.burst_loss_active = False
        self.burst_counter = 0

    def should_drop_packet(
        self,
        base_loss_rate: float = 0.05,
        burst_loss_prob: float = 0.02,
        burst_duration: int = 4,
    ) -> bool:
        """Simulates both i.i.d. random packet loss and Gilbert-Elliott burst dropouts."""
        if self.burst_loss_active:
            self.burst_counter -= 1
            if self.burst_counter <= 0:
                self.burst_loss_active = False
            return True

        if burst_loss_prob > 0.0 and self.rng.uniform(0.0, 1.0) < burst_loss_prob:
            self.burst_loss_active = True
            self.burst_counter = burst_duration
            return True

        return bool(self.rng.uniform(0.0, 1.0) < base_loss_rate)

    def apply_channel_noise(self, payload: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Injects Gaussian channel noise with declared Signal-to-Noise Ratio (SNR in dB)."""
        signal_power = float(np.mean(payload ** 2)) + 1e-12
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self.rng.normal(0.0, math.sqrt(noise_power), size=payload.shape).astype(payload.dtype)
        return payload + noise
