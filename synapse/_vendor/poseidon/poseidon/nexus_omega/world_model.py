"""PROJECT NEXUS-Ω: Ensemble World Model with Epistemic Uncertainty & Comparator Feedback.

Implements an ensemble of transition and reward predictors conditioned on
joint observation and causal memory carrier representations.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import PredictionOutput


class SinglePredictor:
    """Deterministic linear-residual predictor with non-linear feature expansion."""

    def __init__(self, d_in: int, d_obs: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # Hidden dimension
        d_hid = 32
        self.W1 = rng.randn(d_in, d_hid) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hid)
        self.W_obs = rng.randn(d_hid, d_obs) * math.sqrt(2.0 / d_hid)
        self.b_obs = np.zeros(d_obs)
        self.W_rew = rng.randn(d_hid, 1) * math.sqrt(2.0 / d_hid)
        self.b_rew = 0.0

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)
        pred_delta = np.dot(h, self.W_obs) + self.b_obs
        pred_rew = float(np.dot(h, self.W_rew)[0] + self.b_rew)
        return pred_delta, pred_rew


class EnsembleWorldModel:
    """Ensemble of K predictors measuring epistemic disagreement and computing comparator residuals."""

    def __init__(self, d_obs: int, d_mem: int, n_actions: int = 4, n_members: int = 3, base_seed: int = 100):
        self.d_obs = d_obs
        self.d_mem = d_mem
        self.n_actions = n_actions
        self.n_members = n_members

        # Input: obs (d_obs) + memory (d_mem) + one-hot action (n_actions)
        self.d_in = d_obs + d_mem + n_actions
        self.members = [
            SinglePredictor(self.d_in, self.d_obs, seed=base_seed + i)
            for i in range(n_members)
        ]
        self.last_prediction: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def _prepare_input(self, obs: np.ndarray, mem: np.ndarray, action: int) -> np.ndarray:
        a_vec = np.zeros(self.n_actions, dtype=np.float32)
        if 0 <= action < self.n_actions:
            a_vec[action] = 1.0
        return np.concatenate([obs, mem, a_vec], axis=0).astype(np.float32)

    def predict(self, obs: np.ndarray, mem: np.ndarray, action: int) -> PredictionOutput:
        """Predicts next state and reward with ensemble epistemic variance."""
        x = self._prepare_input(obs, mem, action)
        pred_deltas = []
        pred_rews = []

        for m in self.members:
            delta, rew = m.forward(x)
            pred_deltas.append(obs + delta)
            pred_rews.append(rew)

        deltas_stack = np.stack(pred_deltas, axis=0)  # (K, d_obs)
        mean_next_obs = np.mean(deltas_stack, axis=0)
        mean_reward = float(np.mean(pred_rews))

        # Disagreement variance (Epistemic Uncertainty)
        variance_per_dim = np.var(deltas_stack, axis=0)
        epistemic_u = float(np.mean(variance_per_dim))

        self.last_prediction = mean_next_obs.copy()
        self.last_action = action

        return PredictionOutput(
            predicted_obs=mean_next_obs,
            predicted_reward=mean_reward,
            predicted_done=0.0,
            epistemic_uncertainty=epistemic_u,
            ensemble_variance=variance_per_dim,
            comparator_error=None,
        )

    def compute_comparator_residual(self, true_next_obs: np.ndarray) -> np.ndarray:
        """Computes realized prediction error e_t = o_{t+1} - \\hat{o}_{t+1}."""
        if self.last_prediction is None:
            return np.zeros(self.d_obs, dtype=np.float32)
        residual = true_next_obs - self.last_prediction
        return residual.astype(np.float32)
