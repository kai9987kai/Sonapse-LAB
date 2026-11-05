"""Project PROMETHEUS Discovery Lab: Learned Action-Conditioned World Model.

Implements an ensemble world model that learns object dynamics, physical collisions,
and fluid interactions from interaction trajectories without modifying frozen core checkpoints.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class ActionConditionedDynamicsModel:
    """Predicts next state s_{t+1} and reward r_t given current state s_t and manipulation action a_t."""

    def __init__(self, state_dim: int = 24, num_actions: int = 11, ensemble_size: int = 3, seed: int = 42):
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.ensemble_size = int(ensemble_size)
        self.rng = np.random.RandomState(seed)

        # Ensemble weight tensors for residual dynamics delta_s = f(s, a)
        # s_dim + num_actions -> s_dim
        in_dim = self.state_dim + self.num_actions
        hidden_dim = 48

        self.W1 = [self.rng.normal(0.0, 0.05, size=(in_dim, hidden_dim)).astype(np.float32) for _ in range(self.ensemble_size)]
        self.b1 = [np.zeros(hidden_dim, dtype=np.float32) for _ in range(self.ensemble_size)]
        self.W2 = [self.rng.normal(0.0, 0.05, size=(hidden_dim, self.state_dim)).astype(np.float32) for _ in range(self.ensemble_size)]
        self.b2 = [np.zeros(self.state_dim, dtype=np.float32) for _ in range(self.ensemble_size)]

    def _one_hot(self, action: int) -> np.ndarray:
        oh = np.zeros(self.num_actions, dtype=np.float32)
        idx = int(action) % self.num_actions
        oh[idx] = 1.0
        return oh

    def predict(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predicts mean next state and epistemic variance across the ensemble."""
        s_arr = np.asarray(state, dtype=np.float32)
        if len(s_arr) > self.state_dim:
            s_arr = s_arr[:self.state_dim]
        elif len(s_arr) < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:len(s_arr)] = s_arr
            s_arr = padded

        a_oh = self._one_hot(action)
        inp = np.concatenate([s_arr, a_oh])

        preds = []
        for i in range(self.ensemble_size):
            h = np.maximum(0.0, np.dot(inp, self.W1[i]) + self.b1[i])  # ReLU
            delta_s = np.dot(h, self.W2[i]) + self.b2[i]
            preds.append(s_arr + delta_s)

        mean_next_state = np.mean(preds, axis=0)
        # Epistemic variance across ensemble members
        epistemic_var = float(np.mean(np.var(preds, axis=0)))
        return mean_next_state, epistemic_var

    def update(self, state: np.ndarray, action: int, next_state: np.ndarray, lr: float = 0.01) -> float:
        """Online gradient step on transition (s_t, a_t, s_{t+1})."""
        s_arr = np.asarray(state, dtype=np.float32)[:self.state_dim]
        ns_arr = np.asarray(next_state, dtype=np.float32)[:self.state_dim]
        target_delta = ns_arr - s_arr
        inp = np.concatenate([s_arr, self._one_hot(action)])

        total_loss = 0.0
        for i in range(self.ensemble_size):
            h = np.maximum(0.0, np.dot(inp, self.W1[i]) + self.b1[i])
            pred_delta = np.dot(h, self.W2[i]) + self.b2[i]
            diff = pred_delta - target_delta
            loss = float(np.mean(diff ** 2))
            total_loss += loss

            # Backprop updates
            grad_W2 = np.outer(h, diff) / len(diff)
            grad_b2 = diff / len(diff)
            grad_h = np.dot(diff, self.W2[i].T) * (h > 0)
            grad_W1 = np.outer(inp, grad_h) / len(grad_h)
            grad_b1 = grad_h / len(grad_h)

            self.W2[i] -= lr * grad_W2
            self.b2[i] -= lr * grad_b2
            self.W1[i] -= lr * grad_W1
            self.b1[i] -= lr * grad_b1

        return total_loss / self.ensemble_size

    train_step = update
