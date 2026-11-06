"""Active inference and epistemic information-seeking engine for SYNAPSE.

Implements Expected Free Energy (EFE) minimization:
    G(a) = Pragmatic_Value(a) + Epistemic_Value(a)
balancing utility maximization with active reduction of hidden state uncertainty.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ActiveInferenceBelief:
    """Belief state over hidden environment variables."""
    state_mean: np.ndarray          # Expected hidden state (e.g., target location, friction, obstacle class)
    state_cov: np.ndarray           # Epistemic covariance / uncertainty matrix
    prior_preference: np.ndarray    # Preferred observation distribution C
    entropy: float = 0.0

    def compute_entropy(self) -> float:
        # Differential entropy of multivariate Gaussian: 0.5 * ln|2*pi*e*Sigma|
        d = len(self.state_mean)
        sign, logdet = np.linalg.slogdet(self.state_cov + 1e-6 * np.eye(d))
        self.entropy = float(0.5 * (d * (1.0 + np.log(2.0 * np.pi)) + logdet))
        return self.entropy


class ActiveInferenceAgent:
    """Action selection agent governed by the Free Energy Principle."""
    def __init__(self, state_dim: int = 4, action_dim: int = 5, temperature: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = temperature
        
        # Transition model B: s_{t+1} = B * s_t + control
        self.B = np.eye(state_dim)
        # Observation model A: o_t = A * s_t
        self.A = np.eye(state_dim)
        
        # Initial belief
        self.belief = ActiveInferenceBelief(
            state_mean=np.zeros(state_dim),
            state_cov=np.eye(state_dim) * 2.0, # High initial uncertainty
            prior_preference=np.array([1.0, 1.0, 0.0, 0.0]) # Target goal preferences
        )
        self.belief.compute_entropy()

    def update_belief(self, observation: np.ndarray, action: np.ndarray) -> ActiveInferenceBelief:
        """Bayesian belief update given new sensory observation and executed action (Kalman-style filter)."""
        o = np.asarray(observation[:self.state_dim])
        
        # 1. Prediction step
        pred_mean = self.B @ self.belief.state_mean + 0.1 * action[:self.state_dim]
        Q = np.eye(self.state_dim) * 0.05 # Process noise
        pred_cov = self.B @ self.belief.state_cov @ self.B.T + Q
        
        # 2. Innovation / Measurement update
        R = np.eye(self.state_dim) * 0.1 # Measurement noise
        innovation = o - (self.A @ pred_mean)
        S = self.A @ pred_cov @ self.A.T + R
        K = pred_cov @ self.A.T @ np.linalg.inv(S)
        
        updated_mean = pred_mean + K @ innovation
        updated_cov = (np.eye(self.state_dim) - K @ self.A) @ pred_cov
        
        self.belief.state_mean = updated_mean
        self.belief.state_cov = updated_cov
        self.belief.compute_entropy()
        return self.belief

    def compute_expected_free_energy(self, candidate_actions: List[np.ndarray]) -> List[Dict[str, float]]:
        """Computes Expected Free Energy G(a) for a discrete set of candidate actions.
        
        G(a) = - Epistemic_Value(a) - Pragmatic_Value(a)
        (We minimize G, so higher epistemic gain and higher utility decrease G).
        """
        results = []
        for a in candidate_actions:
            # 1. Predict future state mean and covariance under action a
            pred_mean = self.B @ self.belief.state_mean + 0.1 * a[:self.state_dim]
            Q = np.eye(self.state_dim) * 0.05
            pred_cov = self.B @ self.belief.state_cov @ self.B.T + Q
            
            # 2. Epistemic Value: Expected Information Gain (mutual information reduction)
            # Actions that probe unfamiliar regions (e.g., inspecting an object or looking ahead)
            # yield large observation variance reduction.
            # Information gain = 0.5 * log(|Sigma_prior| / |Sigma_post|)
            det_prior = max(1e-6, float(np.linalg.det(self.belief.state_cov)))
            # If action is an exploratory/inspection action (e.g. looking closer, probing),
            # expected measurement noise R_eff is smaller:
            is_probing = bool(abs(a[1]) > 0.3 or abs(a[4]) > 0.5) # Steer to look or interact probe
            meas_noise_scale = 0.02 if is_probing else 0.2
            
            R_sim = np.eye(self.state_dim) * meas_noise_scale
            S_sim = self.A @ pred_cov @ self.A.T + R_sim
            K_sim = pred_cov @ self.A.T @ np.linalg.inv(S_sim)
            post_cov = (np.eye(self.state_dim) - K_sim @ self.A) @ pred_cov
            det_post = max(1e-8, float(np.linalg.det(post_cov)))
            
            info_gain = 0.5 * np.log(det_prior / det_post)
            epistemic_value = max(0.0, float(info_gain))
            
            # 3. Pragmatic Value: negative KL divergence between predicted observations and prior preferences C
            # Closer to preferred target state = higher pragmatic value
            target_pref = self.belief.prior_preference
            predicted_obs = self.A @ pred_mean
            pragmatic_error = np.sum((predicted_obs - target_pref) ** 2)
            pragmatic_value = -float(pragmatic_error)
            
            # Total Expected Free Energy G (to minimize)
            # G = - (alpha * Epistemic + beta * Pragmatic)
            # In partially observable high-uncertainty regimes, Epistemic dominates.
            uncertainty_weight = 1.5 if self.belief.entropy > 0.0 else 0.5
            G = - (uncertainty_weight * epistemic_value + 1.0 * pragmatic_value)
            
            results.append({
                'G': float(G),
                'epistemic_value': float(epistemic_value),
                'pragmatic_value': float(pragmatic_value),
                'predicted_mean': pred_mean.tolist()
            })
            
        return results

    def select_action(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Selects optimal action via Expected Free Energy minimization."""
        # Discrete candidate repertoire: [forward, inspect_left, inspect_right, probe_interact, reverse]
        candidates = [
            np.array([0.8, 0.0, 0.0, 0.0, 0.0]),   # Forward
            np.array([0.3, -0.6, 0.0, 0.0, 0.0]),  # Inspect left
            np.array([0.3, 0.6, 0.0, 0.0, 0.0]),   # Inspect right
            np.array([0.1, 0.0, 0.0, 0.0, 1.0]),   # Probe / interact
            np.array([-0.4, 0.0, 0.0, 0.0, 0.0]),  # Back off
        ]
        
        efe_scores = self.compute_expected_free_energy(candidates)
        g_vals = np.array([res['G'] for res in efe_scores])
        
        # Softmax over negative free energy: P(a) ~ exp(-G / T)
        logits = -g_vals / max(1e-4, self.temperature)
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        
        best_idx = int(np.argmax(probs))
        chosen_action = candidates[best_idx]
        
        # Execute belief update
        self.update_belief(observation, chosen_action)
        
        telemetry = {
            'chosen_action_index': best_idx,
            'efe_scores': efe_scores,
            'selected_epistemic_gain': efe_scores[best_idx]['epistemic_value'],
            'selected_pragmatic_value': efe_scores[best_idx]['pragmatic_value'],
            'belief_entropy': float(self.belief.entropy),
            'action_probabilities': probs.tolist()
        }
        return chosen_action, telemetry
