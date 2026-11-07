"""Precision-Weighted Hierarchical Active Inference Engine.

Grounding:
Based on Karl Friston's Free Energy Principle and Active Inference (Friston et al. 2023;
Da Costa et al. 2020), agents act to minimize Expected Free Energy (EFE), G(pi):
    G(pi) = - ExtrinsicUtility(pi) - beta_epistemic * EpistemicInformationGain(pi)

Components:
1. Pragmatic / Extrinsic Value: Log-likelihood of desired sensory outcomes under target priors.
2. Epistemic Value (Bayesian Surprise / Salience): Expected Kullback-Leibler divergence
   between prior and posterior hidden states, driving purposeful exploration to resolve
   connectome uncertainty.
3. Neuromodulatory Precision Weighting:
   - Acetylcholine (ACh): modulates sensory observation precision (likelihood sharpness).
   - Dopamine (DA): modulates action policy selection precision (inverse temperature).
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActiveInferenceEngine:
    """Precision-weighted active inference decision maker and epistemic explorer."""

    def __init__(
        self,
        observation_dim: int = 155,
        action_dim: int = 5,
        hidden_dim: int = 32,
        horizon: int = 3,
        beta_epistemic: float = 0.35,
        seed: int = 42,
    ):
        self.obs_dim = int(observation_dim)
        self.act_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.beta_epistemic = float(beta_epistemic)
        self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)
        self.updates = 0
        self.last_efe = 0.0
        self.last_epistemic_gain = 0.0
        self.last_pragmatic_value = 0.0

        # State transition model prior & uncertainty covariance
        self.state_prior_mean = np.zeros(self.hidden_dim, dtype=np.float32)
        self.state_uncertainty = np.ones(self.hidden_dim, dtype=np.float32)

    def evaluate_expected_free_energy(
        self,
        candidate_actions: np.ndarray,
        current_features: np.ndarray,
        ach_precision: float = 1.0,
        dopamine_gain: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evaluates EFE for a batch of candidate actions and returns softmax action probabilities."""
        n_candidates = len(candidate_actions)
        efe_values = np.zeros(n_candidates, dtype=np.float32)
        pragmatic_vals = np.zeros(n_candidates, dtype=np.float32)
        epistemic_vals = np.zeros(n_candidates, dtype=np.float32)

        feat = np.asarray(current_features, dtype=np.float32).flatten()[: self.hidden_dim]
        if len(feat) < self.hidden_dim:
            feat = np.pad(feat, (0, self.hidden_dim - len(feat)))

        ach = float(np.clip(ach_precision, 0.2, 3.0))
        da = float(np.clip(dopamine_gain, 0.2, 3.0))

        for i, a in enumerate(candidate_actions):
            act = np.asarray(a, dtype=np.float32)
            # Projected forward state shift
            step_shift = 0.1 * np.dot(feat[: self.act_dim], act)
            # Pragmatic value: movement velocity + positive feature alignment
            prag = float(step_shift + 0.15 * np.sum(np.abs(act)))
            # Epistemic value: expected information gain (reduction in state variance)
            state_cov = self.state_uncertainty / (1.0 + ach * np.abs(act.mean()))
            info_gain = float(0.5 * np.sum(np.log(np.maximum(self.state_uncertainty / np.maximum(state_cov, 1e-5), 1.0))))

            pragmatic_vals[i] = prag
            epistemic_vals[i] = info_gain
            # Expected Free Energy: lower is better (or minimize G)
            efe_values[i] = -prag - self.beta_epistemic * info_gain

        # Softmax probability with dopamine policy precision
        scaled_logits = -da * efe_values
        scaled_logits -= np.max(scaled_logits)
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        best_idx = int(np.argmin(efe_values))
        self.last_efe = float(efe_values[best_idx])
        self.last_pragmatic_value = float(pragmatic_vals[best_idx])
        self.last_epistemic_gain = float(epistemic_vals[best_idx])

        diagnostics = {
            "enabled": True,
            "min_efe": round(float(efe_values[best_idx]), 4),
            "mean_efe": round(float(np.mean(efe_values)), 4),
            "pragmatic_value": round(float(pragmatic_vals[best_idx]), 4),
            "epistemic_gain": round(float(epistemic_vals[best_idx]), 4),
            "ach_precision": round(ach, 3),
            "dopamine_gain": round(da, 3),
            "policy_entropy": round(float(-np.sum(probs * np.log(np.maximum(probs, 1e-8)))), 4),
        }
        return probs, diagnostics

    def select_action(
        self,
        base_action: np.ndarray,
        features: np.ndarray,
        num_candidates: int = 8,
        ach_precision: float = 1.0,
        dopamine_gain: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Deterministically evaluates candidate variations around base_action and selects minimum EFE action."""
        base = np.asarray(base_action, dtype=np.float32).flatten()[: self.act_dim]
        candidates = [base]

        # Deterministic coordinate & diagonal basis perturbations (0 RNG samples consumed)
        basis_shifts = [
            np.array([ 0.15,  0.00,  0.00,  0.00,  0.00], dtype=np.float32),
            np.array([-0.15,  0.00,  0.00,  0.00,  0.00], dtype=np.float32),
            np.array([ 0.00,  0.15,  0.00,  0.00,  0.00], dtype=np.float32),
            np.array([ 0.00, -0.15,  0.00,  0.00,  0.00], dtype=np.float32),
            np.array([ 0.00,  0.00,  0.15,  0.00,  0.00], dtype=np.float32),
            np.array([ 0.00,  0.00, -0.15,  0.00,  0.00], dtype=np.float32),
            np.array([ 0.10,  0.10,  0.00,  0.00,  0.00], dtype=np.float32),
        ]
        for p in basis_shifts[: max(0, num_candidates - 1)]:
            cand = np.clip(base + p[: self.act_dim], -1.0, 1.0)
            candidates.append(cand)

        candidate_arr = np.array(candidates, dtype=np.float32)
        probs, diag = self.evaluate_expected_free_energy(
            candidate_arr, features, ach_precision=ach_precision, dopamine_gain=dopamine_gain
        )
        selected_idx = int(np.argmax(probs))
        return candidate_arr[selected_idx], diag

    def update_beliefs(self, observed_transition_error: float) -> None:
        """Updates internal uncertainty estimates based on prediction accuracy."""
        err = float(np.clip(observed_transition_error, 0.0, 5.0))
        self.state_uncertainty = 0.95 * self.state_uncertainty + 0.05 * (1.0 + err)
        self.updates += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "horizon": self.horizon,
            "beta_epistemic": self.beta_epistemic,
            "last_efe": round(self.last_efe, 4),
            "last_pragmatic_value": round(self.last_pragmatic_value, 4),
            "last_epistemic_gain": round(self.last_epistemic_gain, 4),
            "mean_uncertainty": round(float(np.mean(self.state_uncertainty)), 4),
            "updates": int(self.updates),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "state_prior_mean": self.state_prior_mean.tolist(),
            "state_uncertainty": self.state_uncertainty.tolist(),
            "updates": int(self.updates),
            "last_efe": float(self.last_efe),
            "last_epistemic_gain": float(self.last_epistemic_gain),
            "last_pragmatic_value": float(self.last_pragmatic_value),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "state_prior_mean" in state:
            self.state_prior_mean = np.array(state["state_prior_mean"], dtype=np.float32)
        if "state_uncertainty" in state:
            self.state_uncertainty = np.array(state["state_uncertainty"], dtype=np.float32)
        self.updates = int(state.get("updates", 0))
        self.last_efe = float(state.get("last_efe", 0.0))
        self.last_epistemic_gain = float(state.get("last_epistemic_gain", 0.0))
        self.last_pragmatic_value = float(state.get("last_pragmatic_value", 0.0))

    def run_active_inference_assay(self, trials: int = 10, seed: int = 42) -> Dict[str, Any]:
        """Runs controlled assay evaluating pragmatic vs epistemic trade-offs."""
        rng = np.random.default_rng(seed)
        feat = rng.standard_normal(self.hidden_dim).astype(np.float32)
        base_act = np.zeros(self.act_dim, dtype=np.float32)

        efes = []
        epistemic_ratios = []
        for _ in range(trials):
            act, diag = self.select_action(base_act, feat, num_candidates=8, ach_precision=1.2, dopamine_gain=1.5)
            efes.append(diag["min_efe"])
            denom = abs(diag["pragmatic_value"]) + abs(diag["epistemic_gain"]) + 1e-5
            epistemic_ratios.append(abs(diag["epistemic_gain"]) / denom)

        return {
            "schema": "active-inference-free-energy-v1",
            "status": "completed",
            "trials": trials,
            "mean_min_efe": round(float(np.mean(efes)), 4),
            "mean_epistemic_ratio": round(float(np.mean(epistemic_ratios)), 4),
            "exploration_entropy_mean": round(float(np.mean([diag["policy_entropy"]])), 4),
            "action_norm": round(float(np.linalg.norm(act)), 4),
        }


class ActiveInferenceAgent:
    """Convenience agent interface wrapping ActiveInferenceEngine for active inference assays."""

    def __init__(self, state_dim: int = 155, action_dim: int = 5, seed: int = 42) -> None:
        self.engine = ActiveInferenceEngine(observation_dim=state_dim, action_dim=action_dim, seed=seed)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_act = np.zeros(self.action_dim, dtype=np.float32)
        candidates = [base_act]
        for i in range(self.action_dim):
            p = np.zeros(self.action_dim, dtype=np.float32)
            p[i] = 0.2
            candidates.append(p)
        cand_arr = np.array(candidates, dtype=np.float32)
        probs, diag = self.engine.evaluate_expected_free_energy(cand_arr, obs)
        best_idx = int(np.argmax(probs))
        tele = {
            'efe_scores': [float(diag.get('min_efe', 0.0))] * self.action_dim,
            'selected_epistemic_gain': float(diag.get('epistemic_gain', 0.0)),
            'selected_pragmatic_value': float(diag.get('pragmatic_value', 0.0)),
            'belief_entropy': float(diag.get('policy_entropy', 0.0)),
        }
        return cand_arr[best_idx], tele

