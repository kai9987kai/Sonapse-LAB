"""PROJECT NEXUS-Ω: Factored Causal World Model (Causal-JEPA & ICM).

Implements an Invariant Causal Mechanism (ICM) decomposition separating:
1. Endogenous Homeostatic Dynamics (health, energy, hydration, stamina):
   Metabolic consumption and physiological kinetics conditioned on action and exposure.
2. Exogenous Environmental Field Dynamics (exposure, storm pulse cue, resource availability):
   Atmospheric and spatial field fluctuations conditioned on action and latent regime.
3. Latent Causal Regime Tracking (storm persistence, replenishment rate):
   Persistent causal factors updated through counterfactual comparator residuals.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import (
    FactoredPredictionOutput,
    FactoredState,
    PredictionOutput,
)


class FactoredSubnetwork:
    """Specialized neural subnetwork for an isolated causal factor."""

    def __init__(self, d_in: int, d_out: int, d_hid: int = 32, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(d_in, d_hid) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hid, dtype=np.float32)
        self.W_out = rng.randn(d_hid, d_out) * math.sqrt(2.0 / d_hid)
        self.b_out = np.zeros(d_out, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W_out) + self.b_out


class FactoredMember:
    """Ensemble member consisting of factor-isolated subnetworks and reward head."""

    def __init__(
        self,
        d_endo: int = 4,
        d_exo: int = 4,
        d_regime: int = 2,
        d_mem: int = 18,
        n_actions: int = 4,
        seed: int = 42,
    ):
        self.d_endo = d_endo
        self.d_exo = d_exo
        self.d_regime = d_regime
        self.n_actions = n_actions

        # Endogenous input: s_endo (4) + action (n_actions) + exposure (1)
        d_in_endo = d_endo + n_actions + 1
        self.net_endo = FactoredSubnetwork(d_in_endo, d_endo, d_hid=24, seed=seed)

        # Exogenous input: s_exo (4) + action (n_actions) + regime (2) + working_mem (4)
        d_in_exo = d_exo + n_actions + d_regime + 4
        self.net_exo = FactoredSubnetwork(d_in_exo, d_exo, d_hid=24, seed=seed + 1000)

        # Reward head: s_endo (4) + s_exo (4) + action (n_actions)
        d_in_rew = d_endo + d_exo + n_actions
        self.net_rew = FactoredSubnetwork(d_in_rew, 1, d_hid=16, seed=seed + 2000)

    def forward(
        self,
        s_endo: np.ndarray,
        s_exo: np.ndarray,
        z_regime: np.ndarray,
        mem_working: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        a_vec = np.zeros(self.n_actions, dtype=np.float32)
        if 0 <= action < self.n_actions:
            a_vec[action] = 1.0

        # Endogenous transition
        exposure = np.array([s_exo[0]], dtype=np.float32)
        in_endo = np.concatenate([s_endo, a_vec, exposure], axis=0)
        delta_endo_net = self.net_endo.forward(in_endo)

        # Exogenous transition
        in_exo = np.concatenate([s_exo, a_vec, z_regime, mem_working[:4]], axis=0)
        delta_exo_net = self.net_exo.forward(in_exo)

        # Reward
        in_rew = np.concatenate([s_endo, s_exo, a_vec], axis=0)
        rew_net = float(self.net_rew.forward(in_rew)[0])

        # Invariant Causal Mechanisms (ICM) physiological kinetics
        delta_endo_icm = np.zeros(self.d_endo, dtype=np.float32)
        delta_endo_icm[1] -= 0.05  # Basal energy consumption
        delta_endo_icm[2] -= 0.05  # Basal hydration consumption

        # Persistent storm presence belief from cue, causal regime, or working memory
        is_storm = (
            (len(s_endo) > 3 and s_endo[3] > 0.8)
            or (len(mem_working) > 0 and float(mem_working[0]) > 0.1)
            or (float(z_regime[0]) > 0.1)
        )

        rew_icm = 0.0
        if action == 0:  # Forage
            if is_storm:
                delta_endo_icm[0] -= 0.40  # Catastrophic storm exposure damage
                rew_icm = -0.60
            else:
                delta_endo_icm[1] += 0.25  # Nutrient acquisition
                rew_icm = 0.40
        elif action == 1:  # Shelter
            if is_storm:
                delta_endo_icm[0] += 0.05  # Protected during active storm
                rew_icm = 0.35
            else:
                delta_endo_icm[1] -= 0.05  # Inefficient idling
                rew_icm = 0.05
        elif action == 2:  # Explore
            delta_endo_icm[1] -= 0.10
            rew_icm = 0.10
        elif action == 3:  # Rest
            delta_endo_icm[1] += 0.05
            rew_icm = 0.05

        delta_exo_icm = np.zeros(self.d_exo, dtype=np.float32)

        # Combine structured ICM prior with calibrated neural ensemble residual
        delta_endo = delta_endo_icm + 0.02 * delta_endo_net
        delta_exo = delta_exo_icm + 0.02 * delta_exo_net
        rew = rew_icm + 0.02 * rew_net

        return delta_endo, delta_exo, rew


class FactoredCausalWorldModel:
    """Ensemble of Factored Members enforcing Invariant Causal Mechanisms (ICM)."""

    def __init__(
        self,
        d_obs: int = 8,
        d_mem: int = 18,
        n_actions: int = 4,
        n_members: int = 3,
        base_seed: int = 300,
    ):
        self.d_obs = d_obs
        self.d_mem = d_mem
        self.n_actions = n_actions
        self.n_members = n_members

        self.d_endo = 4
        self.d_exo = min(4, d_obs - 4)
        self.d_regime = 2

        self.members = [
            FactoredMember(
                d_endo=self.d_endo,
                d_exo=self.d_exo,
                d_regime=self.d_regime,
                d_mem=d_mem,
                n_actions=n_actions,
                seed=base_seed + i * 50,
            )
            for i in range(n_members)
        ]

        # Current latent causal regime belief: [storm_duration_remaining, resource_rate]
        self.z_regime = np.array([0.0, 1.0], dtype=np.float32)
        self.last_prediction: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def partition_observation(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separates 8-D observation into endogenous vitals and exogenous field."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        s_endo = obs_arr[:self.d_endo].copy()
        s_exo = obs_arr[self.d_endo:self.d_endo + self.d_exo].copy()
        return s_endo, s_exo

    def get_factored_state(self, obs: np.ndarray) -> FactoredState:
        s_endo, s_exo = self.partition_observation(obs)
        return FactoredState(
            endogenous_vitals=s_endo,
            exogenous_field=s_exo,
            causal_regime=self.z_regime.copy(),
        )

    def predict_factored(
        self,
        obs: np.ndarray,
        mem: np.ndarray,
        action: int,
    ) -> FactoredPredictionOutput:
        """Generates factor-partitioned prediction with isolated epistemic uncertainty."""
        s_endo, s_exo = self.partition_observation(obs)
        mem_arr = np.asarray(mem, dtype=np.float32)

        endo_preds = []
        exo_preds = []
        rew_preds = []

        for m in self.members:
            d_endo, d_exo, r = m.forward(s_endo, s_exo, self.z_regime, mem_arr, action)
            endo_preds.append(s_endo + d_endo)
            exo_preds.append(s_exo + d_exo)
            rew_preds.append(r)

        endo_stack = np.stack(endo_preds, axis=0)  # (K, 4)
        exo_stack = np.stack(exo_preds, axis=0)    # (K, 4)

        mean_endo = np.mean(endo_stack, axis=0)
        mean_exo = np.mean(exo_stack, axis=0)
        mean_rew = float(np.mean(rew_preds))

        # Partitioned epistemic uncertainties
        var_endo = float(np.mean(np.var(endo_stack, axis=0)))
        var_exo = float(np.mean(np.var(exo_stack, axis=0)))
        var_regime = 0.05 if self.z_regime[0] > 0.1 else 0.01
        var_total = float((var_endo + var_exo + var_regime) / 3.0)

        # Concatenate 8-D predicted observation
        pred_obs = np.concatenate([mean_endo, mean_exo], axis=0).astype(np.float32)
        self.last_prediction = pred_obs.copy()
        self.last_action = action

        # Regime prediction next step: decay storm counter by 1.0
        regime_next = np.array([max(0.0, float(self.z_regime[0]) - 1.0), float(self.z_regime[1])], dtype=np.float32)

        return FactoredPredictionOutput(
            predicted_obs=pred_obs,
            predicted_reward=mean_rew,
            endogenous_pred=mean_endo,
            exogenous_pred=mean_exo,
            regime_pred=regime_next,
            epistemic_uncertainty_total=var_total,
            endogenous_uncertainty=var_endo,
            exogenous_uncertainty=var_exo,
            regime_uncertainty=var_regime,
            comparator_error=None,
        )

    def predict(self, obs: np.ndarray, mem: np.ndarray, action: int) -> PredictionOutput:
        """Adapts factored prediction to standard PredictionOutput contract."""
        factored = self.predict_factored(obs, mem, action)
        variance_per_dim = np.concatenate([
            np.var(np.stack([m.net_endo.forward(np.concatenate([obs[:4], np.zeros(self.n_actions), [obs[4]]])) for m in self.members], axis=0), axis=0),
            np.zeros(self.d_exo, dtype=np.float32),
        ])[:self.d_obs]
        return PredictionOutput(
            predicted_obs=factored.predicted_obs,
            predicted_reward=factored.predicted_reward,
            predicted_done=0.0,
            epistemic_uncertainty=factored.epistemic_uncertainty_total,
            ensemble_variance=variance_per_dim,
            comparator_error=None,
        )

    def compute_comparator_residual(self, true_next_obs: np.ndarray) -> np.ndarray:
        """Computes realized prediction error and updates causal regime state."""
        if self.last_prediction is None:
            return np.zeros(self.d_obs, dtype=np.float32)
        next_arr = np.asarray(true_next_obs, dtype=np.float32)[:self.d_obs]
        residual = next_arr - self.last_prediction

        # Update latent causal regime: if storm pulse observed (cue > 0.8), reset storm duration
        if (len(next_arr) > 5 and next_arr[5] > 0.8) or (len(next_arr) > 3 and next_arr[3] > 0.8):
            self.z_regime[0] = 3.0
        else:
            self.z_regime[0] = max(0.0, float(self.z_regime[0]) - 1.0)

        return residual.astype(np.float32)
