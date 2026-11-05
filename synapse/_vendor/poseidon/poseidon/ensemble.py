"""Uncertainty-Aware Dynamics Ensemble and Risk-Sensitive World Modeling.

Implements Experiment A of Supermix Beyond:
1. Ensemble of K separately parameterized dynamics heads:
       \\hat{s}_{t+1}^{(k)} = s_t + \\Delta_{\\theta_k}(s_t, a_t)  for k in {1, ..., K}
2. Epistemic uncertainty estimation via predictive disagreement:
       U(s_t, a_t) = (1 / K) * sum_{k=1}^K || \\hat{s}_{t+1}^{(k)} - \\bar{s}_{t+1} ||^2
3. Heuristic vulnerability score (not a calibrated probability):
       Detects predicted starvation, dehydration, exhaustion, exposure, and predator attacks
       across the imagined ensemble paths.
4. Risk-sensitive Model Predictive Control (MPC):
       a_t^* = argmax_a [ E[R | a] - lambda * U(s_t, a) - beta * P(failure | a) ]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import CoreRuntime, TidalCore, SCENE_LABELS
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass(frozen=True)
class RiskConfig:
    """Hyperparameters for uncertainty-aware, risk-sensitive planning."""
    ensemble_size: int = 3
    horizon: int = 2
    gamma: float = 0.90
    policy_weight: float = 0.45
    mpc_weight: float = 0.55
    uncertainty_penalty: float = 1.50   # lambda: penalize unmodeled epistemic risk
    failure_penalty: float = 5.00       # beta: penalize predicted lethal trajectories
    disagreement_threshold: float = 0.08  # threshold for flag indicating high surprise


class DynamicsHead(nn.Module):
    """Action-conditioned next-observation state delta predictor."""

    def __init__(self, hidden_size: int = 128, obs_size: int = OBS_SIZE, action_count: int = N_ACTIONS, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.action_embedding = nn.Embedding(action_count, 24)
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 24 + obs_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, obs_size),
        )

    def forward(self, state: torch.Tensor, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        act_embed = self.action_embedding(actions)
        features = torch.cat((state, act_embed, obs), dim=-1)
        return self.net(features)


class WorldModelEnsemble(nn.Module):
    """Ensemble of K independently parameterized dynamics predictors.
    
    Construction initializes random predictors. Training and held-out calibration
    must be demonstrated separately; disagreement alone does not establish
    epistemic uncertainty. The default planner uses partially warm-started priors.
    """

    def __init__(self, hidden_size: int = 128, k: int = 3, base_seed: int = 100):
        super().__init__()
        self.k = k
        self.heads = nn.ModuleList([
            DynamicsHead(hidden_size=hidden_size, seed=base_seed + i * 37)
            for i in range(k)
        ])

    def forward(self, state: torch.Tensor, obs: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predicts delta distributions across all ensemble members.
        
        Args:
            state: [batch, hidden_size] latent representation from TidalCore
            obs: [batch, 16] observation vector
            actions: [batch] action indices (0 to 5)
            
        Returns:
            dict containing:
                deltas: [k, batch, 16] individual head delta predictions
                mean_delta: [batch, 16] ensemble mean delta
                variance: [batch, 16] per-feature variance
                disagreement: [batch] epistemic uncertainty metric U(s, a)
        """
        deltas = torch.stack([head(state, obs, actions) for head in self.heads], dim=0)  # [k, batch, 16]
        mean_delta = deltas.mean(dim=0)  # [batch, 16]
        diffs = deltas - mean_delta.unsqueeze(0)  # [k, batch, 16]
        variance = (diffs ** 2).mean(dim=0)  # [batch, 16]
        # Disagreement is the mean squared norm across ensemble predictions:
        disagreement = variance.sum(dim=-1)  # [batch]
        return {
            "deltas": deltas,
            "mean_delta": mean_delta,
            "variance": variance,
            "disagreement": disagreement,
        }


def evaluate_survival_state(obs: Sequence[float], action: int) -> tuple[float, float]:
    """Evaluates reserve value and a hand-written vulnerability score.
    
    Returns:
        (vitality_score, vulnerability_score), with no probability calibration.
    """
    if len(obs) != OBS_SIZE:
        raise ValueError(f"Observation must have {OBS_SIZE} elements")
    health, energy, hydration, stamina, exposure, threat, food, water, shelter, severity, temp, daylight, terrain, scent, last_action, progress = obs

    # Hand-written lethal vulnerability indicators
    failure_risk = 0.0
    if health < 0.20:
        failure_risk += 0.8 * (0.20 - health) / 0.20
    if energy < 0.10:
        failure_risk += 0.7 * (0.10 - energy) / 0.10
    if hydration < 0.10:
        failure_risk += 0.9 * (0.10 - hydration) / 0.10
    if stamina < 0.05:
        failure_risk += 0.5 * (0.05 - stamina) / 0.05
    if exposure > 0.82:
        failure_risk += 0.85 * (exposure - 0.82) / 0.18
    if threat > 0.65 and action not in (3, 5):
        failure_risk += 1.0 * (threat - 0.65) / 0.35

    failure_prob = min(1.0, max(0.0, failure_risk))

    # Positive reserve score
    vitality = (
        1.6 * min(energy, 1.0)
        + 2.0 * min(hydration, 1.0)
        + 0.9 * min(stamina, 1.0)
        + 2.5 * min(health, 1.0)
        - 1.3 * max(0.0, exposure)
    )

    return float(vitality), float(failure_prob)


class UncertaintyAwarePlanner:
    """Supermix Beyond's Risk-Sensitive Uncertainty-Aware MPC Planner.
    
    Integrates:
    1. Base policy prior logits from TidalCore
    2. K-head ensemble world model predictions for all candidate actions
    3. Epistemic disagreement penalty U(s, a)
    4. Failure probability penalty P(failure | a)
    """

    def __init__(
        self,
        core: CoreRuntime,
        ensemble: WorldModelEnsemble | None = None,
        config: RiskConfig | None = None,
    ):
        self.core = core
        self.config = config or RiskConfig()
        h = core.model.config.hidden_size
        if ensemble is not None:
            self.ensemble = ensemble
        else:
            # Partial warm start; later randomly initialized layers are untrained.
            self.ensemble = WorldModelEnsemble(hidden_size=h, k=self.config.ensemble_size)
            self._warm_start_ensemble()

    def _warm_start_ensemble(self) -> None:
        """Initialize ensemble members around the trained core's dynamics weights with perturbed priors."""
        base_state = self.core.model.dynamics_head.state_dict()
        base_embed = self.core.model.action_embedding.state_dict()
        with torch.no_grad():
            for i, head in enumerate(self.ensemble.heads):
                head.action_embedding.load_state_dict(base_embed)
                # Map matching layer weights from base dynamics head
                if "0.weight" in base_state and "0.weight" in head.net.state_dict():
                    head.net[0].weight.copy_(base_state["0.weight"])
                    head.net[0].bias.copy_(base_state["0.bias"])
                if i > 0:
                    # Inject controlled diversity into ensemble members
                    head.net[0].weight.add_(torch.randn_like(head.net[0].weight) * 0.035 * i)
                    head.net[-1].weight.add_(torch.randn_like(head.net[-1].weight) * 0.025 * i)

    @torch.inference_mode()
    def plan(self, observation: Sequence[float]) -> dict[str, Any]:
        """Runs uncertainty-aware batched lookahead planning."""
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        zeros_1 = torch.zeros(1, self.core.model.config.hash_buckets)

        # 1. Base policy prior & shared latent state representation
        core_out = self.core.model(zeros_1, obs_tensor)
        latent_state = core_out["memory"]  # [1, hidden_size]
        policy_logits = core_out["action"][0]
        policy_probs = F.softmax(policy_logits, dim=-1)

        # 2. Batched Horizon-1 imagination across all 6 actions
        obs_batch_6 = obs_tensor.repeat(N_ACTIONS, 1)
        latent_batch_6 = latent_state.repeat(N_ACTIONS, 1)
        actions_0 = torch.arange(N_ACTIONS)

        # Predict across the 3-model dynamics ensemble
        ens_result_1 = self.ensemble(latent_batch_6, obs_batch_6, actions_0)
        mean_delta_1 = ens_result_1["mean_delta"]
        disagreements_1 = ens_result_1["disagreement"]  # [6]

        obs_1 = (obs_batch_6 + mean_delta_1).clamp(0.0, 1.0)

        # Evaluate step 1: reward/vitality, failure risk, and uncertainty
        vitalities_1 = []
        fail_probs_1 = []
        for a in range(N_ACTIONS):
            v, f_p = evaluate_survival_state(obs_1[a].tolist(), a)
            vitalities_1.append(v)
            fail_probs_1.append(f_p)

        vitalities_t = torch.tensor(vitalities_1, dtype=torch.float32)
        fail_probs_t = torch.tensor(fail_probs_1, dtype=torch.float32)

        # 3. Batched Horizon-2 imagination (36 candidate branches)
        if self.config.horizon >= 2:
            n_branches = N_ACTIONS * N_ACTIONS
            obs_batch_36 = obs_1.repeat_interleave(N_ACTIONS, dim=0)
            latent_batch_36 = latent_batch_6.repeat_interleave(N_ACTIONS, dim=0)
            actions_1 = torch.arange(N_ACTIONS).repeat(N_ACTIONS)

            ens_result_2 = self.ensemble(latent_batch_36, obs_batch_36, actions_1)
            mean_delta_2 = ens_result_2["mean_delta"]
            disagreements_2 = ens_result_2["disagreement"]
            obs_2 = (obs_batch_36 + mean_delta_2).clamp(0.0, 1.0)

            # Fold step-2 outcomes into step-1 action Q-values
            q_values = torch.zeros(N_ACTIONS)
            step2_fail_probs = torch.zeros(N_ACTIONS)
            step2_uncertainties = torch.zeros(N_ACTIONS)

            for a0 in range(N_ACTIONS):
                branch_scores = []
                branch_fails = []
                branch_uncs = []
                for a1 in range(N_ACTIONS):
                    idx = a0 * N_ACTIONS + a1
                    v2, f_p2 = evaluate_survival_state(obs_2[idx].tolist(), a1)
                    u2 = float(disagreements_2[idx])
                    # Penalize risky step-2 paths
                    branch_score = v2 - self.config.uncertainty_penalty * u2 - self.config.failure_penalty * f_p2
                    branch_scores.append(branch_score)
                    branch_fails.append(f_p2)
                    branch_uncs.append(u2)

                q_values[a0] = vitalities_t[a0] + self.config.gamma * max(branch_scores)
                step2_fail_probs[a0] = max(branch_fails)
                step2_uncertainties[a0] = max(branch_uncs)
        else:
            q_values = vitalities_t
            step2_fail_probs = fail_probs_t
            step2_uncertainties = disagreements_1

        # 4. Risk-Sensitive Objective with Dynamic Attenuation (solves the Pessimism Trap)
        total_uncertainty = (disagreements_1 + 0.5 * step2_uncertainties)
        total_failure_risk = torch.max(fail_probs_t, step2_fail_probs)

        # When vital reserves drop below critical threshold, attenuate risk penalty to prevent paralysis
        obs_list = list(observation)
        energy_lvl = obs_list[1] if len(obs_list) > 1 else 1.0
        hydr_lvl = obs_list[2] if len(obs_list) > 2 else 1.0
        vital_reserves = min(energy_lvl, hydr_lvl)
        attenuation = min(1.0, max(0.15, vital_reserves / 0.35))
        effective_unc_penalty = self.config.uncertainty_penalty * attenuation

        risk_adjusted_q = (
            q_values
            - effective_unc_penalty * total_uncertainty
            - self.config.failure_penalty * total_failure_risk
        )

        # Normalize risk-adjusted Q-values
        q_norm = (risk_adjusted_q - risk_adjusted_q.mean()) / (risk_adjusted_q.std() + 1e-6)
        mpc_probs = F.softmax(q_norm, dim=-1)

        # 5. Hybridize policy prior with uncertainty-aware lookahead
        combined = self.config.policy_weight * policy_probs + self.config.mpc_weight * mpc_probs
        chosen_action = int(combined.argmax())

        predicted_futures = {}
        for a in range(N_ACTIONS):
            predicted_futures[ACTIONS[a]] = {
                "step_1_obs": [round(x, 4) for x in obs_1[a].tolist()],
                "vitality": round(float(vitalities_t[a]), 4),
                "epistemic_uncertainty": round(float(total_uncertainty[a]), 5),
                "failure_probability": round(float(total_failure_risk[a]), 4),
                "risk_adjusted_q": round(float(risk_adjusted_q[a]), 4),
            }

        return {
            "action": chosen_action,
            "action_name": ACTIONS[chosen_action],
            "policy_action": int(policy_probs.argmax()),
            "risk_action": int(risk_adjusted_q.argmax()),
            "epistemic_uncertainty": float(total_uncertainty[chosen_action]),
            "failure_probability": float(total_failure_risk[chosen_action]),
            "is_high_surprise": bool(total_uncertainty[chosen_action] > self.config.disagreement_threshold),
            "uncertainty_note": "Ensemble disagreement; default heads are partially warm-started untrained priors.",
            "risk_note": "failure_probability is a legacy field name for a heuristic vulnerability score.",
            "policy_probs": [round(float(p), 4) for p in policy_probs.tolist()],
            "combined_scores": [round(float(s), 4) for s in combined.tolist()],
            "predicted_futures": predicted_futures,
        }

    def act(self, observation: Sequence[float]) -> int:
        return self.plan(observation)["action"]
