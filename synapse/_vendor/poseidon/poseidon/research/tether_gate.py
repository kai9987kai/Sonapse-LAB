"""TETHER Reliability Gate & TetherOracleAdapter for Supermix Poseidon.

Connects Tether's intervention-trained reliability gating, dark-knowledge/continuous
forensics, zero-shot corruption detection, and epistemic fallback gates to
Poseidon's ORACLE predictive world-model ensemble without modifying the frozen
Tidal checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .oracle_adapter import ResearchOracleAdapter
from ..world import N_ACTIONS, OBS_SIZE


class TetherReliabilityGate(nn.Module):
    """Neural reliability receiver for K continuous world model members.

    Inputs:
    - obs: [batch, 16]
    - action: [batch]
    - deltas: [K, batch, 16] (predicted state deltas from K members)
    - delta_log_vars: [K, batch, 16] (aleatoric log-variance)
    - rewards: [K, batch] (predicted rewards)
    - deaths: [K, batch] (predicted death logits)

    Outputs:
    - member_weights: [batch, K]
    - gated_delta: [batch, 16]
    - gated_reward: [batch]
    - gated_death_prob: [batch]
    - epistemic_variance: [batch, 16]
    - aleatoric_variance: [batch, 16]
    - total_variance: [batch, 16]
    - epistemic_trust: [batch] in [0, 1]
    - abstain: [batch] bool (True if trust < threshold)
    """

    def __init__(
        self,
        ensemble_size: int = 5,
        obs_size: int = OBS_SIZE,
        num_actions: int = N_ACTIONS,
        hidden_dim: int = 32,
        epistemic_threshold: float = 0.35,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.epistemic_threshold = float(epistemic_threshold)

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        self.member_embedding = nn.Embedding(ensemble_size, 8)

        # Forensic feature size:
        # 16 (delta_k) + 16 (dev_k) + 16 (log_var_k) + 1 (rew_k) + 1 (death_k) + 1 (cos_sim_k) = 51
        forensic_in = 51 + 8 + hidden_dim  # 51 features + 8 member_emb + hidden_dim context
        self.scorer = nn.Sequential(
            nn.Linear(forensic_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deltas: torch.Tensor,
        delta_log_vars: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        deaths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute reliability-weighted forecasts and epistemic trust."""
        batch_size = obs.shape[0]
        K = self.ensemble_size

        if deltas.shape != (K, batch_size, self.obs_size):
            raise ValueError(f"deltas must have shape [{K}, {batch_size}, {self.obs_size}]")

        dtype = obs.dtype
        device = obs.device

        # Defaults for optional terms
        if delta_log_vars is None:
            delta_log_vars = torch.zeros_like(deltas)
        if rewards is None:
            rewards = torch.zeros(K, batch_size, dtype=dtype, device=device)
        if deaths is None:
            deaths = torch.zeros(K, batch_size, dtype=dtype, device=device)

        # 1. State-Action context representation
        context = self.obs_encoder(obs) + self.action_embedding(action)  # [batch, hidden]

        # 2. Consensus centroid and deviations
        centroid = deltas.mean(dim=0, keepdim=True)  # [1, batch, 16]
        diffs = deltas - centroid  # [K, batch, 16]

        # Cosine similarity to centroid
        c_norm = centroid.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [1, batch, 1]
        p_norm = deltas.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # [K, batch, 1]
        cos_sim = (deltas * centroid).sum(dim=-1, keepdim=True) / (p_norm * c_norm)  # [K, batch, 1]

        # 3. Assemble forensic features per member
        scores_list = []
        for k in range(K):
            m_emb = self.member_embedding.weight[k : k + 1].expand(batch_size, -1)  # [batch, 8]
            feat_k = torch.cat(
                (
                    deltas[k],  # 16
                    diffs[k],  # 16
                    delta_log_vars[k],  # 16
                    rewards[k : k + 1].T,  # 1
                    deaths[k : k + 1].T,  # 1
                    cos_sim[k],  # 1
                    m_emb,  # 8
                    context,  # hidden_dim
                ),
                dim=-1,
            )
            s_k = self.scorer(feat_k)  # [batch, 1]
            scores_list.append(s_k)

        scores = torch.cat(scores_list, dim=-1)  # [batch, K]
        weights = F.softmax(scores, dim=-1)  # [batch, K]

        # 4. Gated synthesized outputs
        # deltas: [K, batch, 16] -> weights: [batch, K] -> [batch, 16]
        w_expanded = weights.T.unsqueeze(-1)  # [K, batch, 1]
        gated_delta = (w_expanded * deltas).sum(dim=0)  # [batch, 16]
        gated_reward = (weights * rewards.T).sum(dim=-1)  # [batch]
        gated_death_prob = (weights * deaths.sigmoid().T).sum(dim=-1)  # [batch]

        # 5. Law of total variance
        # Epistemic: sum_k w_k (delta_k - gated_delta)^2
        # Aleatoric: sum_k w_k exp(delta_log_var_k)
        epistemic_var = (w_expanded * (deltas - gated_delta.unsqueeze(0)) ** 2).sum(dim=0)
        aleatoric_var = (w_expanded * delta_log_vars.exp()).sum(dim=0)
        total_var = epistemic_var + aleatoric_var

        # 6. Epistemic Trust Gauge tau in [0, 1]
        mean_epistemic = epistemic_var.mean(dim=-1)
        max_weight = weights.max(dim=-1).values
        trust_tau = (torch.exp(-mean_epistemic * 4.0) * max_weight).clamp(0.0, 1.0)
        abstain = trust_tau < self.epistemic_threshold

        return {
            "weights": weights,
            "gated_delta": gated_delta,
            "gated_reward": gated_reward,
            "gated_death_prob": gated_death_prob,
            "epistemic_variance": epistemic_var,
            "aleatoric_variance": aleatoric_var,
            "total_variance": total_var,
            "epistemic_trust": trust_tau,
            "abstain": abstain,
        }

    def attribute(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        deltas: torch.Tensor,
        delta_log_vars: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        deaths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute mechanistic input x gradient attribution."""
        p = deltas.clone().detach().requires_grad_(True)
        res = self.forward(obs, action, p, delta_log_vars, rewards, deaths)
        gated = res["gated_delta"]
        # Score scalar: sum of gated prediction
        score = gated.sum()
        grads = torch.autograd.grad(score, p, create_graph=False, allow_unused=True)[0]
        if grads is None:
            grads = torch.zeros_like(p)
        saliency = (p.detach() * grads.detach()).abs()  # [K, batch, 16]
        member_saliency = saliency.sum(dim=-1)  # [K, batch]
        feature_saliency = saliency.sum(dim=0)  # [batch, 16]
        return {
            "saliency": saliency,
            "member_saliency": member_saliency,
            "feature_saliency": feature_saliency,
            "weights": res["weights"].detach(),
            "epistemic_trust": res["epistemic_trust"].detach(),
        }


class TetherOracleAdapter(ResearchOracleAdapter):
    """Tether-gated ORACLE adapter that wraps an existing OracleEnsemble artifact.

    - Replaces equal averaging with learned continuous reliability weighting.
    - Reports weighted forecasts and heuristic epistemic trust for diagnostics.
    - The ORACLE artifact calibrates equal averaging, not this changed predictor.
      Planning therefore preserves the frozen policy until weighted-predictor
      calibration evidence is supported.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        gate: Optional[TetherReliabilityGate] = None,
        epistemic_threshold: float = 0.35,
    ):
        super().__init__(artifact_path)
        self.epistemic_threshold = float(epistemic_threshold)
        ensemble_size = len(self.model.members)
        if gate is not None:
            self.gate = gate
        else:
            self.gate = TetherReliabilityGate(
                ensemble_size=ensemble_size,
                obs_size=self.config.obs_size,
                num_actions=self.config.action_count,
                epistemic_threshold=self.epistemic_threshold,
            )
        self.tether_enabled = True
        self.is_calibrated.fill_(False)
        self.eval()

    def status(self) -> Dict[str, Any]:
        base_status = super().status()
        base_status.update({
            "tether_enabled": True,
            "tether_protocol": "tether-v2-continuous-reliability",
            "epistemic_threshold": self.epistemic_threshold,
            "gate_parameters": sum(p.numel() for p in self.gate.parameters()),
            "calibration_available": False,
            "calibration_scope": "weighted predictor uncalibrated; inherited quantiles give diagnostic widths only",
            "gate_artifact_bound": False,
            "corrupted_member_defense": False,
        })
        return base_status

    def planning_support(self, observation: Sequence[float]) -> Tuple[bool, Optional[str]]:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        self._validate_observation(obs)
        if any(count < 8 for count in self.action_counts):
            return False, "insufficient_training_action_exposure"

        # Agreement and a low threshold cannot supply missing calibration for
        # the weighted predictor or bind this gate's parameters to the artifact.
        return False, "tether_weighted_calibration_unavailable"

    @torch.inference_mode()
    def forward(
        self,
        latent_state: Optional[torch.Tensor],
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._validate_observation(obs)
        raw_out = self.model(obs, actions)

        # Apply Tether reliability gate
        gated = self.gate(
            obs=obs,
            action=actions,
            deltas=raw_out["delta_mean"],
            delta_log_vars=raw_out["delta_log_variance"],
            rewards=raw_out["reward_mean"],
            deaths=raw_out["death_logit"],
        )

        state_half_width = self.state_quantile * gated["total_variance"].sqrt()
        member_weights = gated["weights"].transpose(0, 1)  # [members, batch]
        reward_epistemic = (member_weights * (raw_out["reward_mean"] - gated["gated_reward"].unsqueeze(0)) ** 2).sum(dim=0)
        reward_aleatoric = (member_weights * raw_out["reward_log_variance"].exp()).sum(dim=0)
        reward_total_var = (reward_epistemic + reward_aleatoric).clamp_min(1e-8)
        reward_half_width = self.reward_quantile * reward_total_var.sqrt()

        result = {
            "mean_delta": gated["gated_delta"],
            "mean_reward": gated["gated_reward"],
            "mean_failure": gated["gated_death_prob"],
            "raw_disagreement": gated["epistemic_variance"].sum(-1),
            # Legacy compatibility names; status explicitly marks these
            # inherited-quantile widths as uncalibrated for this predictor.
            "calibrated_uncertainty": state_half_width.mean(-1),
            "state_interval_half_width": state_half_width,
            "reward_interval_half_width": reward_half_width,
            "deltas": raw_out["delta_mean"],
            "rewards": raw_out["reward_mean"],
            "failures": raw_out["death_logit"].sigmoid(),
            # Tether-specific telemetry
            "tether_weights": gated["weights"],
            "epistemic_trust": gated["epistemic_trust"],
            "tether_abstain": gated["abstain"],
        }
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("TETHER produced nonfinite predictions or diagnostic widths")
        self.member_forward_calls += len(self.model.members)
        self.prediction_rows += obs.shape[0]
        return result
