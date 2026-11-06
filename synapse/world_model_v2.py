"""World Model v2: Recurrent Latent Dynamics Ensemble, CEM Planner, Fallback Shield & Exploitation Audit.

Grounding:
Upgrades the predictive dynamics into an ensemble of sequence models:
    p_theta(z_{t+1:t+H}, r_{t:t+H}, d_{t:t+H}, h_{t:t+H} | z_t, a_{t:t+H-1})
Features:
1. Recurrent Latent Dynamics Ensemble (M=3 members) with JEPA representation loss.
2. Cross-Entropy Method (CEM) Receding-Horizon Trajectory Optimizer (iterative Gaussian fitting over elites).
3. Risk-aware objective with epistemic uncertainty and homeostatic preservation penalties:
    J(tau) = sum_{k=1}^H gamma^{k-1} [ r_{t+k} - lambda_U U_{t+k} - lambda_Q q_{t+k} - lambda_H Delta H_{t+k} ]
4. Fallback Shield: Invariant fallback to trusted frozen controller when disagreement or risk exceeds threshold.
5. Model Exploitation Metric: G_exploit = R_predicted - R_realised.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLatentDynamicsMember(nn.Module):
    """Individual member in the world model ensemble with recurrent transition and multi-heads."""

    def __init__(self, state_dim: int = 64, action_dim: int = 5, obs_dim: int = 155):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Encoder: maps high-dimensional sensory observation to compact latent z
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )

        # Recurrent state-space transition: (z_t, a_t) -> z_{t+1}
        self.transition = nn.GRUCell(action_dim, state_dim)

        # Predictive Heads
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.homeostatic_head = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4)  # Predicted delta in [energy, stamina, threat, hunger]
        )
        self.obs_ray_decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 33),
            nn.Sigmoid()
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_next = self.transition(a, z)
        rew_in = torch.cat([z_next, a], dim=-1)
        r_pred = self.reward_head(rew_in).squeeze(-1)
        q_risk = self.risk_head(z_next).squeeze(-1)
        delta_h = self.homeostatic_head(z_next)
        obs_rays = self.obs_ray_decoder(z_next)
        return z_next, r_pred, q_risk, delta_h, obs_rays


class RecurrentLatentWorldModelEnsemble:
    """Ensemble of M recurrent latent dynamics models for calibrated epistemic uncertainty."""

    def __init__(
        self,
        num_members: int = 3,
        state_dim: int = 64,
        action_dim: int = 5,
        obs_dim: int = 155,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.num_members = num_members
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        torch.manual_seed(seed)
        self.members = nn.ModuleList([
            SingleLatentDynamicsMember(state_dim, action_dim, obs_dim)
            for _ in range(num_members)
        ])
        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=lr) for m in self.members
        ]
        self.rng = np.random.default_rng(seed)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes observation using the mean latent representation across ensemble members."""
        zs = [m.encode(obs) for m in self.members]
        return torch.stack(zs, dim=0).mean(dim=0)

    def step_ensemble(
        self,
        zs: List[torch.Tensor],
        a: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all ensemble members forward one step.

        Returns:
            (next_zs, mean_reward, mean_risk, mean_homeostatic_delta, epistemic_uncertainty)
        """
        next_zs = []
        rs = []
        qs = []
        hs = []
        for i, m in enumerate(self.members):
            z_next, r_pred, q_risk, d_h, _ = m.step(zs[i], a)
            next_zs.append(z_next)
            rs.append(r_pred)
            qs.append(q_risk)
            hs.append(d_h)

        z_stack = torch.stack(next_zs, dim=0)  # [M, B, D]
        z_mean = z_stack.mean(dim=0)
        # Epistemic uncertainty = mean squared deviation from ensemble mean
        uncertainty = torch.mean((z_stack - z_mean) ** 2, dim=[0, -1])  # [B]

        r_mean = torch.stack(rs, dim=0).mean(dim=0)
        q_mean = torch.stack(qs, dim=0).mean(dim=0)
        h_mean = torch.stack(hs, dim=0).mean(dim=0)

        return next_zs, r_mean, q_mean, h_mean, uncertainty

    def train_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        delta_homeo: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Trains ensemble members on transition batch with JEPA latent representation loss."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        act_t = torch.as_tensor(actions, dtype=torch.float32)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        done_t = torch.as_tensor(dones, dtype=torch.float32)

        dh_t = (
            torch.as_tensor(delta_homeo, dtype=torch.float32)
            if delta_homeo is not None
            else torch.zeros(obs.shape[0], 4, dtype=torch.float32)
        )

        losses = []
        for i, (m, opt) in enumerate(zip(self.members, self.optimizers)):
            opt.zero_grad()
            z_t = m.encode(obs_t)
            # Target representation with stopped gradient (JEPA formulation)
            with torch.no_grad():
                z_target = m.encode(next_obs_t)

            z_next, r_pred, q_risk, d_h, obs_rays = m.step(z_t, act_t)

            # Losses
            loss_jepa = F.mse_loss(z_next, z_target)
            loss_rew = F.mse_loss(r_pred, rew_t)
            loss_risk = F.binary_cross_entropy(q_risk, done_t)
            loss_homeo = F.mse_loss(d_h, dh_t)
            loss_rays = F.mse_loss(obs_rays, next_obs_t[:, :33])

            tot = loss_jepa + 2.0 * loss_rew + 0.5 * loss_risk + 0.3 * loss_homeo + 0.2 * loss_rays
            tot.backward()
            opt.step()
            losses.append(float(tot.item()))

        return {
            'mean_loss': float(np.mean(losses)),
            'min_loss': float(np.min(losses)),
            'max_loss': float(np.max(losses)),
        }


class CrossEntropyMethodPlanner:
    """Cross-Entropy Method (CEM) trajectory optimization with fallback shield and exploitation tracking."""

    def __init__(
        self,
        world_model: RecurrentLatentWorldModelEnsemble,
        action_dim: int = 5,
        default_horizon: int = 5,
        num_samples: int = 32,
        num_elites: int = 6,
        cem_iterations: int = 3,
        gamma: float = 0.95,
        lambda_u: float = 0.4,     # Uncertainty penalty
        lambda_q: float = 2.0,     # Risk/death penalty
        lambda_h: float = 0.5,     # Homeostatic deficit penalty
        uncertainty_threshold: float = 0.85, # Fallback trigger
        risk_threshold: float = 0.60,        # Fallback trigger
        seed: int = 42,
    ):
        self.wm = world_model
        self.action_dim = action_dim
        self.default_horizon = default_horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.gamma = gamma
        self.lambda_u = lambda_u
        self.lambda_q = lambda_q
        self.lambda_h = lambda_h
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_threshold = risk_threshold
        self.rng = np.random.default_rng(seed)

        # Audit and Exploitation metrics
        self.planned_counts = 0
        self.fallback_counts = 0
        self.exploitation_history: List[float] = []

    def plan(
        self,
        current_obs: np.ndarray,
        trusted_action: np.ndarray,
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Optimizes multi-step action trajectory with CEM and applies fallback shield."""
        self.planned_counts += 1
        H = horizon if horizon is not None else self.default_horizon
        obs_t = torch.as_tensor(current_obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            initial_zs = [m.encode(obs_t) for m in self.wm.members]

            # Initialize Gaussian distribution over action sequences: [H, action_dim]
            mean_traj = np.tile(trusted_action.reshape(1, -1), (H, 1))
            std_traj = np.ones((H, self.action_dim), dtype=np.float32) * 0.5

            best_traj_score = -float('inf')
            best_sequence = mean_traj.copy()
            best_mean_unc = 0.0
            best_max_risk = 0.0

            for iter_idx in range(self.cem_iterations):
                # Sample N candidate sequences
                candidates = self.rng.normal(
                    loc=mean_traj, scale=std_traj, size=(self.num_samples, H, self.action_dim)
                ).astype(np.float32)
                # Ensure candidate 0 is the current mean trajectory
                candidates[0] = mean_traj.copy()
                candidates = np.clip(candidates, -1.0, 1.0)

                scores = []
                metrics = []

                for k in range(self.num_samples):
                    traj = candidates[k]
                    zs = [z.clone() for z in initial_zs]
                    traj_return = 0.0
                    uncs = []
                    risks = []

                    for tau in range(H):
                        a_tau = torch.as_tensor(traj[tau], dtype=torch.float32).unsqueeze(0)
                        zs, r_val, q_val, h_val, u_val = self.wm.step_ensemble(zs, a_tau)

                        r_f = float(r_val.item())
                        u_f = float(u_val.item())
                        q_f = float(q_val.item())
                        # Homeostatic deficit penalty = threat + hunger - (energy + stamina)
                        h_pen = float(h_val[0, 2] + h_val[0, 3] - h_val[0, 0] - h_val[0, 1])

                        step_objective = r_f - self.lambda_u * u_f - self.lambda_q * q_f - self.lambda_h * h_pen
                        traj_return += (self.gamma ** tau) * step_objective

                        uncs.append(u_f)
                        risks.append(q_f)

                    scores.append(traj_return)
                    metrics.append({'mean_u': float(np.mean(uncs)), 'max_q': float(np.max(risks))})

                # Sort by score descending and select elites
                elite_indices = np.argsort(scores)[::-1][:self.num_elites]
                elites = candidates[elite_indices]

                # Update Gaussian parameters from elites
                mean_traj = np.mean(elites, axis=0)
                std_traj = np.std(elites, axis=0) + 0.05

                if scores[elite_indices[0]] > best_traj_score:
                    best_traj_score = scores[elite_indices[0]]
                    best_sequence = elites[0].copy()
                    best_mean_unc = metrics[elite_indices[0]]['mean_u']
                    best_max_risk = metrics[elite_indices[0]]['max_q']

            # FALLBACK SHIELD CHECK:
            # If epistemic uncertainty or maximum predicted risk exceeds safe thresholds, fallback!
            shield_triggered = (
                best_mean_unc > self.uncertainty_threshold or
                best_max_risk > self.risk_threshold
            )

            if shield_triggered:
                self.fallback_counts += 1
                selected_action = trusted_action.copy()
                plan_mode = "fallback_trusted_controller"
            else:
                selected_action = best_sequence[0].copy()
                plan_mode = "cem_latent_mpc"

            return {
                'status': 'ok',
                'plan_mode': plan_mode,
                'action': selected_action,
                'shield_triggered': shield_triggered,
                'horizon': H,
                'best_score': round(float(best_traj_score), 4),
                'mean_uncertainty': round(float(best_mean_unc), 4),
                'max_risk': round(float(best_max_risk), 4),
                'planned_trajectory': best_sequence.tolist(),
                'fallback_rate': round(self.fallback_counts / max(1, self.planned_counts), 4)
            }

    def record_realized_outcome(self, predicted_return: float, actual_return: float):
        """Records model exploitation gap: G_exploit = R_predicted - R_realised."""
        gap = float(predicted_return - actual_return)
        self.exploitation_history.append(gap)

    def get_exploitation_metric(self) -> Dict[str, float]:
        if not self.exploitation_history:
            return {'mean_exploitation_gap': 0.0, 'max_exploitation_gap': 0.0}
        return {
            'mean_exploitation_gap': float(np.mean(self.exploitation_history)),
            'std_exploitation_gap': float(np.std(self.exploitation_history)),
            'max_exploitation_gap': float(np.max(self.exploitation_history)),
            'samples': len(self.exploitation_history)
        }
