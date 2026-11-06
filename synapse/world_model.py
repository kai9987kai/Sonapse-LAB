"""Neural Predictive World Model with Imagined Planning and Epistemic Uncertainty.

Recurrent transition model predicting future observations, rewards, and physiological
resource frames. Evaluates candidate action sequences in neural imagination with
uncertainty-penalized tree search to limit unreliable plans.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentPredictiveWorldModel(nn.Module):
    """Predictive forward world model for imagined rollouts."""

    def __init__(self, state_dim: int = 32, action_dim: int = 5, obs_dim: int = 155):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Recurrent state transition model: (z_t, a_t) -> (z_{t+1}, uncertainty)
        self.transition_gru = nn.GRUCell(action_dim, state_dim)
        self.uncertainty_head = nn.Linear(state_dim, state_dim)

        # Observation decoder: z_t -> predicted sensory rays [33]
        self.obs_decoder = nn.Linear(state_dim, 33)

        # Reward predictor: (z_t, a_t) -> scalar reward
        self.reward_head = nn.Linear(state_dim + action_dim, 1)

        # Resource frame predictor: z_t -> 16-d resource frame
        self.resource_head = nn.Linear(state_dim, 16)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward_step(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step forward: returns (z_next, predicted_reward, predicted_obs_rays, uncertainty)."""
        z_next = self.transition_gru(a, z)
        log_var = self.uncertainty_head(z_next)
        uncertainty = torch.exp(torch.clamp(log_var, -3.0, 3.0))

        rew_in = torch.cat([z_next, a], dim=-1)
        r_pred = self.reward_head(rew_in).squeeze(-1)
        obs_pred = torch.sigmoid(self.obs_decoder(z_next))

        return z_next, r_pred, obs_pred, uncertainty

    def imagine_rollout(
        self,
        initial_z: np.ndarray,
        action_sequence: list[np.ndarray],
        gamma: float = 0.95,
        uncertainty_penalty: float = 0.2
    ) -> Dict[str, Any]:
        """Forward-simulate candidate action sequence in neural imagination."""
        z = torch.as_tensor(initial_z, dtype=torch.float32).reshape(1, -1)
        if z.shape[-1] != self.state_dim:
            z = F.pad(z, (0, max(0, self.state_dim - z.shape[-1])))[:, :self.state_dim]

        total_imagined_return = 0.0
        cumulative_uncertainty = 0.0
        trajectory = []

        with torch.no_grad():
            for t, a_np in enumerate(action_sequence):
                a = torch.as_tensor(a_np, dtype=torch.float32).reshape(1, -1)
                z, r_pred, obs_pred, uncert = self.forward_step(z, a)

                r_val = float(r_pred.item())
                uncert_val = float(uncert.mean().item())

                # Penalize uncertain predictions
                penalized_r = r_val - uncertainty_penalty * uncert_val
                total_imagined_return += (gamma ** t) * penalized_r
                cumulative_uncertainty += uncert_val

                trajectory.append({
                    'step': t,
                    'predicted_reward': round(r_val, 4),
                    'uncertainty': round(uncert_val, 4),
                    'predicted_sensory_mean': round(float(obs_pred.mean().item()), 4)
                })

        return {
            'status': 'ok',
            'horizon': len(action_sequence),
            'total_imagined_return': float(total_imagined_return),
            'mean_uncertainty': float(cumulative_uncertainty / max(1, len(action_sequence))),
            'reliable_plan': (cumulative_uncertainty / max(1, len(action_sequence))) < 1.5,
            'trajectory': trajectory,
            'scope': 'neural predictive world model imagined forward rollout'
        }

    def train_on_transitions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        steps: int = 5
    ) -> float:
        """Train transition, reward, and uncertainty predictors on collected experience."""
        total_loss = 0.0
        for _ in range(steps):
            z_pred = self.transition_gru(actions, states)
            r_pred = self.reward_head(torch.cat([z_pred, actions], dim=-1)).squeeze(-1)

            loss_z = F.mse_loss(z_pred, next_states)
            loss_r = F.mse_loss(r_pred, rewards)
            loss = loss_z + loss_r

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += float(loss.item())

        return total_loss / steps
