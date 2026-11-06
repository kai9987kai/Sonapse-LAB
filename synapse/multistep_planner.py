"""Recurrent Multistep World Model and Uncertainty-Calibrated Predictive Planning.

Grounding:
Replaces isolated 1-step heuristic planning with a trainable recurrent latent dynamics model:
    p_theta(z_{t+1}, r_t, d_t, o_{t+1} | z_t, a_t)
Provides multi-step Model Predictive Control (MPC) with horizons H in {1, 3, 5, 10},
calibrated epistemic uncertainty penalties, and comparative horizon benchmarking.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentLatentDynamics(nn.Module):
    """Recurrent transition, reward, termination, and observation prediction network."""

    def __init__(self, state_dim: int = 32, action_dim: int = 5, obs_dim: int = 155):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # Encoder: obs -> latent state z
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

        # Recurrent Latent Transition: (z_t, a_t) -> z_{t+1}
        self.transition = nn.GRUCell(action_dim, state_dim)

        # Reward Head: (z_{t+1}, a_t) -> scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Termination Head: z_{t+1} -> termination logit
        self.term_head = nn.Linear(state_dim, 1)

        # Epistemic Uncertainty Head: z_{t+1} -> variance estimate
        self.uncertainty_head = nn.Linear(state_dim, state_dim)

        # Sensory Ray Reconstruction Decoder: z_{t+1} -> 33 ray distances
        self.obs_decoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 33)
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(obs))

    def step(self, z: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward 1 latent step: returns (z_next, r_pred, term_prob, obs_rays, uncertainty)."""
        z_next = self.transition(a, z)
        rew_in = torch.cat([z_next, a], dim=-1)
        r_pred = self.reward_head(rew_in).squeeze(-1)
        term_prob = torch.sigmoid(self.term_head(z_next)).squeeze(-1)
        obs_rays = torch.sigmoid(self.obs_decoder(z_next))
        log_var = self.uncertainty_head(z_next)
        uncertainty = torch.exp(torch.clamp(log_var, -3.0, 3.0)).mean(dim=-1)
        return z_next, r_pred, term_prob, obs_rays, uncertainty


class MultistepRecurrentPlanner:
    """Model Predictive Control over recurrent latent trajectories with uncertainty penalties."""

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 5,
        obs_dim: int = 155,
        default_horizon: int = 5,
        gamma: float = 0.95,
        beta_unc: float = 0.3,
        lambda_term: float = 2.0,
        num_candidates: int = 16,
        lr: float = 1e-3,
        seed: int = 42
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.default_horizon = default_horizon
        self.gamma = gamma
        self.beta_unc = beta_unc
        self.lambda_term = lambda_term
        self.num_candidates = num_candidates

        self.model = RecurrentLatentDynamics(state_dim, action_dim, obs_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.training_steps = 0
        self.plans_generated = 0

    def train_step(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        reward_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        done_batch: np.ndarray
    ) -> Dict[str, float]:
        """Train recurrent dynamics on transition batch with multi-objective loss."""
        self.training_steps += 1
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32)
        rew_t = torch.as_tensor(reward_batch, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs_batch, dtype=torch.float32)
        done_t = torch.as_tensor(done_batch, dtype=torch.float32)

        self.optimizer.zero_grad()
        z_t = self.model.encode(obs_t)
        z_next_target = self.model.encode(next_obs_t).detach()

        z_next, r_pred, term_prob, obs_rays, uncert = self.model.step(z_t, act_t)

        # Losses
        loss_trans = F.mse_loss(z_next, z_next_target)
        loss_rew = F.mse_loss(r_pred, rew_t)
        loss_term = F.binary_cross_entropy(term_prob, done_t)
        loss_obs = F.mse_loss(obs_rays, next_obs_t[:, :33])
        # Uncertainty calibration: variance penalizes residual error
        res_error = (r_pred - rew_t).detach().abs()
        loss_unc = F.mse_loss(uncert, res_error)

        total_loss = loss_trans + 2.0 * loss_rew + 0.5 * loss_term + 0.5 * loss_obs + 0.2 * loss_unc
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss_total': float(total_loss.item()),
            'loss_trans': float(loss_trans.item()),
            'loss_reward': float(loss_rew.item()),
            'loss_term': float(loss_term.item()),
            'loss_obs': float(loss_obs.item())
        }

    def plan(
        self,
        current_obs: np.ndarray,
        baseline_action: np.ndarray,
        horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate multi-step imagined rollouts and select uncertainty-penalized optimal action."""
        self.plans_generated += 1
        H = horizon if horizon is not None else self.default_horizon
        obs_t = torch.as_tensor(current_obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            z_0 = self.model.encode(obs_t)  # [1, state_dim]

            # Generate K candidate action trajectories over horizon H
            # Candidate 0 is always the baseline action repeated
            candidate_trajs = []
            cand_0 = np.tile(baseline_action.reshape(1, -1), (H, 1))
            candidate_trajs.append(cand_0)

            # Candidates 1..K-1: local stochastic perturbations
            for _ in range(self.num_candidates - 1):
                pert = self.rng.normal(0.0, 0.25, size=(H, self.action_dim)).astype(np.float32)
                cand = np.clip(cand_0 + pert, -1.0, 1.0)
                candidate_trajs.append(cand)

            scores = []
            trajectories_diag = []

            for k, traj in enumerate(candidate_trajs):
                z = z_0.clone()
                traj_score = 0.0
                step_records = []

                for tau in range(H):
                    a_tau = torch.as_tensor(traj[tau], dtype=torch.float32).unsqueeze(0)
                    z, r_pred, term_prob, _, uncert = self.model.step(z, a_tau)

                    r_val = float(r_pred.item())
                    u_val = float(uncert.item())
                    d_val = float(term_prob.item())

                    # Uncertainty & termination penalization
                    penalized_r = r_val - self.beta_unc * u_val - self.lambda_term * d_val
                    traj_score += (self.gamma ** tau) * penalized_r

                    step_records.append({
                        'tau': tau,
                        'r_pred': r_val,
                        'uncert': u_val,
                        'term_prob': d_val
                    })

                scores.append(traj_score)
                trajectories_diag.append(step_records)

            best_idx = int(np.argmax(scores))
            best_action = candidate_trajs[best_idx][0]  # Take first action of best sequence
            baseline_score = scores[0]
            best_score = scores[best_idx]
            gain = best_score - baseline_score

        return {
            'action': best_action,
            'horizon': H,
            'best_candidate_idx': best_idx,
            'best_predicted_value': best_score,
            'baseline_predicted_value': baseline_score,
            'predicted_plan_gain': gain,
            'used_plan': best_idx != 0 and gain > 0.02,
            'candidates_evaluated': len(candidate_trajs)
        }

    def summary(self) -> Dict[str, Any]:
        return {
            'training_steps': self.training_steps,
            'plans_generated': self.plans_generated,
            'default_horizon': self.default_horizon,
            'gamma': self.gamma,
            'beta_uncertainty': self.beta_unc,
            'lambda_termination': self.lambda_term
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'training_steps': self.training_steps,
            'plans_generated': self.plans_generated,
            'default_horizon': self.default_horizon,
            'weights': {k: v.cpu().numpy().tolist() for k, v in self.model.state_dict().items()}
        }

    def import_state(self, state: Dict[str, Any]):
        self.training_steps = int(state.get('training_steps', 0))
        self.plans_generated = int(state.get('plans_generated', 0))
        self.default_horizon = int(state.get('default_horizon', self.default_horizon))
        weights = {k: torch.tensor(v, dtype=torch.float32) for k, v in state['weights'].items()}
        self.model.load_state_dict(weights)


def compare_planning_horizons(
    planner: MultistepRecurrentPlanner,
    test_obs: np.ndarray,
    baseline_action: np.ndarray,
    horizons: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Evaluate multi-step planner across different planning horizons."""
    results = {}
    for H in horizons:
        plan_out = planner.plan(test_obs, baseline_action, horizon=H)
        results[f"horizon_{H}"] = {
            'best_score': plan_out['best_predicted_value'],
            'gain': plan_out['predicted_plan_gain'],
            'used_plan': plan_out['used_plan'],
            'action_norm': float(np.linalg.norm(plan_out['action']))
        }
    return {
        'status': 'ok',
        'horizons_tested': horizons,
        'results': results,
        'scope': 'multi-step recurrent planning horizon comparison'
    }
