"""Direct Preference Optimization (DPO) over Machine-Verifiable Outcomes.

Grounding:
Replaces ungrounded LLM self-judging with first-party machine-verifiable preference pairs:
1. Safe vs. Unsafe trajectory outcomes (surviving vs hazard collision / death).
2. High-return vs. Low-return planner rollouts.
3. Well-calibrated vs. Overconfident forecasts.
4. Unit-test passing vs. Failing patch proposals.

Optimizes policy parameters directly via DPO:
    L_DPO(theta; theta_ref) = - E_{(x, y_w, y_l)} [ log sigma( beta * log(pi_theta(y_w|x)/pi_ref(y_w|x))
                                                             - beta * log(pi_theta(y_l|x)/pi_ref(y_l|x)) ) ]
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PreferencePair:
    pair_id: str
    context: np.ndarray      # Observation / state x
    winner_action: np.ndarray # y_w (preferred machine-verified action)
    loser_action: np.ndarray  # y_l (dispreferred failing / unsafe action)
    criterion: str           # 'safety', 'return', 'calibration', 'unit_tests'
    delta_utility: float     # Measured utility advantage


class DPOPolicyNetwork(nn.Module):
    """Gaussian policy network emitting action mean and log-std."""

    def __init__(self, obs_dim: int = 155, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = torch.tanh(self.mean_head(h))
        std = torch.exp(torch.clamp(self.log_std, -2.0, 1.0))
        return mean, std

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, std = self.forward(obs)
        var = std.pow(2)
        log_scale = torch.log(std)
        # Gaussian log probability: -0.5 * sum( ((a - mu)/std)^2 + 2*log(std) + log(2*pi) )
        log_p = -0.5 * torch.sum(
            ((actions - mean) ** 2) / var + 2.0 * log_scale + math.log(2.0 * math.pi),
            dim=-1
        )
        return log_p


class MachineVerifiableDPOEngine:
    """Collects machine-checkable preference pairs and trains policy via DPO."""

    def __init__(
        self,
        obs_dim: int = 155,
        action_dim: int = 5,
        beta: float = 0.1,
        lr: float = 1e-4,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.beta = beta

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = DPOPolicyNetwork(obs_dim, action_dim)
            # Reference policy is a frozen copy of the initial policy
            self.ref_policy = copy.deepcopy(self.policy)
            for param in self.ref_policy.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)
        self.preference_buffer: List[PreferencePair] = []
        self.last_margin: float = 0.0

    def record_preference(
        self,
        context: np.ndarray,
        winner_action: np.ndarray,
        loser_action: np.ndarray,
        criterion: str,
        delta_utility: float,
    ):
        """Appends a machine-verified preference pair to the training buffer."""
        pair_id = f"pref_{len(self.preference_buffer) + 1}_{criterion}"
        pair = PreferencePair(
            pair_id=pair_id,
            context=context.copy().astype(np.float32),
            winner_action=winner_action.copy().astype(np.float32),
            loser_action=loser_action.copy().astype(np.float32),
            criterion=criterion,
            delta_utility=float(delta_utility)
        )
        self.preference_buffer.append(pair)

    def train_dpo_step(self, batch_size: int = 16) -> Dict[str, float]:
        """Executes one DPO training step over collected verifiable preference pairs."""
        if len(self.preference_buffer) < 4:
            return {'status': 'insufficient_data', 'loss': 0.0, 'buffer_size': len(self.preference_buffer)}

        indices = self.rng.choice(len(self.preference_buffer), size=min(batch_size, len(self.preference_buffer)), replace=False)
        batch = [self.preference_buffer[i] for i in indices]

        obs_batch = torch.as_tensor(np.stack([p.context for p in batch]), dtype=torch.float32)
        act_w_batch = torch.as_tensor(np.stack([p.winner_action for p in batch]), dtype=torch.float32)
        act_l_batch = torch.as_tensor(np.stack([p.loser_action for p in batch]), dtype=torch.float32)

        self.optimizer.zero_grad()

        # Policy log probabilities
        pi_w = self.policy.log_prob(obs_batch, act_w_batch)
        pi_l = self.policy.log_prob(obs_batch, act_l_batch)

        # Frozen reference policy log probabilities
        with torch.no_grad():
            ref_w = self.ref_policy.log_prob(obs_batch, act_w_batch)
            ref_l = self.ref_policy.log_prob(obs_batch, act_l_batch)

        # Log-ratio differences
        log_ratio_w = pi_w - ref_w
        log_ratio_l = pi_l - ref_l

        # DPO loss: -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
        logits = self.beta * (log_ratio_w - log_ratio_l)
        loss = -F.logsigmoid(logits).mean()

        loss.backward()
        self.optimizer.step()

        # Implicit reward margin
        reward_margin = (logits / self.beta).detach().mean().item()

        return {
            'loss_dpo': float(loss.item()),
            'implicit_reward_margin': float(reward_margin),
            'batch_size': len(batch),
            'buffer_size': len(self.preference_buffer),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'policy': copy.deepcopy(self.policy.state_dict()),
            'ref_policy': copy.deepcopy(self.ref_policy.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'rng': copy.deepcopy(self.rng.bit_generator.state),
            'last_margin': float(getattr(self, 'last_margin', 0.0)),
            'preference_buffer': [
                {
                    'pair_id': p.pair_id,
                    'context': p.context.copy(),
                    'winner_action': p.winner_action.copy(),
                    'loser_action': p.loser_action.copy(),
                    'criterion': p.criterion,
                    'delta_utility': float(p.delta_utility),
                }
                for p in self.preference_buffer
            ],
        }

    def import_state(self, state: Dict[str, Any]):
        if 'policy' in state:
            self.policy.load_state_dict(copy.deepcopy(state['policy']))
        if 'ref_policy' in state:
            self.ref_policy.load_state_dict(copy.deepcopy(state['ref_policy']))
        if 'optimizer' in state:
            self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        else:
            self.optimizer.state.clear()
        if 'rng' in state:
            self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.last_margin = float(state.get('last_margin', 0.0))
        self.preference_buffer = [
            PreferencePair(
                pair_id=str(p['pair_id']),
                context=np.asarray(p['context'], dtype=np.float32).copy(),
                winner_action=np.asarray(p['winner_action'], dtype=np.float32).copy(),
                loser_action=np.asarray(p['loser_action'], dtype=np.float32).copy(),
                criterion=str(p['criterion']),
                delta_utility=float(p['delta_utility']),
            )
            for p in state.get('preference_buffer', [])
        ]
