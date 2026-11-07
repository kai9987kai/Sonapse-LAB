"""Causal World Model for Mechanism-Aware State Transitions.

Implements structural causal models (SCMs) that predict transitions under both
observational conditioning P(s' | s, a) and explicit do-calculus interventions P(s' | do(a)).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_hypotheses import CausalHypothesisBank


class CausalWorldModel(nn.Module):
    """Predicts next state and disentangles agent causal agency from passive dynamics."""

    def __init__(
        self,
        state_dim: int = 48,
        act_dim: int = 5,
        action_dim: Optional[int] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = action_dim if action_dim is not None else act_dim
        self.seed = int(seed)
        self.hypothesis_bank = CausalHypothesisBank(seed=seed)

        # Disentangled dynamics networks:
        # 1. Passive environmental drift net: f_env(s)
        self.env_drift_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, state_dim),
        )
        # 2. Agent interventional effect net: f_agent(s, do(a))
        self.agent_effect_net = nn.Sequential(
            nn.Linear(state_dim + act_dim, 64),
            nn.SiLU(),
            nn.Linear(64, state_dim),
        )

    def forward_observational(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """P(s' | s, a) = s + f_env(s) + f_agent(s, a)."""
        drift = self.env_drift_net(s)
        effect = self.agent_effect_net(torch.cat([s, a], dim=-1))
        return s + drift + effect

    def forward_interventional(self, s: torch.Tensor, do_a: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """P(s' | do(a), s) with counterfactual attribution of delta_s."""
        drift = self.env_drift_net(s)
        effect = self.agent_effect_net(torch.cat([s, do_a], dim=-1))
        next_s = s + drift + effect
        agent_fraction = float((effect.norm() / (drift.norm() + effect.norm() + 1e-6)).item())
        return next_s, {
            "agent_causal_attribution": round(agent_fraction, 4),
            "environmental_drift_norm": round(float(drift.norm().item()), 4),
            "intervention_action_norm": round(float(do_a.norm().item()), 4),
        }

    def observe(self, s: Any, a: Any, next_s: Any) -> Dict[str, Any]:
        """Record transition and return causal attribution diagnostics."""
        with torch.no_grad():
            s_t = torch.as_tensor(s, dtype=torch.float32).flatten()
            a_t = torch.as_tensor(a, dtype=torch.float32).flatten()
            ns_t = torch.as_tensor(next_s, dtype=torch.float32).flatten()

            if s_t.numel() < self.state_dim:
                s_t = F.pad(s_t, (0, self.state_dim - s_t.numel()))
            elif s_t.numel() > self.state_dim:
                s_t = s_t[:self.state_dim]

            if a_t.numel() < self.act_dim:
                a_t = F.pad(a_t, (0, self.act_dim - a_t.numel()))
            elif a_t.numel() > self.act_dim:
                a_t = a_t[:self.act_dim]

            pred_s, info = self.forward_interventional(s_t, a_t)
            delta_actual = ns_t[:self.state_dim] - s_t
            delta_pred = pred_s - s_t
            err = float(torch.norm(delta_actual - delta_pred).item())

            info["prediction_error"] = round(err, 4)
            info["hypotheses"] = {k: h.to_dict() for k, h in self.hypothesis_bank.hypotheses.items()}
            return info
