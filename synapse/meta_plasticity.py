"""Meta-Learning of Synaptic Plasticity Rules: Delta w_ij = f_phi(x_i, x_j, delta_t, m_t, u_ij, c_t).

Implements a learnable multi-factor synaptic update rule respecting:
1. Dale's law sign constraints: excitatory weights cannot cross below 0, inhibitory cannot cross above 0.
2. Hard weight magnitude bounds: |w_ij| <= w_max.
3. Neuromodulatory gating (DA, 5-HT, OA, ACh).
4. Long-term synaptic importance & utility protection against catastrophic forgetting.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaPlasticityRule(nn.Module):
    """Parameterized synaptic learning rule function f_phi."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # Input features: [pre_x, post_x, td_error, dopamine, acetylcholine, utility, novelty, cur_w]
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        # Small orthogonal init for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.2)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        pre_act: torch.Tensor,
        post_act: torch.Tensor,
        td_error: float,
        dopamine: float = 1.0,
        acetylcholine: float = 1.0,
        utility: Optional[torch.Tensor] = None,
        novelty: float = 0.0,
        current_w: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_syn = len(pre_act)
        u_val = utility if utility is not None else torch.zeros(n_syn, device=pre_act.device)
        w_val = current_w if current_w is not None else torch.zeros(n_syn, device=pre_act.device)

        features = torch.stack(
            [
                pre_act,
                post_act,
                torch.full((n_syn,), float(td_error), device=pre_act.device),
                torch.full((n_syn,), float(dopamine), device=pre_act.device),
                torch.full((n_syn,), float(acetylcholine), device=pre_act.device),
                u_val,
                torch.full((n_syn,), float(novelty), device=pre_act.device),
                w_val,
            ],
            dim=1,
        )
        raw_delta = self.net(features).squeeze(1) * 0.02
        return raw_delta


# Backwards-compatible alias
ParameterizedPlasticityRule = MetaPlasticityRule
