"""Learned Protocol Selector for Sonapse NeuroWave-X2.

Predicts protocol distribution p(protocol | z_t) = softmax(f_theta(z_t)) conditioned on
uncertainty, surprise, decision value, compute/bandwidth costs, and regional state.
Supports soft mixture training, Gumbel-Softmax exploration, and deterministic evaluation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol_registry import PROTOCOLS, list_protocols_up_to_level


class ProtocolSelector(nn.Module):
    """Context-conditioned policy for dynamic neural communication protocol selection."""

    def __init__(
        self,
        context_dim: int = 32,
        hidden_dim: int = 32,
        temperature: float = 1.0,
        mode: str = "learned",  # 'learned', 'gumbel', 'fixed', 'cost_aware'
        default_protocol: str = "vector",
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.mode = str(mode)
        self.default_protocol = str(default_protocol)

        self.protocol_names: List[str] = list(PROTOCOLS.keys())
        self.num_protocols = len(self.protocol_names)

        # Policy network: maps context vector -> protocol logits
        self.net = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_protocols),
        )

        # Conservative initialization: default protocol has slightly higher logit
        nn.init.normal_(self.net[0].weight, std=0.05)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[3].weight, std=0.05)
        bias_init = torch.zeros(self.num_protocols)
        if self.default_protocol in self.protocol_names:
            def_idx = self.protocol_names.index(self.default_protocol)
            bias_init[def_idx] = 1.5
        self.net[3].bias.data.copy_(bias_init)

        # Telemetry history buffer
        self.last_probs: np.ndarray = np.ones(self.num_protocols, dtype=np.float32) / self.num_protocols
        self.last_selected_protocol = self.default_protocol
        self.selection_counts: Dict[str, int] = {p: 0 for p in self.protocol_names}

    def forward(
        self,
        context: torch.Tensor,
        max_development_level: int = 6,
        deterministic: bool = False,
    ) -> Tuple[str, torch.Tensor, Dict[str, float]]:
        """Evaluates protocol selection policy.
        
        Args:
            context: (context_dim,) or (B, context_dim)
            max_development_level: maximum complexity level eligible (0 to 6)
            deterministic: if True, chooses argmax over probabilities
        Returns:
            (selected_protocol_id, protocol_probabilities, diagnostics_dict)
        """
        if context.dim() == 1:
            c = context.unsqueeze(0)
        else:
            c = context

        logits = self.net(c)[0]  # shape (num_protocols,)

        # Mask out protocols above current developmental complexity level
        eligible_protocols = set(list_protocols_up_to_level(max_development_level))
        mask = torch.zeros_like(logits)
        for i, p_name in enumerate(self.protocol_names):
            if p_name not in eligible_protocols:
                mask[i] = -1e9
        masked_logits = (logits + mask) / max(0.1, self.temperature)

        if self.mode == "gumbel" and self.training and not deterministic:
            probs = F.gumbel_softmax(masked_logits, tau=self.temperature, hard=False)
            chosen_idx = int(torch.argmax(probs).item())
        else:
            probs = F.softmax(masked_logits, dim=-1)
            if deterministic:
                chosen_idx = int(torch.argmax(probs).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                chosen_idx = int(dist.sample().item())

        selected_protocol = self.protocol_names[chosen_idx]
        self.last_probs = probs.detach().cpu().numpy()
        self.last_selected_protocol = selected_protocol
        self.selection_counts[selected_protocol] = self.selection_counts.get(selected_protocol, 0) + 1

        # Protocol entropy H(p) = -sum p log p
        safe_probs = torch.clamp(probs, min=1e-12)
        entropy = -float(torch.sum(probs * torch.log(safe_probs)).item())

        diag: Dict[str, float] = {
            f"p_{p}": float(probs[i].item()) for i, p in enumerate(self.protocol_names)
        }
        diag["protocol_entropy"] = entropy
        diag["effective_protocols"] = float(math.exp(entropy))

        return selected_protocol, probs, diag
