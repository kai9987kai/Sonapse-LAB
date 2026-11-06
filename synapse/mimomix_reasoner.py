"""MiMoMix Dual-Process Deliberative Reasoner & Cognitive Verifier.

Adapted from Kai9987kai's Supermix v53/v56/v93 MiMoMix line.
Fuses fast intuitive reflexive action with deliberative multi-step latent reasoning:
1. **System 1 (Reflexive Mode)**: Sub-millisecond sensorimotor path straight from
   proxy connectome and motor head when uncertainty and free energy are low.
2. **System 2 (Deliberative Mode)**: Activated when Free Energy, Sensory Surprisal,
   or Threat exceed an epistemic deliberation threshold. Halts raw reflex and engages
   adaptive recurrent mental steps with an ACT (Adaptive Computation Time) ponder cost.
3. **Attention Sink & Auxiliary-Loss-Free Balancing**: Employs an attention sink null-slot
   to prevent forced attention over noisy inputs.
4. **Calibrated Action Refinement**: Emits an uncertainty-weighted delta correction
   applied to the primary motor action.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiMoMixReasoningTrace:
    system_engaged: str = "System 1 (Reflex)"
    ponder_steps: int = 0
    ponder_cost: float = 0.0
    epistemic_trigger: float = 0.0
    action_delta_l2: float = 0.0
    confidence: float = 1.0


class MiMoMixCognitiveCore(nn.Module):
    """Dual-process deliberative latent reasoner with adaptive ponder halting."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 5,
        max_ponder_steps: int = 4,
        ponder_cost_rate: float = 0.02,
        deliberation_threshold: float = 0.45,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_ponder_steps = max_ponder_steps
        self.ponder_cost_rate = ponder_cost_rate
        self.deliberation_threshold = deliberation_threshold

        # Latent state transition recurrent step
        self.recurrent_cell = nn.GRUCell(latent_dim, latent_dim)

        # Attention Sink: Learnable bias vector allowing null attention
        self.attention_sink = nn.Parameter(torch.zeros(latent_dim))

        # Halting probability head (ACT)
        self.halting_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Action refinement delta head
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def deliberate(
        self,
        latent_state: torch.Tensor,
        free_energy: float,
        surprisal: float,
        threat: float = 0.0,
    ) -> Tuple[torch.Tensor, MiMoMixReasoningTrace]:
        """Evaluates whether to fire fast reflex (System 1) or engage mental deliberation (System 2)."""
        epistemic_signal = float(0.5 * free_energy + 0.3 * surprisal + 0.5 * threat)
        trace = MiMoMixReasoningTrace(epistemic_trigger=epistemic_signal)

        # System 1: Low uncertainty reflex -> zero action delta, 0 ponder steps
        if epistemic_signal < self.deliberation_threshold:
            trace.system_engaged = "System 1 (Reflex)"
            trace.confidence = float(np.clip(1.0 - epistemic_signal, 0.5, 1.0))
            return torch.zeros(self.action_dim, dtype=torch.float32), trace

        # System 2: Deliberative Thinking engaged
        trace.system_engaged = "System 2 (Deliberation)"
        current_h = latent_state.clone()
        if current_h.ndim == 1:
            current_h = current_h.unsqueeze(0)

        accumulated_halt = 0.0
        ponder_steps = 0
        total_delta = torch.zeros(1, self.action_dim, dtype=torch.float32)

        for step in range(self.max_ponder_steps):
            ponder_steps += 1
            # Incorporate attention sink bias for stabilization
            input_with_sink = current_h + 0.1 * self.attention_sink.unsqueeze(0)
            current_h = self.recurrent_cell(input_with_sink, current_h)

            # Compute halting probability
            halt_prob = float(self.halting_head(current_h).item())
            step_delta = self.delta_head(current_h) * 0.25  # Scaled action correction

            weight = min(halt_prob, 1.0 - accumulated_halt)
            total_delta += weight * step_delta
            accumulated_halt += halt_prob

            if accumulated_halt >= 0.95:
                break

        action_delta = total_delta.squeeze(0).detach()
        trace.ponder_steps = ponder_steps
        trace.ponder_cost = round(ponder_steps * self.ponder_cost_rate, 4)
        trace.action_delta_l2 = round(float(torch.norm(action_delta).item()), 4)
        trace.confidence = round(float(np.clip(1.0 - 0.2 * ponder_steps, 0.4, 0.95)), 4)

        return action_delta, trace
