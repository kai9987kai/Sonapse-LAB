"""Coconut-Style Continuous Latent Recurrent Reasoner and Adaptive Depth Engine.

Grounding:
Replaces textual or discrete token deliberation with continuous hidden-state recurrence:
    h^{think}_{i+1} = F(h^{think}_i, z_t)
Features:
1. Continuous Latent Recurrence: Multi-step continuous mental simulation without discrete token bottlenecks.
2. Adaptive Test-Time Depth: Dynamically scales recurrent depth with learned ponder halting.
3. Ponder Cost Optimization: Loss penalizes unnecessary computation:
    L = L_task + lambda_ponder * E[steps]
4. Anytime Performance Curve: Evaluates A(c) = accuracy after c recurrent steps vs equal-compute feedforward baseline.
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
class CoconutThoughtTrace:
    engaged: bool
    recurrent_depth: int
    ponder_cost: float
    accumulated_halt: float
    latent_thought_norm: float
    anytime_trajectory: List[Dict[str, float]] = field(default_factory=list)


class ContinuousLatentReasoner(nn.Module):
    """Coconut-inspired recurrent latent reasoner with continuous thought recycling."""

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 5,
        max_depth: int = 8,
        ponder_cost_rate: float = 0.015,
        halt_threshold: float = 0.95,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_depth = max_depth
        self.ponder_cost_rate = ponder_cost_rate
        self.halt_threshold = halt_threshold

        # Recurrent thought transition: (h_think, z_input) -> h_think_next
        self.thought_cell = nn.GRUCell(latent_dim, latent_dim)

        # Non-linear thought refinement MLP block
        self.thought_refiner = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Learned Halting policy head
        self.halting_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Output Action & Value decoder
        self.action_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # Feed-forward equal-FLOP baseline for anytime comparison
        self.ff_baseline = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, action_dim),
            nn.Tanh()
        )

    def forward_continuous_thinking(
        self,
        z_input: torch.Tensor,
        forced_depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, CoconutThoughtTrace]:
        """Runs continuous latent recurrence without decoding intermediate tokens."""
        if z_input.ndim == 1:
            z_input = z_input.unsqueeze(0)
        B = z_input.shape[0]

        h_think = z_input.clone()
        accumulated_halt = torch.zeros(B, 1, dtype=torch.float32, device=z_input.device)
        total_action = torch.zeros(B, self.action_dim, dtype=torch.float32, device=z_input.device)

        anytime_trace = []
        depth_executed = 0

        max_steps = forced_depth if forced_depth is not None else self.max_depth

        for depth in range(max_steps):
            depth_executed += 1
            # 1. Recurrent update
            h_think = self.thought_cell(z_input, h_think)
            # 2. Residual thought refinement
            h_think = h_think + 0.2 * self.thought_refiner(h_think)

            # 3. Halting probability
            halt_prob = self.halting_head(h_think)
            # Step weight is bounded by remaining probability mass
            step_weight = torch.min(halt_prob, 1.0 - accumulated_halt)

            # 4. Anytime action candidate
            step_action = self.action_head(h_think)
            total_action = total_action + step_weight * step_action
            accumulated_halt = accumulated_halt + halt_prob

            anytime_trace.append({
                'depth': depth + 1,
                'halt_prob': round(float(halt_prob.mean().item()), 4),
                'thought_norm': round(float(h_think.norm(dim=-1).mean().item()), 4),
                'step_action_l2': round(float(step_action.norm(dim=-1).mean().item()), 4)
            })

            # Early stopping if dynamic pondering saturated
            if forced_depth is None and bool((accumulated_halt >= self.halt_threshold).all()):
                break

        # If loop ended with remaining probability mass, add final step action
        remaining_mass = torch.clamp(1.0 - accumulated_halt, min=0.0)
        if bool((remaining_mass > 0).any()):
            final_action = self.action_head(h_think)
            total_action = total_action + remaining_mass * final_action

        trace = CoconutThoughtTrace(
            engaged=True,
            recurrent_depth=depth_executed,
            ponder_cost=round(depth_executed * self.ponder_cost_rate, 4),
            accumulated_halt=round(float(accumulated_halt.mean().item()), 4),
            latent_thought_norm=round(float(h_think.norm(dim=-1).mean().item()), 4),
            anytime_trajectory=anytime_trace
        )

        return total_action, trace

    def compute_anytime_curve(
        self,
        test_inputs: torch.Tensor,
        target_actions: torch.Tensor,
        depths: List[int] = [1, 2, 4, 6, 8]
    ) -> Dict[str, Any]:
        """Computes accuracy/loss across varying test-time recurrent depth A(c) vs feedforward baseline."""
        self.eval()
        anytime_results = {}

        with torch.no_grad():
            # Feedforward equal-FLOP baseline
            ff_pred = self.ff_baseline(test_inputs)
            ff_loss = float(F.mse_loss(ff_pred, target_actions).item())

            for d in depths:
                act_pred, _ = self.forward_continuous_thinking(test_inputs, forced_depth=d)
                loss_d = float(F.mse_loss(act_pred, target_actions).item())
                anytime_results[f'depth_{d}'] = {
                    'depth': d,
                    'mse_loss': round(loss_d, 4),
                    'fl_advantage_vs_ff': round(ff_loss - loss_d, 4)
                }

        return {
            'status': 'ok',
            'feedforward_baseline_mse': round(ff_loss, 4),
            'anytime_recurrent_curve': anytime_results,
            'scope': 'coconut anytime accuracy curve vs recurrent depth'
        }
