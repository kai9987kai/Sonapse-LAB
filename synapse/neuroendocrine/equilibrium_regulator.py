"""Affective Equilibrium and Homeostatic Regulator.

Computes target affective states e_t^* = f_psi(o_t, b_t, g_t, p_t),
evaluates allostatic regulation cost L_eq = (e_t - e_t^*)^T Q_t (e_t - e_t^*),
tracks physiological burden L_body, and synthesizes restorative steering currents
u_t^{eq} into the WPAC equilibrium_control population.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.neuroendocrine.mixed_affective_state import AFFECTIVE_DIMENSIONS
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class AffectiveEquilibriumRegulator(nn.Module):
    """Monitors affective trajectory and applies compensatory steering towards equilibrium."""

    def __init__(
        self,
        num_emotions: int = len(AFFECTIVE_DIMENSIONS),  # 32
        obs_dim: int = 64,
        body_dim: int = 8,
        gland_dim: int = 12,
        control_dim: int = 32,  # Matches WPAC equilibrium_control population (32 neurons)
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_emotions = num_emotions
        self.obs_dim = obs_dim
        self.body_dim = body_dim
        self.gland_dim = gland_dim
        self.control_dim = control_dim

        # Target setpoint predictor f_psi(o_t, b_t, g_t)
        in_dim = obs_dim + body_dim + gland_dim
        self.net_target = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_emotions),
            nn.Sigmoid(),
        )

        # Baseline baseline equilibrium attractor (default baseline state: calm, moderate contentment)
        base_target = torch.full((num_emotions,), 0.05)
        idx = {name: i for i, name in enumerate(AFFECTIVE_DIMENSIONS)}
        if "calmness" in idx:
            base_target[idx["calmness"]] = 0.60
        if "contentment" in idx:
            base_target[idx["contentment"]] = 0.50
        if "curiosity" in idx:
            base_target[idx["curiosity"]] = 0.40
        if "restfulness" in idx:
            base_target[idx["restfulness"]] = 0.40
        if "trust" in idx:
            base_target[idx["trust"]] = 0.50
        self.register_buffer("base_target", base_target)

        # Importance diagonal Q_t (penalties on persistent negative dysregulation)
        q_weights = torch.ones(num_emotions)
        for neg in ["fear", "anxiety", "dread", "frustration", "anger", "hate_aversive", "sadness", "stress_arousal"]:
            if neg in idx:
                q_weights[idx[neg]] = 2.5
        self.register_buffer("q_diag", q_weights)

        # Compensatory steering generator: maps affective error (e_t - e_t^*) to control current
        self.net_control = nn.Sequential(
            nn.Linear(num_emotions, 64),
            nn.Tanh(),
            nn.Linear(64, control_dim),
        )

        # Running history buffers for telemetry
        self.register_buffer("last_target", base_target.clone())
        self.register_buffer("last_error", torch.zeros(num_emotions))
        self.register_buffer("running_cost", torch.tensor(0.0))

    def compute_target(
        self,
        obs: Optional[torch.Tensor] = None,
        body_state: Optional[torch.Tensor] = None,
        glands: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts context-adaptive equilibrium target e_t^* in [0, 1]^32."""
        device = self.base_target.device
        if obs is None:
            obs = torch.zeros(self.obs_dim, device=device)
        if body_state is None:
            body_state = torch.zeros(self.body_dim, device=device)
        if glands is None:
            glands = torch.zeros(self.gland_dim, device=device)

        inputs = torch.cat([obs.flatten(), body_state.flatten(), glands.flatten()], dim=0)
        delta_target = self.net_target(inputs.unsqueeze(0)).squeeze(0)

        # Convex blend with baseline setpoint
        e_target = 0.7 * self.base_target + 0.3 * delta_target
        self.last_target = e_target.detach()
        return e_target

    def evaluate_cost(
        self,
        e_t: torch.Tensor,
        e_target: torch.Tensor,
        body_state: Optional[torch.Tensor] = None,
        glands: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates L_eq, L_body, and total allostatic cost.

        L_eq = (e_t - e_target)^T Q (e_t - e_target)
        L_body = metabolic / allostatic strain penalty (e.g. high sustained stress/exhaustion)
        """
        err = e_t - e_target
        self.last_error = err.detach()

        # Weighted squared affective error
        l_eq = torch.sum(self.q_diag * (err ** 2))

        # Physiological cost: penalized by extreme gland levels (e.g. cortisol, extreme adrenaline)
        l_body = torch.tensor(0.0, device=e_t.device)
        if glands is not None and glands.numel() >= 12:
            # Gland 6: cortisol, Gland 4: adrenaline, Gland 11: adenosine
            cortisol = glands[6] if glands.dim() == 1 else glands[0, 6]
            adrenaline = glands[4] if glands.dim() == 1 else glands[0, 4]
            adenosine = glands[11] if glands.dim() == 1 else glands[0, 11]
            l_body = 0.5 * (cortisol ** 2) + 0.3 * (adrenaline ** 2) + 0.2 * (adenosine ** 2)

        total_cost = l_eq + l_body
        self.running_cost = 0.9 * self.running_cost + 0.1 * total_cost.detach()
        return total_cost, l_eq, l_body

    def synthesize_control(self, e_t: torch.Tensor, e_target: torch.Tensor) -> torch.Tensor:
        """Computes restorative steering control current u_t^{eq} for WPAC."""
        err = e_t - e_target
        u_t = self.net_control(err.unsqueeze(0)).squeeze(0)
        return torch.clamp(u_t, -1.0, 1.0)

    def forward(
        self,
        e_t: torch.Tensor,
        obs: Optional[torch.Tensor] = None,
        body_state: Optional[torch.Tensor] = None,
        glands: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Full regulatory step: computes target, costs, and compensatory control."""
        target = self.compute_target(obs, body_state, glands)
        cost, l_eq, l_body = self.evaluate_cost(e_t, target, body_state, glands)
        control = self.synthesize_control(e_t, target)

        receipt = EvidenceReceipt(
            item_id="affective_equilibrium_step",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="AffectiveEquilibriumRegulator",
            sha256_digest=compute_tensor_digest(control),
            metadata={
                "cost_total": float(cost.item()),
                "l_eq": float(l_eq.item()),
                "l_body": float(l_body.item()),
            },
        )

        return {
            "target": target,
            "control": control,
            "cost_total": cost,
            "l_eq": l_eq,
            "l_body": l_body,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        """Resets running state."""
        self.last_target.copy_(self.base_target)
        self.last_error.zero_()
        self.running_cost.zero_()
