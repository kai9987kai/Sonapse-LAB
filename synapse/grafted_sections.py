"""Specialized Grafted Neural Sections & Modern Architecture Components.

Includes:
1. OverflowMiniBrain: 256-neuron weight-programmed mini-brain with Dale-signed connectivity and spectral norm <= 0.95.
2. GraftedMiniDiffusionModel: Mini score-based diffusion policy & trajectory prior with DDIM reverse sampling.
3. SelectiveStateSpaceSection: Input-dependent Selective State-Space (Mamba-style) recurrent layer.
4. HopfieldAssociativeSection: Modern continuous Hopfield associative energy memory.
5. MultiParadigmRLTrainer: Dual-mode SAC Twin-Q and IQL expectile value regression.

Scientific note: These neural modules provide additive computational capacity and
control priors; they do not replace the biological connectome graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverflowMiniBrain(nn.Module):
    """256-neuron weight-programmed mini-brain with Dale-signed connectivity and spectral norm <= 0.95."""

    def __init__(self, num_neurons: int = 256, sensory_dim: int = 155, motor_dim: int = 5):
        super().__init__()
        self.num_neurons = num_neurons
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim

        # Input and motor projections
        self.w_sensory = nn.Linear(sensory_dim, num_neurons, bias=False)
        self.w_motor_feed = nn.Linear(motor_dim, num_neurons, bias=False)
        self.w_motor_out = nn.Linear(num_neurons, motor_dim, bias=False)

        # 256x256 internal recurrent connectivity with Dale's law
        # 80% Excitatory (indices 0..204), 20% Inhibitory (indices 205..255)
        raw_rec = torch.randn(num_neurons, num_neurons) * 0.05
        # Dale sign mask
        dale_sign = torch.ones(num_neurons)
        dale_sign[int(num_neurons * 0.8) :] = -1.0
        self.register_buffer("dale_sign", dale_sign)

        # Enforce positive magnitude
        self.raw_weights = nn.Parameter(torch.abs(raw_rec))

        # Output projection back to primary connectome
        self.overflow_proj = nn.Linear(num_neurons, 64, bias=False)

        # State buffer
        self.register_buffer("state", torch.zeros(num_neurons))
        self.spectral_norm_limit = 0.95

    def get_effective_weights(self) -> torch.Tensor:
        """Construct signed weight matrix with Dale's law and enforce spectral norm <= 0.95."""
        # W = |W| * sign(dale)
        w_signed = torch.abs(self.raw_weights) * self.dale_sign.unsqueeze(0)

        # Spectral normalization estimate via power iteration
        with torch.no_grad():
            v = torch.ones(self.num_neurons, 1, device=w_signed.device) / math.sqrt(self.num_neurons)
            for _ in range(5):
                v = torch.matmul(w_signed, v)
                norm = torch.norm(v)
                if norm < 1e-8:
                    break
                v = v / norm
            spectral_radius = float(torch.norm(torch.matmul(w_signed, v)).item())

        if spectral_radius > self.spectral_norm_limit:
            scale = self.spectral_norm_limit / max(spectral_radius, 1e-8)
            w_signed = w_signed * scale

        return w_signed

    def forward(
        self,
        sensory_input: torch.Tensor,
        motor_feedback: Optional[torch.Tensor] = None,
        tau: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one recurrent microstep.

        Returns:
            overflow_current: [64] additive auxiliary current for primary connectome
            motor_proposal: [5] mini-brain motor action suggestion
            current_state: [256] internal activations
        """
        # Align sensory input
        if sensory_input.ndim == 1:
            s_in = sensory_input.unsqueeze(0)
        else:
            s_in = sensory_input

        in_drive = self.w_sensory(s_in).squeeze(0)
        if motor_feedback is not None:
            in_drive = in_drive + self.w_motor_feed(motor_feedback)

        w_eff = self.get_effective_weights()
        rec_drive = torch.mv(w_eff, self.state)

        # Leaky integration with tanh non-linearity
        new_state = (1.0 - tau) * self.state + tau * torch.tanh(rec_drive + in_drive)
        self.state = new_state.detach()

        overflow_current = self.overflow_proj(new_state)
        motor_proposal = torch.tanh(self.w_motor_out(new_state))

        return overflow_current, motor_proposal, new_state

    def reset_state(self) -> None:
        self.state.zero_()


class GraftedMiniDiffusionModel(nn.Module):
    """Mini score-based diffusion policy model with DDIM sampling clamped to action trust regions."""

    def __init__(self, action_dim: int = 5, cond_dim: int = 64, hidden_dim: int = 128, num_steps: int = 5, seed: int = 42):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(seed)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, 64)

        # Score network
        self.net = nn.Sequential(
            nn.Linear(action_dim + 32 + 64, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Precompute variance schedule (linear beta from 1e-4 to 0.02)
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict noise / score vector epsilon_theta(x_t, t, cond)."""
        t_emb = self.time_embed(t_norm.view(-1, 1))
        c_emb = self.cond_proj(cond.view(-1, self.cond_dim))
        inp = torch.cat([x_t.view(-1, self.action_dim), t_emb, c_emb], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def sample_action(self, cond: torch.Tensor, trust_region_limit: float = 1.0) -> torch.Tensor:
        """Fast 5-step DDIM reverse sampling with trust-region clamping."""
        device = cond.device
        batch_size = cond.shape[0] if cond.ndim > 1 else 1

        # Start from standard normal noise using private generator
        x = torch.randn(batch_size, self.action_dim, generator=self.generator).to(device)

        for step in reversed(range(self.num_steps)):
            t_val = torch.tensor([step / max(self.num_steps - 1, 1)], device=device).expand(batch_size, 1)
            eps_pred = self.forward(x, t_val, cond)

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]

            # Approximate reverse step
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            # Clamp to trust region
            pred_x0 = torch.clamp(pred_x0, -trust_region_limit, trust_region_limit)

            if step > 0:
                alpha_bar_prev = self.alphas_cumprod[step - 1]
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
            else:
                x = pred_x0

        return torch.clamp(x.squeeze(0), -trust_region_limit, trust_region_limit)

    def export_state(self) -> Dict[str, Any]:
        return {
            "weights": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            "generator": self.generator.get_state(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "weights" in state:
            self.load_state_dict(state["weights"])
        if "generator" in state:
            self.generator.set_state(state["generator"])


class SelectiveStateSpaceSection(nn.Module):
    """Selective State-Space (Mamba-style) section with input-dependent selection gates."""

    def __init__(self, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent projections for Delta, B, C
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)

        # Base recurrent transition A (diagonal initialization: -exp(A_log))
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)))
        self.d = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer("h_state", torch.zeros(d_model, d_state))

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step selective state-space scan."""
        # x: [d_model]
        x_flat = x.view(-1)
        params = self.x_proj(x_flat)
        delta_raw = params[0]
        b = params[1 : 1 + self.d_state]  # [d_state]
        c = params[1 + self.d_state :]    # [d_state]

        # Discretization
        delta = F.softplus(delta_raw)
        a_diag = -torch.exp(self.a_log).squeeze(0)  # [d_state]
        a_bar = torch.exp(delta * a_diag)           # [d_state]
        b_bar = delta * b                           # [d_state]

        # Recurrent state update: h = A_bar * h + B_bar * x
        # Outer product of x and b_bar: [d_model, d_state]
        self.h_state = self.h_state * a_bar.unsqueeze(0) + torch.outer(x_flat, b_bar)

        # Output projection: y = h * C + D * x
        y_ssm = torch.mv(self.h_state, c) + self.d * x_flat
        y = self.out_proj(y_ssm)
        return y

    def reset_state(self) -> None:
        self.h_state.zero_()


class HopfieldAssociativeSection(nn.Module):
    """Modern continuous Hopfield network (Demircigil / Ramsauer et al.) for associative memory."""

    def __init__(self, pattern_dim: int = 64, max_patterns: int = 32, beta: float = 4.0):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.beta = beta

        # Stored patterns matrix: [max_patterns, pattern_dim]
        self.register_buffer("patterns", torch.zeros(max_patterns, pattern_dim))
        self.register_buffer("num_stored", torch.tensor(0, dtype=torch.long))

    def store_pattern(self, pattern: torch.Tensor) -> None:
        """Store a new normalized memory pattern."""
        idx = int(self.num_stored.item()) % self.max_patterns
        p = pattern.detach().clone().float()
        p = p / (torch.norm(p, p=2) + 1e-8)
        self.patterns[idx] = p
        self.num_stored += 1

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Continuous Hopfield associative retrieval via softmax attention over stored patterns."""
        if self.num_stored.item() == 0:
            return query, 0.0

        n = min(int(self.num_stored.item()), self.max_patterns)
        active_patterns = self.patterns[:n]  # [n, dim]

        q = query.detach().clone().float()
        q_norm = q / (torch.norm(q, p=2) + 1e-8)

        # Dot products
        sims = torch.mv(active_patterns, q_norm)  # [n]
        weights = F.softmax(self.beta * sims, dim=0)

        # Retrieved pattern
        retrieved = torch.mv(active_patterns.t(), weights)

        # Hopfield energy: E(x) = -1/beta * logsumexp(beta * X x) + 1/2 ||x||^2
        energy = float((-1.0 / self.beta * torch.logsumexp(self.beta * sims, dim=0) + 0.5 * torch.norm(q) ** 2).item())

        return retrieved, round(energy, 4)


class MultiParadigmRLTrainer:
    """Dual-mode RL trainer implementing SAC Twin-Q and IQL expectile value regression."""

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 5,
        iql_expectile: float = 0.7,
        discount: float = 0.99,
        tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.iql_expectile = iql_expectile
        self.discount = discount
        self.tau = tau

        # Twin Q-networks
        self.q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.v_net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1))

        # Target networks
        self.q1_target = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q2_target = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def compute_iql_value_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute IQL expectile loss for value function: |tau - 1(diff < 0)| * diff^2."""
        with torch.no_grad():
            inp = torch.cat([states, actions], dim=-1)
            target_q = torch.min(self.q1_target(inp), self.q2_target(inp))

        v_pred = self.v_net(states)
        diff = target_q - v_pred
        weight = torch.where(diff > 0, self.iql_expectile, 1.0 - self.iql_expectile)
        return torch.mean(weight * (diff ** 2))

    def compute_q_loss(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Bellman error using target V-network."""
        with torch.no_grad():
            target_v = self.v_net(next_states)
            target_q = rewards + (1.0 - dones) * self.discount * target_v

        inp = torch.cat([states, actions], dim=-1)
        q1_loss = F.mse_loss(self.q1(inp), target_q)
        q2_loss = F.mse_loss(self.q2(inp), target_q)
        return q1_loss + q2_loss
