"""Active Dendritic Liquid Time-Constant (LTC / CfC) Neuromorphic Sub-Graft.

Grounding:
Inspired by Hasani et al. (Nature Machine Intelligence, Closed-form Continuous-time
Neural Networks) and biological two-compartment pyramidal/projection neurons with
basal feedforward inputs and apical contextual/neuromodulatory tuft inputs:
    tau_i(t) = tau_min + (tau_max - tau_min) * sigmoid(W_tau [x_basal, c_apical] + b_tau)
    v_i(t + dt) = exp(-dt / tau_i(t)) * v_i(t)
                  + (1 - exp(-dt / tau_i(t))) * tanh(I_basal,i * (1 + alpha_NMDA * sigmoid(I_apical,i)))

The readout projection starts at zero so enabling the sub-graft preserves exact
baseline actor behavior until online or supervised updates modify it.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DendriticLiquidSubGraft(nn.Module):
    """Two-compartment dendritic liquid time-constant module with zero-init residual readout."""

    def __init__(
        self,
        basal_dim: int = 32,
        apical_dim: int = 26,
        hidden_dim: int = 32,
        tau_min: float = 0.15,
        tau_max: float = 4.0,
        dt: float = 0.25,
        lr: float = 1e-3,
        seed: int = 7,
    ):
        super().__init__()
        self.basal_dim = int(basal_dim)
        self.apical_dim = int(apical_dim)
        self.hidden_dim = int(hidden_dim)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.dt = float(dt)
        self.seed = int(seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 3109)
            self.basal_proj = nn.Linear(self.basal_dim, self.hidden_dim)
            self.apical_proj = nn.Linear(self.apical_dim, self.hidden_dim)
            self.tau_gate = nn.Linear(self.basal_dim + self.apical_dim, self.hidden_dim)
            self.readout = nn.Linear(self.hidden_dim, self.basal_dim, bias=False)
            nn.init.orthogonal_(self.basal_proj.weight, gain=0.6)
            nn.init.zeros_(self.basal_proj.bias)
            nn.init.orthogonal_(self.apical_proj.weight, gain=0.5)
            nn.init.zeros_(self.apical_proj.bias)
            nn.init.normal_(self.tau_gate.weight, std=0.1)
            nn.init.zeros_(self.tau_gate.bias)
            # Zero-initialized residual readout guarantees unchanged untrained output.
            nn.init.zeros_(self.readout.weight)

        self.register_buffer("voltage", torch.zeros(self.hidden_dim))
        self.register_buffer("elig", torch.zeros(self.basal_dim, self.hidden_dim))
        self.register_buffer("last_tau", torch.full((self.hidden_dim,), 0.5 * (self.tau_min + self.tau_max)))
        self.register_buffer("last_nmda", torch.zeros(self.hidden_dim))
        self.updates = 0
        self.last_loss = 0.0

    def reset_state(self) -> None:
        self.voltage.zero_()
        self.elig.zero_()
        self.last_tau.fill_(0.5 * (self.tau_min + self.tau_max))
        self.last_nmda.zero_()

    def step(
        self,
        basal: torch.Tensor,
        apical: torch.Tensor,
        ach_precision: float = 0.5,
        astrocyte_ca: float = 0.1,
        learn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Advance dendritic compartments one step and return (residual_delta, diagnostics)."""
        b = basal.detach().reshape(-1)[: self.basal_dim]
        if b.numel() < self.basal_dim:
            b = F.pad(b, (0, self.basal_dim - b.numel()))
        a = apical.detach().reshape(-1)[: self.apical_dim]
        if a.numel() < self.apical_dim:
            a = F.pad(a, (0, self.apical_dim - a.numel()))

        joint = torch.cat([b, a], dim=0)
        # Cholinergic precision speeds integration (smaller tau); glial calcium stabilizes slow memory (larger tau).
        mod_shift = float(np.clip(0.35 * astrocyte_ca - 0.35 * (ach_precision - 0.5), -1.0, 1.0))
        tau_sig = torch.sigmoid(self.tau_gate(joint) + mod_shift)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_sig

        i_basal = self.basal_proj(b)
        i_apical = self.apical_proj(a)
        nmda_gate = torch.sigmoid(i_apical)
        # Nonlinear apical amplification (dendritic plateau potential)
        equilibrium_drive = torch.tanh(i_basal * (1.0 + 0.65 * nmda_gate))

        decay = torch.exp(-self.dt / tau.clamp(min=1e-3))
        next_v = decay * self.voltage + (1.0 - decay) * equilibrium_drive
        if learn:
            self.voltage.copy_(next_v.detach())
            self.last_tau.copy_(tau.detach())
            self.last_nmda.copy_(nmda_gate.detach())
            # Outer-product dendritic eligibility trace between post-target basal activity and dendritic voltage
            outer = torch.outer(b, self.voltage)
            self.elig.mul_(0.88).add_(outer, alpha=0.12)

        delta = self.readout(next_v)
        nmda_plateau = float((nmda_gate > 0.65).float().mean().item())
        return delta, {
            "enabled": True,
            "mean_tau": round(float(tau.mean().item()), 4),
            "min_tau": round(float(tau.min().item()), 4),
            "max_tau": round(float(tau.max().item()), 4),
            "nmda_plateau_fraction": round(nmda_plateau, 4),
            "voltage_norm": round(float(next_v.norm().item()), 4),
            "readout_norm": round(float(self.readout.weight.norm().item()), 5),
            "updates": int(self.updates),
        }

    def update_from_outcome(self, reward: float, prediction_error: float, lr: float = 2e-3) -> Dict[str, float]:
        """Reward-modulated dendritic plasticity update with weight decay stabilization."""
        modulator = float(np.clip(np.tanh(reward + 0.5 * prediction_error), -1.0, 1.0))
        with torch.no_grad():
            self.readout.weight.mul_(0.998).add_(self.elig, alpha=lr * modulator)
            self.readout.weight.clamp_(-0.5, 0.5)
        self.updates += 1
        self.last_loss = float(abs(prediction_error))
        return {
            "updates": int(self.updates),
            "modulator": round(modulator, 4),
            "readout_norm": round(float(self.readout.weight.norm().item()), 5),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "hidden_dim": self.hidden_dim,
            "mean_tau": round(float(self.last_tau.mean().item()), 4),
            "min_tau": round(float(self.last_tau.min().item()), 4),
            "max_tau": round(float(self.last_tau.max().item()), 4),
            "nmda_plateau_fraction": round(float((self.last_nmda > 0.65).float().mean().item()), 4),
            "voltage_norm": round(float(self.voltage.norm().item()), 4),
            "readout_norm": round(float(self.readout.weight.norm().item()), 5),
            "updates": int(self.updates),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "state_dict": deepcopy(self.state_dict()),
            "updates": int(self.updates),
            "last_loss": float(self.last_loss),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "state_dict" in state:
            self.load_state_dict(state["state_dict"])
        self.updates = int(state.get("updates", 0))
        self.last_loss = float(state.get("last_loss", 0.0))
