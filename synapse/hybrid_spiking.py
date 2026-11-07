"""Hybrid Continuous-Rate and Spiking Neural Interface.

Combines a continuous-rate recurrent core with event-driven Leaky Integrate-and-Fire (LIF)
spiking peripheral interfaces.

Scientific note: Spike trains and membrane potentials are discrete digital approximations
for computational efficiency and event-driven gating; not biophysical Hodgkin-Huxley simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSigmoidSurrogate(torch.autograd.Function):
    """Fast sigmoid surrogate gradient for backward pass through Heaviside step."""

    @staticmethod
    def forward(ctx, v_sub_vth: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
        ctx.save_for_backward(v_sub_vth)
        ctx.alpha = alpha
        return (v_sub_vth >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (v_sub_vth,) = ctx.saved_tensors
        alpha = ctx.alpha
        # Surrogate derivative: alpha / (2 * (1 + |alpha * x|)^2)
        grad_input = grad_output * (alpha / (2.0 * (1.0 + torch.abs(alpha * v_sub_vth)) ** 2))
        return grad_input, None


@dataclass
class HybridSpikingState:
    membrane_potentials: torch.Tensor
    refractory_counts: torch.Tensor
    spike_history: torch.Tensor


class HybridSpikingPeripheral(nn.Module):
    """Spiking peripheral layer that converts continuous inputs to spike events and vice versa."""

    def __init__(
        self,
        num_spiking_neurons: int = 128,
        v_threshold: float = 1.0,
        tau_m: float = 0.9,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.num_neurons = num_spiking_neurons
        self.v_threshold = v_threshold
        self.tau_m = tau_m
        self.v_reset = v_reset
        self.surrogate = FastSigmoidSurrogate.apply

        self.register_buffer("v_mem", torch.zeros(num_spiking_neurons))
        self.register_buffer("refractory", torch.zeros(num_spiking_neurons))

    def step(self, current_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one simulation micro-step of LIF dynamics.

        Args:
            current_input: [N] tensor of injected synaptic currents

        Returns:
            spikes: [N] binary float tensor in {0.0, 1.0}
            v_mem: [N] current membrane potential
        """
        # Decay and integrate
        v = self.tau_m * self.v_mem + current_input

        # Spike generation via surrogate
        spikes = self.surrogate(v - self.v_threshold, 4.0)

        # Soft reset: reset by subtraction
        self.v_mem = v - spikes * self.v_threshold

        return spikes, self.v_mem

    def reset_state(self) -> None:
        self.v_mem.zero_()
        self.refractory.zero_()


class HybridRateSpikingBridge(nn.Module):
    """Bi-directional interface connecting continuous rate core to spiking peripherals."""

    def __init__(self, rate_dim: int = 256, spiking_dim: int = 128):
        super().__init__()
        self.rate_dim = rate_dim
        self.spiking_dim = spiking_dim

        # Projections
        self.rate_to_spiking = nn.Linear(rate_dim, spiking_dim, bias=False)
        self.spiking_to_rate = nn.Linear(spiking_dim, rate_dim, bias=False)
        self.spiking_layer = HybridSpikingPeripheral(spiking_dim)

        # Initialize small
        nn.init.normal_(self.rate_to_spiking.weight, std=0.01)
        nn.init.normal_(self.spiking_to_rate.weight, std=0.01)

    def forward(self, rate_activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Project rate activations to spiking periphery, simulate step, and project back.

        Returns:
            feedback_rate: [rate_dim] continuous feedback current
            spikes: [spiking_dim] binary spikes
            telemetry: sparsity and active counts
        """
        injected = self.rate_to_spiking(rate_activations)
        spikes, v_mem = self.spiking_layer.step(injected)
        feedback_rate = self.spiking_to_rate(spikes)

        active = float(spikes.sum().item())
        sparsity = 1.0 - (active / max(self.spiking_dim, 1))

        telemetry = {
            "spiking_sparsity": round(sparsity, 4),
            "active_spikes": active,
            "mean_v_mem": round(float(v_mem.mean().item()), 4),
        }
        return feedback_rate, spikes, telemetry

    def reset_state(self) -> None:
        self.spiking_layer.reset_state()
