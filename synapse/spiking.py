"""Spiking-Neuron Experimental Branch: LIF & AdEx Dynamics with Surrogate Gradients.

Implements biophysical Leaky Integrate-and-Fire (LIF) and Adaptive Exponential
Integrate-and-Fire (AdEx) neurons with surrogate gradient e-prop learning compatibility.
Tracks real Synaptic Operations (SynOps) to benchmark neuromorphic hardware efficiency
against continuous rate models.
"""
from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class FastSigmoidSurrogate(torch.autograd.Function):
    """Surrogate gradient for non-differentiable heaviside spike function:

    Forward: s = 1 if v >= v_th else 0
    Backward: ds/dv = 1 / (1 + beta * |v - v_th|)^2
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, v_th: float, beta: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.beta = beta
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (v,) = ctx.saved_tensors
        v_th = ctx.v_th
        beta = ctx.beta
        # Fast sigmoid derivative
        grad_v = grad_output / (1.0 + beta * torch.abs(v - v_th)) ** 2
        return grad_v, None, None


def surrogate_spike(v: torch.Tensor, v_th: float = 1.0, beta: float = 10.0) -> torch.Tensor:
    return FastSigmoidSurrogate.apply(v, v_th, beta)


class LIFSpikingLayer(nn.Module):
    """Leaky Integrate-and-Fire (LIF) recurrent neural layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_m: float = 20.0,  # ms membrane time constant
        dt: float = 1.0,      # ms simulation step
        v_th: float = 1.0,
        v_reset: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = float(math.exp(-dt / tau_m))
        self.v_th = v_th
        self.v_reset = v_reset

        self.w_in = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.w_rec = nn.Parameter(torch.randn(out_features, out_features) * (0.5 / math.sqrt(out_features)))

        self.register_buffer('v', torch.zeros(out_features))
        self.register_buffer('spikes', torch.zeros(out_features))
        self.total_synops = 0

    def reset_state(self):
        self.v.zero_()
        self.spikes.zero_()
        self.total_synops = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step LIF integration:

        I(t) = W_in x(t) + W_rec s(t-1)
        v(t) = alpha * v(t-1) * (1 - s(t-1)) + I(t)
        s(t) = Heaviside(v(t) - v_th)
        """
        # Effective current:
        syn_in = torch.matmul(self.w_in, x)
        syn_rec = torch.matmul(self.w_rec, self.spikes)
        i_t = syn_in + syn_rec

        # Reset membrane potential if spiked in previous step
        v_prev = self.v * (1.0 - self.spikes)
        v_next = self.alpha * v_prev + i_t
        s_next = surrogate_spike(v_next, self.v_th)

        # Count SynOps: non-zero spikes * fan-out
        active_spikes = int((s_next > 0.0).sum().item())
        self.total_synops += active_spikes * self.out_features

        self.v.copy_(v_next.detach())
        self.spikes.copy_(s_next.detach())
        return s_next, v_next


class AdExSpikingLayer(nn.Module):
    """Adaptive Exponential Integrate-and-Fire (AdEx) recurrent layer.

    Models biological subthreshold adaptation and spike-frequency adaptation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        c_m: float = 200.0,   # pF
        g_l: float = 10.0,    # nS
        e_l: float = -70.0,   # mV
        v_t: float = -50.0,   # mV
        delta_t: float = 2.0, # mV
        a: float = 2.0,       # nS subthreshold adaptation
        tau_w: float = 30.0,  # ms adaptation time constant
        b: float = 60.0,      # pA spike-triggered adaptation
        v_peak: float = 20.0, # mV
        v_reset: float = -60.0,
        dt: float = 0.5,      # ms
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c_m = c_m
        self.g_l = g_l
        self.e_l = e_l
        self.v_t = v_t
        self.delta_t = delta_t
        self.a = a
        self.tau_w = tau_w
        self.b = b
        self.v_peak = v_peak
        self.v_reset = v_reset
        self.dt = dt

        self.w_in = nn.Parameter(torch.randn(out_features, in_features) * 5.0)
        self.w_rec = nn.Parameter(torch.randn(out_features, out_features) * 2.0)

        self.register_buffer('v', torch.full((out_features,), e_l))
        self.register_buffer('w', torch.zeros(out_features))
        self.register_buffer('spikes', torch.zeros(out_features))
        self.total_synops = 0

    def reset_state(self):
        self.v.fill_(self.e_l)
        self.w.zero_()
        self.spikes.zero_()
        self.total_synops = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step AdEx integration:

        dv/dt = ( -g_L(v - E_L) + g_L * delta_T * exp((v - V_T)/delta_T) - w + I ) / C_m
        dw/dt = ( a(v - E_L) - w ) / tau_w
        """
        syn_in = torch.matmul(self.w_in, x)
        syn_rec = torch.matmul(self.w_rec, self.spikes)
        i_ext = syn_in + syn_rec

        # Reset states from previous spikes
        v_curr = torch.where(self.spikes > 0.5, torch.full_like(self.v, self.v_reset), self.v)
        w_curr = self.w + self.spikes * self.b

        # Exp term with safety clamping
        v_clamped = torch.clamp(v_curr, -100.0, self.v_peak)
        exp_term = self.g_l * self.delta_t * torch.exp(torch.clamp((v_clamped - self.v_t) / self.delta_t, -10.0, 5.0))

        dv = (-self.g_l * (v_clamped - self.e_l) + exp_term - w_curr + i_ext) / self.c_m
        dw = (self.a * (v_clamped - self.e_l) - w_curr) / self.tau_w

        v_next = v_curr + self.dt * dv
        w_next = w_curr + self.dt * dw

        s_next = surrogate_spike(v_next, self.v_peak)

        active_spikes = int((s_next > 0.0).sum().item())
        self.total_synops += active_spikes * self.out_features

        self.v.copy_(v_next.detach())
        self.w.copy_(w_next.detach())
        self.spikes.copy_(s_next.detach())
        return s_next, v_next


class ContinuousRateLayer(nn.Module):
    """Standard continuous rate tanh recurrent layer for baseline comparison."""

    def __init__(self, in_features: int, out_features: int, tau: float = 0.8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.w_in = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.w_rec = nn.Parameter(torch.randn(out_features, out_features) * (0.5 / math.sqrt(out_features)))
        self.register_buffer('r', torch.zeros(out_features))
        self.total_macs = 0

    def reset_state(self):
        self.r.zero_()
        self.total_macs = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i_t = torch.matmul(self.w_in, x) + torch.matmul(self.w_rec, self.r)
        r_next = self.tau * self.r + (1.0 - self.tau) * torch.tanh(i_t)
        # In dense continuous rate, all weights undergo multiply-accumulate every step
        self.total_macs += self.in_features * self.out_features + self.out_features * self.out_features
        self.r.copy_(r_next.detach())
        return r_next


def benchmark_spiking_vs_rate(
    num_neurons: int = 64,
    in_features: int = 16,
    steps: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Side-by-side empirical benchmark of Continuous Rate vs LIF Spiking vs AdEx Spiking.

    Measures CPU latency, activity sparsity, and Synaptic Operations (SynOps).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    rate_layer = ContinuousRateLayer(in_features, num_neurons)
    lif_layer = LIFSpikingLayer(in_features, num_neurons)
    adex_layer = AdExSpikingLayer(in_features, num_neurons)

    inputs = [torch.tensor(rng.normal(0.0, 1.0, size=(in_features,)), dtype=torch.float32) for _ in range(steps)]

    # 1. Rate Layer
    rate_layer.reset_state()
    t0 = time.perf_counter()
    for x in inputs:
        rate_layer(x)
    t_rate = (time.perf_counter() - t0) * 1000.0  # ms

    # 2. LIF Spiking
    lif_layer.reset_state()
    t0 = time.perf_counter()
    lif_spikes_list = []
    for x in inputs:
        s, _ = lif_layer(x)
        lif_spikes_list.append(s.detach().cpu().numpy())
    t_lif = (time.perf_counter() - t0) * 1000.0  # ms

    # 3. AdEx Spiking
    adex_layer.reset_state()
    t0 = time.perf_counter()
    adex_spikes_list = []
    for x in inputs:
        s, _ = adex_layer(x)
        adex_spikes_list.append(s.detach().cpu().numpy())
    t_adex = (time.perf_counter() - t0) * 1000.0  # ms

    lif_all_spikes = np.array(lif_spikes_list)
    adex_all_spikes = np.array(adex_spikes_list)

    lif_sparsity = float(1.0 - (np.count_nonzero(lif_all_spikes) / lif_all_spikes.size))
    adex_sparsity = float(1.0 - (np.count_nonzero(adex_all_spikes) / adex_all_spikes.size))

    # Standard neuromorphic energy estimation:
    # 32-bit Float MAC ~ 4.6 pJ; Neuromorphic SynOp ~ 0.9 pJ
    joules_per_mac = 4.6e-12
    joules_per_synop = 0.9e-12

    energy_rate_nj = (rate_layer.total_macs * joules_per_mac) * 1e9
    energy_lif_nj = (lif_layer.total_synops * joules_per_synop) * 1e9
    energy_adex_nj = (adex_layer.total_synops * joules_per_synop) * 1e9

    return {
        'steps': steps,
        'neurons': num_neurons,
        'rate': {
            'total_macs': rate_layer.total_macs,
            'latency_ms': round(t_rate, 3),
            'estimated_energy_nJ': round(energy_rate_nj, 4),
            'sparsity': 0.0,
        },
        'lif_spiking': {
            'total_synops': lif_layer.total_synops,
            'latency_ms': round(t_lif, 3),
            'estimated_energy_nJ': round(energy_lif_nj, 4),
            'sparsity': round(lif_sparsity, 4),
            'synops_reduction_ratio': round(rate_layer.total_macs / max(1, lif_layer.total_synops), 2),
        },
        'adex_spiking': {
            'total_synops': adex_layer.total_synops,
            'latency_ms': round(t_adex, 3),
            'estimated_energy_nJ': round(energy_adex_nj, 4),
            'sparsity': round(adex_sparsity, 4),
            'synops_reduction_ratio': round(rate_layer.total_macs / max(1, adex_layer.total_synops), 2),
        },
    }
