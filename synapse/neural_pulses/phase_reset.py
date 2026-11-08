"""Neural Phase Reset Protocol.

Implements soft phase entrainment and reset driven by traveling wave pulses:
theta_{j, t+1} = theta_{j, t} + omega_j dt + epsilon_j * sin(phi_pulse - theta_{j, t})

Computes the Kuramoto order parameter R to evaluate global and regional coherence:
R * exp(i * Psi) = (1 / N) * sum_j exp(i * theta_j)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class NeuralPhaseResetProtocol(nn.Module):
    """Tracks regional oscillator phases and applies wave-induced phase entrainment."""

    def __init__(
        self,
        num_regions: int = 8,
        base_freq_hz: float = 10.0,
        reset_sensitivity: float = 0.40,
        dt: float = 0.05,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_regions = num_regions
        self.base_freq_hz = base_freq_hz
        self.reset_sensitivity = reset_sensitivity
        self.dt = dt

        # Natural angular frequencies omega_j = 2 * pi * f_j
        # Small regional frequency diversity (+- 10%)
        freqs = base_freq_hz * (1.0 + 0.1 * (torch.rand(num_regions) - 0.5))
        omegas = 2.0 * math.pi * freqs
        self.register_buffer("omegas", omegas)

        # Region-specific phase reset coupling epsilon_j
        epsilons = torch.full((num_regions,), reset_sensitivity)
        self.register_buffer("epsilons", epsilons)

        # Current phase state theta_j in [-pi, pi]
        initial_phases = (torch.rand(num_regions) * 2.0 - 1.0) * math.pi
        self.register_buffer("phases", initial_phases)

        # Telemetry
        self.register_buffer("coherence_history", torch.zeros(100))

    def step(
        self,
        wave_displacement: Optional[torch.Tensor] = None,
        wave_velocity: Optional[torch.Tensor] = None,
        dt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advances oscillator phases and entrains them to local traveling wave dynamics."""
        dt = dt_override if dt_override is not None else self.dt
        device = self.phases.device

        # Autonomous phase progression
        d_theta = self.omegas * dt

        # If wave signals are present, wave phase phi = atan2(v, x)
        if wave_displacement is not None and wave_velocity is not None:
            x = wave_displacement.to(device).flatten()
            v = wave_velocity.to(device).flatten()
            wave_phi = torch.atan2(v, x + 1e-6)

            # Wave amplitude gating: entrainment strength scales with wave amplitude
            wave_amp = torch.sqrt(x ** 2 + v ** 2)
            gate = torch.tanh(wave_amp)

            # Phase reset adjustment: eps * gate * sin(wave_phi - theta)
            entrainment = self.epsilons * gate * torch.sin(wave_phi - self.phases)
            new_phases = self.phases + d_theta + entrainment
        else:
            new_phases = self.phases + d_theta

        # Wrap phases into [-pi, pi]
        new_phases = torch.remainder(new_phases + math.pi, 2.0 * math.pi) - math.pi
        self.phases.copy_(new_phases)

        # Compute Kuramoto order parameter R and mean phase Psi
        cos_sum = torch.sum(torch.cos(self.phases))
        sin_sum = torch.sum(torch.sin(self.phases))
        n = float(self.num_regions)
        r_coherence = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
        psi_mean = torch.atan2(sin_sum, cos_sum)

        receipt = EvidenceReceipt(
            item_id="phase_reset_step",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="NeuralPhaseResetProtocol",
            sha256_digest=compute_tensor_digest(self.phases),
            metrics={
                "kuramoto_coherence": float(r_coherence.item()),
                "mean_phase_rad": float(psi_mean.item()),
            },
        )

        return {
            "phases": self.phases.clone(),
            "coherence": r_coherence,
            "mean_phase": psi_mean,
            "receipt": receipt.to_dict(),
        }

    def reset(self) -> None:
        """Randomizes or zeroes phases."""
        self.phases.zero_()
