"""Multi-Compartmental Apical Amplification & Pyramidal Calcium Dynamics Engine.

Scientific Basis:
1. Two-Compartment Pyramidal Microcircuits (Larkum et al., Science 1999; 2007):
   Pyramidal neurons in cortical and high-order connectome neuropils feature segregated
   somatic and apical dendritic compartments:
   - Somatic compartment (Vs): Integrates bottom-up feedforward sensory drive.
   - Apical dendritic compartment (Va): Integrates top-down contextual, predictive,
     and neuromodulatory attention (ACh, DA).
2. Apical Amplification (Phillips et al., Brain & Cognition 2016; Aru et al., Trends Cogn Sci 2020):
   Apical input alone does not easily drive somatic action potentials, but when coincident
   with somatic depolarization, it triggers a regenerative dendritic Calcium spike
   (Ca2+ plateau potential with duration tau_ca ~ 20-30 ms), multiplying somatic firing
   frequency into high-frequency burst firing (coincidence gain multiplier Lambda in [1.0, 3.5]).
3. Neuromodulatory Gain Gating:
   - Acetylcholine (ACh): Sharpening of apical calcium sensitivity.
   - Dopamine (DA): Modulation of bursting threshold and plasticity eligibility.

Invariants:
- Default OFF unless explicitly enabled in configuration.
- Completely deterministic when seed is fixed.
- Pure PyTorch / NumPy implementation with zero external binary dependencies.
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
class ApicalTelemetry:
    """Live telemetry payload for Apical Amplification microcircuit."""
    mean_amplification_gain: float = 1.0
    calcium_spike_count: int = 0
    coincidence_burst_ratio: float = 0.0
    mean_somatic_voltage: float = 0.0
    mean_apical_voltage: float = 0.0
    active_plateaus: int = 0
    recent_somatic_trace: List[float] = field(default_factory=list)
    recent_apical_trace: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_amplification_gain': round(float(self.mean_amplification_gain), 4),
            'calcium_spike_count': int(self.calcium_spike_count),
            'coincidence_burst_ratio': round(float(self.coincidence_burst_ratio), 4),
            'mean_somatic_voltage': round(float(self.mean_somatic_voltage), 4),
            'mean_apical_voltage': round(float(self.mean_apical_voltage), 4),
            'active_plateaus': int(self.active_plateaus),
            'recent_somatic_trace': [round(float(v), 3) for v in self.recent_somatic_trace[-32:]],
            'recent_apical_trace': [round(float(v), 3) for v in self.recent_apical_trace[-32:]],
            'biological_basis': 'Larkum two-compartment pyramidal coincidence detection & Ca2+ dendritic plateaus (abstract digital model)',
        }


class ApicalAmplificationEngine(nn.Module):
    """Two-compartment somatic-apical pyramidal microcircuit.

    Args:
        num_units: Number of pyramidal units modeled (default 64 representative or full).
        tau_somatic: Time constant for somatic leak (ms, default 12.0).
        tau_apical: Time constant for apical dendritic compartment (ms, default 24.0).
        axial_conductance: Axial coupling conductance between compartments gc (default 0.35).
        ca_threshold: Voltage threshold to trigger regenerative Ca2+ plateau (default 0.65).
        ca_duration: Number of timesteps Ca2+ plateau remains active (default 4).
        max_gain: Maximum coincidence amplification factor Lambda (default 3.0).
    """

    def __init__(
        self,
        num_units: int = 64,
        tau_somatic: float = 12.0,
        tau_apical: float = 24.0,
        axial_conductance: float = 0.35,
        ca_threshold: float = 0.65,
        ca_duration: int = 4,
        max_gain: float = 3.0,
        seed: int = 42,
    ):
        super().__init__()
        self.num_units = max(8, int(num_units))
        self.tau_somatic = float(tau_somatic)
        self.tau_apical = float(tau_apical)
        self.axial_conductance = float(axial_conductance)
        self.ca_threshold = float(ca_threshold)
        self.ca_duration = max(1, int(ca_duration))
        self.max_gain = float(max_gain)
        self.seed = int(seed)

        # State tensors
        self.register_buffer('v_somatic', torch.zeros(self.num_units, dtype=torch.float32))
        self.register_buffer('v_apical', torch.zeros(self.num_units, dtype=torch.float32))
        self.register_buffer('ca_timer', torch.zeros(self.num_units, dtype=torch.int32))

        # Context projection: projects high-dimensional contextual/Tidal latents into apical drive
        self.context_proj = nn.Linear(32, self.num_units, bias=False)
        with torch.no_grad():
            torch.manual_seed(seed)
            nn.init.orthogonal_(self.context_proj.weight, gain=0.5)

        self.telemetry = ApicalTelemetry()
        self._soma_history: List[float] = []
        self._apical_history: List[float] = []

    def reset(self):
        """Reset membrane states and history buffers."""
        self.v_somatic.zero_()
        self.v_apical.zero_()
        self.ca_timer.zero_()
        self.telemetry = ApicalTelemetry()
        self._soma_history.clear()
        self._apical_history.clear()

    def forward(
        self,
        somatic_input: torch.Tensor,
        apical_context: Optional[torch.Tensor] = None,
        neuromod_ach: float = 1.0,
        neuromod_da: float = 1.0,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, ApicalTelemetry]:
        """Integrate two-compartment dynamics and apply coincidence amplification.

        Args:
            somatic_input: Tensor of shape (..., num_units) representing bottom-up basal input.
            apical_context: Tensor of shape (..., context_dim) or None for top-down contextual input.
            neuromod_ach: Acetylcholine level (sharpens apical calcium threshold).
            neuromod_da: Dopamine level (modulates burst amplification gain).
            dt: Integration time step (ms).

        Returns:
            amplified_output: Somatic output multiplied by apical coincidence gain Lambda.
            telemetry: ApicalTelemetry dataclass.
        """
        # Ensure correct shape
        if somatic_input.ndim == 1:
            inp = somatic_input[:self.num_units]
            if inp.shape[0] < self.num_units:
                inp = F.pad(inp, (0, self.num_units - inp.shape[0]))
        else:
            inp = somatic_input[..., :self.num_units]

        device = inp.device
        if self.v_somatic.device != device:
            self.to(device)

        # Compute apical top-down drive
        if apical_context is not None:
            ctx = apical_context.to(device=device, dtype=torch.float32)
            if ctx.ndim == 1:
                ctx = ctx.unsqueeze(0)
            if ctx.shape[-1] >= 32:
                ctx_in = ctx[..., :32]
            else:
                ctx_in = F.pad(ctx, (0, 32 - ctx.shape[-1]))
            apical_drive = torch.tanh(self.context_proj(ctx_in)).squeeze(0)
        else:
            apical_drive = torch.zeros(self.num_units, device=device)

        # Modulate Ca2+ threshold by Acetylcholine (higher ACh -> lower threshold -> higher attentional gain)
        ach_scale = max(0.2, min(2.0, float(neuromod_ach)))
        effective_ca_thresh = self.ca_threshold / ach_scale

        # Two-compartment differential equations
        gc = self.axial_conductance
        # Somatic leak + basal input + axial coupling from apical
        dv_s = (-self.v_somatic + inp + gc * (self.v_apical - self.v_somatic)) / self.tau_somatic
        # Apical leak + top-down context + axial coupling from soma
        dv_a = (-self.v_apical + apical_drive + gc * (self.v_somatic - self.v_apical)) / self.tau_apical

        self.v_somatic = torch.clamp(self.v_somatic + dv_s * dt, -1.5, 2.0)
        self.v_apical = torch.clamp(self.v_apical + dv_a * dt, -1.5, 2.0)

        # Check for regenerative Ca2+ plateau triggers
        new_spikes = (self.v_apical > effective_ca_thresh) & (self.ca_timer == 0)
        self.ca_timer = torch.where(new_spikes, torch.full_like(self.ca_timer, self.ca_duration), self.ca_timer)
        self.ca_timer = torch.clamp(self.ca_timer - 1, min=0)

        is_plateau = self.ca_timer > 0

        # Coincidence detection: Apical plateau AND somatic depolarization (Vs > 0.3)
        somatic_active = self.v_somatic > 0.3
        coincidence = is_plateau & somatic_active

        # Burst gain multiplier Lambda
        da_scale = max(0.5, min(2.5, float(neuromod_da)))
        effective_max_gain = 1.0 + (self.max_gain - 1.0) * (da_scale / 1.0)
        coincidence_gain = torch.where(
            coincidence,
            torch.full_like(self.v_somatic, effective_max_gain),
            torch.where(is_plateau, torch.full_like(self.v_somatic, 1.25), torch.ones_like(self.v_somatic))
        )

        # Amplified somatic output
        amplified = self.v_somatic * coincidence_gain

        # Record telemetry
        mean_gain = float(coincidence_gain.mean().item())
        calcium_count = int(new_spikes.sum().item())
        burst_ratio = float(coincidence.float().mean().item())
        mean_vs = float(self.v_somatic.mean().item())
        mean_va = float(self.v_apical.mean().item())
        active_plat = int(is_plateau.sum().item())

        self._soma_history.append(mean_vs)
        self._apical_history.append(mean_va)
        if len(self._soma_history) > 64:
            self._soma_history.pop(0)
            self._apical_history.pop(0)

        self.telemetry = ApicalTelemetry(
            mean_amplification_gain=mean_gain,
            calcium_spike_count=calcium_count,
            coincidence_burst_ratio=burst_ratio,
            mean_somatic_voltage=mean_vs,
            mean_apical_voltage=mean_va,
            active_plateaus=active_plat,
            recent_somatic_trace=list(self._soma_history),
            recent_apical_trace=list(self._apical_history),
        )

        return amplified, self.telemetry

    def export_state(self) -> Dict[str, Any]:
        """Export state dictionary for checkpointing."""
        return {
            'v_somatic': self.v_somatic.detach().cpu().numpy().tolist(),
            'v_apical': self.v_apical.detach().cpu().numpy().tolist(),
            'ca_timer': self.ca_timer.detach().cpu().numpy().tolist(),
            'soma_history': list(self._soma_history),
            'apical_history': list(self._apical_history),
            'telemetry': self.telemetry.to_dict(),
        }

    def import_state(self, state: Dict[str, Any]):
        """Import state dictionary from checkpoint."""
        if not state:
            return
        if 'v_somatic' in state:
            self.v_somatic.copy_(torch.tensor(state['v_somatic'], dtype=torch.float32))
        if 'v_apical' in state:
            self.v_apical.copy_(torch.tensor(state['v_apical'], dtype=torch.float32))
        if 'ca_timer' in state:
            self.ca_timer.copy_(torch.tensor(state['ca_timer'], dtype=torch.int32))
        if 'soma_history' in state:
            self._soma_history = [float(x) for x in state['soma_history']]
        if 'apical_history' in state:
            self._apical_history = [float(x) for x in state['apical_history']]
        if 'telemetry' in state and isinstance(state['telemetry'], dict):
            import dataclasses
            valid = {f.name for f in dataclasses.fields(ApicalTelemetry) if f.init}
            filtered = {k: v for k, v in state['telemetry'].items() if k in valid}
            self.telemetry = ApicalTelemetry(**filtered)
