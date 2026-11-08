"""Phase-Coherence Communication Adapter & Wave Channel for Sonapse NeuroWave-X2.

Integrates with Sonapse's existing ``OscillatoryCoherenceEngine`` (synapse/oscillatory_coherence.py),
providing non-negative Communication-Through-Coherence (CTC) gating across
theta (4-8 Hz), beta (15-30 Hz), and gamma (30-80 Hz) bands while strictly preserving
Dale's law and synthetic/biological E/I synaptic identities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BAND_FREQUENCIES: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "beta": (15.0, 30.0),
    "gamma": (30.0, 80.0),
}


class PhaseCoherenceAdapter(nn.Module):
    """Adapter bridging Sonapse's OscillatoryCoherenceEngine with NeuroWave-X2.
    
    Maintains regional phase variables theta_r(t) across multiple frequency bands
    and computes bounded CTC transmission gains:
        g_ij = (1 + cos(theta_i - theta_j)) / 2
    """

    def __init__(
        self,
        num_regions: int = 8,
        bands: Tuple[str, ...] = ("theta", "beta", "gamma"),
        dt: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_regions = max(1, int(num_regions))
        self.bands = tuple(bands)
        self.dt = float(dt)

        # Base angular frequencies omega = 2 * pi * f for each band and region
        # Initialized near center frequencies
        omegas: Dict[str, torch.Tensor] = {}
        for b in self.bands:
            f_low, f_high = BAND_FREQUENCIES.get(b, (8.0, 16.0))
            f_mid = 0.5 * (f_low + f_high)
            # Slight regional variation in natural frequencies
            var = 1.0 + 0.1 * (torch.arange(self.num_regions, dtype=torch.float32) / float(self.num_regions) - 0.5)
            omegas[b] = 2.0 * math.pi * f_mid * var

        for b, om in omegas.items():
            self.register_buffer(f"omega_{b}", om)
            self.register_buffer(f"phase_{b}", torch.zeros(self.num_regions, dtype=torch.float32))

    def reset(self, initial_phases: Optional[Dict[str, Sequence[float]]] = None) -> None:
        """Resets regional phase states to zero or user-provided angles."""
        for b in self.bands:
            buf = getattr(self, f"phase_{b}")
            if initial_phases is not None and b in initial_phases:
                init_t = torch.tensor(initial_phases[b], dtype=torch.float32, device=buf.device)
                buf.copy_(init_t[:self.num_regions])
            else:
                buf.zero_()

    def step(self, external_coherence_phases: Optional[torch.Tensor] = None) -> None:
        """Advances regional phase oscillators.
        If external_coherence_phases is supplied from Sonapse's OscillatoryCoherenceEngine,
        aligns theta band directly with it.
        """
        for b in self.bands:
            buf = getattr(self, f"phase_{b}")
            omega = getattr(self, f"omega_{b}")
            if b == "theta" and external_coherence_phases is not None:
                # Synchronize theta band directly from OscillatoryCoherenceEngine
                n = min(self.num_regions, external_coherence_phases.numel())
                buf[:n].copy_(external_coherence_phases[:n].to(buf.device))
            else:
                buf.add_(omega * self.dt)
                buf.remainder_(2.0 * math.pi)

    def get_phase(self, region_id: int, band: str = "theta") -> float:
        """Returns the current phase angle in radians for a given region."""
        r = min(max(0, region_id), self.num_regions - 1)
        buf = getattr(self, f"phase_{band}", getattr(self, "phase_theta", None))
        if buf is None:
            return 0.0
        return float(buf[r].item())

    def compute_coherence_gain(
        self,
        src_region: int,
        dst_region: int,
        band: str = "theta",
    ) -> float:
        """Computes non-negative CTC alignment gain g_ij in [0.0, 1.0]."""
        theta_src = self.get_phase(src_region, band=band)
        theta_dst = self.get_phase(dst_region, band=band)
        d_theta = theta_src - theta_dst
        return float(0.5 * (1.0 + math.cos(d_theta)))

    def compute_multi_band_gains(self, src_region: int, dst_region: int) -> Dict[str, float]:
        """Returns alignment gains across all configured frequency bands."""
        return {b: self.compute_coherence_gain(src_region, dst_region, band=b) for b in self.bands}

    def export_state(self) -> Dict[str, Any]:
        """Serializes regional phase buffer states."""
        return {b: getattr(self, f"phase_{b}").detach().cpu().numpy().tolist() for b in self.bands}

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restores regional phase buffer states."""
        if not state:
            return
        for b, vals in state.items():
            if hasattr(self, f"phase_{b}"):
                buf = getattr(self, f"phase_{b}")
                t = torch.tensor(vals, dtype=torch.float32, device=buf.device)
                n = min(buf.numel(), t.numel())
                buf[:n].copy_(t[:n])


class WaveChannel(nn.Module):
    """NeuroWave-X2 Combined Phase + Spectral Multiplexing Wave Channel.
    
    Transforms regional activation m_i:
        m_i -> spectral mode decomposition -> phase-gated carrier transport -> receiver reconstruction.
    """

    def __init__(
        self,
        message_dim: int = 16,
        num_modes: int = 8,
        num_regions: int = 8,
    ) -> None:
        super().__init__()
        self.message_dim = int(message_dim)
        self.num_modes = int(num_modes)
        self.num_regions = int(num_regions)
        self.phase_adapter = PhaseCoherenceAdapter(num_regions=num_regions)
        # Mode selector projection: maps message to spectral mode coefficients
        self.mode_encoder = nn.Linear(self.message_dim, self.num_modes)
        nn.init.orthogonal_(self.mode_encoder.weight, gain=0.8)
        nn.init.zeros_(self.mode_encoder.bias)

    def encode_wave_packet_payload(
        self,
        message: torch.Tensor,
        src_region: int,
        band: str = "theta",
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Projects message onto spectral modes and attaches source phase metadata.
        
        Returns:
            (mode_coefficients_tuple, phase_metadata_tuple)
        """
        coeffs = torch.tanh(self.mode_encoder(message))
        src_phase = self.phase_adapter.get_phase(src_region, band=band)
        c_tuple = tuple(float(c.item()) for c in coeffs)
        p_tuple = (float(src_phase),)
        return c_tuple, p_tuple

    def export_state(self) -> Dict[str, Any]:
        return {
            "phase_adapter": self.phase_adapter.export_state(),
            "mode_encoder": {k: v.cpu() for k, v in self.mode_encoder.state_dict().items()},
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if "phase_adapter" in state:
            self.phase_adapter.import_state(state["phase_adapter"])
        if "mode_encoder" in state:
            self.mode_encoder.load_state_dict(state["mode_encoder"])
