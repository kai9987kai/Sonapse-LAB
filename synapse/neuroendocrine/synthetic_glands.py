"""Synthetic Gland Bank.

Implements the 12-channel synthetic neurochemical and endocrine gland bank:
1. dopamine
2. serotonin
3. octopamine
4. acetylcholine
5. adrenaline
6. noradrenaline
7. cortisol
8. oxytocin
9. endogenous_opioid
10. melatonin
11. orexin
12. adenosine

Differential clearance dynamics:
g_{k, t+dt} = exp(-dt / tau_k) * g_{k, t} + (1 - exp(-dt / tau_k)) * q_{k, t}
with finite metabolic reservoir synthesis, release gating, and hard clipping in [0, 1].
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)

GLAND_NAMES: List[str] = [
    "dopamine",
    "serotonin",
    "octopamine",
    "acetylcholine",
    "adrenaline",
    "noradrenaline",
    "cortisol",
    "oxytocin",
    "endogenous_opioid",
    "melatonin",
    "orexin",
    "adenosine",
]

# Baseline biological decay time constants (seconds) and baseline resting levels in [0, 1]
GLAND_SPECS: Dict[str, Tuple[float, float]] = {
    # (tau_seconds, baseline_level)
    "dopamine": (5.0, 0.20),
    "serotonin": (20.0, 0.50),
    "octopamine": (8.0, 0.10),
    "acetylcholine": (2.5, 0.30),
    "adrenaline": (4.0, 0.05),
    "noradrenaline": (6.0, 0.20),
    "cortisol": (60.0, 0.15),
    "oxytocin": (15.0, 0.30),
    "endogenous_opioid": (10.0, 0.10),
    "melatonin": (45.0, 0.05),
    "orexin": (12.0, 0.40),
    "adenosine": (80.0, 0.10),
}


class SyntheticGlandBank(nn.Module):
    """12-channel synthetic hormonal and neuromodulatory gland system."""

    def __init__(
        self,
        wpac_dim: int = 256,
        dt: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_glands = len(GLAND_NAMES)
        self.dt = dt

        # Projection from WPAC 256-d hidden state to 12 gland secretory drives
        self.proj_secretory = nn.Linear(wpac_dim, self.num_glands)
        nn.init.orthogonal_(self.proj_secretory.weight, gain=0.3)
        nn.init.zeros_(self.proj_secretory.bias)

        # Build baseline and tau vectors
        baselines = torch.tensor([GLAND_SPECS[name][1] for name in GLAND_NAMES], dtype=torch.float32)
        taus = torch.tensor([GLAND_SPECS[name][0] for name in GLAND_NAMES], dtype=torch.float32)
        self.register_buffer("baselines", baselines)
        self.register_buffer("taus", taus)

        # Dynamic state buffers
        self.register_buffer("levels", baselines.clone())
        self.register_buffer("reservoirs", torch.ones(self.num_glands, dtype=torch.float32))  # Full reservoirs
        self.register_buffer("synthesis_rate", torch.full((self.num_glands,), 0.05))

        # Direct external pulse input buffer
        self.register_buffer("external_pulse", torch.zeros(self.num_glands))

    def step(
        self,
        wpac_hidden: Optional[torch.Tensor] = None,
        external_pulse: Optional[torch.Tensor] = None,
        dt_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Advances gland dynamics by dt.

        Args:
            wpac_hidden: (256,) affective connectome hidden state.
            external_pulse: (12,) direct shock/pulse inputs.
            dt_override: optional custom dt.
        """
        dt = dt_override if dt_override is not None else self.dt
        device = self.levels.device

        # Compute raw secretory drive q_k in [0, 1]
        if wpac_hidden is not None:
            if isinstance(wpac_hidden, (tuple, list)):
                wpac_hidden = wpac_hidden[0]
            raw_drive = torch.sigmoid(self.proj_secretory(wpac_hidden.to(device).unsqueeze(0)).squeeze(0))
        else:
            raw_drive = self.baselines.clone()

        # Add external pulses if provided
        if external_pulse is not None:
            raw_drive = torch.clamp(raw_drive + external_pulse.to(device), 0.0, 1.0)

        # Reservoir gating: can only secrete what is in the reservoir
        effective_secretion = raw_drive * torch.clamp(self.reservoirs, 0.0, 1.0)

        # Replenish reservoir via synthesis and deplete by secretion
        synth = self.synthesis_rate * dt
        deplete = effective_secretion * dt * 0.5
        new_reservoirs = torch.clamp(self.reservoirs + synth - deplete, 0.0, 1.0)
        self.reservoirs.copy_(new_reservoirs)

        # Exponential decay towards target release
        # g_{t+dt} = alpha * g_t + (1 - alpha) * q_t
        alpha = torch.exp(-dt / self.taus)
        new_levels = alpha * self.levels + (1.0 - alpha) * effective_secretion
        new_levels = torch.clamp(new_levels, 0.0, 1.0)
        self.levels.copy_(new_levels)

        # Gland level deltas from resting baseline
        deltas = self.levels - self.baselines

        receipt = EvidenceReceipt(
            item_id="synthetic_gland_bank_step",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source="SyntheticGlandBank",
            sha256_digest=compute_tensor_digest(self.levels),
            metadata={
                "dt": dt,
                "dopamine": float(self.levels[0].item()),
                "serotonin": float(self.levels[1].item()),
                "adrenaline": float(self.levels[4].item()),
                "cortisol": float(self.levels[6].item()),
                "adenosine": float(self.levels[11].item()),
            },
        )

        return {
            "levels": self.levels.clone(),
            "deltas": deltas,
            "reservoirs": self.reservoirs.clone(),
            "secretory_drive": raw_drive,
            "gland_dict": {name: float(self.levels[i].item()) for i, name in enumerate(GLAND_NAMES)},
            "receipt": receipt.to_dict(),
        }

    def get_level(self, gland_name: str) -> float:
        """Returns the current level of a specific gland by name."""
        if gland_name not in GLAND_NAMES:
            raise KeyError(f"Unknown gland: {gland_name}. Valid glands: {GLAND_NAMES}")
        idx = GLAND_NAMES.index(gland_name)
        return float(self.levels[idx].item())

    def inject_pulse(self, gland_name: str, magnitude: float) -> None:
        """Directly injects a chemical pulse into a gland channel."""
        if gland_name not in GLAND_NAMES:
            raise KeyError(f"Unknown gland: {gland_name}")
        idx = GLAND_NAMES.index(gland_name)
        self.levels[idx] = torch.clamp(self.levels[idx] + magnitude, 0.0, 1.0)

    def reset(self) -> None:
        """Resets all gland levels to their biological resting baseline."""
        self.levels.copy_(self.baselines)
        self.reservoirs.fill_(1.0)
        self.external_pulse.zero_()

    reset_state = reset
