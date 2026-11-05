"""Central Complex (CX) Ring Attractor Compass & Optomotor Stabilization.

Inspired by Drosophila neuropil models (FLY-DIAMOND-NEXUS, Central Complex E-PG ring),
this module maintains an online neural ring attractor representing agent heading theta in [0, 2pi).
It computes a Population Vector Average (PVA) and PVA coherence metric R in [0, 1].
When environmental vorticity or turbulence causes coherence collapse (R < R_crit),
an optomotor stabilization reflex triggers counter-torque actions to restore orientation.
"""
from __future__ import annotations

import math
from typing import Any, Tuple
import numpy as np


class CXRingCompass:
    """16-wedge recurrent neural ring attractor compass."""

    NUM_WEDGES = 16
    R_STABILIZATION_THRESHOLD = 0.35

    def __init__(self, num_wedges: int = 16, initial_heading: float = 0.0) -> None:
        self.num_wedges = num_wedges
        self.angles = np.linspace(0, 2 * math.pi, num_wedges, endpoint=False)
        # Recurrent connectivity: Mexican-hat / cosine profile (local excitation, global inhibition)
        diffs = self.angles[:, None] - self.angles[None, :]
        self.w_rec = -0.5 + 1.5 * np.cos(diffs)
        self.state = np.zeros(num_wedges, dtype=np.float64)
        self.reset(initial_heading)

    def reset(self, heading: float = 0.0) -> None:
        """Initialize the ring with an activity bump centered at heading."""
        wrapped = heading % (2 * math.pi)
        diffs = np.angle(np.exp(1j * (self.angles - wrapped)))
        # Gaussian bump profile around heading
        self.state = np.exp(-0.5 * (diffs / 0.5) ** 2)
        self.normalize()

    def normalize(self) -> None:
        """Enforce non-negative firing rates and normalized total activity."""
        self.state = np.clip(self.state, 0.0, None)
        total = np.sum(self.state)
        if total > 1e-8:
            self.state = self.state / total
        else:
            self.state = np.ones(self.num_wedges, dtype=np.float64) / self.num_wedges

    def step(self, angular_velocity: float, sensory_input: np.ndarray | None = None, dt: float = 0.1) -> Tuple[float, float]:
        """Update ring attractor with angular velocity integration and sensory drive.
        
        Returns:
            (pva_heading, pva_coherence):
                pva_heading in [0, 2pi)
                pva_coherence in [0, 1]
        """
        # 1. Recurrent relaxation
        recurrent_drive = np.dot(self.w_rec, self.state)
        
        # 2. Asymmetric velocity shift (P-EN style left/right lateral shift)
        # Positive velocity shifts bump counter-clockwise, negative shifts clockwise
        shift_amount = angular_velocity * dt
        shifted_diffs = np.angle(np.exp(1j * (self.angles[:, None] - self.angles[None, :] - shift_amount)))
        w_shift = np.exp(-0.5 * (shifted_diffs / 0.6) ** 2)
        shift_drive = np.dot(w_shift, self.state)
        
        # 3. Update state
        new_state = 0.4 * self.state + 0.3 * recurrent_drive + 0.3 * shift_drive
        if sensory_input is not None and len(sensory_input) == self.num_wedges:
            new_state += 0.2 * sensory_input
            
        self.state = np.clip(new_state, 0.0, None)
        self.normalize()
        
        return self.readout()

    def readout(self) -> Tuple[float, float]:
        """Compute Population Vector Average (PVA) and coherence metric R."""
        vx = float(np.sum(self.state * np.cos(self.angles)))
        vy = float(np.sum(self.state * np.sin(self.angles)))
        magnitude = math.sqrt(vx * vx + vy * vy)
        total_activity = float(np.sum(self.state))
        
        pva_heading = math.atan2(vy, vx) % (2 * math.pi)
        coherence = magnitude / (total_activity + 1e-8)
        coherence = min(1.0, max(0.0, coherence))
        
        return pva_heading, coherence

    def is_disoriented(self) -> bool:
        """True if PVA coherence drops below the stabilization threshold."""
        _, coherence = self.readout()
        return coherence < self.R_STABILIZATION_THRESHOLD

    def optomotor_reflex_action(self, current_action: int, orientation_error: float) -> int:
        """If disoriented, override action to stabilize heading towards target flow."""
        # Action space: 0=rest, 1=up, 2=down, 3=left, 4=right, 5=forage
        # In TidePool: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        if orientation_error > 0.3:
            return 4  # Turn/move right
        elif orientation_error < -0.3:
            return 3  # Turn/move left
        return current_action

    def export_state(self) -> dict[str, Any]:
        pva, coh = self.readout()
        return {
            "num_wedges": self.num_wedges,
            "wedge_activations": [round(float(x), 4) for x in self.state],
            "pva_heading": round(float(pva), 4),
            "pva_coherence": round(float(coh), 4),
            "disoriented": bool(self.is_disoriented()),
        }
