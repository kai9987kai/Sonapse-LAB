"""INTERMITTENT: Passive Energy Harvesting & NTAG Ephemeral Resuscitation.

Synthesizes intermittent computing, passive RF carrier harvesting, and NTAG 215
ephemeral frame checkpointing (inspired by Flipper-Zero-Periodic-NTAG-Emulator)
to safeguard agent state across energy brownouts and near-death metabolic crises.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Optional
import numpy as np


class NtagHarvestEngine:
    """Intermittent computing engine with passive ambient energy harvesting

    and compact binary NTAG ephemeral state checkpointing for crisis resuscitation.
    """

    def __init__(
        self,
        capacitor_max: float = 1.0,
        harvest_efficiency: float = 0.035,
        resuscitation_threshold: float = 0.08,
        resuscitation_boost: float = 0.18,
    ):
        self.capacitor_max = float(capacitor_max)
        self.harvest_efficiency = float(harvest_efficiency)
        self.resuscitation_threshold = float(resuscitation_threshold)
        self.resuscitation_boost = float(resuscitation_boost)

        self.virtual_capacitor = 0.15
        self.total_harvested_energy = 0.0
        self.total_checkpoints = 0
        self.total_resuscitations = 0
        self.last_checkpoint_frame: bytes = b""

    def reset(self) -> None:
        """Reset virtual capacitor and checkpoint records."""
        self.virtual_capacitor = 0.15  # Baseline quiescent charge
        self.total_harvested_energy = 0.0
        self.total_checkpoints = 0
        self.total_resuscitations = 0
        self.last_checkpoint_frame = b""

    def update_harvest(
        self,
        obs: List[float],
        action: int,
    ) -> float:
        """Harvest ambient potential from shelter or resource proximity and resting."""
        # Proximity to food (obs[0]), water (obs[1]), shelter (obs[14] if available)
        ambient_field = 0.0
        if len(obs) >= 2:
            ambient_field += (float(obs[0]) + float(obs[1])) * 0.5
        # Stationary resting or shelter seeking boosts inductive coupling
        if action == 3:  # rest
            ambient_field += 0.8
        elif action == 4:  # shelter
            ambient_field += 1.0

        harvested = self.harvest_efficiency * ambient_field
        self.virtual_capacitor = min(self.capacitor_max, self.virtual_capacitor + harvested)
        self.total_harvested_energy += harvested
        return harvested

    def pack_ntag_frame(
        self,
        step: int,
        phases: List[float],
        lattice_pos: List[float],
        health: float,
        stamina: float,
    ) -> bytes:
        """Pack an NFC Type-2 / NTAG 215 compatible 48-byte binary checkpoint frame."""
        # Header: 4 bytes ASCII 'NTAG'
        # Step: 4 bytes unsigned int
        # Health & stamina: 2 x uint16 (scaled 0-65535)
        # Lattice (x, y, z): 3 x int16 (scaled 100x)
        # Phases (6 actions): 6 x uint16 (scaled 0-65535 for 0-2pi)
        # Checksum: 8 bytes SHA-256 truncation
        header = b"NTAG"
        step_bytes = struct.pack(">I", int(step) & 0xFFFFFFFF)
        health_u16 = int(max(0.0, min(1.0, health)) * 65535)
        stamina_u16 = int(max(0.0, min(1.0, stamina)) * 65535)
        vitals_bytes = struct.pack(">HH", health_u16, stamina_u16)

        lx = int(np.clip(lattice_pos[0] if len(lattice_pos) > 0 else 0.0, -320.0, 320.0) * 100)
        ly = int(np.clip(lattice_pos[1] if len(lattice_pos) > 1 else 0.0, -320.0, 320.0) * 100)
        lz = int(np.clip(lattice_pos[2] if len(lattice_pos) > 2 else 0.0, -320.0, 320.0) * 100)
        lattice_bytes = struct.pack(">hhh", lx, ly, lz)

        phase_words = []
        two_pi = 2.0 * np.pi
        for i in range(6):
            phi = float(phases[i]) if i < len(phases) else 0.0
            norm_phi = (phi % two_pi) / two_pi
            phase_words.append(int(norm_phi * 65535))
        phases_bytes = struct.pack(">6H", *phase_words)

        payload = header + step_bytes + vitals_bytes + lattice_bytes + phases_bytes
        chk = hashlib.sha256(payload).digest()[:8]
        frame = payload + chk
        self.last_checkpoint_frame = frame
        self.total_checkpoints += 1
        return frame

    def evaluate_resuscitation(
        self,
        stamina: float,
        health: float,
    ) -> Tuple[bool, float]:
        """Check if metabolic brownout occurs and discharge virtual capacitor if needed."""
        # If stamina or health drops into critical brownout zone and capacitor has reserve
        if (stamina < self.resuscitation_threshold or health < self.resuscitation_threshold) and self.virtual_capacitor >= 0.20:
            discharge = min(self.virtual_capacitor, self.resuscitation_boost)
            self.virtual_capacitor -= discharge
            self.total_resuscitations += 1
            return True, discharge
        return False, 0.0

    def step(
        self,
        step: int,
        obs: List[float],
        action: int,
        phases: List[float],
        lattice_pos: List[float],
    ) -> Dict[str, Any]:
        """Execute one step of intermittent carrier harvest, checkpointing, and resuscitation."""
        health = float(obs[2]) if len(obs) > 2 else 1.0
        stamina = float(obs[3]) if len(obs) > 3 else 1.0

        # 1. Update passive RF/ambient harvesting
        harvested = self.update_harvest(obs, action)

        # 2. Checkpoint if entering risk or periodically
        checkpointed = False
        if stamina < 0.25 or health < 0.35 or (step % 16 == 0):
            self.pack_ntag_frame(step, phases, lattice_pos, health, stamina)
            checkpointed = True

        # 3. Check for resuscitation trigger
        resuscitated, energy_boost = self.evaluate_resuscitation(stamina, health)

        return {
            "virtual_capacitor": round(self.virtual_capacitor, 4),
            "harvested_step": round(harvested, 4),
            "total_harvested": round(self.total_harvested_energy, 4),
            "total_checkpoints": self.total_checkpoints,
            "total_resuscitations": self.total_resuscitations,
            "resuscitated": resuscitated,
            "energy_boost": round(energy_boost, 4),
            "checkpointed": checkpointed,
            "ntag_frame_hex": self.last_checkpoint_frame.hex() if self.last_checkpoint_frame else "",
        }
