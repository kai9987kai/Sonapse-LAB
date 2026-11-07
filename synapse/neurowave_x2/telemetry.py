"""Live Telemetry Aggregator for Sonapse NeuroWave-X2 Observatory.

Collects real-time communication edge activity, protocol usage distributions,
phase alignment, spectral mode energies, wire costs, and dropped packet metrics
for the web workbench observatory and API endpoints.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class NeuroWaveTelemetryAggregator:
    """Aggregates step-by-step metrics into a structured telemetry payload."""

    def __init__(self, history_len: int = 50) -> None:
        self.history_len = int(history_len)
        self.step_counter = 0

        # Ring buffers for time-series charts
        self.wire_bits_history: List[int] = []
        self.flops_history: List[float] = []
        self.protocol_history: List[str] = []
        self.residual_magnitude_history: List[float] = []

        # Latest snapshot cache
        self.latest_active_routes: List[Dict[str, Any]] = []
        self.latest_spectral_energies: List[float] = []
        self.latest_phase_alignments: Dict[str, float] = {}
        self.latest_protocol_probs: Dict[str, float] = {}
        self.dropped_packets_total = 0
        self.transmitted_packets_total = 0
        self.topology_version = 1
        self.basis_version = 1

    def record_step(
        self,
        active_routes: List[Tuple[int, int]],
        protocol: str,
        wire_bits: int,
        flops: float,
        residual_norm: float,
        spectral_energies: Optional[List[float]] = None,
        phase_alignments: Optional[Dict[str, float]] = None,
        protocol_probs: Optional[Dict[str, float]] = None,
        dropped: int = 0,
        transmitted: int = 1,
        topology_version: int = 1,
        basis_version: int = 1,
    ) -> None:
        """Records a single communication step."""
        self.step_counter += 1
        self.wire_bits_history.append(int(wire_bits))
        self.flops_history.append(float(flops))
        self.protocol_history.append(str(protocol))
        self.residual_magnitude_history.append(float(residual_norm))

        if len(self.wire_bits_history) > self.history_len:
            self.wire_bits_history.pop(0)
            self.flops_history.pop(0)
            self.protocol_history.pop(0)
            self.residual_magnitude_history.pop(0)

        self.latest_active_routes = [{"src": s, "dst": d} for s, d in active_routes]
        if spectral_energies is not None:
            self.latest_spectral_energies = [round(float(e), 4) for e in spectral_energies]
        if phase_alignments is not None:
            self.latest_phase_alignments = {k: round(float(v), 4) for k, v in phase_alignments.items()}
        if protocol_probs is not None:
            self.latest_protocol_probs = {k: round(float(v), 4) for k, v in protocol_probs.items()}

        self.dropped_packets_total += int(dropped)
        self.transmitted_packets_total += int(transmitted)
        self.topology_version = int(topology_version)
        self.basis_version = int(basis_version)

    def export_observatory_payload(self) -> Dict[str, Any]:
        """Formats real-time observatory dictionary for JSON serialization in /api/state."""
        mean_bits = float(np.mean(self.wire_bits_history)) if self.wire_bits_history else 0.0
        mean_flops = float(np.mean(self.flops_history)) if self.flops_history else 0.0
        drop_rate = float(self.dropped_packets_total / max(1, self.transmitted_packets_total))

        return {
            "step": self.step_counter,
            "active_routes": self.latest_active_routes,
            "num_active_routes": len(self.latest_active_routes),
            "current_protocol": self.protocol_history[-1] if self.protocol_history else "vector",
            "protocol_distribution": self.latest_protocol_probs,
            "spectral_mode_energies": self.latest_spectral_energies,
            "phase_alignments": self.latest_phase_alignments,
            "wire_bits_recent_avg": round(mean_bits, 1),
            "flops_recent_avg": round(mean_flops, 1),
            "drop_rate": round(drop_rate, 4),
            "topology_version": self.topology_version,
            "basis_version": self.basis_version,
            "history_wire_bits": self.wire_bits_history[-20:],
            "history_residual_norm": [round(x, 4) for x in self.residual_magnitude_history[-20:]],
        }
