"""Configuration schema for Sonapse NeuroWave-X2 architecture.

Follows strict non-regression invariant:
``enabled=False`` and ``residual_gain=0.0`` by default, guaranteeing zero behavioural deviation
from baseline Sonapse connectome dynamics unless explicitly opted in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


VALID_PROTOCOLS: Set[str] = {
    "no_comm",
    "scalar",
    "vector",
    "quantized",
    "attention",
    "spectral",
    "phase",
    "phase_spectral",
    "phase_spectral_dendritic",
    "adaptive",
}

VALID_BASES: Set[str] = {
    "graph",
    "learned",
    "hybrid",
    "dct",
    "hadamard",
    "random_orthogonal",
    "identity",
}


@dataclass
class NeuroWaveX2Config:
    """Configuration container for the NeuroWave-X2 communication subsystem."""

    # Master switches - default OFF for non-regression
    enabled: bool = False
    residual_gain: float = 0.0  # gamma parameter; 0.0 guarantees exact baseline identity

    # Communication protocol & representation
    protocol: str = "adaptive"
    message_dimension: int = 16
    precision_bits: int = 32
    spectral_modes: int = 8
    spectral_basis: str = "hybrid"

    # Phase coherence channel
    phase_enabled: bool = True
    phase_bands: Tuple[str, ...] = ("theta", "beta", "gamma")

    # Cadence and resource constraints
    communication_every: int = 1
    max_messages: int = 8
    max_payload_bits: int = 2048
    max_latency: int = 0
    packet_loss: float = 0.0
    noise: float = 0.0

    # Topology and plasticity
    topology_strategy: str = "hybrid_causal"
    structural_plasticity: bool = False
    development_mode: bool = False
    development_level: int = 0
    protocol_selector: str = "learned"  # 'fixed', 'learned', 'gumbel', 'cost_aware'

    # Compute and bandwidth budgets
    compute_budget: float = 1e6
    bandwidth_budget: int = 4096
    causal_validation: bool = True
    read_only: bool = False
    trace_packets: bool = False

    # Dendritic integration compartment
    dendritic_target: str = "apical"  # 'apical', 'basal', 'both'

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if not (-1e-5 <= self.residual_gain <= 5.0):
            raise ValueError(f"residual_gain must be in [0.0, 5.0], got {self.residual_gain}")
        if self.protocol not in VALID_PROTOCOLS:
            raise ValueError(f"protocol '{self.protocol}' not in {sorted(VALID_PROTOCOLS)}")
        if self.spectral_basis not in VALID_BASES:
            raise ValueError(f"spectral_basis '{self.spectral_basis}' not in {sorted(VALID_BASES)}")
        if self.message_dimension < 1 or self.message_dimension > 512:
            raise ValueError(f"message_dimension must be in [1, 512], got {self.message_dimension}")
        if self.precision_bits not in (2, 4, 8, 16, 32):
            raise ValueError(f"precision_bits must be in (2, 4, 8, 16, 32), got {self.precision_bits}")
        if self.spectral_modes < 1 or self.spectral_modes > 128:
            raise ValueError(f"spectral_modes must be in [1, 128], got {self.spectral_modes}")
        if not (0.0 <= self.packet_loss <= 1.0):
            raise ValueError(f"packet_loss must be in [0.0, 1.0], got {self.packet_loss}")
        if self.max_latency < 0 or self.max_latency > 32:
            raise ValueError(f"max_latency must be in [0, 32], got {self.max_latency}")
        if self.development_level < 0 or self.development_level > 6:
            raise ValueError(f"development_level must be in [0, 6], got {self.development_level}")
        if self.dendritic_target not in ("apical", "basal", "both"):
            raise ValueError(f"dendritic_target must be in ('apical', 'basal', 'both'), got {self.dendritic_target}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "residual_gain": float(self.residual_gain),
            "protocol": self.protocol,
            "message_dimension": int(self.message_dimension),
            "precision_bits": int(self.precision_bits),
            "spectral_modes": int(self.spectral_modes),
            "spectral_basis": self.spectral_basis,
            "phase_enabled": self.phase_enabled,
            "phase_bands": list(self.phase_bands),
            "communication_every": int(self.communication_every),
            "max_messages": int(self.max_messages),
            "max_payload_bits": int(self.max_payload_bits),
            "max_latency": int(self.max_latency),
            "packet_loss": float(self.packet_loss),
            "noise": float(self.noise),
            "topology_strategy": self.topology_strategy,
            "structural_plasticity": self.structural_plasticity,
            "development_mode": self.development_mode,
            "development_level": int(self.development_level),
            "protocol_selector": self.protocol_selector,
            "compute_budget": float(self.compute_budget),
            "bandwidth_budget": int(self.bandwidth_budget),
            "causal_validation": self.causal_validation,
            "read_only": self.read_only,
            "trace_packets": self.trace_packets,
            "dendritic_target": self.dendritic_target,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroWaveX2Config":
        d = dict(data)
        if "phase_bands" in d and isinstance(d["phase_bands"], list):
            d["phase_bands"] = tuple(d["phase_bands"])
        return cls(**d)
