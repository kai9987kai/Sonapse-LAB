"""Protocol Registry for Sonapse NeuroWave-X2.

Defines candidate neural communication protocols, wire costs, and dispatch specifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ProtocolSpec:
    """Specification of a NeuroWave-X2 communication protocol."""

    protocol_id: str
    dimension: int
    precision_bits: int
    uses_spectral: bool
    uses_phase: bool
    uses_dendritic: bool
    complexity_level: int  # 0 to 6 in developmental hierarchy
    base_wire_cost_bits: int
    description: str


PROTOCOLS: Dict[str, ProtocolSpec] = {
    "no_comm": ProtocolSpec(
        protocol_id="no_comm",
        dimension=0,
        precision_bits=0,
        uses_spectral=False,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=0,
        base_wire_cost_bits=0,
        description="Zero communication (dormant or suppressed channel)",
    ),
    "scalar": ProtocolSpec(
        protocol_id="scalar",
        dimension=1,
        precision_bits=32,
        uses_spectral=False,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=1,
        base_wire_cost_bits=32,
        description="Single-scalar bottleneck transmission",
    ),
    "vector": ProtocolSpec(
        protocol_id="vector",
        dimension=16,
        precision_bits=32,
        uses_spectral=False,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=2,
        base_wire_cost_bits=512,
        description="Dense real-valued vector message",
    ),
    "quantized": ProtocolSpec(
        protocol_id="quantized",
        dimension=16,
        precision_bits=8,
        uses_spectral=False,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=3,
        base_wire_cost_bits=128,
        description="Low-precision quantized vector (8-bit or 4-bit)",
    ),
    "attention": ProtocolSpec(
        protocol_id="attention",
        dimension=16,
        precision_bits=32,
        uses_spectral=False,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=3,
        base_wire_cost_bits=512,
        description="Attentive multi-query routed vector",
    ),
    "spectral": ProtocolSpec(
        protocol_id="spectral",
        dimension=8,
        precision_bits=32,
        uses_spectral=True,
        uses_phase=False,
        uses_dendritic=False,
        complexity_level=4,
        base_wire_cost_bits=256,
        description="Graph Laplacian eigenmode decomposition",
    ),
    "phase": ProtocolSpec(
        protocol_id="phase",
        dimension=16,
        precision_bits=32,
        uses_spectral=False,
        uses_phase=True,
        uses_dendritic=False,
        complexity_level=4,
        base_wire_cost_bits=544,
        description="Communication-Through-Coherence (CTC) phase-aligned channel",
    ),
    "phase_spectral": ProtocolSpec(
        protocol_id="phase_spectral",
        dimension=8,
        precision_bits=32,
        uses_spectral=True,
        uses_phase=True,
        uses_dendritic=False,
        complexity_level=5,
        base_wire_cost_bits=288,
        description="Combined wave multiplexing (Laplacian eigenmodes + phase alignment)",
    ),
    "phase_spectral_dendritic": ProtocolSpec(
        protocol_id="phase_spectral_dendritic",
        dimension=8,
        precision_bits=32,
        uses_spectral=True,
        uses_phase=True,
        uses_dendritic=True,
        complexity_level=6,
        base_wire_cost_bits=288,
        description="Full NeuroWave-X2: Wave multiplexing targeting apical dendritic compartments",
    ),
}


def get_protocol_spec(proto_id: str) -> ProtocolSpec:
    """Retrieves specification for a protocol, defaulting to vector if unknown."""
    if proto_id in PROTOCOLS:
        return PROTOCOLS[proto_id]
    raise KeyError(f"Unknown protocol '{proto_id}'. Valid protocols: {list(PROTOCOLS.keys())}")


def list_protocols_up_to_level(level: int) -> List[str]:
    """Returns protocol IDs eligible under developmental hierarchy level (0 to 6)."""
    return [p_id for p_id, spec in PROTOCOLS.items() if spec.complexity_level <= level]
