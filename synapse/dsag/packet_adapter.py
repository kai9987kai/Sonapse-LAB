"""Internal DSAG packet envelopes; the NeuroWave-X2 wire format is untouched."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

from ..neurowave_x2.packets import NeuralPacket, PacketValidationError


@dataclass(frozen=True)
class DSAGLocalEnvelope:
    graft_id: str
    destination_kind: str
    destination_index: int
    event_tick: int
    source_tick: int
    age_ticks: int
    ttl: int
    payload_vector: tuple[float, ...]
    source_kind: str = "local-graft"
    source_sequence_id: int = 0
    source_region: int = 0
    receiver_region: int = 0
    parent_packet_checksum: str | None = None
    phase_vector: tuple[float, ...] = ()
    spectral_amplitude: tuple[float, ...] = ()
    modulation_vector: tuple[float, ...] = ()
    validity_mask: tuple[bool, ...] = ()
    basis_version: int = 0
    topology_version: int = 0
    quantization_bits: int = 32
    units: str = "host_rate"
    provenance: str = "explicit-local-input"
    schema_version: int = 1
    checksum: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PacketValidationError("DSAGLocalEnvelope schema_version must be 1")
        if not self.graft_id or not self.source_kind or not self.provenance or not self.units:
            raise PacketValidationError("graft_id, source_kind, provenance, and units are required")
        if self.destination_kind not in {"soma_a", "soma_b", "axon_segment", "axon_branch"}:
            raise PacketValidationError("unsupported DSAG envelope destination")
        for name in ("destination_index", "event_tick", "source_tick", "age_ticks", "ttl",
                     "source_sequence_id", "source_region", "receiver_region", "basis_version",
                     "topology_version"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise PacketValidationError(f"{name} must be a non-negative integer")
        if self.event_tick < self.source_tick or self.age_ticks != self.event_tick - self.source_tick:
            raise PacketValidationError("age_ticks must equal event_tick - source_tick")
        if self.quantization_bits not in (2, 4, 8, 16, 32):
            raise PacketValidationError("quantization_bits must be one of 2, 4, 8, 16, 32")
        for name in ("payload_vector", "phase_vector", "spectral_amplitude", "modulation_vector"):
            vector = tuple(float(v) for v in getattr(self, name))
            if len(vector) > 128:
                raise PacketValidationError(f"{name} exceeds the 128-value sidecar budget")
            if any(not math.isfinite(v) for v in vector):
                raise PacketValidationError(f"{name} cannot contain NaN or infinity")
            object.__setattr__(self, name, vector)
        mask = tuple(self.validity_mask) if self.validity_mask else tuple(True for _ in self.payload_vector)
        if len(mask) != len(self.payload_vector) or any(type(v) is not bool for v in mask):
            raise PacketValidationError("validity_mask must contain one boolean per payload value")
        object.__setattr__(self, "validity_mask", mask)
        expected = self._compute_checksum()
        if self.checksum and self.checksum != expected:
            raise PacketValidationError("DSAG envelope checksum mismatch")
        object.__setattr__(self, "checksum", expected)

    def _payload_for_hash(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("checksum", None)
        return data

    def _compute_checksum(self) -> str:
        encoded = json.dumps(self._payload_for_hash(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_for_hash(), "checksum": self.checksum}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DSAGLocalEnvelope":
        if not isinstance(data, dict):
            raise PacketValidationError("DSAG envelope must be a mapping")
        values = dict(data)
        for key in ("payload_vector", "phase_vector", "spectral_amplitude", "modulation_vector", "validity_mask"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def from_packet(
        cls,
        packet: NeuralPacket,
        *,
        graft_id: str,
        destination_kind: str = "axon_branch",
        destination_index: int = 0,
        event_tick: int | None = None,
        ttl: int = 64,
        source_kind: str = "neurowave",
        provenance: str | None = None,
    ) -> "DSAGLocalEnvelope":
        if not isinstance(packet, NeuralPacket):
            raise PacketValidationError("from_packet requires an immutable NeuralPacket")
        tick = packet.emission_step if event_tick is None else int(event_tick)
        if tick < packet.emission_step:
            raise PacketValidationError("event_tick cannot precede packet emission")
        return cls(
            graft_id=graft_id,
            destination_kind=destination_kind,
            destination_index=destination_index,
            event_tick=tick,
            source_tick=packet.emission_step,
            age_ticks=tick - packet.emission_step,
            ttl=ttl,
            payload_vector=tuple(packet.payload),
            source_kind=source_kind,
            source_sequence_id=packet.sequence_id,
            source_region=packet.sender_region,
            receiver_region=packet.receiver_region,
            parent_packet_checksum=packet.checksum,
            phase_vector=tuple(packet.phase_metadata),
            validity_mask=tuple(True for _ in packet.payload),
            basis_version=packet.basis_version,
            topology_version=packet.topology_version,
            quantization_bits=packet.precision_bits,
            provenance=provenance or f"{packet.protocol_id}:{packet.checksum}",
        )
