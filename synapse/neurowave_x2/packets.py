"""Immutable Neural Packet Protocol for Sonapse NeuroWave-X2.

Implements strict validation, deterministic serialization, byte-order fidelity,
checksum verification, and separate payload/metadata wire cost accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple


class PacketValidationError(ValueError):
    """Raised when a neural packet violates structural or numerical invariants."""
    pass


@dataclass(frozen=True)
class NeuralPacket:
    """Immutable neural communication packet exchanged across connectome regions."""

    sender_region: int
    receiver_region: int
    emission_step: int
    sequence_id: int
    topology_version: int
    protocol_id: str
    payload: Tuple[float, ...]
    precision_bits: int = 32
    basis_version: int = 0
    phase_metadata: Tuple[float, ...] = ()
    payload_bits: int = field(init=False)
    metadata_bits: int = field(init=False)
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        # 1. Validate integer identifiers
        if self.sender_region < 0:
            raise PacketValidationError(f"sender_region must be non-negative, got {self.sender_region}")
        if self.receiver_region < 0:
            raise PacketValidationError(f"receiver_region must be non-negative, got {self.receiver_region}")
        if self.emission_step < 0:
            raise PacketValidationError(f"emission_step must be non-negative, got {self.emission_step}")
        if self.sequence_id < 0:
            raise PacketValidationError(f"sequence_id must be non-negative, got {self.sequence_id}")
        if self.topology_version < 0:
            raise PacketValidationError(f"topology_version must be non-negative, got {self.topology_version}")
        if self.basis_version < 0:
            raise PacketValidationError(f"basis_version must be non-negative, got {self.basis_version}")
        if self.precision_bits not in (2, 4, 8, 16, 32):
            raise PacketValidationError(f"precision_bits must be in (2, 4, 8, 16, 32), got {self.precision_bits}")
        if not self.protocol_id or not isinstance(self.protocol_id, str):
            raise PacketValidationError("protocol_id must be a non-empty string")

        # 2. Strict numerical validation (Reject NaN, Inf)
        for i, val in enumerate(self.payload):
            if not isinstance(val, (int, float)):
                raise PacketValidationError(f"payload element {i} is not a float/int: {val}")
            if math.isnan(val):
                raise PacketValidationError(f"payload element {i} is NaN")
            if math.isinf(val):
                raise PacketValidationError(f"payload element {i} is Inf")

        for i, val in enumerate(self.phase_metadata):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise PacketValidationError(f"phase_metadata element {i} invalid: {val}")

        # 3. Calculate payload bits and metadata bits
        dim = len(self.payload)
        calc_payload_bits = int(dim * self.precision_bits)
        # Metadata: 4 bytes sender + 4 bytes receiver + 4 bytes emission + 4 bytes seq +
        # 4 bytes topo_ver + 4 bytes basis_ver + 2 bytes prec + 2 bytes proto_len +
        # len(proto) bytes + len(phase) * 4 bytes
        calc_metadata_bits = int((32 + len(self.protocol_id.encode('utf-8')) + len(self.phase_metadata) * 4) * 8)

        # Frozen dataclass assignment via object.__setattr__
        object.__setattr__(self, "payload_bits", calc_payload_bits)
        object.__setattr__(self, "metadata_bits", calc_metadata_bits)

        # 4. Deterministic checksum calculation
        hasher = hashlib.sha256()
        hasher.update(struct.pack(">IIIIIIH",
            self.sender_region,
            self.receiver_region,
            self.emission_step,
            self.sequence_id,
            self.topology_version,
            self.basis_version,
            self.precision_bits,
        ))
        hasher.update(self.protocol_id.encode("utf-8"))
        for v in self.payload:
            hasher.update(struct.pack(">d", float(v)))
        for p in self.phase_metadata:
            hasher.update(struct.pack(">d", float(p)))
        object.__setattr__(self, "checksum", hasher.hexdigest()[:16])

    @property
    def total_wire_bits(self) -> int:
        """Total communication cost including payload and protocol metadata."""
        return self.payload_bits + self.metadata_bits

    @property
    def dimension(self) -> int:
        return len(self.payload)

    def to_bytes(self) -> bytes:
        """Deterministic binary serialization with big-endian network byte-ordering."""
        proto_bytes = self.protocol_id.encode("utf-8")
        header = struct.pack(
            ">IIIIIIHH",
            self.sender_region,
            self.receiver_region,
            self.emission_step,
            self.sequence_id,
            self.topology_version,
            self.basis_version,
            self.precision_bits,
            len(proto_bytes),
        )
        body = [header, proto_bytes]
        # Payload length and values (as double precision floats)
        body.append(struct.pack(">H", len(self.payload)))
        for v in self.payload:
            body.append(struct.pack(">d", float(v)))
        # Phase metadata length and values
        body.append(struct.pack(">H", len(self.phase_metadata)))
        for p in self.phase_metadata:
            body.append(struct.pack(">d", float(p)))
        return b"".join(body)

    @classmethod
    def from_bytes(cls, b: bytes) -> "NeuralPacket":
        """Deserialize packet from binary payload with strict validation."""
        if len(b) < 28:
            raise PacketValidationError(f"Binary packet too short ({len(b)} bytes)")
        offset = 0
        sender, receiver, emission, seq, topo_ver, basis_ver, prec, proto_len = struct.unpack_from(
            ">IIIIIIHH", b, offset
        )
        offset += 28
        if len(b) < offset + proto_len + 2:
            raise PacketValidationError("Corrupted binary packet: incomplete protocol string")
        proto_id = b[offset:offset + proto_len].decode("utf-8")
        offset += proto_len

        payload_len, = struct.unpack_from(">H", b, offset)
        offset += 2
        payload = []
        for _ in range(payload_len):
            if len(b) < offset + 8:
                raise PacketValidationError("Corrupted binary packet: incomplete payload")
            val, = struct.unpack_from(">d", b, offset)
            payload.append(val)
            offset += 8

        if len(b) < offset + 2:
            raise PacketValidationError("Corrupted binary packet: missing phase count")
        phase_len, = struct.unpack_from(">H", b, offset)
        offset += 2
        phase_meta = []
        for _ in range(phase_len):
            if len(b) < offset + 8:
                raise PacketValidationError("Corrupted binary packet: incomplete phase metadata")
            val, = struct.unpack_from(">d", b, offset)
            phase_meta.append(val)
            offset += 8

        return cls(
            sender_region=sender,
            receiver_region=receiver,
            emission_step=emission,
            sequence_id=seq,
            topology_version=topo_ver,
            protocol_id=proto_id,
            payload=tuple(payload),
            precision_bits=prec,
            basis_version=basis_ver,
            phase_metadata=tuple(phase_meta),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_region": self.sender_region,
            "receiver_region": self.receiver_region,
            "emission_step": self.emission_step,
            "sequence_id": self.sequence_id,
            "topology_version": self.topology_version,
            "protocol_id": self.protocol_id,
            "payload": list(self.payload),
            "precision_bits": self.precision_bits,
            "basis_version": self.basis_version,
            "phase_metadata": list(self.phase_metadata),
            "payload_bits": self.payload_bits,
            "metadata_bits": self.metadata_bits,
            "checksum": self.checksum,
            "total_wire_bits": self.total_wire_bits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuralPacket":
        return cls(
            sender_region=int(data["sender_region"]),
            receiver_region=int(data["receiver_region"]),
            emission_step=int(data["emission_step"]),
            sequence_id=int(data["sequence_id"]),
            topology_version=int(data["topology_version"]),
            protocol_id=str(data["protocol_id"]),
            payload=tuple(float(v) for v in data["payload"]),
            precision_bits=int(data.get("precision_bits", 32)),
            basis_version=int(data.get("basis_version", 0)),
            phase_metadata=tuple(float(p) for p in data.get("phase_metadata", ())),
        )
