"""Unit tests verifying NeuralPacket protocol invariants for Sonapse NeuroWave-X2."""
import math
import pytest
from synapse.neurowave_x2.packets import NeuralPacket, PacketValidationError


def test_packet_construction_and_immutability():
    pkt = NeuralPacket(
        sender_region=0,
        receiver_region=1,
        emission_step=10,
        sequence_id=42,
        topology_version=1,
        protocol_id="vector",
        payload=(0.5, -0.2, 0.9, -0.8),
        precision_bits=32,
        basis_version=1,
        phase_metadata=(0.785,),
    )
    assert pkt.sender_region == 0
    assert pkt.receiver_region == 1
    assert pkt.dimension == 4
    assert pkt.payload_bits == 4 * 32
    assert pkt.metadata_bits > 0
    assert pkt.total_wire_bits == pkt.payload_bits + pkt.metadata_bits
    assert len(pkt.checksum) == 16

    # Verify immutability
    with pytest.raises(Exception):
        pkt.sender_region = 2
    with pytest.raises(Exception):
        pkt.payload = (0.0,)


def test_packet_rejects_nan_and_inf():
    with pytest.raises(PacketValidationError):
        NeuralPacket(0, 1, 0, 0, 1, "vector", (0.1, float("nan")))

    with pytest.raises(PacketValidationError):
        NeuralPacket(0, 1, 0, 0, 1, "vector", (0.1, float("inf")))

    with pytest.raises(PacketValidationError):
        NeuralPacket(0, 1, 0, 0, 1, "vector", (0.1, float("-inf")))

    with pytest.raises(PacketValidationError):
        NeuralPacket(0, 1, 0, 0, 1, "phase", (0.1,), phase_metadata=(float("nan"),))


def test_packet_rejects_invalid_precision_and_negative_ids():
    with pytest.raises(PacketValidationError):
        NeuralPacket(-1, 1, 0, 0, 1, "vector", (0.1,))

    with pytest.raises(PacketValidationError):
        NeuralPacket(0, -1, 0, 0, 1, "vector", (0.1,))

    with pytest.raises(PacketValidationError):
        NeuralPacket(0, 1, 0, 0, 1, "vector", (0.1,), precision_bits=7)


def test_packet_binary_serialization_roundtrip():
    original = NeuralPacket(
        sender_region=2,
        receiver_region=5,
        emission_step=105,
        sequence_id=999,
        topology_version=3,
        protocol_id="phase_spectral",
        payload=(0.12345, -0.6789, 0.4321, -0.9876),
        precision_bits=16,
        basis_version=2,
        phase_metadata=(1.57079, 3.14159),
    )
    b = original.to_bytes()
    assert isinstance(b, bytes)
    reconstructed = NeuralPacket.from_bytes(b)

    assert reconstructed == original
    assert reconstructed.checksum == original.checksum
    assert reconstructed.total_wire_bits == original.total_wire_bits


def test_packet_dict_serialization_roundtrip():
    original = NeuralPacket(
        sender_region=1,
        receiver_region=3,
        emission_step=20,
        sequence_id=5,
        topology_version=1,
        protocol_id="quantized",
        payload=(0.25, -0.75),
        precision_bits=8,
    )
    d = original.to_dict()
    reconstructed = NeuralPacket.from_dict(d)
    assert reconstructed == original
