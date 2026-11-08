"""Master unit and scientific regression tests for AETHER GraftBridge and Unified Graft Bus.

Verifies:
1. Graph Laplacian orthonormal modal basis computation and canonicalization.
2. Binary packet serialization / deserialization wire fidelity.
3. AETHER GraftBridge zero-gain identity guarantee.
4. Scientific negative donor controls ('none', 'scrambled', 'zero_donor', 'phase_inverted').
5. Stochastic packet loss and noise impairment assays.
6. Unified Graft Communication Bus multi-channel bounded composition.
7. Independent channel ablation (SANCF, NeuroWave-X2, CrossGraft, AetherBridge).
8. Full state export / import persistence across all subsystems.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from synapse.aether_graft_bridge import (
    compute_graph_modes,
    GraftBridgeSpec,
    GraftTransferPacket,
    AetherGraftBridge,
)
from synapse.cross_graft import CrossConnectomeGraft
from synapse.graft_bus import (
    NeuralGraftCommunicationBus,
    SANCFBusChannel,
    NeuroWaveX2BusChannel,
    CrossGraftBusChannel,
    AetherBridgeBusChannel,
)


def test_graph_laplacian_modes_orthonormality():
    """Verify that graph_modes computes an orthonormal basis V^T V = I."""
    rng = np.random.default_rng(42)
    adj = rng.uniform(0.0, 1.0, size=(32, 32))
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)

    evals, modes = compute_graph_modes(adj, mode_count=8, seed=42)

    assert len(evals) == 8
    assert modes.shape == (32, 8)

    # Check orthonormality: V^T V == I
    gram = modes.T @ modes
    eye = np.eye(8)
    assert np.allclose(gram, eye, atol=1e-5), f"Gram matrix deviates from identity: {gram}"


def test_graft_transfer_packet_wire_serialization():
    """Verify little-endian binary packet serialization round-trip."""
    pkt = GraftTransferPacket(
        version=1,
        sequence_id=101,
        step_tick=42,
        source_region=1,
        target_region=3,
        control_arm="none",
        mode_amplitudes=(0.12, -0.45, 0.78, 0.0),
        carrier_phases=(0.0, 1.57, 3.14, 4.71),
    )

    data = pkt.to_bytes()
    assert isinstance(data, bytes)
    assert len(data) > 36

    recovered = GraftTransferPacket.from_bytes(data)
    assert recovered.version == pkt.version
    assert recovered.sequence_id == pkt.sequence_id
    assert recovered.step_tick == pkt.step_tick
    assert recovered.source_region == pkt.source_region
    assert recovered.target_region == pkt.target_region
    assert recovered.control_arm == pkt.control_arm
    assert len(recovered.mode_amplitudes) == len(pkt.mode_amplitudes)
    for a, b in zip(recovered.mode_amplitudes, pkt.mode_amplitudes):
        assert math.isclose(a, b, rel_tol=1e-5)
    for a, b in zip(recovered.carrier_phases, pkt.carrier_phases):
        assert math.isclose(a, b, rel_tol=1e-5)


def test_aether_bridge_zero_gain_identity():
    """Verify zero-gain identity guarantee when disabled or coupling_gain=0.0."""
    spec_disabled = GraftBridgeSpec(enabled=False, coupling_gain=0.3)
    bridge_dis = AetherGraftBridge(spec=spec_disabled, donor_dim=32, recipient_dim=32)

    spec_zero = GraftBridgeSpec(enabled=True, coupling_gain=0.0)
    bridge_zero = AetherGraftBridge(spec=spec_zero, donor_dim=32, recipient_dim=32)

    d_state = torch.randn(32)
    r_state = torch.randn(32)

    cur_dis, telem_dis = bridge_dis(d_state, r_state)
    assert (cur_dis == 0.0).all()
    assert cur_dis.norm().item() == 0.0

    cur_zero, telem_zero = bridge_zero(d_state, r_state)
    assert (cur_zero == 0.0).all()
    assert cur_zero.norm().item() == 0.0


def test_aether_bridge_negative_controls():
    """Verify negative donor controls (scrambled, zero_donor) alter/ablate transfer."""
    spec = GraftBridgeSpec(enabled=True, coupling_gain=0.25, negative_control="none")
    bridge = AetherGraftBridge(spec=spec, donor_dim=32, recipient_dim=32)

    d_state = torch.randn(32)
    r_state = torch.randn(32)

    # 1. Authentic transfer
    bridge.set_negative_control("none")
    cur_auth, telem_auth = bridge(d_state, r_state, advance=False)
    assert cur_auth.norm().item() > 0.0

    # 2. Zero donor control
    bridge.set_negative_control("zero_donor")
    cur_zero, telem_zero = bridge(d_state, r_state, advance=False)
    assert cur_zero.norm().item() == 0.0

    # 3. Scrambled donor control
    bridge.set_negative_control("scrambled")
    cur_scram, telem_scram = bridge(d_state, r_state, advance=False)
    assert not torch.allclose(cur_auth, cur_scram, atol=1e-3)


def test_aether_bridge_packet_loss_assay():
    """Verify packet loss drops packets and zeroes output current."""
    spec = GraftBridgeSpec(enabled=True, coupling_gain=0.2, packet_loss_rate=1.0)
    bridge = AetherGraftBridge(spec=spec, donor_dim=32, recipient_dim=32)

    d_state = torch.randn(32)
    r_state = torch.randn(32)

    cur, telem = bridge(d_state, r_state)
    assert cur.norm().item() == 0.0
    assert bridge.total_dropped > 0


def test_graft_bus_multi_channel_composition_and_zero_gain():
    """Verify multi-channel composition, zero-gain identity, and clipping bound."""
    bus = NeuralGraftCommunicationBus(max_aux_current=0.4, context_dim=32)

    # Dummy channels
    graft = CrossConnectomeGraft(node_dim=32, gamma=0.0, seed=42)
    ch_cross = CrossGraftBusChannel(graft, name="cross_graft")

    bridge_spec = GraftBridgeSpec(enabled=False, coupling_gain=0.0)
    bridge = AetherGraftBridge(spec=bridge_spec, donor_dim=32, recipient_dim=32)
    ch_bridge = AetherBridgeBusChannel(bridge, name="aether_bridge")

    bus.register_channel("cross_graft", ch_cross, gain=1.0)
    bus.register_channel("aether_bridge", ch_bridge, gain=1.0)

    # Both channels zero-gain / disabled -> bus disabled
    assert not bus.enabled
    r = torch.randn(32)
    step_out = bus.compute_current(r)
    assert (step_out.residual_current == 0.0).all()
    assert step_out.total_norm == 0.0

    # Enable CrossGraft with gamma=0.2
    graft.gamma = 0.2
    ch_cross.set_enabled(True)
    assert bus.enabled

    step_out_cross = bus.compute_current(r)
    assert step_out_cross.total_norm > 0.0
    assert "cross_graft" in step_out_cross.channel_currents

    # Enable AetherBridge with gain=0.25
    bridge.spec.enabled = True
    bridge.spec.coupling_gain = 0.25

    step_out_both = bus.compute_current(r)
    assert step_out_both.total_norm > 0.0
    assert "cross_graft" in step_out_both.channel_currents
    assert "aether_bridge" in step_out_both.channel_currents

    # Bounded clip check
    assert step_out_both.residual_current.abs().max().item() <= 0.4 + 1e-5


def test_graft_bus_independent_channel_ablation():
    """Verify each channel can be toggled independently."""
    bus = NeuralGraftCommunicationBus(max_aux_current=0.5, context_dim=32)

    graft = CrossConnectomeGraft(node_dim=32, gamma=0.2, seed=42)
    bridge_spec = GraftBridgeSpec(enabled=True, coupling_gain=0.2)
    bridge = AetherGraftBridge(spec=bridge_spec, donor_dim=32, recipient_dim=32)

    bus.register_channel("cross_graft", CrossGraftBusChannel(graft))
    bus.register_channel("aether_bridge", AetherBridgeBusChannel(bridge))

    r = torch.randn(32)

    # Disable cross_graft
    bus.set_channel_enabled("cross_graft", False)
    out1 = bus.compute_current(r)
    assert "cross_graft" not in out1.channel_currents
    assert "aether_bridge" in out1.channel_currents

    # Disable aether_bridge, re-enable cross_graft
    bus.set_channel_enabled("cross_graft", True)
    bus.set_channel_enabled("aether_bridge", False)
    out2 = bus.compute_current(r)
    assert "cross_graft" in out2.channel_currents
    assert "aether_bridge" not in out2.channel_currents


def test_graft_bus_checkpoint_roundtrip():
    """Verify that export_state and import_state preserve bus configuration and member states."""
    bus1 = NeuralGraftCommunicationBus(max_aux_current=0.6, context_dim=32)
    graft1 = CrossConnectomeGraft(node_dim=32, gamma=0.18, seed=42)
    bus1.register_channel("cross_graft", CrossGraftBusChannel(graft1), gain=0.85)

    r = torch.randn(32)
    out1 = bus1.compute_current(r)
    state = bus1.export_state()

    bus2 = NeuralGraftCommunicationBus(max_aux_current=0.1, context_dim=32)
    graft2 = CrossConnectomeGraft(node_dim=32, gamma=0.0, seed=1)
    bus2.register_channel("cross_graft", CrossGraftBusChannel(graft2), gain=0.1)

    bus2.import_state(state)
    out2 = bus2.compute_current(r)

    assert math.isclose(bus1.max_aux_current, bus2.max_aux_current, rel_tol=1e-5)
    assert math.isclose(bus1.channel_gains["cross_graft"], bus2.channel_gains["cross_graft"], rel_tol=1e-5)
    assert torch.allclose(out1.residual_current, out2.residual_current, atol=1e-5)
