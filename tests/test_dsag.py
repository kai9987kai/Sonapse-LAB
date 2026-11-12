from dataclasses import replace

import pytest
import torch

from synapse.config import SynapseConfig
from synapse.dsag.bridge import DSAGBusChannel
from synapse.dsag.config import DSAGSpecV1
from synapse.dsag.model import DualSomaAxodendriticGraft
from synapse.dsag.packet_adapter import DSAGLocalEnvelope
from synapse.dsag.plasticity import DSAGTopologyController
from synapse.dsag.state import DSAGCheckpointError, canonical_state_digest, seal_checkpoint
from synapse.dsag.transaction import DSAGCandidateRejected, DSAGTrainingTransaction
from synapse.dsag.validation import validate_spec
from synapse.graft_bus import NeuralGraftCommunicationBus, NeuralGraftProtocol
from synapse.neurowave_x2.packets import NeuralPacket, PacketValidationError
from synapse.runtime import LabRuntime


def _spec(**overrides):
    base = DSAGSpecV1(
        enabled=True,
        gain=0.05,
        max_gain=0.1,
        ramp_steps=4,
        soma_dim=4,
        segment_count=3,
        branch_segments=(1, 2),
        max_branches=4,
        branch_delays=(0, 1),
        context_dim=8,
        message_dim=4,
    )
    return replace(base, **overrides)


def test_default_off_config_keeps_legacy_config_export_and_hash_scope():
    config = SynapseConfig(nodes_per_source=16)
    assert "dsag" not in config.to_dict()
    assert config.matches_saved(config.to_dict())
    assert config.matches_saved({key: value for key, value in config.to_dict().items() if key != "dsag"})
    assert DSAGSpecV1().enabled is False
    assert DSAGSpecV1().gain == 0.0


def test_dsag_default_is_additive_when_loading_a_legacy_runtime_checkpoint(tmp_path):
    legacy = LabRuntime(SynapseConfig(nodes_per_source=16), output_root=tmp_path)
    legacy.step(1)
    legacy.save('legacy')

    enabled_config = replace(SynapseConfig(nodes_per_source=16), dsag=_spec())
    restored = LabRuntime(enabled_config, output_root=tmp_path)
    initial_dsag_state = restored.agent.dsag_channel.export_state()
    snapshot = restored.load('legacy')

    assert snapshot['steps'] == 1
    assert restored.agent.dsag_channel.enabled is True
    assert canonical_state_digest(restored.agent.dsag_channel.export_state()) == canonical_state_digest(initial_dsag_state)
    assert SynapseConfig(nodes_per_source=16).matches_saved(enabled_config.to_dict()) is False
    assert enabled_config.matches_saved(SynapseConfig(nodes_per_source=16).to_dict()) is True

    advanced = restored.step(1)
    assert advanced['steps'] == 2
    assert restored.agent.dsag_channel.get_telemetry_dict()['enabled'] is True


def test_spec_digest_covers_numeric_topology_and_feature_flags():
    spec = _spec()
    assert spec.spec_digest != replace(spec, eta_soma_a=0.4).spec_digest
    assert spec.spec_digest != replace(spec, branch_segments=(0, 2)).spec_digest
    assert spec.spec_digest != replace(spec, phase_gate_enabled=True).spec_digest


def test_model_initialization_uses_independent_rng_and_supports_batch_shapes():
    torch.manual_seed(991)
    before = torch.random.get_rng_state().clone()
    model = DualSomaAxodendriticGraft(16, _spec())
    assert torch.equal(before, torch.random.get_rng_state())
    rates = torch.linspace(0, 1, 48).reshape(3, 16)
    first = model(rates, advance=True, microstep=0)
    second = model(rates, advance=True, microstep=1)
    assert first.shape == (3, 16)
    assert second.shape == (3, 16)
    assert torch.isfinite(second).all()
    assert torch.count_nonzero(first) == 0
    assert torch.count_nonzero(second) > 0


def test_soma_updates_are_synchronous_and_match_hand_calculation():
    spec = DSAGSpecV1(
        enabled=True, gain=0.0, max_gain=0.1, ramp_steps=0,
        soma_dim=2, segment_count=1, branch_segments=(), max_branches=0,
        branch_delays=(), context_dim=0, message_dim=2,
    )
    model = DualSomaAxodendriticGraft(3, spec)
    old_a = torch.tensor([[0.2, -0.4]])
    old_b = torch.tensor([[0.6, 0.1]])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.soma_a_from_b.copy_(torch.eye(2) * 0.7)
        model.soma_b_from_a.copy_(torch.eye(2) * 0.4)
    model._runtime["soma_a"] = old_a.clone()
    model._runtime["soma_b"] = old_b.clone()
    model(torch.zeros(3), advance=True, microstep=0)
    expected_a = 0.5 * old_a + 0.5 * torch.tanh(0.7 * old_b)
    expected_b = 0.75 * old_b + 0.25 * torch.tanh(0.4 * old_a)
    assert torch.allclose(model._runtime["soma_a"], expected_a, atol=1e-7)
    assert torch.allclose(model._runtime["soma_b"], expected_b, atol=1e-7)


@pytest.mark.parametrize("training", [True, False])
def test_gain_zero_and_observe_only_emit_exact_zero(training):
    zero_gain = DSAGBusChannel(_spec(gain=0.0), input_dim=16)
    zero_gain.model.train(training)
    current = zero_gain.compute_current(torch.ones(16), microstep=0, advance=True, adapt=True)
    assert torch.equal(current, torch.zeros_like(current))

    shadow = DSAGBusChannel(_spec(observe_only=True), input_dim=16)
    shadow.model.train(training)
    before = shadow.model._runtime["soma_a"].clone()
    current = shadow.compute_current(torch.ones(16), microstep=0, advance=True, adapt=True)
    assert torch.equal(current, torch.zeros_like(current))
    assert not torch.equal(before, shadow.model._runtime["soma_a"])


def test_advance_false_is_transactionally_read_only_including_metrics_and_queue():
    channel = DSAGBusChannel(_spec(), input_dim=16)
    packet = NeuralPacket(1, 2, 0, 0, 0, "neurowave_x2", (0.1, 0.2), phase_metadata=(0.5,))
    envelope = DSAGLocalEnvelope.from_packet(
        packet, graft_id="dsag-001", destination_kind="axon_branch", destination_index=0,
    )
    channel.enqueue_envelope(envelope)
    before = canonical_state_digest(channel.export_state())
    result = channel.compute_current(torch.ones(16), microstep=0, advance=False, adapt=True)
    after = canonical_state_digest(channel.export_state())
    assert result.shape == (16,)
    assert before == after
    assert len(channel.model._queue) == 1
    assert channel.model._runtime["tick"] == -1


def test_packet_envelope_preserves_source_provenance_and_rejects_tampering():
    packet = NeuralPacket(2, 3, 4, 9, 0, "neurowave_x2", (0.1, -0.2), phase_metadata=(1.25,))
    envelope = DSAGLocalEnvelope.from_packet(
        packet, graft_id="dsag-001", destination_kind="axon_branch", destination_index=1,
        event_tick=5, ttl=8,
    )
    assert envelope.parent_packet_checksum == packet.checksum
    assert envelope.source_sequence_id == 9
    assert envelope.age_ticks == 1
    assert envelope.phase_vector == (1.25,)
    assert envelope.provenance.endswith(packet.checksum)
    changed = envelope.to_dict()
    changed["payload_vector"] = [0.9, -0.2]
    with pytest.raises(PacketValidationError, match="checksum"):
        DSAGLocalEnvelope.from_dict(changed)


def test_queue_rejects_duplicate_out_of_order_and_stale_topology_packets():
    channel = DSAGBusChannel(_spec(), input_dim=16)
    p0 = NeuralPacket(1, 2, 0, 1, 0, "neurowave_x2", (0.1,))
    env0 = DSAGLocalEnvelope.from_packet(p0, graft_id="dsag-001", destination_index=0)
    channel.enqueue_envelope(env0)
    with pytest.raises(ValueError, match="duplicate"):
        channel.enqueue_envelope(env0)
    p1_old = NeuralPacket(1, 2, 0, 0, 0, "neurowave_x2", (0.2,))
    env1_old = DSAGLocalEnvelope.from_packet(p1_old, graft_id="dsag-001", destination_index=0)
    with pytest.raises(ValueError, match="out-of-order"):
        channel.enqueue_envelope(env1_old)
    stale = replace(env0, topology_version=1, checksum="")
    with pytest.raises(ValueError, match="stale"):
        channel.enqueue_envelope(stale)


def test_adapt_false_freezes_plasticity_and_explicit_modulator_updates_it():
    channel = DSAGBusChannel(_spec(plasticity_enabled=True), input_dim=16)
    packet0 = NeuralPacket(1, 2, 0, 0, 0, "local", (0.4, 0.2))
    env0 = DSAGLocalEnvelope.from_packet(
        packet0, graft_id="dsag-001", destination_kind="axon_branch", destination_index=0,
    )
    env0 = replace(env0, modulation_vector=(0.8,), checksum="")
    channel.enqueue_envelope(env0)
    weights = channel.model.branch_to_segment.detach().clone()
    channel.compute_current(torch.linspace(0, 1, 16), microstep=0, advance=True, adapt=False)
    assert torch.equal(weights, channel.model.branch_to_segment)
    assert channel.model._runtime["plasticity_updates"] == 0

    packet1 = NeuralPacket(1, 2, 1, 1, 0, "local", (0.2, 0.8))
    env1 = DSAGLocalEnvelope.from_packet(
        packet1, graft_id="dsag-001", destination_kind="axon_branch", destination_index=0,
    )
    env1 = replace(env1, modulation_vector=(0.8,), checksum="")
    channel.enqueue_envelope(env1)
    channel.compute_current(torch.linspace(0, 1, 16), microstep=1, advance=True, adapt=True)
    assert channel.model._runtime["plasticity_updates"] == 1
    assert not torch.equal(weights, channel.model.branch_to_segment)


def test_structural_branch_edits_have_exact_rollback_and_budgets():
    model = DualSomaAxodendriticGraft(16, _spec(structural_growth_enabled=True))
    controller = DSAGTopologyController(model)
    before_topology = model.topology_digest
    before_weights = model.branch_to_segment.detach().clone()
    receipt = controller.add_axon_input_branch(0)
    assert len(model._active_branch_segments) == 3
    assert model.topology_version == 1
    controller.rollback(receipt)
    assert model.topology_digest == before_topology
    assert torch.equal(model.branch_to_segment, before_weights)

    with pytest.raises(ValueError, match="branch budget"):
        full = DualSomaAxodendriticGraft(16, _spec(max_branches=2, branch_segments=(1, 2), branch_delays=(0, 0), structural_growth_enabled=True))
        full.apply_topology("add_axon_input_branch", segment_index=0)


def test_candidate_transaction_requires_commit_and_restores_full_dsag_state():
    channel = DSAGBusChannel(_spec(), input_dim=16)
    before_checkpoint = channel.export_state()
    before = canonical_state_digest(before_checkpoint)
    with DSAGTrainingTransaction(channel):
        channel.compute_current(torch.ones(16), microstep=0, advance=True, adapt=True)
    assert canonical_state_digest(channel.export_state()) == before

    with pytest.raises(DSAGCandidateRejected, match="validator"):
        with DSAGTrainingTransaction(channel, validator=lambda candidate: False) as transaction:
            channel.compute_current(torch.ones(16), microstep=0, advance=True, adapt=True)
            transaction.commit()
    assert canonical_state_digest(channel.export_state()) == before

    with DSAGTrainingTransaction(channel) as transaction:
        channel.compute_current(torch.ones(16), microstep=0, advance=True, adapt=True)
        receipt = transaction.commit()
    assert receipt.accepted is True
    assert receipt.parent_checkpoint_digest == before_checkpoint["state_digest"]
    assert receipt.candidate_checkpoint_digest != receipt.parent_checkpoint_digest


def test_checkpoint_round_trip_is_exact_and_rejects_stale_parent_and_corruption():
    spec = _spec(parent_commit="base-a")
    channel = DSAGBusChannel(spec, input_dim=16)
    rates = torch.linspace(0, 1, 16)
    channel.compute_current(rates, microstep=0, advance=True)
    state = channel.export_state()
    restored = DSAGBusChannel(spec, input_dim=16)
    restored.import_state(state)
    left = channel.compute_current(rates, microstep=1, advance=True)
    right = restored.compute_current(rates, microstep=1, advance=True)
    assert torch.equal(left, right)
    assert canonical_state_digest(channel.export_state()) == canonical_state_digest(restored.export_state())

    payload = {key: value for key, value in state.items() if key != "state_digest"}
    payload["parent_commit"] = "base-b"
    stale = seal_checkpoint(payload)
    with pytest.raises(DSAGCheckpointError, match="parent commit"):
        restored.import_state(stale)
    corrupted = dict(state)
    corrupted["runtime_state"] = {**state["runtime_state"], "topology_version": 99}
    with pytest.raises(DSAGCheckpointError, match="digest mismatch"):
        restored.import_state(corrupted)


class _FixedCurrent(NeuralGraftProtocol):
    name = "fixed"
    enabled = True

    def compute_current(self, r, **kwargs):
        return torch.full_like(r, 1.0)

    def get_telemetry_dict(self):
        return {}


def test_dsag_shares_the_existing_final_bus_cap_with_legacy_channels():
    spec = _spec(gain=0.05, ramp_steps=1)
    dsag = DSAGBusChannel(spec, input_dim=16)
    bus = NeuralGraftCommunicationBus(max_aux_current=0.2, context_dim=8)
    bus.register_channel("dsag", dsag, gain=1.0)
    bus.register_channel("fixed", _FixedCurrent(), gain=1.0)
    result = bus.compute_current(torch.zeros(16), microstep=0, advance=True, adapt=False)
    assert torch.all(result.residual_current <= 0.2)
    assert torch.all(result.residual_current >= -0.2)
    assert torch.equal(result.residual_current, torch.full((16,), 0.2))
    assert dsag.get_telemetry_dict()["effective_current_inf_norm"] == pytest.approx(0.2)


def test_config_file_and_resource_validation_are_reproducible():
    spec = DSAGSpecV1.from_file("configs/dsag/rate_pilot.yaml")
    report = validate_spec(spec, input_dim=32)
    assert spec.enabled is True
    assert spec.gain == 0.05
    assert report["valid"] is True
    assert report["parameter_count_upper_bound"] > 0
    assert report["spec_digest"] == spec.spec_digest
    with pytest.raises(ValueError, match="target_indices"):
        validate_spec(replace(spec, target_indices=(32,)), input_dim=32)
    with pytest.raises(ValueError, match="input_dim"):
        validate_spec(spec, input_dim=100_001)
