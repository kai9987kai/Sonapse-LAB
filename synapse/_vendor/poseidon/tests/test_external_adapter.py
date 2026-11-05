"""Tests for Poseidon Stage 7: Typed External Adapter Protocol & 3D Interactive Validation."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from poseidon.external_adapter import (
    AdapterHandshake,
    Compact3DReplayVerifier,
    ExternalActionResponse,
    ExternalAdapterBridge,
    ExternalAdapterIdentity,
    ExternalTickMessage,
    MemoryInfluenceTelemetry,
    SUPPORTED_COORDINATES,
    SUPPORTED_UNITS,
)
from poseidon.runtime import Poseidon
from poseidon.world import N_ACTIONS, OBS_SIZE


def test_external_adapter_identity_validation():
    # Valid identities
    ident_si = ExternalAdapterIdentity(
        adapter_id="unreal-engine-01",
        units="SI_METRIC",
        ticks_per_second=60.0,
        coordinate_frame="FLU",
        spatial_bounds=(100.0, 100.0, 20.0),
        max_linear_velocity=15.0,
    )
    assert ident_si.units == "SI_METRIC"
    assert ident_si.coordinate_frame == "FLU"

    ident_norm = ExternalAdapterIdentity(
        adapter_id="threejs-webgl",
        units="NORMALIZED_01",
        ticks_per_second=30.0,
        coordinate_frame="ENU",
    )
    assert ident_norm.units == "NORMALIZED_01"

    # Rejection of invalid units
    with pytest.raises(ValueError, match="unsupported units"):
        ExternalAdapterIdentity(adapter_id="bad-units", units="IMPERIAL_FEET")

    # Rejection of invalid coordinates
    with pytest.raises(ValueError, match="unsupported coordinate_frame"):
        ExternalAdapterIdentity(adapter_id="bad-coord", coordinate_frame="XYZ_UNKNOWN")

    # Rejection of empty adapter_id
    with pytest.raises(ValueError, match="non-empty string"):
        ExternalAdapterIdentity(adapter_id="")

    # Rejection of out-of-range tick rate
    with pytest.raises(ValueError, match="ticks_per_second"):
        ExternalAdapterIdentity(adapter_id="bad-hz", ticks_per_second=0.5)

    with pytest.raises(ValueError, match="ticks_per_second"):
        ExternalAdapterIdentity(adapter_id="bad-hz2", ticks_per_second=300.0)


def test_adapter_handshake_serialization():
    ident = ExternalAdapterIdentity(
        adapter_id="godot-client",
        units="SI_METRIC",
        ticks_per_second=20.0,
        coordinate_frame="NED",
        spatial_bounds=(80.0, 80.0, 25.0),
        max_linear_velocity=10.0,
    )
    bridge = ExternalAdapterBridge()
    handshake = bridge.negotiate_handshake(
        client_id="godot-4",
        engine_name="GodotEngine",
        identity=ident,
    )
    assert handshake.accepted is True
    assert handshake.client_id == "godot-4"
    assert handshake.engine_name == "GodotEngine"

    d = handshake.to_dict()
    assert d["session_token"] == handshake.session_token
    assert d["protocol_version"] == "poseidon-external-adapter-v1"

    # Roundtrip from_dict
    reconstructed = AdapterHandshake.from_dict(d)
    assert reconstructed.session_token == handshake.session_token
    assert reconstructed.identity.adapter_id == "godot-client"
    assert reconstructed.identity.coordinate_frame == "NED"


def test_monotonic_tick_progression():
    bridge = ExternalAdapterBridge()
    ident = ExternalAdapterIdentity(adapter_id="tick-test")
    handshake = bridge.negotiate_handshake("c1", "eng", ident)
    token = handshake.session_token

    class DummyAgent:
        def act(self, obs):
            return 1

    msg0 = ExternalTickMessage(session_token=token, tick=0, delta_time=0.05, observation_16d=[0.5] * OBS_SIZE)
    resp0 = bridge.process_tick(msg0, DummyAgent())
    assert resp0.tick == 0
    assert resp0.action == 1

    # Second tick: monotonic advance passes
    msg1 = ExternalTickMessage(session_token=token, tick=1, delta_time=0.05, observation_16d=[0.5] * OBS_SIZE)
    resp1 = bridge.process_tick(msg1, DummyAgent())
    assert resp1.tick == 1

    # Stale/duplicate tick fails
    msg_dup = ExternalTickMessage(session_token=token, tick=1, delta_time=0.05, observation_16d=[0.5] * OBS_SIZE)
    with pytest.raises(ValueError, match="non-monotonic tick received"):
        bridge.process_tick(msg_dup, DummyAgent())

    # Regression tick fails
    msg_reg = ExternalTickMessage(session_token=token, tick=0, delta_time=0.05, observation_16d=[0.5] * OBS_SIZE)
    with pytest.raises(ValueError, match="non-monotonic tick received"):
        bridge.process_tick(msg_reg, DummyAgent())


def test_kinematic_velocity_bound_check():
    bridge = ExternalAdapterBridge()
    ident = ExternalAdapterIdentity(adapter_id="kine-test", max_linear_velocity=10.0)
    handshake = bridge.negotiate_handshake("c1", "eng", ident)
    token = handshake.session_token

    class DummyAgent:
        def act(self, obs):
            return 0

    # Start at origin
    msg0 = ExternalTickMessage(
        session_token=token, tick=0, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE, position=(0.0, 0.0, 0.0)
    )
    bridge.process_tick(msg0, DummyAgent())

    # Move 0.5m in 0.1s -> speed 5.0 m/s (below max 10.0 m/s) -> OK
    msg1 = ExternalTickMessage(
        session_token=token, tick=1, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE, position=(0.5, 0.0, 0.0)
    )
    resp1 = bridge.process_tick(msg1, DummyAgent())
    assert resp1.tick == 1

    # Move 5.0m in 0.1s -> speed 50.0 m/s (exceeds max 10.0 * 1.5 = 15.0 m/s) -> Raises ValueError
    msg_teleport = ExternalTickMessage(
        session_token=token, tick=2, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE, position=(5.5, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="kinematic speed violation"):
        bridge.process_tick(msg_teleport, DummyAgent())


def test_spatial_bounds_and_normalized_units():
    bridge = ExternalAdapterBridge()
    ident = ExternalAdapterIdentity(adapter_id="norm-test", units="NORMALIZED_01")
    handshake = bridge.negotiate_handshake("c1", "eng", ident)
    token = handshake.session_token

    class DummyAgent:
        def act(self, obs):
            return 0

    # Position in [0, 1] passes
    msg_valid = ExternalTickMessage(
        session_token=token, tick=0, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE, position=(0.5, 0.5, 0.5)
    )
    resp = bridge.process_tick(msg_valid, DummyAgent())
    assert resp.tick == 0

    # Position out of [0, 1] fails
    msg_bad = ExternalTickMessage(
        session_token=token, tick=1, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE, position=(1.5, 0.5, 0.5)
    )
    with pytest.raises(ValueError, match="out of \\[0, 1\\] range for NORMALIZED_01 units"):
        bridge.process_tick(msg_bad, DummyAgent())


def test_session_cancellation_responsive():
    bridge = ExternalAdapterBridge()
    ident = ExternalAdapterIdentity(adapter_id="cancel-test")
    handshake = bridge.negotiate_handshake("c1", "eng", ident)
    token = handshake.session_token

    assert token in bridge.active_sessions
    assert token not in bridge.cancelled_sessions

    # Cancel session
    ok = bridge.cancel_session(token)
    assert ok is True
    assert token not in bridge.active_sessions
    assert token in bridge.cancelled_sessions

    # Repeat cancel on already cancelled session returns False
    assert bridge.cancel_session(token) is False

    # Process tick on cancelled session raises RuntimeError
    class DummyAgent:
        def act(self, obs):
            return 0

    msg = ExternalTickMessage(session_token=token, tick=0, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE)
    with pytest.raises(RuntimeError, match="was cancelled"):
        bridge.process_tick(msg, DummyAgent())

    # Process tick on non-existent session raises KeyError
    msg_unknown = ExternalTickMessage(session_token="unknown-token", tick=0, delta_time=0.1, observation_16d=[0.0] * OBS_SIZE)
    with pytest.raises(KeyError, match="unknown session token"):
        bridge.process_tick(msg_unknown, DummyAgent())


def test_telemetry_memory_influence_with_cognitive_agent():
    bridge = ExternalAdapterBridge()
    ident = ExternalAdapterIdentity(adapter_id="telemetry-test")
    handshake = bridge.negotiate_handshake("c1", "eng", ident)
    token = handshake.session_token

    from poseidon.cognitive_agent import CognitiveAgent
    agent = CognitiveAgent()

    msg = ExternalTickMessage(
        session_token=token,
        tick=0,
        delta_time=0.05,
        observation_16d=[1.0, 0.8, 0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    resp = bridge.process_tick(msg, agent)

    assert resp.session_token == token
    assert 0 <= resp.action < N_ACTIONS
    assert resp.latency_ms >= 0.0
    assert resp.memory_influence.readiness is True
    assert resp.memory_influence.readiness_scope == "calibrated-stage6-core"
    assert len(resp.memory_influence.dv_gem_deltas) == N_ACTIONS

    # Bridge latest_telemetry matches
    assert bridge.latest_telemetry == resp.memory_influence


def test_compact_3d_replay_verifier(tmp_path):
    replay = {
        "schema": "poseidon-3d-replay-v1",
        "adapter_identity": {
            "adapter_id": "test-3d-sim",
            "max_linear_velocity": 12.0,
        },
        "total_reward": 5.0,
        "keyframes": [
            {"tick": 0, "delta_time": 0.05, "position": [0.0, 0.0, 0.0], "action": 0, "reward": 1.0},
            {"tick": 1, "delta_time": 0.05, "position": [0.2, 0.0, 0.0], "action": 1, "reward": 2.0},
            {"tick": 2, "delta_time": 0.05, "position": [0.4, 0.1, 0.0], "action": 4, "reward": 2.0},
        ],
    }

    receipt = Compact3DReplayVerifier.verify_replay(replay)
    assert receipt["verified"] is True
    assert receipt["transitions_verified"] == 3
    assert receipt["total_reward"] == 5.0
    assert receipt["final_tick"] == 2
    assert len(receipt["receipt_sha256"]) == 64

    # Save to disk and test verify_replay_file
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(replay), encoding="utf-8")
    file_receipt = Compact3DReplayVerifier.verify_replay_file(p)
    assert file_receipt["receipt_sha256"] == receipt["receipt_sha256"]

    # Arithmetic mismatch error
    corrupt_reward = dict(replay, total_reward=999.0)
    with pytest.raises(ValueError, match="reward arithmetic mismatch"):
        Compact3DReplayVerifier.verify_replay(corrupt_reward)

    # Monotonic tick error
    corrupt_ticks = json.loads(json.dumps(replay))
    corrupt_ticks["keyframes"][2]["tick"] = 1
    with pytest.raises(ValueError, match="monotonic tick failure"):
        Compact3DReplayVerifier.verify_replay(corrupt_ticks)

    # Kinematic speed anomaly error
    corrupt_speed = json.loads(json.dumps(replay))
    corrupt_speed["keyframes"][2]["position"] = [50.0, 0.0, 0.0]  # Jump 49.6m in 0.05s
    with pytest.raises(ValueError, match="kinematic speed anomaly"):
        Compact3DReplayVerifier.verify_replay(corrupt_speed)

    # Invalid schema error
    corrupt_schema = dict(replay, schema="unsupported-schema")
    with pytest.raises(ValueError, match="invalid replay schema"):
        Compact3DReplayVerifier.verify_replay(corrupt_schema)


def test_runtime_external_adapter_status():
    rt = Poseidon()
    status = rt.external_adapter_status()
    assert status["ready"] is True
    assert status["protocol_version"] == "poseidon-external-adapter-v1"
    assert "SI_METRIC" in status["supported_units"]
    assert "FLU" in status["supported_coordinates"]
    assert status["active_sessions_count"] == 0

    full_status = rt.status()
    assert "external_adapter" in full_status
    assert full_status["external_adapter"]["ready"] is True
