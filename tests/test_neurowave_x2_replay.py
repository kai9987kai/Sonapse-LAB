import pytest
import numpy as np

from synapse.neurowave_x2.packets import NeuralPacket
from synapse.neurowave_x2.replay import (
    ReplayRecorder,
    verify_replay_consistency,
    ReplayVerificationError,
)


def test_replay_recorder_and_verification_pass():
    recorder = ReplayRecorder()
    pkt = NeuralPacket(
        sender_region=0,
        receiver_region=1,
        emission_step=1,
        sequence_id=0,
        topology_version=1,
        protocol_id="vector",
        payload=(0.1, -0.2),
        precision_bits=16,
    )
    recorder.record_step(action=np.array([0.5, -0.5]), reward=1.0, packets=[pkt])

    trace1 = recorder.export_trace()
    trace2 = recorder.export_trace()

    passed, max_err, diag = verify_replay_consistency(trace1, trace2, tolerance=1e-5)
    assert passed is True
    assert max_err == 0.0
    assert diag["packets_matched"] is True


def test_replay_verification_failure_action():
    recorder = ReplayRecorder()
    recorder.record_step(action=np.array([0.5, -0.5]), reward=1.0, packets=[])
    trace1 = recorder.export_trace()

    recorder_divergent = ReplayRecorder()
    recorder_divergent.record_step(action=np.array([0.5, 0.9]), reward=1.0, packets=[])
    trace2 = recorder_divergent.export_trace()

    with pytest.raises(ReplayVerificationError, match="Replay divergence"):
        verify_replay_consistency(trace1, trace2, tolerance=1e-4)


def test_replay_verification_length_mismatch():
    trace1 = {"actions": [[0.1]], "rewards": [1.0], "packet_count": 0}
    trace2 = {"actions": [[0.1], [0.2]], "rewards": [1.0, 1.0], "packet_count": 0}

    with pytest.raises(ReplayVerificationError, match="length mismatch"):
        verify_replay_consistency(trace1, trace2)
