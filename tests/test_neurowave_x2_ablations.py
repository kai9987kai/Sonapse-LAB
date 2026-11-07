import pytest
import numpy as np

from synapse.neurowave_x2.packets import NeuralPacket
from synapse.neurowave_x2.ablations import (
    CausalAblationController,
    evaluate_route_causal_importance,
    construct_route_importance_matrix,
)


def _sample_packets():
    p1 = NeuralPacket(
        sender_region=0,
        receiver_region=1,
        emission_step=1,
        sequence_id=0,
        topology_version=1,
        protocol_id="vector",
        payload=(1.0, 2.0),
        phase_metadata=(0.5,),
    )
    p2 = NeuralPacket(
        sender_region=1,
        receiver_region=2,
        emission_step=1,
        sequence_id=1,
        topology_version=1,
        protocol_id="vector",
        payload=(3.0, 4.0),
        phase_metadata=(1.5,),
    )
    return [p1, p2]


def test_ablation_none():
    ctrl = CausalAblationController("none")
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 2
    assert abl[0].payload == (1.0, 2.0)


def test_ablation_zero_comm():
    ctrl = CausalAblationController("zero_comm")
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 0


def test_ablation_lesion_route():
    ctrl = CausalAblationController("lesion_route", target_route=(0, 1))
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 1
    assert abl[0].sender_region == 1
    assert abl[0].receiver_region == 2


def test_ablation_shuffle_messages():
    ctrl = CausalAblationController("shuffle_messages", seed=42)
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 2
    # Verify payloads are still drawn from the original set
    payloads = [p.payload for p in abl]
    assert (1.0, 2.0) in payloads
    assert (3.0, 4.0) in payloads


def test_ablation_randomize_phase():
    ctrl = CausalAblationController("randomize_phase", seed=42)
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 2
    # Phase metadata was randomized
    assert abl[0].phase_metadata != (0.5,)


def test_ablation_freeze_phase():
    ctrl = CausalAblationController("freeze_phase")
    pkts = _sample_packets()
    abl = ctrl.apply_ablation_to_packets(pkts)
    assert len(abl) == 2
    assert abl[0].phase_metadata == (0.0,)
    assert abl[1].phase_metadata == (0.0,)


def test_route_causal_importance_and_matrix():
    routes = [(0, 1), (1, 2)]
    baseline = 10.0

    def mock_eval(route):
        if route == (0, 1):
            return 6.0  # lesion drops performance by 4.0
        return 9.5      # lesion drops performance by 0.5

    imp = evaluate_route_causal_importance(mock_eval, routes, baseline)
    assert imp[(0, 1)] == 4.0
    assert imp[(1, 2)] == 0.5

    mat = construct_route_importance_matrix(3, imp)
    assert mat.shape == (3, 3)
    assert mat[0, 1] == 4.0
    assert mat[1, 2] == 0.5
    assert mat[0, 0] == 0.0
