"""Unit tests for Project NEUROGRAFT-II Engine (Stage 31).

Tests Successor Representations (SR-TD), Sharp-Wave Ripple (SWR) Reverse Replay,
Graph Laplacian Topological Spatial Representations, and Trainable Graph Circuit Routing.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from poseidon.neurograft_engine import (
    SuccessorRepresentationEngine,
    SharpWaveRippleReplay,
    GraphLaplacianTopologicalEngine,
    TrainableGraphCircuitRouter,
    NeurograftEngine,
    SCHEMA_ID,
)


def test_successor_representation_td_learning():
    num_states = 6
    num_actions = 4
    discount = 0.85
    lr = 0.20

    sr = SuccessorRepresentationEngine(
        num_states=num_states,
        num_actions=num_actions,
        discount=discount,
        learning_rate=lr,
    )

    # Initial state representation is identity matrix
    assert np.allclose(sr.M_state, np.eye(num_states))

    # Single transition s=1, a=2, next_s=2
    info = sr.update_sr_td(state=1, action=2, next_state=2, reward=1.0)
    assert info["state_td_error_norm"] > 0.0
    assert info["action_td_error_norm"] > 0.0

    # M_state[1, 2] should have increased
    assert sr.M_state[1, 2] > 0.0

    # Execute cyclic loop 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
    for _ in range(50):
        for s in range(num_states):
            next_s = (s + 1) % num_states
            sr.update_sr_td(state=s, action=0, next_state=next_s)

    # Successor state occupancy should decay with graph distance
    assert sr.M_state[0, 1] > sr.M_state[0, 3]


def test_successor_reward_revaluation_zero_shot():
    num_states = 5
    sr = SuccessorRepresentationEngine(
        num_states=num_states,
        num_actions=2,
        discount=0.8,
        learning_rate=0.25,
    )

    # Linear chain 0 -> 1 -> 2 -> 3 -> 4
    for _ in range(40):
        for s in range(num_states - 1):
            sr.update_sr_td(s, 0, s + 1)

    # Reward 1: goal at state 4
    r1 = [0.0, 0.0, 0.0, 0.0, 10.0]
    v1 = sr.revalue_state_values(r1)
    # State values should increase as we get closer to state 4
    assert v1[0] < v1[1] < v1[2] < v1[3] < v1[4]

    # Reward 2: goal shifted to state 1 (Zero-Shot Reward Revaluation!)
    r2 = [0.0, 10.0, 0.0, 0.0, 0.0]
    v2 = sr.revalue_state_values(r2)
    # State values should peak at state 1 immediately without retraining transitions!
    assert v2[1] > v2[3]
    assert v2[1] > v2[4]

    adapt = sr.evaluate_zero_shot_adaptation(r1, r2)
    assert adapt["zero_shot_revalued"] is True
    assert "v_old_mean" in adapt
    assert "v_new_mean" in adapt


def test_sharp_wave_ripple_reverse_replay():
    replay = SharpWaveRippleReplay(capacity=64)
    sr = SuccessorRepresentationEngine(num_states=8, num_actions=2, discount=0.85)

    # Push episodic transitions: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 (goal with high reward & surprise)
    for t in range(7):
        rew = 10.0 if t == 6 else 0.0
        surp = 2.0 if t == 6 else 0.1
        replay.push(t, 0, t + 1, rew, surp)

    assert len(replay) == 7

    # Trigger SWR sweeps
    swr_result = replay.trigger_swr_reverse_replay(sr, num_sweeps=4)
    assert swr_result["sweeps_executed"] == 4
    assert swr_result["total_updates"] > 0
    assert swr_result["swr_burst_active"] is True
    assert swr_result["replay_direction"] == "reverse_salient"


def test_graph_laplacian_topological_engine():
    # Canonical spatial graph
    engine = GraphLaplacianTopologicalEngine.create_canonical_spatial_graph(num_nodes=6)
    spec = engine.compute_laplacian_spectrum()

    assert spec["num_nodes"] == 6
    assert len(spec["eigenvalues"]) == 6
    # Smallest eigenvalue of normalized Laplacian is 0
    assert np.isclose(spec["eigenvalues"][0], 0.0, atol=1e-5)
    # Fiedler eigenvalue lambda_1 > 0 for connected graph
    assert spec["fiedler_value"] > 0.0
    assert spec["algebraic_connectivity_positive"] is True
    assert len(spec["fiedler_vector"]) == 6
    assert spec["mean_diffusion_distance"] > 0.0


def test_trainable_graph_circuit_router():
    torch.manual_seed(42)
    num_sources = 4
    hidden_dim = 16
    top_k = 2

    router = TrainableGraphCircuitRouter(
        num_sources=num_sources,
        hidden_dim=hidden_dim,
        top_k=top_k,
        null_route_bias=0.5,
    )

    inputs = torch.randn(num_sources, hidden_dim)
    out_states, hidden_states, telem = router(inputs)

    assert out_states.shape == (num_sources, hidden_dim)
    assert hidden_states.shape == (num_sources, hidden_dim)
    assert telem["routing_entropy"] > 0.0
    assert 0.0 <= telem["mean_null_weight"] <= 1.0
    assert 0.0 <= telem["mean_gate"] <= 1.0
    assert telem["num_parameters"] > 0

    # Verify gradients flow through router
    loss = out_states.sum()
    loss.backward()
    assert router.q_proj.weight.grad is not None
    assert router.null_logit.grad is not None


def test_neurograft_engine_synthesis_and_receipt():
    engine = NeurograftEngine(num_states=12, num_actions=4, discount=0.90, learning_rate=0.15)
    res = engine.synthesize(seed=42, num_simulation_steps=24)

    assert res["project"] == "NEUROGRAFT"
    assert res["schema"] == SCHEMA_ID
    assert res["verified"] is True
    assert len(res["receipt_sha256"]) == 64
    assert res["sha256"] == res["receipt_sha256"]

    sr = res["successor_representation"]
    assert sr["num_states"] == 12
    assert sr["num_actions"] == 4
    assert sr["M_state_norm"] > 0.0

    swr = res["sharp_wave_ripple_replay"]
    assert swr["sweeps_executed"] == 6
    assert swr["total_updates"] > 0
    assert swr["swr_burst_active"] is True

    lap = res["graph_laplacian_topological"]
    assert lap["algebraic_connectivity_positive"] is True
    assert lap["fiedler_value"] > 0.0

    circuit = res["trainable_graph_circuit"]
    assert circuit["routing_entropy"] > 0.0
    assert circuit["num_parameters"] > 0
