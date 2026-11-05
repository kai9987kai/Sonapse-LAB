"""Unit tests for SuccessorMemoryAdapter & TidePool Discretization (Stage 31).

Tests the coupling between Poseidon's Mnemosyne memory and NEUROGRAFT's Successor Representation.
"""
from __future__ import annotations

import numpy as np
import pytest

from poseidon.mnemosyne import MnemosyneConfig, MnemosyneMemory
from poseidon.research.successor_memory_adapter import (
    SuccessorMemoryAdapter,
    discretize_obs_to_spatial_state,
)
from poseidon.world import ACTIONS, OBSERVATION_NAMES


def test_discretize_obs_to_spatial_state():
    # 16-D continuous TidePool vector
    obs = np.zeros(16, dtype=np.float32)
    s0 = discretize_obs_to_spatial_state(obs, num_spatial_states=16)
    assert 0 <= s0 < 16

    obs[12] = 0.9  # High terrain difficulty
    s_high_terrain = discretize_obs_to_spatial_state(obs, num_spatial_states=16)
    assert 0 <= s_high_terrain < 16

    obs[8] = 0.8  # High shelter
    s_shelter = discretize_obs_to_spatial_state(obs, num_spatial_states=16)
    assert 0 <= s_shelter < 16


def test_successor_spatial_state_uses_local_patch_features_not_energy():
    observation = [0.0] * 16
    state = discretize_obs_to_spatial_state(observation)
    observation[OBSERVATION_NAMES.index("energy")] = 1.0
    assert discretize_obs_to_spatial_state(observation) == state
    observation[OBSERVATION_NAMES.index("resource_scent")] = 1.0
    assert discretize_obs_to_spatial_state(observation) == 5


@pytest.mark.parametrize("action_name, surprise, expected_replay", [
    ("rest", 0.0, True), ("flee", 0.0, False), ("forage", 0.8, True),
])
def test_successor_replay_triggers_on_rest_or_surprise(action_name, surprise, expected_replay):
    adapter = SuccessorMemoryAdapter()
    observation = [0.5] * 16
    result = adapter.record_step(observation, ACTIONS.index(action_name), observation,
                                 surprise=surprise)
    assert result["swr_triggered"] is expected_replay
    assert adapter.total_swr_triggers == int(expected_replay)


def test_successor_memory_adapter_workflow():
    cfg = MnemosyneConfig(episodic_capacity=32)
    memory = MnemosyneMemory(config=cfg)

    adapter = SuccessorMemoryAdapter(
        memory=memory,
        num_spatial_states=16,
        num_actions=6,
        discount=0.85,
        learning_rate=0.20,
        surprise_threshold=0.5,
    )

    assert adapter.num_spatial_states == 16
    assert adapter.num_actions == 6

    # Simulate steps
    for step in range(20):
        obs = np.random.uniform(0.1, 0.9, size=(16,)).astype(np.float32)
        next_obs = np.random.uniform(0.1, 0.9, size=(16,)).astype(np.float32)
        action = step % 6
        reward = 5.0 if step == 15 else 0.0

        res = adapter.record_step(obs, action, next_obs, reward)
        assert "spatial_state" in res
        assert "next_spatial_state" in res
        assert "sr_td_error" in res

    # SWR replay should have been triggered or can be explicitly triggered
    swr = adapter.trigger_offline_swr_consolidation(num_sweeps=4)
    assert swr["sweeps_executed"] == 4
    assert swr["total_updates"] >= 0

    # Test action preferences from SR
    test_obs = np.ones(16, dtype=np.float32) * 0.5
    pref = adapter.get_action_preferences(test_obs)
    assert len(pref) == 6
    assert isinstance(pref, list)

    # Test zero-shot goal revaluation audit
    audit = adapter.audit_zero_shot_goal_revaluation(
        old_goal_state=2,
        new_goal_state=14,
    )
    assert audit["zero_shot_revalued"] is True
    assert "old_values" in audit
    assert "new_values" in audit
    assert len(audit["old_values"]) == 16
    assert len(audit["new_values"]) == 16
