import math
import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.alias_world import ObservationAliasTidePool
from poseidon.research.decision_value_memory import (
    DECISION_MEMORY_MODES,
    DecisionValueAgent,
    DecisionValueMemory,
    DecisionValueMemoryConfig,
    EpisodicEntry,
    fit_decision_value_memory,
)
from poseidon.world import OBS_SIZE


def test_episodic_entry_validation():
    # Valid entry
    entry = EpisodicEntry(
        tick=0,
        observation=tuple([0.5] * OBS_SIZE),
        action=1,
        reward=0.8,
        next_observation=tuple([0.6] * OBS_SIZE),
        residual=tuple([0.0] * OBS_SIZE),
    )
    entry.validate()

    # Invalid action
    with pytest.raises(ValueError, match="action"):
        EpisodicEntry(
            tick=0,
            observation=tuple([0.5] * OBS_SIZE),
            action=9,
            reward=0.8,
            next_observation=tuple([0.6] * OBS_SIZE),
            residual=tuple([0.0] * OBS_SIZE),
        ).validate()

    # Non-finite reward
    with pytest.raises(ValueError, match="reward"):
        EpisodicEntry(
            tick=0,
            observation=tuple([0.5] * OBS_SIZE),
            action=1,
            reward=float("nan"),
            next_observation=tuple([0.6] * OBS_SIZE),
            residual=tuple([0.0] * OBS_SIZE),
        ).validate()


def test_config_validation():
    with pytest.raises(ValueError, match="mode"):
        DecisionValueMemoryConfig(mode="unsupported_mode")

    with pytest.raises(ValueError, match="capacity"):
        DecisionValueMemoryConfig(capacity=0)


def test_empty_or_erased_memory_gate_is_strictly_closed():
    for mode in ("erased", "baseline"):
        mem = DecisionValueMemory(DecisionValueMemoryConfig(mode=mode))
        # Add some records
        mem.add_entry([0.5] * OBS_SIZE, 1, 0.8, [0.5] * OBS_SIZE)
        decision = mem.query([0.5] * OBS_SIZE)
        assert decision.admission_gate_open is False
        assert all(d == 0.0 for d in decision.gated_deltas)
        assert decision.retrieved_count == 0


def test_distance_support_and_conflict_gating():
    mem = DecisionValueMemory(DecisionValueMemoryConfig(
        mode="intact",
        max_candidate_distance=0.10,  # Strict distance threshold
        conflict_threshold=0.01,
    ))

    # Add an entry that is very far from the query
    mem.add_entry([0.0] * OBS_SIZE, 1, 0.8, [0.0] * OBS_SIZE)
    # Query with ones vector: distance is sqrt(16)/sqrt(16) = 1.0 > 0.10
    decision = mem.query([1.0] * OBS_SIZE)
    # Gate must be closed because candidate is unsupported
    assert decision.admitted_actions[1] is False
    assert decision.gated_deltas[1] == 0.0

    # Test conflict: add two entries for same action with diverging rewards
    mem_conflict = DecisionValueMemory(DecisionValueMemoryConfig(
        mode="intact",
        max_candidate_distance=1.0,
        conflict_threshold=0.01,
    ))
    mem_conflict.add_entry([0.5] * OBS_SIZE, 1, 1.0, [0.5] * OBS_SIZE)
    mem_conflict.add_entry([0.5] * OBS_SIZE, 1, -1.0, [0.5] * OBS_SIZE)
    dec_conflict = mem_conflict.query([0.5] * OBS_SIZE)
    # Conflict score must be positive and gate must close
    assert dec_conflict.conflict_score > 0.01
    assert dec_conflict.admission_gate_open is False


def test_all_six_lesion_modes():
    env = ObservationAliasTidePool(seed=123, max_steps=10)
    core = CoreRuntime(active_core_path())

    for mode in DECISION_MEMORY_MODES:
        agent = DecisionValueAgent(core=core, mode=mode)
        if mode == "donor":
            # Provide donor pool
            agent.memory.set_donor_pool([
                EpisodicEntry(0, tuple([0.5]*OBS_SIZE), 1, 0.8, tuple([0.5]*OBS_SIZE), tuple([0.0]*OBS_SIZE)),
            ])
        res = agent.rollout(env, max_steps=5)
        assert res["mode"] == mode
        assert res["steps"] > 0
        assert "total_reward" in res
        assert "trajectory" in res


def test_serialization_and_deserialization():
    mem = DecisionValueMemory(DecisionValueMemoryConfig(hidden_dim=32, belief_dim=16))
    state = mem.state_dict()
    assert state["schema"] == "poseidon-decision-value-memory-v1"

    mem_loaded = DecisionValueMemory(DecisionValueMemoryConfig(hidden_dim=32, belief_dim=16))
    mem_loaded.load_state_dict(state)

    # Check identical parameters
    for p1, p2 in zip(mem.scorer.parameters(), mem_loaded.scorer.parameters()):
        assert torch.equal(p1, p2)
