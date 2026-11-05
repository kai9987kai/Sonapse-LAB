import math
import pytest
import torch

from poseidon.mnemosyne import (
    MNEMOSYNE_MODES,
    EpisodicRecord,
    EpisodicStore,
    EpisodicStoreConfig,
    MnemosyneConfig,
    MnemosyneMemory,
    PatchState,
    SpatialCognitiveMap,
    WorkingMemory,
    _fingerprint,
)
from poseidon.world import N_ACTIONS, OBS_SIZE


def test_working_memory_update():
    wm = WorkingMemory(obs_size=OBS_SIZE, action_count=N_ACTIONS, hidden_size=32)
    h0 = wm.init_state(1)
    assert h0.shape == (1, 32)
    assert (h0 == 0.0).all()

    obs = torch.randn(1, OBS_SIZE)
    prev_act = torch.tensor([2], dtype=torch.long)
    surprise = torch.tensor([0.45], dtype=torch.float32)

    h1 = wm(obs, prev_act, surprise, h0)
    assert h1.shape == (1, 32)
    assert not torch.allclose(h0, h1)


def test_episodic_store_retrieval_and_modes():
    cfg = EpisodicStoreConfig(capacity=5)
    store = EpisodicStore(cfg)

    # Insert 4 dummy records
    for i in range(4):
        obs = [0.1 * i] * OBS_SIZE
        store.add(
            EpisodicRecord(
                step=i,
                observation=obs,
                action=i % N_ACTIONS,
                reward=float(i),
                next_observation=obs,
                surprise=0.1 * i,
                context=[0.0] * 16,
            )
        )

    assert len(store.records) == 4

    # Intact retrieval
    query = [0.3] * OBS_SIZE
    retrieved = store.retrieve(query, current_step=5, top_k=2, mode="intact")
    assert len(retrieved) == 2
    # Highest reward/surprise should be preferred
    assert retrieved[0]["step"] == 3

    # Shifted retrieval
    shifted = store.retrieve(query, current_step=5, top_k=2, mode="shifted")
    assert len(shifted) == 2

    # Random retrieval
    random_res = store.retrieve(query, current_step=5, top_k=2, mode="random_retrieval")
    assert len(random_res) == 2

    # Erased & memory_free retrieval
    assert store.retrieve(query, current_step=5, top_k=2, mode="erased") == []
    assert store.retrieve(query, current_step=5, top_k=2, mode="memory_free") == []

    # Capacity overflow test
    for i in range(10):
        store.add(
            EpisodicRecord(
                step=10 + i,
                observation=[0.5] * OBS_SIZE,
                action=0,
                reward=1.0,
                next_observation=[0.5] * OBS_SIZE,
                surprise=0.1,
                context=[0.0] * 16,
            )
        )
    assert len(store.records) == 5


def test_spatial_cognitive_map_bayesian_updates():
    spatial = SpatialCognitiveMap(lcb_z=1.5)
    obs1 = [0.0] * OBS_SIZE
    obs1[12] = 0.4  # terrain
    obs1[8] = 0.6   # shelter
    obs1[6] = 0.5   # food
    obs1[7] = 0.5   # water

    patch = spatial.register(obs1, current_tick=0)
    assert patch.visit_count == 1
    assert patch.revisit_count == 0

    # Simulate revisit after 10 ticks with replenished resources
    obs1_revisit = list(obs1)
    obs1_revisit[6] = 0.8  # food increased by 0.3 over 10 ticks -> rate 0.03
    obs1_revisit[7] = 0.7  # water increased by 0.2 over 10 ticks -> rate 0.02
    spatial.register(obs1_revisit, current_tick=10)

    assert patch.visit_count == 2
    assert patch.revisit_count == 1
    assert patch.food_lcb(current_tick=10) > 0.0
    assert patch.water_lcb(current_tick=10) > 0.0

    # Frontier detection & coverage
    frontiers = spatial.frontier_patches()
    assert len(frontiers) == 1
    assert spatial.topological_coverage(36) == 1 / 36
    assert spatial.best_replenished_patch(current_tick=10, need="food") == patch.patch_id


def test_mnemosyne_integrated_lesion_assays():
    for mode in MNEMOSYNE_MODES:
        config = MnemosyneConfig(hidden_size=32, mode=mode)
        mnemo = MnemosyneMemory(config)

        obs = [0.2] * OBS_SIZE
        step_out = mnemo.step_observation(obs, predicted_observation=[0.25] * OBS_SIZE)

        assert step_out["mode"] == mode
        assert "surprise" in step_out
        assert len(step_out["working_memory"]) == 32

        if mode in ("erased", "memory_free"):
            assert all(v == 0.0 for v in step_out["working_memory"])
            assert step_out["retrieved_episodes"] == []
        else:
            assert any(v != 0.0 for v in step_out["working_memory"])

        # Record transition
        mnemo.record_transition(
            observation=obs,
            action=1,
            reward=0.5,
            next_observation=[0.3] * OBS_SIZE,
        )
        assert mnemo.current_tick == 1
