"""Tests for Multi-Scale Sleep, SWR Replay & Synaptic Homeostasis (SHY)."""
import numpy as np
import pytest
import torch

from synapse.sleep_replay.consolidation_engine import (
    MemoryConsolidationAndSHYEngine,
    ConsolidationReceipt,
    ReplayEpisode,
)


def test_consolidation_engine_initialization():
    engine = MemoryConsolidationAndSHYEngine(
        gamma_shy=0.08,
        swr_compression_factor=10.0,
        dream_branching_factor=3,
        seed=42,
    )
    assert engine.gamma_shy == 0.08
    assert engine.swr_compression == 10.0
    assert engine.branching == 3
    assert len(engine.episode_buffer) == 0
    assert len(engine.consolidation_history) == 0


def test_shy_synaptic_downscaling_proportionality():
    engine = MemoryConsolidationAndSHYEngine(gamma_shy=0.10, seed=42)
    # Random synaptic weight tensor with mixed positive/negative values
    weights = torch.randn(16, 16) * 0.5
    pre_norm = float(torch.norm(weights).item())

    scaled_w, pre_n, post_n = engine.execute_shy_synaptic_downscaling(weights)
    assert post_n < pre_n  # Norm is downscaled
    assert scaled_w.shape == weights.shape

    # Dale's law / sign preservation: sign(scaled_w) == sign(weights)
    assert torch.all(torch.sign(scaled_w) == torch.sign(weights))

    # Strong synapses are compressed proportionally more than near-zero noise
    pct_down = (pre_n - post_n) / pre_n * 100.0
    assert 2.0 <= pct_down <= 25.0


def test_episode_recording_and_swr_replay():
    engine = MemoryConsolidationAndSHYEngine(seed=88)

    # Record 3 mock episodes with varying surprises
    for i in range(3):
        obs = [np.random.randn(10).astype(np.float32) for _ in range(8)]
        acts = [np.random.randn(3).astype(np.float32) for _ in range(8)]
        rews = [float(0.1 * j) for j in range(8)]
        surp = [float(0.05 * j * (i + 1)) for j in range(8)]
        ep_id = engine.record_episode(obs, acts, rews, surp)
        assert ep_id.startswith("ep_")

    assert len(engine.episode_buffer) == 3

    # Execute 10x-accelerated SWR playback
    visited = []
    def dummy_step(o, a):
        visited.append(o)

    replayed = engine.execute_swr_replay(connectome_step_fn=dummy_step, max_episodes=2)
    assert replayed > 0
    assert len(visited) == replayed


def test_counterfactual_dreaming_and_full_consolidation():
    engine = MemoryConsolidationAndSHYEngine(seed=99)
    obs = [np.random.randn(10).astype(np.float32) for _ in range(6)]
    acts = [np.random.randn(3).astype(np.float32) for _ in range(6)]
    rews = [0.1, 0.2, 0.8, 0.4, 0.1, 0.0]
    engine.record_episode(obs, acts, rews)

    # Counterfactual dreams
    dreams = engine.synthesize_counterfactual_dreams(num_dreams=4)
    assert len(dreams) == 4
    for d in dreams:
        assert "dream_id" in d
        assert "simulated_reward" in d
        assert "novelty" in d

    # Full consolidation cycle
    weights = torch.randn(8, 8) * 0.4
    scaled_w, receipt = engine.run_full_consolidation_cycle(weights)
    assert isinstance(receipt, ConsolidationReceipt)
    assert receipt.percent_downscaled > 0.0
    assert receipt.replayed_transitions >= 0
    assert receipt.synthetic_dreams_generated == 4

    summary = engine.get_summary()
    assert summary["buffered_episodes"] == 1
    assert summary["completed_consolidation_cycles"] == 1
    assert summary["last_receipt"] is not None
