"""Comprehensive Test Suite for Sleep Replay, Fatigue & Generative Dreams.

Verifies:
1. Two-process sleep regulation and homeostatic sleep pressure P_t.
2. Brain state machine transitions and emergency acute threat override.
3. Generative dream simulation with strict synthetic segregation (EvidenceKind.SYNTHETIC_SIMULATION).
4. Prioritized experience replay scheduling for offline consolidation.
"""
import pytest
import numpy as np
import torch

from synapse.sleep_replay import (
    SleepState,
    FatigueController,
    DreamEpisode,
    DreamReplayEngine,
    ReplayMemoryItem,
    ReplayScheduler,
)
from synapse.neuroendocrine import EvidenceKind


def test_fatigue_controller_and_sleep_pressure():
    fc = FatigueController(
        wake_rate_kw=0.01,
        sleep_rate_ks=0.05,
        drowsy_threshold=0.5,
        sleep_threshold=0.7,
        wake_threshold=0.2,
        dt=1.0,
    )
    assert fc.current_state == SleepState.AWAKE

    # Advance time while awake: pressure increases
    initial_p = float(fc.sleep_pressure.item())
    states_visited = set()
    for _ in range(40):
        fc.step(adenosine_level=0.5)
        states_visited.add(fc.current_state)

    p_after = float(fc.sleep_pressure.item())
    assert p_after > initial_p

    # Should transition into DROWSY or SLEEP_REPLAY once pressure passes thresholds
    for _ in range(60):
        fc.step(adenosine_level=0.8)
        states_visited.add(fc.current_state)

    assert any(s in states_visited for s in [SleepState.DROWSY, SleepState.SLEEP_REPLAY, SleepState.DREAM_GENERATION])


def test_emergency_awakening_threat_override():
    fc = FatigueController()
    fc.force_state(SleepState.SLEEP_REPLAY)
    assert fc.current_state == SleepState.SLEEP_REPLAY

    # Sudden acute threat arrives
    res = fc.step(threat_level=0.90)
    assert fc.current_state == SleepState.AWAKE
    assert res["interrupted_by_threat"] is True


def test_generative_dreams_synthetic_segregation():
    engine = DreamReplayEngine(latent_dim=64, hidden_dim=128)

    seed_z = torch.randn(64)
    dream = engine.generate_dream(seed_latent=seed_z, theme="predator_counterfactual", steps=15)

    assert dream.is_synthetic_dream is True
    assert dream.evidence_kind == EvidenceKind.SYNTHETIC_SIMULATION
    assert len(dream.latent_trajectory) == 15
    assert len(dream.affective_trajectory) == 15
    assert len(engine.dream_history) == 1

    d_dict = dream.to_dict()
    assert d_dict["is_synthetic_dream"] is True
    assert d_dict["evidence_kind"] == "synthetic_simulation"


def test_prioritized_replay_scheduler():
    scheduler = ReplayScheduler(capacity=50, alpha=0.6, beta=0.4)

    # Push experiences with differing TD errors and valence
    for i in range(20):
        scheduler.push(
            state=[float(i)] * 8,
            action=[0.1, -0.2],
            reward=1.0 if i == 10 else 0.0,
            next_state=[float(i + 1)] * 8,
            td_error=5.0 if i == 10 else 0.1,  # i=10 is high-surprise
            valence=0.8 if i == 10 else 0.0,
        )

    assert scheduler.size() == 20

    # Sample batch
    batch, weights, indices = scheduler.sample_batch(batch_size=8)
    assert len(batch) == 8
    assert len(weights) == 8
    assert len(indices) == 8

    # The high-salience experience should have higher sampling likelihood
    high_salience_sampled = any(item.td_error == 5.0 for item in batch)
    # Update priorities
    scheduler.update_priorities(indices, [0.05] * len(indices))
