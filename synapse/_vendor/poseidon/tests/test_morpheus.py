"""Tests for Morpheus Offline Oneiric Sleep Engine & Memory Consolidation."""
import pytest
from poseidon.causeway import CausewaySuperpositionEngine
from poseidon.hologram import HolographicAssociativeMemory
from poseidon.lattice import DiamondLatticeNeuropil
from poseidon.morpheus import MorpheusDreamEngine, SleepState


def test_morpheus_initialization_and_reset():
    engine = MorpheusDreamEngine(buffer_capacity=32)
    assert engine.current_state == SleepState.WAKE
    assert engine.total_sleep_cycles == 0
    assert len(engine.memory_buffer) == 0

    obs = [0.5] * 16
    engine.record_waking_step(0, obs, 2, 0.8, obs, [0.0, 0.0, 0.0])
    assert len(engine.memory_buffer) == 1

    engine.reset()
    assert len(engine.memory_buffer) == 0
    assert engine.total_sleep_cycles == 0


def test_morpheus_sleep_onset_detection():
    engine = MorpheusDreamEngine()
    # Waking state with high stamina and non-rest action (action=1: forage)
    assert not engine.is_sleep_indicated(action=1, stamina=0.8)

    # Triggered by rest action (action=0: rest)
    assert engine.is_sleep_indicated(action=0, stamina=0.8)

    # Triggered by exhaustion
    assert engine.is_sleep_indicated(action=1, stamina=0.10)

    # Triggered by consecutive rests
    assert engine.is_sleep_indicated(action=1, stamina=0.5, consecutive_rests=2)


def test_morpheus_sws_and_rem_cycles():
    engine = MorpheusDreamEngine(buffer_capacity=16, rem_replay_depth=4)
    causeway = CausewaySuperpositionEngine(num_actions=6)
    lattice = DiamondLatticeNeuropil()
    hologram = HolographicAssociativeMemory(dimension=64)

    # Populate memory buffer with salient transitions
    for step in range(8):
        obs = [0.2 + 0.05 * step] * 16
        reward = 0.5 if step % 2 == 0 else 0.0
        engine.record_waking_step(step, obs, step % 6, reward, obs, [0.1, 0.2, 0.3], predicted_reward=0.1)

    initial_phases = list(causeway.phases)

    # Step 1-2 in sleep: should be SWS
    t1 = engine.process_cycle(causeway, lattice, hologram, action=0, stamina=0.10)
    assert t1["sleep_state"] == SleepState.SWS.value
    assert t1["sleep_tick"] == 1

    t2 = engine.process_cycle(causeway, lattice, hologram, action=0, stamina=0.10)
    assert t2["sleep_state"] == SleepState.SWS.value
    assert t2["sleep_tick"] == 2

    # Step 3 in sleep: should transition to REM
    t3 = engine.process_cycle(causeway, lattice, hologram, action=0, stamina=0.10)
    assert t3["sleep_state"] == SleepState.REM.value
    assert t3["sleep_tick"] == 3
    assert t3["rem_replays"] >= 1
