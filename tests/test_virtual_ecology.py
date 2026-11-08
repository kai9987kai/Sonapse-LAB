"""Tests for SONAPSE-EVO Virtual Ecology & Multi-Agent Biome."""
import math
import numpy as np
import pytest

from synapse.virtual_ecology import (
    VirtualEcologyBiome,
    EcologyAgentState,
    EcologicalResource,
    VisualEcologySnapshot,
)


def test_virtual_ecology_initialization():
    biome = VirtualEcologyBiome(
        biome_half=10.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=1,
        seed=101,
    )
    assert biome.biome_half == 10.0
    assert len(biome.agents) == 9
    assert len(biome.resources) >= 16

    niches = [a.niche for a in biome.agents]
    assert niches.count("forager") == 4
    assert niches.count("sentinel") == 2
    assert niches.count("artisan") == 2
    assert niches.count("predator") == 1

    for a in biome.agents:
        assert a.alive is True
        assert 0.0 <= a.energy <= 100.0
        assert abs(a.x) <= biome.biome_half
        assert abs(a.y) <= biome.biome_half


def test_virtual_ecology_kinematics_and_resource_consumption():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=3,
        num_sentinels=1,
        num_artisans=1,
        num_predators=1,
        seed=202,
    )
    initial_step = biome.step_count
    snap0 = biome.step()
    assert biome.step_count == initial_step + 1
    assert isinstance(snap0, VisualEcologySnapshot)
    assert snap0.population_count == len(biome.agents)
    assert 0.0 <= snap0.swarm_coherence <= 1.0
    assert snap0.biomass_energy > 0.0

    # Place a forager right on a flora bloom to test feeding
    forager = next(a for a in biome.agents if a.niche == "forager")
    food = next(r for r in biome.resources if r.kind == "flora_bloom")
    forager.x = food.x
    forager.y = food.y
    forager.energy = 50.0
    food.amount = 0.8

    snap1 = biome.step()
    assert forager.energy > 50.0  # Consumed food and gained energy
    assert food.amount < 0.8      # Food amount decreased


def test_virtual_ecology_predator_evasion_and_pulses():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=2,
        num_sentinels=1,
        num_artisans=0,
        num_predators=1,
        seed=303,
    )
    predator = next(a for a in biome.agents if a.niche == "predator")
    forager = next(a for a in biome.agents if a.niche == "forager")

    # Place predator near forager to trigger evasion
    predator.x = 0.0
    predator.y = 0.0
    forager.x = 1.0
    forager.y = 0.0
    forager.adrenaline = 0.1

    snap = biome.step()
    # Evasion should increase adrenaline and arousal
    assert forager.arousal >= 0.2
    assert forager.adrenaline > 0.1


def test_virtual_ecology_external_stimulus():
    biome = VirtualEcologyBiome(
        biome_half=10.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=1,
        seed=404,
    )
    # 1. Trigger alarm pulse
    snap_alarm = biome.step(external_stimulus={"type": "broadcast_alarm", "x": 0.0, "y": 0.0})
    assert len(snap_alarm.active_pulses) >= 1
    pulse = snap_alarm.active_pulses[-1]
    assert pulse["x"] == 0.0
    assert pulse["y"] == 0.0

    # 2. Induce sleep cycle
    snap_sleep = biome.step(external_stimulus={"type": "induce_sleep_cycle"})
    assert any(a["sleep_state"] == "sws_sleep" for a in snap_sleep.agents)

    # 3. Seed resources
    min_flora_before = min(r.amount for r in biome.resources)
    biome.step(external_stimulus={"type": "seed_resources"})
    min_flora_after = min(r.amount for r in biome.resources)
    assert min_flora_after >= min_flora_before
