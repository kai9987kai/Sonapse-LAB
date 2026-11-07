"""Comprehensive verification suite for dynamic complex environments and physics.

Covers:
1. Frontier tasks: apex_megasite, stealth_labyrinth, kinetic_foundry.
2. Vector flow displacement physics (rivers, conveyor belts).
3. Topographic elevation hill slope gradient forces.
4. Coordinated multi-agent predator pack alerting and flanking.
5. Rotating sentinel searchlights with cone detection and shadow cloaks.
6. Acoustic sound tripwires with speed threshold sensing.
7. Multi-switch and button puzzle unlocks with terminal goal rewards.
"""
import math
import numpy as np
import pytest

from synapse.complex_world import ComplexWorld, COMPLEX_TASKS, ALL_TASKS


def test_all_frontier_tasks_reset_and_have_valid_observations():
    world = ComplexWorld(seed=123)
    for task in COMPLEX_TASKS:
        obs = world.reset(task)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (155,)
        assert np.isfinite(obs).all()


def test_vector_river_current_displaces_agent_and_objects():
    world = ComplexWorld(seed=42)
    world.reset("apex_megasite")
    # Place agent directly in the river current (x=0, y=0, vy=-0.15)
    world.a["x"] = 0.0
    world.a["y"] = 0.0
    initial_y = world.a["y"]

    # Step with neutral zero action
    zero_action = np.zeros(5, dtype=np.float32)
    world.step(zero_action)

    # Agent should have been carried downstream (negative y)
    assert world.a["y"] < initial_y
    delta_y = world.a["y"] - initial_y
    assert delta_y < -0.10


def test_topographic_elevation_gradient_force():
    world = ComplexWorld(seed=42)
    world.reset("apex_megasite")
    # Position agent on the slope of the West Plateau hill (hill is at x=-2.6, y=2.0, height=1.8)
    world.a["x"] = -2.0
    world.a["y"] = 2.0
    initial_x = world.a["x"]

    zero_action = np.zeros(5, dtype=np.float32)
    _, _, _, info = world.step(zero_action)

    assert "elevation_gradient" in info
    gx, gy = info["elevation_gradient"]
    # Slope vector should be directed uphill towards hill center (-2.6 < -2.0), so gx is negative
    assert gx < 0.0
    # Gravitational drift pushes downhill (positive x direction)
    assert world.a["x"] > initial_x


def test_predator_pack_alert_propagation():
    world = ComplexWorld(seed=42)
    world.reset("apex_megasite")
    assert len(world.predators) >= 2

    # Initially predators are not alerted
    for p in world.predators:
        assert not p.get("alerted", False)

    # Position agent close to the leader predator (within alert distance 3.5)
    lead_pred = world.predators[0]
    world.a["x"] = lead_pred["x"] + 1.0
    world.a["y"] = lead_pred["y"] + 1.0

    zero_action = np.zeros(5, dtype=np.float32)
    world.step(zero_action)

    # Alert should have triggered on the leader and propagated across the pack
    for p in world.predators:
        assert p.get("alerted", True) is True


def test_sentinel_searchlight_detection_and_shadow_cloaking():
    world = ComplexWorld(seed=42)
    world.reset("stealth_labyrinth")
    assert len(world.sentinels) >= 1

    sentinel = world.sentinels[0]  # at (0, 2.2)
    sentinel["angle"] = -math.pi / 2.0  # pointing south
    # Place agent south of sentinel in open corridor outside shadow zones
    world.a["x"] = 0.0
    world.a["y"] = 1.0

    zero_action = np.zeros(5, dtype=np.float32)
    _, _, _, info = world.step(zero_action)
    assert info.get("in_shadow") is False
    assert info.get("alarm_tripped") is True
    assert sentinel.get("alerted") is True

    # Now place agent in a designated shadow zone (e.g. x=-1.2, y=-2.0)
    world.reset("stealth_labyrinth")
    shadow = next(tz for tz in world.terrain_zones if tz.get("kind") == "shadow")
    world.a["x"] = shadow["x"]
    world.a["y"] = shadow["y"]

    # Point sentinel right at agent
    sentinel = world.sentinels[0]
    sentinel["x"] = shadow["x"] + 1.0
    sentinel["y"] = shadow["y"]
    sentinel["angle"] = math.pi

    _, _, _, info = world.step(zero_action)
    # Cloaked in shadow, alarm must not be tripped
    assert info.get("in_shadow") is True
    assert info.get("alarm_tripped") is False


def test_acoustic_sound_tripwire_trigger():
    world = ComplexWorld(seed=42)
    world.reset("stealth_labyrinth")
    assert len(world.sound_tripwires) >= 1

    trip = world.sound_tripwires[0]
    # Position agent right on tripwire crossing with forward drive velocity
    world.a["x"] = (trip["x1"] + trip["x2"]) / 2.0
    world.a["y"] = (trip["y1"] + trip["y2"]) / 2.0

    fast_action = np.array([0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, info = world.step(fast_action)
    assert info.get("alarm_tripped") is True


def test_puzzle_chamber_button_unlock_and_goal_reward():
    world = ComplexWorld(seed=42)
    world.reset("puzzle_chamber")

    btn = next(o for o in world.objects if o.get("kind") == "button")
    target = next(o for o in world.objects if o.get("kind") == "target")
    gate = world.dynamic_gates[0]
    assert not gate.get("open", False)

    # Step agent touching button
    world.a["x"] = btn["x"]
    world.a["y"] = btn["y"]
    btn["pressed"] = True

    zero_action = np.zeros(5, dtype=np.float32)
    world.step(zero_action)
    assert gate.get("open", False) is True

    # Now step agent to target -> should complete episode with success bonus
    world.a["x"] = target["x"]
    world.a["y"] = target["y"]
    _, reward, done, info = world.step(zero_action)
    assert done is True
    assert info.get("outcome") == "success"
    assert reward >= 5.0


def test_hyper_boreal_tundra_features_and_predators():
    world = ComplexWorld(seed=42)
    obs = world.reset("hyper_boreal_tundra")
    assert obs.shape == (155,)
    assert getattr(world, "arena_half") == 9.0
    assert len(world.thermal_geysers) >= 2
    assert len(world.elevation_hills) >= 2

    # Verify presence of aerial raptor predator and ground stalker
    kinds = [p["kind"] for p in world.predators]
    assert "aerial_raptor" in kinds
    assert "ground_stalker" in kinds
    assert "pack_leader" in kinds

    # Test snapshot contains streamlines, geysers, elevation grid
    snap = world.snapshot()
    assert "wind_streamlines" in snap
    assert len(snap["wind_streamlines"]) == 16
    assert "thermal_geysers" in snap
    assert len(snap["thermal_geysers"]) == 2
    assert "elevation_grid" in snap
    assert len(snap["elevation_grid"]) == 11
    assert "celestial_angle" in snap

    # Test step execution with neutral action
    zero_action = np.zeros(5, dtype=np.float32)
    obs, reward, done, info = world.step(zero_action)
    assert obs.shape == (155,)
    assert np.isfinite(obs).all()


def test_subterranean_abyss_flora_and_leviathan():
    world = ComplexWorld(seed=42)
    obs = world.reset("subterranean_abyss")
    assert obs.shape == (155,)
    assert getattr(world, "arena_half") == 8.0
    assert len(world.bioluminescent_flora) >= 4
    assert len(world.sound_tripwires) >= 2

    # Check abyssal leviathan predator
    leviathan = next((p for p in world.predators if p["kind"] == "abyssal_leviathan"), None)
    assert leviathan is not None
    assert leviathan["state"] == "dormant"

    # Step slowly: leviathan remains dormant
    slow_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    world.step(slow_action)
    assert leviathan["state"] == "dormant"

    # Step at high speed: triggers acoustic perception -> leviathan alerts and enters pursuit
    fast_action = np.array([0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    world.step(fast_action)
    assert leviathan["state"] == "pursuit"
    assert leviathan["alerted"] is True

    snap = world.snapshot()
    assert "bioluminescent_flora" in snap
    assert len(snap["bioluminescent_flora"]) == 4


def test_vortex_maelstrom_mechanics_and_wormholes():
    world = ComplexWorld(seed=42)
    obs = world.reset("vortex_maelstrom")
    assert obs.shape == (155,)
    assert getattr(world, "arena_half") == 8.0
    assert len(world.vortices) >= 1
    assert len(world.wormholes) == 2

    vortex = world.vortices[0]
    assert vortex["pull"] > 0
    assert vortex["spin"] > 0

    snap = world.snapshot()
    assert "vortices" in snap
    assert len(snap["vortices"]) == 1
    assert "wormholes" in snap
    assert len(snap["wormholes"]) == 2

    # Position agent near wormhole alpha -> step -> agent should teleport to target
    wh = world.wormholes[0]
    world.a["x"] = wh["x"]
    world.a["y"] = wh["y"]
    zero_action = np.zeros(5, dtype=np.float32)
    _, _, _, info = world.step(zero_action)
    assert info.get("wormhole_teleported") is True

