import copy
import hashlib
import json
from pathlib import Path
import random

import pytest
import torch

from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.horizon import HorizonAtlas, HorizonConfig
from poseidon.odyssey import CognitiveMap, CognitivePatch, OdysseyAtlas, OdysseyConfig
from poseidon.world import TidePool


@pytest.fixture
def core(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    return CoreRuntime(path)


def fit_small_horizon(core):
    return HorizonAtlas.fit(
        core,
        train_seeds=[1011, 1012],
        calibration_seeds=[2011, 2012],
        anchors_per_episode=3,
        max_steps=6,
        config=HorizonConfig(horizon=4, scarcity_levels=(1.0,)),
    )


def fit_small_odyssey(core, **settings):
    return OdysseyAtlas.fit(
        core,
        train_seeds=[1011, 1012],
        calibration_seeds=[2011, 2012],
        anchors_per_episode=3,
        max_steps=6,
        config=OdysseyConfig(horizon=4, scarcity_levels=(1.0,), **settings),
    )


def test_cognitive_patch_replenishment():
    patch = CognitivePatch(
        patch_id="p1",
        terrain=0.5,
        shelter=0.8,
        first_seen_tick=10,
        last_seen_tick=10,
        visit_count=1,
        observed_food=0.0,
        observed_water=0.0,
        food_cap=0.4,
        water_cap=0.5,
        last_threat=0.1,
    )
    # At tick 10 (0 elapsed), food and water are 0
    assert patch.estimated_food(10, scarcity=2.0) == 0.0
    assert patch.estimated_water(10, scarcity=2.0) == 0.0

    # At tick 30 (20 elapsed):
    # food += 20 * (0.007 / 2.0) = 20 * 0.0035 = 0.07
    # water += 20 * (0.014 / 2.0) = 20 * 0.007 = 0.14
    assert pytest.approx(patch.estimated_food(30, scarcity=2.0), 1e-4) == 0.07
    assert pytest.approx(patch.estimated_water(30, scarcity=2.0), 1e-4) == 0.14

    # Caps apply
    assert patch.estimated_food(1000, scarcity=2.0) == 0.4
    assert patch.estimated_water(1000, scarcity=2.0) == 0.5


def test_cognitive_map_topology_and_waypoints():
    cmap = CognitiveMap(scarcity=2.0)
    # Step 0: patch A
    obs_a = [1.0, 0.4, 0.4, 0.8, 0.1, 0.0, 0.5, 0.6, 0.7, 0.1, 0.5, 0.5, 0.11, 0.4, 0.0, 0.0]
    p_a = cmap.update(obs_a, tick=1)
    cmap.record_action(4)  # explore

    # Step 1: patch B
    obs_b = [1.0, 0.38, 0.38, 0.7, 0.1, 0.0, 0.0, 0.0, 0.3, 0.1, 0.5, 0.5, 0.22, 0.1, 0.8, 0.01]
    p_b = cmap.update(obs_b, tick=2)

    assert cmap.total_edges == 1
    assert cmap.shortest_distance(p_a.patch_id, p_b.patch_id) == 1

    # At patch B, food and water are depleted (0.0). Evaluate waypoints.
    waypoints = cmap.evaluate_waypoints(obs_b)
    assert len(waypoints) == 1
    best = waypoints[0]
    assert best["patch_id"] == p_a.patch_id
    assert best["distance"] == 1
    assert best["viable"] is True
    assert best["score"] > 0


def test_odyssey_fit_and_reproducibility(core):
    before_python = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    atlas, receipt = fit_small_odyssey(core)
    repeated, repeated_receipt = fit_small_odyssey(core)

    assert atlas.artifact == repeated.artifact
    assert receipt == repeated_receipt
    assert random.getstate() == before_python
    assert torch.equal(before_torch, torch.random.get_rng_state())
    assert receipt["training_samples"] == receipt["calibration_samples"] == 36
    assert atlas.artifact["schema"] == "poseidon-odyssey-atlas-v5"


def test_odyssey_from_horizon(core):
    horizon_atlas, _ = fit_small_horizon(core)
    odyssey = OdysseyAtlas.from_horizon(horizon_atlas, scarcity=2.5)

    assert odyssey.artifact["schema"] == "poseidon-odyssey-atlas-v5"
    assert odyssey.artifact["config"]["scarcity"] == 2.5
    assert odyssey.config.horizon == horizon_atlas.config.horizon

    # Planning works
    obs = [0.8, 0.5, 0.5, 0.5, 0.2, 0.1, 0.3, 0.3, 0.5, 0.1, 0.5, 0.5, 0.2, 0.3, 0.0, 0.1]
    plan = odyssey.plan(obs)
    assert plan["backend"] == "odyssey-atlas-v5"
    assert "cognitive_map" in plan


def test_save_load_roundtrip(core, tmp_path):
    atlas, _ = fit_small_odyssey(core)
    target = tmp_path / "odyssey.json"
    saved_path = atlas.save(target)
    assert Path(saved_path).is_file()

    loaded = OdysseyAtlas.load(core, target)
    assert loaded.artifact == atlas.artifact

    obs = [0.8, 0.5, 0.5, 0.5, 0.2, 0.1, 0.3, 0.3, 0.5, 0.1, 0.5, 0.5, 0.2, 0.3, 0.0, 0.1]
    atlas.reset()
    loaded.reset()
    assert atlas.plan(obs) == loaded.plan(obs)


def test_pretransit_rest_guard(core):
    atlas, _ = fit_small_odyssey(core)
    # Simulate step 1 at rich patch A
    obs1 = [1.0, 0.6, 0.6, 0.8, 0.1, 0.0, 0.7, 0.7, 0.8, 0.1, 0.5, 0.5, 0.15, 0.4, 0.0, 0.0]
    atlas.plan(obs1)
    atlas.cognitive_map.record_action(4)

    # Step 2 at depleted patch B with VERY low stamina (0.10) but adequate energy/hyd (0.45)
    obs2 = [1.0, 0.45, 0.45, 0.10, 0.1, 0.0, 0.01, 0.01, 0.3, 0.1, 0.5, 0.5, 0.35, 0.02, 0.8, 0.05]
    plan = atlas.plan(obs2)
    # With stamina 0.10 < 0.20 and depleted local patch with known waypoint, pre-transit rest should activate
    assert plan["pretransit_rest"] is True
    assert plan["action"] == 0  # rest


def test_partition_overlap_rejected(core):
    with pytest.raises(ValueError, match="must be disjoint"):
        OdysseyAtlas.fit(
            core,
            train_seeds=[1011, 1012],
            calibration_seeds=[1012, 1013],
            anchors_per_episode=2,
            max_steps=4,
        )


def test_invalid_planning_gate(core):
    atlas, _ = fit_small_odyssey(core)
    obs = [0.8, 0.5, 0.5, 0.5, 0.2, 0.1, 0.3, 0.3, 0.5, 0.1, 0.5, 0.5, 0.2, 0.3, 0.0, 0.1]
    with pytest.raises(ValueError, match="gate must be calibrated or uncalibrated"):
        atlas.plan(obs, gate="invalid_gate")


def test_storm_shelter_override(core):
    atlas, _ = fit_small_odyssey(core)
    # High exposure (0.75), high weather severity (0.80), good local shelter (0.75)
    obs = [0.9, 0.6, 0.6, 0.7, 0.75, 0.1, 0.2, 0.2, 0.75, 0.80, 0.5, 0.5, 0.2, 0.3, 0.0, 0.1]
    plan = atlas.plan(obs)
    assert plan["storm_shelter"] is True
    assert plan["action"] == 3  # shelter


def test_cognitive_map_unreachable_path():
    cmap = CognitiveMap(scarcity=2.0)
    obs1 = [1.0, 0.5, 0.5, 0.8, 0.1, 0.0, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.11, 0.4, 0.0, 0.0]
    obs2 = [1.0, 0.5, 0.5, 0.8, 0.1, 0.0, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.22, 0.4, 0.0, 0.0]
    p1 = cmap.update(obs1, tick=1)
    # Notice no movement action recorded between them
    p2 = cmap.update(obs2, tick=2)
    assert cmap.shortest_distance(p1.patch_id, p2.patch_id) == 999
    assert cmap.shortest_path(p1.patch_id, p2.patch_id) is None

