"""Unit tests for Odysseus cognitive navigator and Bayesian replenishment."""
import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.odysseus import (
    EmpiricalCognitiveMap,
    EmpiricalPatch,
    OdysseusAtlas,
    OdysseusConfig,
    fit_odysseus,
    validate_odysseus_artifact,
)
from poseidon.world import TidePool


def test_empirical_patch_bayesian_revisit_updates():
    patch = EmpiricalPatch(
        patch_id="0.5_0.5",
        terrain=0.5,
        shelter=0.5,
        first_seen_tick=10,
        last_seen_tick=10,
        visit_count=1,
        observed_food=0.20,
        observed_water=0.30,
        food_cap=1.0,
        water_cap=1.0,
        last_threat=0.0,
    )

    # First revisit at tick 20 with food=0.35 (delta=0.15 over 10 ticks => rate 0.015)
    patch.update_revisit(current_tick=20, food=0.35, water=0.40)
    assert patch.revisit_count == 1
    assert patch.visit_count == 2
    assert patch.food_rate_mean > 0.003
    assert patch.estimated_food_lcb(20) == 0.35

    # Conservative LCB at future tick 30
    est_food = patch.estimated_food_lcb(current_tick=30, z=1.0)
    assert est_food >= 0.35


def test_empirical_cognitive_map_transitions():
    cmap = EmpiricalCognitiveMap()
    obs1 = [0.0]*16
    obs1[12] = 0.5  # terrain
    obs1[8] = 0.4   # shelter

    patch1 = cmap.update(obs1, tick=0)
    pid1 = patch1.patch_id

    # Record action explore (4)
    cmap.record_action(4)

    obs2 = [0.0]*16
    obs2[12] = 0.8
    obs2[8] = 0.2
    patch2 = cmap.update(obs2, tick=1)
    pid2 = patch2.patch_id

    # Transition probability P(pid2 | pid1, action=4) should be 1.0
    p = cmap.transition_probability(pid1, 4, pid2)
    assert p == 1.0

    # Unseen action transition should be 0.0
    p_unseen = cmap.transition_probability(pid1, 5, pid2)
    assert p_unseen == 0.0


from unittest.mock import patch as mock_patch


def test_odysseus_planning_and_rest_guard(tmp_path):
    torch.set_num_threads(2)
    core = CoreRuntime(active_core_path())

    train_seeds = [131000001, 131000002]
    cal_seeds = [132000001, 132000002]

    atlas, receipt = fit_odysseus(core, train_seeds=train_seeds, calibration_seeds=cal_seeds, max_steps=16)
    assert receipt["schema"] == "poseidon-odysseus-atlas-v1"
    assert "error_radius" in receipt
    assert receipt["error_radius"] >= 0.0

    # Test normal plan evaluation
    obs = [0.5] * 16
    plan = atlas.plan(obs, mode="calibrated")
    assert 0 <= plan["action"] <= 5
    assert "reason" in plan

    # Test pre-transit rest guard: low stamina (<0.20) overrides movement action (4) to rest (0)
    with mock_patch.object(core, "act", return_value=4):
        obs_low_stam = [0.5] * 16
        obs_low_stam[3] = 0.10  # stamina low
        plan_guarded = atlas.plan(obs_low_stam, mode="calibrated")
        assert plan_guarded["action"] == 0  # rest guard enforced
        assert plan_guarded["overridden"] is True
        assert plan_guarded["reason"] == "pre_transit_stamina_rest_guard"

