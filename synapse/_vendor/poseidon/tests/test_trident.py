import math
import pytest

from poseidon.core import CoreRuntime, active_core_path
from poseidon.trident import TridentController, _fingerprint
from poseidon.world import TidePool
from poseidon.world_controls import rollout_with_observed_transitions


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_plan_requires_observed_transition(core):
    controller = TridentController(core=core)
    controller.reset(scarcity=2.5, seed=7, max_steps=32)
    obs = TidePool(seed=7, max_steps=32).observe()
    first = controller.plan(obs)
    assert 0 <= first["action"] <= 5
    assert first["schema"] == "poseidon-trident-controller-v1"
    with pytest.raises(RuntimeError, match="observe_transition"):
        controller.plan(obs)


def test_unobserved_directed_travel_is_refused_after_discovery(core):
    controller = TridentController(core=core)
    controller.reset(scarcity=2.5, seed=11, max_steps=32)
    obs = TidePool(seed=11, max_steps=32).observe()
    fingerprint = _fingerprint(obs)
    controller._patches[fingerprint] = {
        "visits": 4, "last_tick": 0, "last_food": obs[6], "last_water": obs[7],
        "food_on_leave": obs[6], "water_on_leave": obs[7],
        "food_cap": obs[6], "water_cap": obs[7],
        "replenish_food_sum": 0.0, "replenish_water_sum": 0.0, "replenish_samples": 0,
    }
    controller._current_fp = fingerprint
    decision = controller.plan(obs)
    travel = [item for item in decision["candidates"] if item["action"] in (4, 5)]
    assert travel
    assert all(item["reason"] == "unobserved_directed_step" and item["admissible"] is False for item in travel)


def test_no_topology_mode_does_not_block_unobserved_travel(core):
    intact = TridentController(core=core, mode="intact")
    open_map = TridentController(core=core, mode="no_topology")
    obs = TidePool(seed=13, max_steps=32).observe()
    fingerprint = _fingerprint(obs)
    for controller in (intact, open_map):
        controller.reset(scarcity=2.5, seed=13, max_steps=32)
        controller._patches[fingerprint] = {
            "visits": 4, "last_tick": 0, "last_food": obs[6], "last_water": obs[7],
            "food_on_leave": obs[6], "water_on_leave": obs[7],
            "food_cap": obs[6], "water_cap": obs[7],
            "replenish_food_sum": 0.0, "replenish_water_sum": 0.0, "replenish_samples": 0,
        }
        controller._current_fp = fingerprint
    blocked = intact.plan(obs)
    allowed = open_map.plan(obs)
    assert all(item["reason"] == "unobserved_directed_step" for item in blocked["candidates"] if item["action"] in (4, 5))
    assert all(item["reason"] != "unobserved_directed_step" for item in allowed["candidates"] if item["action"] in (4, 5))


def test_erased_and_shifted_change_residual_identity(core):
    env = TidePool(seed=21, scarcity=2.5, max_steps=8)
    obs = env.observe()
    intact = TridentController(core=core, mode="intact")
    intact.reset(scarcity=2.5, seed=21, max_steps=8)
    first = intact.plan(obs)
    nxt, reward, _, info = env.step(first["action"])
    intact.observe_transition(obs, first["action"], reward, nxt, info)
    assert any(abs(value) > 1e-12 for value in intact._ewma)

    erased = TridentController(core=core, mode="erased")
    shifted = TridentController(core=core, mode="shifted")
    erased.reset(scarcity=2.5, seed=21, max_steps=8)
    shifted.reset(scarcity=2.5, seed=21, max_steps=8)
    erased._ewma = list(intact._ewma)
    shifted._ewma = list(intact._ewma)
    intact_decision = intact.plan(nxt)
    erased_decision = erased.plan(nxt)
    shifted_decision = shifted.plan(nxt)
    assert intact_decision["residual_norm"] > 0
    assert erased_decision["residual_norm"] == 0.0
    assert math.isclose(shifted_decision["residual_norm"], intact_decision["residual_norm"], rel_tol=1e-9, abs_tol=1e-9)
    assert shifted_decision["mode"] == "shifted"
    assert erased_decision["mode"] == "erased"


def test_trident_rollout_records_transitions(core):
    controller = TridentController(core=core)
    episode = rollout_with_observed_transitions(controller, seed=42, max_steps=12, scarcity=2.5)
    assert episode["steps"] == 12
    assert controller._pending_transition is None
    assert controller.step == 12
    assert all("decision" in event for event in episode["trajectory"])
    assert all(event["decision"]["backend"] == "trident-identity-directed-v1" for event in episode["trajectory"])
    assert episode["reward"] == episode["reward"]
