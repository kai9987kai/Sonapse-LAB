"""Paired frozen evaluations and auditable credit-growth controls."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same_tree(a, b)
    else:
        assert left == right


def test_credit_scenarios_balance_actual_conditions_and_exclude_training_seeds():
    from graftbrain.credit_experiments import build_credit_scenarios
    from graftbrain.world import World

    scenarios = build_credit_scenarios(["go_nogo", "forage"], [7, 8], training_seeds=[9, 10])
    assert scenarios == build_credit_scenarios(["go_nogo", "forage"], [7, 8], training_seeds=[9, 10])
    assert len({row["scenario_id"] for row in scenarios}) == 6
    go_rows = [row for row in scenarios if row["task"] == "go_nogo"]
    assert [row["condition"] for row in go_rows].count("go") == 2
    assert [row["condition"] for row in go_rows].count("nogo") == 2
    assert all(row["world_seed"] not in {9, 10} for row in scenarios)
    assert len({row["world_seed"] for row in go_rows}) == 4
    for row in go_rows:
        world = World(seed=row["world_seed"], curriculum="go_nogo")
        assert row["condition"] == ("go" if world.st["go"] else "nogo")
        assert world.t == 0


@pytest.mark.parametrize("kwargs", [
    {"train_steps": 0}, {"train_steps": True}, {"max_neurons": 0}, {"microsteps": 0},
    {"tasks": []}, {"tasks": ["maze"]}, {"seeds": [1, 1]}, {"eval_seeds": [True]},
    {"seeds": [4], "eval_seeds": [4]},
    {"seeds": [[4]]}, {"eval_seeds": 4}, {"seeds": [2**64]}, {"eval_seeds": [2**64]},
])
def test_credit_benchmark_rejects_invalid_arguments_before_output(tmp_path, kwargs):
    from graftbrain.credit_experiments import benchmark_graft_credit

    out = tmp_path / "absent"
    with pytest.raises(ValueError):
        benchmark_graft_credit(out, **kwargs)
    assert not out.exists()


def test_credit_benchmark_refuses_existing_output(tmp_path):
    from graftbrain.credit_experiments import benchmark_graft_credit

    protected = tmp_path / "keep.txt"
    protected.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark_graft_credit(tmp_path, train_steps=1)
    assert protected.read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


def test_balanced_scenario_scan_wraps_within_supported_torch_seed_range():
    from graftbrain.credit_experiments import build_credit_scenarios

    rows = build_credit_scenarios(["go_nogo"], [2**64 - 1], training_seeds=[0])
    assert {row["condition"] for row in rows} == {"go", "nogo"}
    assert all(0 <= row["world_seed"] <= 2**64 - 1 and row["world_seed"] != 0 for row in rows)


def test_unchanged_frozen_credit_policy_has_zero_delta_and_preserves_source_and_rng():
    from graftbrain import connectomes as cx
    from graftbrain.brain import Agent
    from graftbrain.credit_experiments import build_credit_scenarios, evaluate_credit, paired_credit_changes
    from graftbrain.graft import build
    from graftbrain.plasticity import PlasticityConfig
    from graftbrain.world import World

    graph = build([cx.load("proxy:flywire", 24, 3), cx.load("proxy:manc", 24, 3)], seed=3)
    agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, rollout=16, microsteps=1, seed=3,
                  plasticity_config=PlasticityConfig(mode="adaptive", growth_rule="gradient", growth_init="zero"))
    scenarios = build_credit_scenarios(["go_nogo"], [101], training_seeds=[3])
    source = copy.deepcopy(agent.state_dict())
    torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    before = evaluate_credit(agent, graph, scenarios, train_seed=3, microsteps=1, phase="pre")
    after = evaluate_credit(agent, graph, scenarios, train_seed=3, microsteps=1, phase="post")
    _same_tree(source, agent.state_dict())
    assert torch.equal(torch_state, torch.get_rng_state())
    _same_tree(numpy_state, np.random.get_state())
    assert python_state == random.getstate()
    assert len(before) == len(after) == 2
    assert {row["condition"] for row in before} == {"go", "nogo"}
    assert all(row["terminal"] and row["phase"] == "pre" for row in before)
    changes = paired_credit_changes(before, after)
    assert changes["mean_paired_return_delta"] == 0.0
    assert all(row["return_delta"] == 0.0 for row in changes["scenarios"])
    assert changes["conditions"]["go"]["episodes"] == 1
    assert changes["conditions"]["nogo"]["episodes"] == 1
    assert all("behavior" in row for row in before + after)
    assert changes["mean_paired_behavior_delta"]
    assert all(value == 0.0 for value in changes["mean_paired_behavior_delta"].values())
    assert all(value == 0.0 for row in changes["scenarios"] for value in row["behavior_delta"].values())
    for condition in ("go", "nogo"):
        assert all(value == 0.0 for value in changes["conditions"][condition]["mean_paired_behavior_delta"].values())


def test_tiny_credit_benchmark_matches_controls_and_records_actual_source(tmp_path):
    from graftbrain.credit_experiments import benchmark_graft_credit

    out = benchmark_graft_credit(tmp_path / "pilot", max_neurons=24, train_steps=1,
                                tasks=["go_nogo"], seeds=[3], eval_seeds=[101], microsteps=1)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    episodes = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((out / "structural_events.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["config"]["rollout"] == 16
    assert manifest["training_seeds"] == [3]
    assert manifest["evaluation_seeds"] == [101]
    assert {"python", "torch", "numpy", "graftbrain"} <= manifest["versions"].keys()
    root = Path(__file__).resolve().parents[1]
    for filename in ("graftbrain/credit_experiments.py", "graftbrain/brain.py", "graftbrain/plasticity.py", "graftbrain/world.py"):
        assert manifest["source_sha256"][filename] == hashlib.sha256((root / filename).read_bytes()).hexdigest()
    arms = manifest["arms"]
    assert len(arms) == len(summary["arms"]) == 7
    assert len({arm["initial_support_digest"] for arm in arms}) == 1
    assert len({arm["graph_digest"] for arm in arms}) == 1
    assert len({arm["initial_model_digest"] for arm in arms}) == 1
    assert all(arm["plasticity_config"]["growth_init"] == "zero" for arm in arms)
    assert {(arm["plasticity_config"]["growth_rule"], arm["plasticity_config"]["min_age"])
            for arm in arms if arm["rewiring_enabled"]} == {
                ("coactivity", 0), ("coactivity", 20), ("gradient", 0), ("gradient", 20), ("random", 0), ("random", 20)}
    assert len([arm for arm in arms if not arm["rewiring_enabled"]]) == 1
    assert all(arm["plasticity_config"]["mode"] == "adaptive" for arm in arms)
    assert len([row for row in episodes if row["phase"] == "reference"]) == 2
    for condition, outcome in (("go", "fail"), ("nogo", "success")):
        assert next(row for row in episodes if row["phase"] == "reference" and row["condition"] == condition)["outcome"] == outcome
    for arm in summary["arms"]:
        rows = [row for row in episodes if row.get("arm") == arm["arm"]]
        assert len([row for row in rows if row["phase"] == "pre"]) == 2
        assert len([row for row in rows if row["phase"] == "post"]) == 2
        assert arm["training_complete"]["episodes"] == 0
        assert arm["training_incomplete"]["episodes"] == 1
        assert arm["training_incomplete"]["mean_return"] is not None
        assert arm["paired"]["conditions"]["go"]["episodes"] == 1
        assert arm["paired"]["conditions"]["nogo"]["episodes"] == 1
    assert isinstance(events, list)
    assert {row["phase"] for row in episodes if "behavior" in row} == {"reference", "pre", "train", "post"}
    assert all("behavior" in row for row in episodes)
    assert manifest["behavior_metrics"]["reporter_only"] is True
    assert summary["inference"]["independent_training_seeds"] == 1
    assert summary["inference"]["inferential_statistics"] is False


def test_zero_action_reference_reports_stationary_distance_without_task_credit():
    from graftbrain.credit_experiments import build_credit_scenarios, zero_action_references

    rows = zero_action_references(build_credit_scenarios(["go_nogo"], [1000]))
    assert {row["condition"] for row in rows} == {"go", "nogo"}
    for row in rows:
        assert row["behavior"] == {
            "initial_target_distance": 5.0, "min_target_distance": 5.0, "final_target_distance": 5.0,
            "path_length": 0.0, "mean_abs_drive": 0.0, "mean_abs_turn": 0.0,
            "mean_squared_action_norm": 0.0, "action_saturation_fraction": 0.0,
        }


def test_behavior_trace_measures_actual_motion_and_clipped_executed_actions():
    from graftbrain import credit_experiments as credit
    from graftbrain.world import World

    world = World(seed=1002, curriculum="go_nogo")
    trace = credit._BehaviorTrace(world)
    action = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
    world.step(action)
    trace.observe(world, action)
    assert trace.summary() == pytest.approx({
        "initial_target_distance": 5.0, "min_target_distance": 4.86, "final_target_distance": 4.86,
        "path_length": 0.14, "mean_abs_drive": 1.0, "mean_abs_turn": 0.0,
        "mean_squared_action_norm": 1.0, "action_saturation_fraction": 0.2,
    })
    world.step(-action)
    trace.observe(world, -action)
    assert trace.summary()["final_target_distance"] == pytest.approx(5.0)
    assert trace.summary()["min_target_distance"] == pytest.approx(4.86)
    assert trace.summary()["path_length"] == pytest.approx(0.28)


def test_training_trace_resets_at_episode_boundary_and_includes_unfinished_episode():
    from graftbrain import connectomes as cx
    from graftbrain.brain import Agent
    from graftbrain.credit_experiments import _train_credit
    from graftbrain.graft import build
    from graftbrain.world import World

    graph = build([cx.load("proxy:flywire", 24, 0), cx.load("proxy:manc", 24, 0)], seed=0)
    agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, seed=0, lr=0.0, rollout=16, microsteps=1)
    with torch.no_grad():
        agent.iface.mu.weight.zero_()
        agent.iface.mu.bias.zero_()
        agent.iface.mu.bias[0] = torch.atanh(torch.tensor(0.05))
        agent.iface.log_std.fill_(-10)
    rows, _, _ = _train_credit(agent, "go_nogo", 0, 81)
    assert [(row["condition"], row["steps"], row["terminal"]) for row in rows] == [
        ("nogo", 80, True), ("go", 1, False)]
    for row in rows:
        assert row["behavior"]["initial_target_distance"] == 5.0
    assert rows[0]["behavior"]["path_length"] == pytest.approx(0.56, abs=0.001)
    assert rows[1]["behavior"]["path_length"] == pytest.approx(0.007, abs=0.0001)
    assert rows[1]["behavior"]["final_target_distance"] == pytest.approx(4.993, abs=0.0001)


def test_paired_behavior_changes_are_signed_descriptive_and_legacy_rows_stay_valid():
    from graftbrain.credit_experiments import paired_credit_changes

    before = [{"scenario_id": "go", "condition": "go", "world_seed": 1, "task": "go_nogo",
               "training_seed": 0, "return": -1.0, "outcome": "fail", "behavior": {
                   "final_target_distance": 5.0, "path_length": 0.0}},
              {"scenario_id": "nogo", "condition": "nogo", "world_seed": 2, "task": "go_nogo",
               "training_seed": 0, "return": 3.0, "outcome": "success", "behavior": {
                   "final_target_distance": 5.0, "path_length": 0.0}}]
    after = copy.deepcopy(before)
    after[0]["behavior"].update(final_target_distance=4.0, path_length=1.0)
    after[1]["behavior"].update(final_target_distance=6.0, path_length=2.0)
    changes = paired_credit_changes(before, after)
    assert changes["mean_paired_behavior_delta"] == {"final_target_distance": 0.0, "path_length": 1.5}
    assert changes["conditions"]["go"]["mean_paired_behavior_delta"]["final_target_distance"] == -1.0
    assert changes["conditions"]["nogo"]["mean_paired_behavior_delta"]["final_target_distance"] == 1.0
    for row in before + after:
        row.pop("behavior")
    legacy = paired_credit_changes(before, after)
    assert legacy["mean_paired_return_delta"] == 0.0
    assert "mean_paired_behavior_delta" not in legacy
