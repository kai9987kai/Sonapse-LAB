"""Controls and auditable artifacts for the bounded plasticity comparison."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from graftbrain import __main__ as cli


def test_cli_preserves_saved_settings_unless_an_override_is_given():
    # Accidentally creating defaults for an ordinary resume changes its saved mode.
    implicit = cli.parse_args(["train", "--steps", "1"])
    assert implicit.plasticity_config is None
    explicit = cli.parse_args(["train", "--steps", "1", "--graft-mode", "adaptive", "--graft-density", ".5"])
    assert explicit.plasticity_config.mode == "adaptive"
    assert explicit.plasticity_config.density == .5
    assert explicit.plasticity_config.interval == 10


@pytest.mark.parametrize("flag,value", [
    ("--graft-density", "0"), ("--graft-density", "1.1"), ("--graft-density", "nan"),
    ("--graft-interval", "0"), ("--graft-turnover", "-0.1"), ("--graft-turnover", "1.1"),
    ("--graft-warmup", "-1"), ("--graft-router-lr", "0"), ("--microsteps", "0"),
    ("--graft-mode", "unknown"), ("--steps", "-1"),
])
def test_cli_rejects_invalid_numbers_before_creating_a_simulation(flag, value):
    with pytest.raises(SystemExit) as err:
        cli.parse_args(["train", flag, value])
    assert err.value.code == 2


def test_benchmark_rejects_overwrite_and_seed_leakage_without_writing(tmp_path):
    from graftbrain.experiments import benchmark_plasticity
    existing = tmp_path / "existing"
    existing.mkdir()
    protected = existing / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark_plasticity(existing, train_steps=1, seeds=[2], eval_seeds=[20])
    assert protected.read_text(encoding="utf-8") == "keep"
    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="disjoint"):
        benchmark_plasticity(absent, train_steps=1, seeds=[2], eval_seeds=[2])
    assert not absent.exists()


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


def test_evaluation_restores_each_episode_and_preserves_the_trained_agent():
    from graftbrain import connectomes as cx
    from graftbrain.brain import Agent
    from graftbrain.experiments import evaluate_frozen
    from graftbrain.graft import build
    from graftbrain.plasticity import PlasticityConfig
    from graftbrain.world import World

    graph = build([cx.load("proxy:flywire", 400, 3), cx.load("proxy:manc", 400, 3)], seed=3)
    agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, microsteps=1, seed=3,
                  plasticity_config=PlasticityConfig(mode="adaptive"))
    before = copy.deepcopy(agent.state_dict())
    rows = evaluate_frozen(agent, graph, "go_nogo", [101, 101], train_seed=3, microsteps=1)
    _same_tree(before, agent.state_dict())
    assert rows[0]["return"] == rows[1]["return"]
    assert rows[0]["outcome"] == rows[1]["outcome"]
    assert rows[0]["steps"] == rows[1]["steps"]
    assert rows[0]["terminal"] is True
    assert rows[0]["stage"] == "eval"


def test_small_benchmark_records_paired_graphs_masks_actual_outcomes_and_cost(tmp_path):
    from graftbrain.experiments import benchmark_plasticity

    out = tmp_path / "pilot"
    benchmark_plasticity(out, specs=["proxy:flywire", "proxy:manc"], max_neurons=400,
                         train_steps=2, tasks=["go_nogo"], seeds=[3], eval_seeds=[101], microsteps=1)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((out / "episodes.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert manifest["training_seeds"] == [3]
    assert manifest["evaluation_seeds"] == [101]
    assert "revision" in manifest["code"] and "dirty" in manifest["code"]
    arms = manifest["arms"]
    assert {arm["mode"] for arm in arms} == {"legacy", "static", "routing", "structural", "adaptive"}
    assert len({arm["graph_digest"] for arm in arms}) == 1
    assert len({arm["initial_support_digest"] for arm in arms if arm["mode"] != "legacy"}) == 1
    assert len(rows) == 10
    assert all({"mode", "task", "stage", "seed", "return", "steps", "outcome", "terminal"} <= row.keys() for row in rows)
    assert all(row["outcome"] in {"success", "fail", "timeout"} for row in rows if row["stage"] == "eval")
    assert all(row["outcome"] == "budget_exhausted" and not row["terminal"] for row in rows if row["stage"] == "train")
    assert len(summary["arms"]) == 5
    for arm in summary["arms"]:
        assert arm["candidate_records"] > 0
        assert arm["active_records"] > 0
        assert arm["wall_seconds"] > 0
        assert arm["train_steps"] == 2
        assert {"swaps", "gate_spread", "evaluation_outcome_rates"} <= arm.keys()
        assert sum(arm["evaluation_outcome_rates"].values()) == pytest.approx(1)
