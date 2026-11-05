import copy
import hashlib
import json
import random

import pytest
import torch

from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.world import TidePool


@pytest.fixture
def core(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    return CoreRuntime(path)


def fit_small(core, **settings):
    from poseidon.contrast import ContrastAtlas, ContrastConfig
    return ContrastAtlas.fit(
        core, train_seeds=[1011, 1012], selection_seeds=[2011, 2012],
        calibration_seeds=[3011, 3012], anchors_per_episode=3, max_steps=6,
        config=ContrastConfig(scarcity_levels=(1.0,), **settings),
    )


def rehash(payload):
    payload.pop("sha256", None)
    payload["sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    return payload


def test_fit_freezes_distinct_partitions_and_preserves_rng_and_core(core):
    before_python = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    before_weights = {name: value.clone() for name, value in core.model.state_dict().items()}
    atlas, receipt = fit_small(core)
    repeated, repeated_receipt = fit_small(core)
    assert atlas.artifact == repeated.artifact
    assert receipt == repeated_receipt
    assert random.getstate() == before_python
    assert torch.equal(before_torch, torch.random.get_rng_state())
    assert all(torch.equal(before_weights[name], value) for name, value in core.model.state_dict().items())
    assert receipt["training_samples"] == receipt["selection_samples"] == receipt["calibration_samples"] == 36
    assert receipt["selection"]["episode_count"] == 2
    assert len(receipt["selection"]["alphas"]) == 16
    assert {record["episode_seed"] for record in atlas.artifact["records"]} == {1011, 1012}
    for mode in ("memory", "unfiltered", "base"):
        paired = receipt["calibration"]["paired"][mode]
        assert paired["episode_count"] == 2
        assert paired["anchor_count"] == 6
        assert len(paired["episode_max_errors"]) == 2


@pytest.mark.parametrize("train,selection,calibration", [([1], [1], [2]), ([1], [2], [1]), ([1], [2], [2])])
def test_fit_rejects_any_cross_partition_episode_overlap(core, train, selection, calibration):
    from poseidon.contrast import ContrastAtlas
    with pytest.raises(ValueError, match="disjoint"):
        ContrastAtlas.fit(core, train_seeds=train, selection_seeds=selection, calibration_seeds=calibration)


def test_feature_selection_balances_episodes_and_rejects_harmful_channels():
    from poseidon.contrast import ContrastConfig, _select_features
    rows, predictions = [], []
    for seed, copies, target in ((1, 1, 1.0), (2, 9, 0.0)):
        for _ in range(copies):
            rows.append({"episode_seed": seed, "base_observation": [0.0] * 16,
                         "next_observation": [target, 0.0] + [0.0] * 14})
            predictions.append([1.0, 1.0] + [0.0] * 14)
    selected = _select_features(rows, predictions, [1, 2], ContrastConfig(shrinkage_grid=(0, .5, 1)))
    # Equal episode weight gives .5*(1-alpha)^2 + .5*alpha^2.
    assert selected["alphas"][0] == .5
    assert selected["base_mse"][0] == .5
    assert selected["admitted_mse"][0] == .25
    assert selected["alphas"][1:] == [0.0] * 15


def test_serving_cannot_access_hidden_simulator_or_teacher(core, monkeypatch):
    import poseidon.contrast as contrast
    atlas, _ = fit_small(core)
    obs = atlas.artifact["records"][0]["observation"]
    before = atlas.artifact
    def forbidden(*args, **kwargs):
        raise AssertionError("controller accessed hidden simulator or teacher")
    monkeypatch.setattr(TidePool, "from_snapshot", forbidden)
    monkeypatch.setattr(TidePool, "step", forbidden)
    monkeypatch.setattr(contrast, "teacher_action", forbidden)
    decision = atlas.plan(obs)
    assert decision["policy_action"] == core.act(obs)
    assert len(decision["candidates"]) == 6
    assert atlas.artifact == before


def test_memory_erasure_and_unfiltered_ablation_have_distinct_frozen_modes(core):
    atlas, _ = fit_small(core)
    obs = atlas.artifact["records"][0]["observation"]
    erased = atlas.plan(obs, use_memory=False)
    raw = atlas.plan(obs, use_selection=False)
    admitted = atlas.plan(obs)
    assert erased["calibration_mode"] == "base"
    assert raw["calibration_mode"] == "unfiltered"
    assert admitted["calibration_mode"] == "memory"
    assert erased["feature_alphas"] == [0.0] * 16
    assert raw["feature_alphas"] == [1.0] * 16
    for candidate in erased["candidates"]:
        assert candidate["predicted_observation"] == candidate["base_observation"]
        assert candidate["source_ids"] == []
    assert any(candidate["predicted_observation"] != candidate["base_observation"] for candidate in raw["candidates"])


def test_live_core_mutation_and_rehashed_invalid_artifacts_are_rejected(core, tmp_path):
    from poseidon.contrast import ContrastAtlas
    atlas, _ = fit_small(core)
    path = tmp_path / "contrast.json"
    atlas.save(path)
    assert ContrastAtlas.load(core, path).artifact == atlas.artifact
    changed = atlas.artifact
    changed["selection"]["alphas"][0] = float("nan")
    with pytest.raises(ValueError):
        ContrastAtlas(core, changed)
    changed = atlas.artifact
    changed["records"].pop()
    with pytest.raises(ValueError, match="six matched"):
        ContrastAtlas(core, rehash(changed))
    changed = atlas.artifact
    changed["partition"]["selection_seeds"] = changed["partition"]["train_seeds"]
    with pytest.raises(ValueError, match="disjoint"):
        ContrastAtlas(core, rehash(changed))
    with torch.no_grad():
        next(core.model.parameters()).add_(.1)
    with pytest.raises(ValueError, match="changed"):
        atlas.plan([.5] * 16)


def test_no_support_preserves_the_incumbent_and_invalid_gate_is_rejected(core):
    atlas, _ = fit_small(core, support_radius=.000001)
    decision = atlas.plan([0.0] * 16)
    assert decision["action"] == core.act([0.0] * 16)
    assert decision["fallback_reason"] == "outside_fitted_support"
    assert decision["override_accepted"] is False
    with pytest.raises(ValueError):
        atlas.plan([.5] * 16, gate="greedy")
    with pytest.raises(ValueError):
        atlas.plan([.5] * 16, use_selection=1)


def test_utility_uses_vital_channels_and_reserve_penalty(core):
    atlas, _ = fit_small(core)
    assert atlas.utility([1.0, 1.0, 1.0, 1.0, 0.0] + [0.0] * 11) == pytest.approx(6.3)
    assert atlas.utility([0.0] * 16) == pytest.approx(-.688)
    left = [.5] * 5 + [0.0] * 11
    right = [.5] * 5 + [1.0] * 11
    assert atlas.utility(left) == atlas.utility(right)


def test_public_reserve_utility_uses_frozen_costs_and_plan_exposes_policy_probabilities(core):
    from poseidon.contrast import reserve_utility
    assert reserve_utility([1.0, 1.0, 1.0, 1.0, 0.0] + [0.0] * 11,
                           .5, policy_weight=.4, margin_penalty=2) == pytest.approx(6.5)
    assert reserve_utility([0.0] * 16, margin_penalty=1) == pytest.approx(-.344)
    atlas, _ = fit_small(core)
    decision = atlas.plan(atlas.artifact["records"][0]["observation"])
    assert len(decision["policy_probabilities"]) == 6
    assert sum(decision["policy_probabilities"]) == pytest.approx(1)
    assert [row["policy_probability"] for row in decision["candidates"]] == decision["policy_probabilities"]


def test_pair_gate_accepts_resolved_advantage_when_absolute_margins_overlap(core):
    from poseidon.contrast import ContrastAtlas
    # Real Tidal core with action-independent base predictions and incumbent rest.
    with torch.no_grad():
        for parameter in core.model.parameters():
            parameter.zero_()
        core.model.action_head.bias[0] = 1.0
    save_checkpoint(core.path, core.model, receipt={})
    atlas, _ = fit_small(core)
    payload = atlas.artifact
    obs = [.5] * 16
    for record in payload["records"]:
        record["observation"] = obs[:]
        record["base_observation"] = obs[:]
        record["next_observation"] = obs[:]
        record["next_observation"][0] = .6
        if record["action"] == 2:
            record["next_observation"][2] = .7
    payload["selection"]["alphas"] = [1.0] * 16
    payload["selection"]["base_mse"] = [1.0] * 16
    payload["selection"]["admitted_mse"] = [0.0] * 16
    for mode in ("memory", "unfiltered", "base"):
        for item in payload["calibration"][mode]:
            item["supported_count"] = item["count"]
            item["error_radius"] = .1
        payload["calibration"]["paired"][mode]["episode_max_errors"] = [0.0, 0.0]
        payload["calibration"]["paired"][mode]["error_radius"] = 0.0
        payload["calibration"]["paired"][mode]["mean_error"] = 0.0
    controlled = ContrastAtlas(core, rehash(payload))
    paired = controlled.plan(obs)
    absolute = controlled.plan(obs, gate="absolute")
    assert paired["policy_action"] == 0
    assert paired["action"] == 2
    assert paired["override_accepted"] is True
    assert paired["empirical_advantage_margin"] > .02
    assert absolute["proposed_action"] == 2
    assert absolute["action"] == 0
    assert absolute["empirical_advantage_margin"] < 0
    assert absolute["fallback_reason"] == "counterfactual_advantage_unresolved"


def test_calibration_uses_worst_anchor_per_episode_and_detaches_public_artifact(core):
    atlas, receipt = fit_small(core)
    for mode in ("memory", "unfiltered", "base"):
        maxima = []
        for seed in (3011, 3012):
            values = [row[mode + "_overestimation"] for row in receipt["paired_anchor_rows"]
                      if row["episode_seed"] == seed]
            assert len(values) == 3
            maxima.append(max(values))
        summary = receipt["calibration"]["paired"][mode]
        assert summary["episode_max_errors"] == maxima
        # At n=2, the observed 90th percentile is the larger episode maximum.
        assert summary["error_radius"] == max(maxima)
    detached = atlas.artifact
    detached["selection"]["alphas"][0] = .123
    assert atlas.artifact["selection"]["alphas"][0] != .123

