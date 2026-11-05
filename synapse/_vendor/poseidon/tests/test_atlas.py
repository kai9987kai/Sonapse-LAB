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
        torch.manual_seed(7)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    return CoreRuntime(path)


def fit_small(core, **config):
    from poseidon.atlas import AtlasConfig, CounterfactualAtlas
    return CounterfactualAtlas.fit(
        core, train_seeds=[1001, 1002], calibration_seeds=[2001, 2002],
        anchors_per_episode=3, max_steps=6,
        config=AtlasConfig(scarcity_levels=(1.0,), **config),
    )


def test_matched_branches_are_exact_and_leave_anchor_untouched():
    from poseidon.atlas import matched_branches
    env = TidePool(seed=81, max_steps=10)
    env.step(2)
    original = env.snapshot()
    branches = matched_branches(original)
    assert original == env.snapshot()
    assert len(branches) == 6
    assert len({row["anchor_id"] for row in branches}) == 1
    for action, row in enumerate(branches):
        reference = TidePool.from_snapshot(original)
        assert row["action"] == action
        assert row["next_observation"] == reference.step(action)[0]


def test_fit_is_deterministic_and_does_not_change_global_rng(core):
    before_python = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    atlas, receipt = fit_small(core)
    repeated, repeated_receipt = fit_small(core)
    assert atlas.artifact == repeated.artifact
    assert receipt == repeated_receipt
    assert random.getstate() == before_python
    assert torch.equal(before_torch, torch.random.get_rng_state())
    assert receipt["training_samples"] == 36
    assert all(item["count"] == 6 for item in receipt["calibration"]["memory"])


def test_fit_rejects_overlap_and_unbounded_request(core):
    from poseidon.atlas import CounterfactualAtlas
    with pytest.raises(ValueError, match="disjoint"):
        CounterfactualAtlas.fit(core, train_seeds=[1], calibration_seeds=[1])
    with pytest.raises(ValueError):
        CounterfactualAtlas.fit(core, train_seeds=[1], calibration_seeds=[2], anchors_per_episode=100000)


def test_ablation_removes_correction_and_predictions_have_sources(core):
    atlas, _ = fit_small(core)
    obs = atlas.artifact["records"][0]["observation"]
    with_memory = atlas.plan(obs)
    without_memory = atlas.plan(obs, use_memory=False)
    assert len(with_memory["candidates"]) == 6
    assert with_memory["policy_action"] == core.act(obs)
    assert any(a["predicted_observation"] != b["predicted_observation"]
               for a, b in zip(with_memory["candidates"], without_memory["candidates"]))
    for row in without_memory["candidates"]:
        assert row["predicted_observation"] == row["base_observation"]
        assert row["source_ids"] == []
    for row in with_memory["candidates"]:
        assert row["source_ids"]
        assert all(0 <= value <= 1 for value in row["predicted_observation"])


def test_unsupported_observation_falls_back_to_actual_core(core):
    atlas, _ = fit_small(core, support_radius=0.000001)
    decision = atlas.plan([0.0] * 16)
    assert decision["action"] == core.act([0.0] * 16)
    assert decision["trusted"] is False
    assert decision["fallback_reason"] == "outside_fitted_support"
    with pytest.raises(ValueError):
        atlas.plan([float("nan")] * 16)


def test_artifact_is_bound_to_checkpoint_and_rejects_tampering(core, tmp_path):
    from poseidon.atlas import CounterfactualAtlas
    atlas, _ = fit_small(core)
    path = tmp_path / "atlas.json"
    atlas.save(path)
    restored = CounterfactualAtlas.load(core, path)
    assert restored.artifact == atlas.artifact
    original = atlas.artifact
    changed = copy.deepcopy(original)
    changed["records"][0]["next_observation"][0] = 0.0
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        CounterfactualAtlas.load(core, path)
    changed = copy.deepcopy(original)
    changed["checkpoint_sha256"] = "0" * 64
    changed.pop("sha256")
    changed["sha256"] = hashlib.sha256(json.dumps(changed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="checkpoint"):
        CounterfactualAtlas(core, changed)


def test_plan_does_not_read_simulator_or_mutate_artifact(core, monkeypatch):
    atlas, _ = fit_small(core)
    obs = atlas.artifact["records"][0]["observation"]
    before = atlas.artifact
    def forbidden(*args, **kwargs):
        raise AssertionError("serving must not access simulator")
    monkeypatch.setattr(TidePool, "from_snapshot", forbidden)
    monkeypatch.setattr(TidePool, "step", forbidden)
    atlas.plan(obs)
    assert atlas.artifact == before


def test_policy_override_requires_nonoverlapping_empirical_value_margins(core):
    atlas, _ = fit_small(core)
    for record in atlas.artifact["records"][::6]:
        decision = atlas.plan(record["observation"])
        if decision["override_accepted"]:
            challenger = decision["candidates"][decision["action"]]
            incumbent = decision["candidates"][decision["policy_action"]]
            assert challenger["trusted"] and incumbent["trusted"]
            assert challenger["value"] > incumbent["upper_value"] + atlas.config.override_margin
        else:
            assert decision["action"] == decision["policy_action"]


def test_core_infinite_delta_is_rejected_before_clipping(core, monkeypatch):
    atlas, _ = fit_small(core)
    original = core.model.forward
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        result["delta"] = torch.full_like(result["delta"], float("inf"))
        return result
    monkeypatch.setattr(core.model, "forward", corrupt)
    with pytest.raises(ValueError, match="non-finite"):
        atlas.plan([0.5] * 16)


def test_core_inplace_weight_change_invalidates_live_atlas(core):
    atlas, _ = fit_small(core)
    with torch.no_grad():
        next(core.model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="changed"):
        atlas.plan([0.5] * 16)


def test_rehashed_malformed_record_and_missing_branch_are_rejected(core):
    from poseidon.atlas import CounterfactualAtlas
    atlas, _ = fit_small(core)
    for field, invalid in (("observation", [float("nan")] * 16), ("action", True)):
        damaged = atlas.artifact
        damaged["records"][0][field] = invalid
        if field == "action":
            damaged.pop("sha256")
            damaged["sha256"] = hashlib.sha256(json.dumps(damaged, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with pytest.raises(ValueError):
            CounterfactualAtlas(core, damaged)
    damaged = atlas.artifact
    damaged["records"].pop()
    damaged.pop("sha256")
    damaged["sha256"] = hashlib.sha256(json.dumps(damaged, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError, match="six matched"):
        CounterfactualAtlas(core, damaged)


def test_supported_but_large_prediction_error_falls_back(core):
    from poseidon.atlas import CounterfactualAtlas
    atlas, _ = fit_small(core)
    payload = atlas.artifact
    for mode in ("memory", "base"):
        for row in payload["calibration"][mode]:
            row["error_radius"] = 1.0
            row["supported_count"] = row["count"]
    payload.pop("sha256")
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    conservative = CounterfactualAtlas(core, payload)
    obs = payload["records"][0]["observation"]
    decision = conservative.plan(obs)
    assert all(row["supported"] for row in decision["candidates"])
    assert not any(row["trusted"] for row in decision["candidates"])
    assert decision["fallback_reason"] == "prediction_error_exceeds_budget"
    assert decision["action"] == core.act(obs)
