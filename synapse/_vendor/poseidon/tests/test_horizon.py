import copy
import hashlib
import json
from pathlib import Path
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
    from poseidon.horizon import HorizonAtlas, HorizonConfig
    return HorizonAtlas.fit(
        core,
        train_seeds=[1011, 1012],
        calibration_seeds=[2011, 2012],
        anchors_per_episode=3,
        max_steps=6,
        config=HorizonConfig(horizon=4, scarcity_levels=(1.0,), **settings),
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
    assert receipt["training_samples"] == receipt["calibration_samples"] == 36
    assert {record["episode_seed"] for record in atlas.artifact["records"]} == {1011, 1012}
    for mode in ("memory", "no_memory", "uncalibrated"):
        cal = receipt["calibration"][mode]
        assert cal["episode_count"] == 2
        assert cal["anchor_count"] == 6
        assert len(cal["episode_max_errors"]) == 2


def test_save_load_roundtrip_preserves_checksum_and_decisions(core, tmp_path):
    atlas, _ = fit_small(core)
    target = tmp_path / "horizon.json"
    saved_path = atlas.save(target)
    assert Path(saved_path).is_file()
    from poseidon.horizon import HorizonAtlas
    loaded = HorizonAtlas.load(core, target)
    assert loaded.artifact == atlas.artifact

    obs = [0.8, 0.5, 0.5, 0.5, 0.2, 0.1, 0.3, 0.3, 0.5, 0.1, 0.5, 0.5, 0.2, 0.3, 0.0, 0.1]
    decision_orig = atlas.plan(obs)
    decision_loaded = loaded.plan(obs)
    assert decision_orig == decision_loaded


def test_depletion_trap_detection(core):
    atlas, _ = fit_small(core)
    # Mock observation where food is 0.0, water is 0.0
    obs = [1.0, 0.3, 0.5, 0.8, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.5, 0.5, 0.1, 0.4, 0.0, 0.2]
    # If incumbent wants to forage (1), depletion trap should fire
    dec = atlas.plan(obs)
    assert "depletion_trap" in dec
    assert "advantage_error_radius" in dec
    assert "point_advantage" in dec
    assert "margin" in dec


def test_partition_overlap_rejected(core):
    from poseidon.horizon import HorizonAtlas
    with pytest.raises(ValueError, match="must be disjoint"):
        HorizonAtlas.fit(
            core,
            train_seeds=[1011, 1012],
            calibration_seeds=[1012, 1013],
            anchors_per_episode=2,
            max_steps=4,
        )


def test_tampered_payload_rejected(core):
    from poseidon.horizon import HorizonAtlas
    atlas, _ = fit_small(core)
    payload = atlas.artifact
    payload["calibration"]["memory"]["error_radius"] = 99.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        HorizonAtlas(core, payload)
