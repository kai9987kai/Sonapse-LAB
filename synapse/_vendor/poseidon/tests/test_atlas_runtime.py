"""Isolated integration checks; never load or replace production checkpoints."""
import json

import pytest
import torch

from poseidon.atlas import AtlasConfig, CounterfactualAtlas
from poseidon.core import CoreConfig, TidalCore, save_checkpoint
from poseidon.experiments import load_and_verify
from poseidon.runtime import Poseidon


@pytest.fixture
def atlas_runtime(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(311)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    checkpoint = tmp_path / "runs/tidal/core.pt"
    save_checkpoint(checkpoint, model, receipt={})
    runtime = Poseidon(tmp_path)
    atlas, _ = CounterfactualAtlas.fit(
        runtime.core(), train_seeds=[101], calibration_seeds=[201],
        anchors_per_episode=2, max_steps=8,
        config=AtlasConfig(scarcity_levels=(1.0,)),
    )
    atlas.save(tmp_path / "outputs/atlas/atlas.json")
    return runtime


def test_status_reports_missing_atlas_without_loading_models(tmp_path):
    runtime = Poseidon(tmp_path)
    result = runtime.status()
    assert result["atlas"]["ready"] is False
    assert runtime._core is None
    assert runtime.language.model is None


def test_runtime_experiment_writes_verified_receipt_and_replay(atlas_runtime):
    runtime = atlas_runtime
    assert runtime.status()["atlas"]["ready"] is True
    result = runtime.experiment(seed=301, episodes=1, max_steps=8)
    assert result["verification"]["verified"] is True
    assert result["verification"]["episodes_replayed"] == 6
    assert result["replay"]["controller"] == "atlas"
    assert result["artifact_url"].startswith("/artifacts/experiments/")
    receipt_path = runtime.root / "outputs" / result["artifact_url"].removeprefix("/artifacts/")
    assert load_and_verify(receipt_path) == result["verification"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["promoted"] is False
    assert runtime.language.model is None


def test_runtime_overlap_fails_before_publishing_receipt(atlas_runtime):
    with pytest.raises(ValueError, match="overlap"):
        atlas_runtime.experiment(seed=101, episodes=1, max_steps=8)
    assert not (atlas_runtime.root / "outputs/experiments").exists()


def test_runtime_rejects_atlas_after_active_checkpoint_replacement(atlas_runtime):
    runtime = atlas_runtime
    assert runtime.atlas_status()["ready"] is True
    with torch.random.fork_rng():
        torch.manual_seed(312)
        replacement = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    save_checkpoint(runtime.root / "runs/tidal/core.pt", replacement, receipt={})
    result = runtime.atlas_status()
    assert result["ready"] is False
    assert "checkpoint mismatch" in result["error"]
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        runtime.experiment(seed=301, episodes=1, max_steps=8)


def test_runtime_exposes_corrupt_artifact_as_unavailable(atlas_runtime):
    path = atlas_runtime.root / "outputs/atlas/atlas.json"
    path.write_text('{"schema": "invalid"}', encoding="utf-8")
    result = atlas_runtime.atlas_status()
    assert result["ready"] is False
    assert "schema" in result["error"].lower()
