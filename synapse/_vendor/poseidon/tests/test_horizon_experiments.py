import copy
import json
from pathlib import Path

import pytest
import torch

from poseidon.atlas import AtlasConfig, CounterfactualAtlas
from poseidon.contrast import ContrastAtlas, ContrastConfig
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.experiments import ExperimentSpec, digest, write_receipt
from poseidon.horizon import HorizonAtlas, HorizonConfig
from poseidon.horizon_experiments import (
    ARMS,
    load_and_verify,
    run_experiment,
    verify_receipt,
)


@pytest.fixture
def fitted(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    core = CoreRuntime(path)
    legacy, _ = CounterfactualAtlas.fit(
        core, train_seeds=[11, 12], calibration_seeds=[21, 22],
        anchors_per_episode=2, max_steps=8, config=AtlasConfig(scarcity_levels=(1.0,))
    )
    contrast, _ = ContrastAtlas.fit(
        core, train_seeds=[31, 32], selection_seeds=[41, 42],
        calibration_seeds=[51, 52], anchors_per_episode=2, max_steps=8,
        config=ContrastConfig(scarcity_levels=(1.0,))
    )
    horizon, _ = HorizonAtlas.fit(
        core, train_seeds=[61, 62], calibration_seeds=[71, 72],
        anchors_per_episode=2, max_steps=8, config=HorizonConfig(horizon=4, scarcity_levels=(1.0,))
    )
    return core, legacy, contrast, horizon


def test_horizon_experiment_and_weightless_replay(fitted, tmp_path, monkeypatch):
    core, legacy, contrast, horizon = fitted
    spec = ExperimentSpec(seed=81, episodes=1, max_steps=8)
    receipt = run_experiment(core, legacy, contrast, horizon, spec)
    assert set(receipt["summary"]) == set(ARMS)
    assert receipt["horizon"]["ready"] is True

    # Replay must NOT invoke any neural model
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted to invoke neural models")
    monkeypatch.setattr(core, "act", forbidden)
    monkeypatch.setattr(horizon, "plan", forbidden)

    checked = verify_receipt(receipt)
    assert checked["verified"] is True
    assert checked["episodes_replayed"] == len(ARMS)
    assert checked["transitions_replayed"] == len(ARMS) * 8
    assert checked["counterfactual_branches_replayed"] == len(ARMS) * 8 * 6

    path = write_receipt(receipt, tmp_path / "horizon_receipts")
    assert load_and_verify(path) == checked


def test_replay_rejects_altered_receipt(fitted):
    core, legacy, contrast, horizon = fitted
    spec = ExperimentSpec(seed=81, episodes=1, max_steps=8)
    receipt = run_experiment(core, legacy, contrast, horizon, spec)
    tampered = copy.deepcopy(receipt)
    tampered["summary"]["horizon"]["mean_reward"] += 1.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_receipt(tampered)
