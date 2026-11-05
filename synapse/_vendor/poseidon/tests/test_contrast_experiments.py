import copy
import json

import pytest
import torch

from poseidon.atlas import AtlasConfig, CounterfactualAtlas
from poseidon.contrast import ContrastAtlas, ContrastConfig
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.contrast_experiments import ARMS, run_experiment, verify_receipt, load_and_verify, write_receipt
from poseidon.experiments import ExperimentSpec, digest


@pytest.fixture
def fitted(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    core = CoreRuntime(path)
    legacy, _ = CounterfactualAtlas.fit(core, train_seeds=[11, 12], calibration_seeds=[21, 22],
        anchors_per_episode=2, max_steps=8, config=AtlasConfig(scarcity_levels=(1.0,)))
    contrast, _ = ContrastAtlas.fit(core, train_seeds=[31, 32], selection_seeds=[41, 42],
        calibration_seeds=[51, 52], anchors_per_episode=2, max_steps=8,
        config=ContrastConfig(scarcity_levels=(1.0,)))
    return core, legacy, contrast


def signed(receipt):
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def test_nine_arm_matched_probe_and_all_action_replay(fitted, tmp_path, monkeypatch):
    core, legacy, contrast = fitted
    receipt = run_experiment(core, legacy, contrast, ExperimentSpec(seed=71, episodes=1, max_steps=8))
    assert set(receipt["summary"]) == set(ARMS)
    assert set(receipt["prediction_comparison"]) == {"base", "unfiltered", "memory"}
    assert receipt["prediction_comparison"]["base"]["branches"] == 48
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay loaded a model")
    monkeypatch.setattr(contrast, "plan", forbidden)
    monkeypatch.setattr(core, "act", forbidden)
    checked = verify_receipt(receipt)
    assert checked["episodes_replayed"] == 9
    assert checked["transitions_replayed"] == 72
    assert checked["counterfactual_branches_replayed"] == 432
    path = write_receipt(receipt, tmp_path / "receipts")
    assert load_and_verify(path) == checked


@pytest.mark.parametrize("kind", ["branch", "candidate", "margin", "probe", "summary", "status", "promotion",
                                  "fit_identity", "fit_partition", "fit_selection", "fit_calibration", "fit_count"])
def test_replay_rejects_resigned_evidence_tampering(fitted, kind):
    receipt = run_experiment(*fitted, ExperimentSpec(seed=71, episodes=1, max_steps=8))
    changed = json.loads(json.dumps(receipt))
    episode = next(row for row in changed["episodes"] if row["controller"] == "contrast")
    event = episode["trajectory"][0]
    if kind == "branch":
        event["audit"]["next_observations"][0][0] -= .01
    elif kind == "candidate":
        event["decision"]["candidates"][0]["action"] = 1
    elif kind == "margin":
        event["decision"]["empirical_advantage_margin"] += .01
    elif kind == "probe":
        policy = next(row for row in changed["episodes"] if row["controller"] == "policy")
        policy["trajectory"][0]["prediction_probe"]["base"].pop()
    elif kind == "summary":
        changed["prediction_comparison"]["memory"]["mse"] += .01
    elif kind == "status":
        changed["status"] = "failed"
    elif kind == "promotion":
        changed["promoted"] = True
    elif kind == "fit_identity":
        changed["contrast"]["artifact_sha256"] = "0" * 64
    elif kind == "fit_partition":
        changed["contrast"]["fit_receipt"]["partition"]["selection_seeds"] = [71]
    elif kind == "fit_selection":
        changed["contrast"]["fit_receipt"]["selection"]["alphas"][0] = .123
    elif kind == "fit_calibration":
        changed["contrast"]["fit_receipt"]["calibration"]["paired"]["memory"]["error_radius"] += .01
    else:
        changed["contrast"]["fit_receipt"]["training_samples"] += 6
    with pytest.raises(ValueError):
        verify_receipt(signed(changed))


def test_experiment_rejects_selection_leakage_and_work_budget(fitted):
    for seed in (11, 21, 31, 41, 51):
        with pytest.raises(ValueError, match="overlap"):
            run_experiment(*fitted, ExperimentSpec(seed=seed, episodes=1, max_steps=8))
    with pytest.raises(ValueError, match="budget"):
        run_experiment(*fitted, ExperimentSpec(seed=71, episodes=32, max_steps=512))
