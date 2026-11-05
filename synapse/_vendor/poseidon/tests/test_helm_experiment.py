import copy
import importlib
import importlib.util

import pytest
import torch

from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.experiments import digest


def experiment_module():
    assert importlib.util.find_spec("poseidon.helm_experiment") is not None, "Helm experiment contract is missing"
    return importlib.import_module("poseidon.helm_experiment")


def resign(receipt):
    receipt["receipt_sha256"] = digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    return receipt


def test_experiment_contract_is_available():
    module = experiment_module()
    assert callable(module.run_helm_experiment)
    assert callable(module.verify_helm_receipt)


@pytest.fixture
def fitted(tmp_path):
    module = experiment_module()
    from poseidon.helm import HelmConfig, HelmCritic
    with torch.random.fork_rng():
        torch.manual_seed(19)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    core = CoreRuntime(path)
    critic, _ = HelmCritic.fit(
        core, train_seeds=[11, 12], selection_seeds=[21, 22], calibration_seeds=[31, 32],
        anchors_per_episode=1, max_steps=4,
        config=HelmConfig(reservoir_size=8, bootstrap_heads=2),
    )
    return module, core, critic


@pytest.fixture
def tiny_receipt(fitted):
    module, core, critic = fitted
    return module.run_helm_experiment(core, critic, seeds=[81, 82], max_steps=4, anchor_interval=2)


def test_complete_paired_episodes_and_weightless_branch_replay(tiny_receipt, fitted, monkeypatch):
    module, core, critic = fitted
    assert {episode["controller"] for episode in tiny_receipt["episodes"]} == set(module.ARMS)
    assert len(tiny_receipt["episodes"]) == len(module.ARMS) * 2
    def forbidden(*args, **kwargs):
        raise AssertionError("Weightless replay invoked a neural controller")
    monkeypatch.setattr(core, "act", forbidden)
    monkeypatch.setattr(critic, "plan", forbidden)
    checked = module.verify_helm_receipt(tiny_receipt)
    assert checked["verified"] is True
    assert checked["episodes_replayed"] == len(module.ARMS) * 2
    assert checked["branches_replayed"] == len(tiny_receipt["anchors"]) * 12
    assert checked["transitions_replayed"] == sum(row["steps"] for row in tiny_receipt["rows"])
    assert tiny_receipt["promoted"] is False
    assert set(tiny_receipt["paired"]) == set(module.ARMS) - {"policy"}
    assert all(row["wins"] + row["ties"] + row["losses"] == 2 for row in tiny_receipt["paired"].values())


@pytest.mark.parametrize("collection", ["episodes", "rows"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_replay_rejects_missing_and_duplicate_pairs(tiny_receipt, fitted, collection, mutation):
    module, _, _ = fitted
    broken = copy.deepcopy(tiny_receipt)
    if mutation == "missing":
        broken[collection].pop()
    else:
        broken[collection][-1] = copy.deepcopy(broken[collection][0])
    with pytest.raises(ValueError):
        module.verify_helm_receipt(resign(broken))


@pytest.mark.parametrize("mutation", ["missing_anchor", "duplicate_anchor", "missing_branch", "duplicate_branch", "reward", "continuation", "partial"])
def test_replay_rejects_resigned_branch_tampering(tiny_receipt, fitted, mutation):
    module, _, _ = fitted
    broken = copy.deepcopy(tiny_receipt)
    if mutation == "missing_anchor":
        broken["anchors"].pop()
    elif mutation == "duplicate_anchor":
        broken["anchors"][-1] = copy.deepcopy(broken["anchors"][0])
    elif mutation == "missing_branch":
        broken["anchors"][0]["branches"]["16"].pop()
    elif mutation == "duplicate_branch":
        broken["anchors"][0]["branches"]["16"][-1] = copy.deepcopy(broken["anchors"][0]["branches"]["16"][0])
    elif mutation == "reward":
        broken["anchors"][0]["branches"]["4"][0]["return"] += 1
    elif mutation == "continuation":
        broken["anchors"][0]["branches"]["16"][0]["actions"][1] = 99
    else:
        broken["episodes"][0]["trajectory"].pop()
    with pytest.raises(ValueError):
        module.verify_helm_receipt(resign(broken))


def test_seed_partitions_are_disjoint_and_inputs_are_bounded(fitted):
    module, core, critic = fitted
    for seeds in ([11], [21], [31], [81, 81], [True]):
        with pytest.raises(ValueError):
            module.run_helm_experiment(core, critic, seeds=seeds, max_steps=4)
    for options in ({"max_steps": True}, {"scarcity": float("nan")}, {"anchor_interval": 0}):
        with pytest.raises(ValueError):
            module.run_helm_experiment(core, critic, seeds=[81], **options)


def test_null_exposure_and_predeclared_descriptive_thresholds(tiny_receipt, fitted):
    module, _, _ = fitted
    risk = module.helm_risk_coverage(tiny_receipt)
    assert risk == tiny_receipt["risk_coverage"]
    assert risk["thresholds"] == list(module.RISK_THRESHOLDS)
    assert risk["selection_from_evaluation"] is False
    for mode in module.ARMS[1:]:
        for row in risk["modes"][mode]["override"]:
            if row["exposure"] == 0:
                assert row["adverse_rate"] is None
                assert row["mean_actual_advantage"] is None


def test_progress_cancels_collection_and_replay(tiny_receipt, fitted):
    module, core, critic = fitted
    class Cancelled(RuntimeError):
        pass
    events = []
    def cancel(event):
        events.append(event)
        if len(events) >= 3:
            raise Cancelled("requested")
    with pytest.raises(Cancelled):
        module.run_helm_experiment(core, critic, seeds=[81], max_steps=4, progress=cancel)
    assert len(events) == 3
    events.clear()
    with pytest.raises(Cancelled):
        module.verify_helm_receipt(tiny_receipt, progress=cancel)
    assert len(events) == 3


def test_bindings_and_outcome_arithmetic_reject_tampering(tiny_receipt, fitted):
    module, _, _ = fitted
    for field in ("checkpoint_sha256", "model_sha256", "critic_sha256"):
        broken = copy.deepcopy(tiny_receipt)
        broken["protocol"][field] = "0" * 64
        broken["experiment_id"] = digest(broken["protocol"])
        with pytest.raises(ValueError):
            module.verify_helm_receipt(resign(broken))
    broken = copy.deepcopy(tiny_receipt)
    broken["paired"]["selected"]["mean_reward_delta"] += 1
    with pytest.raises(ValueError):
        module.verify_helm_receipt(resign(broken))
