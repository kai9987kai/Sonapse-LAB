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
from poseidon.odyssey import OdysseyAtlas, OdysseyConfig
from poseidon.odyssey_experiments import (
    ARMS,
    load_and_verify,
    run_experiment,
    verify_receipt,
)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    torch.set_num_threads(2)
    tmp_path = tmp_path_factory.mktemp("odyssey-evidence")
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
    odyssey, _ = OdysseyAtlas.fit(
        core, train_seeds=[81, 82], calibration_seeds=[91, 92],
        anchors_per_episode=2, max_steps=8, config=OdysseyConfig(horizon=4, scarcity_levels=(1.0,))
    )
    return core, legacy, contrast, horizon, odyssey


def test_odyssey_experiment_and_weightless_replay(fitted, tmp_path, monkeypatch):
    core, legacy, contrast, horizon, odyssey = fitted
    spec = ExperimentSpec(seed=101, episodes=1, max_steps=8)
    receipt = run_experiment(core, legacy, contrast, horizon, odyssey, spec)
    assert set(receipt["summary"]) == set(ARMS)
    assert receipt["odyssey"]["ready"] is True

    # Replay must NOT invoke any neural model
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted to invoke neural models")
    monkeypatch.setattr(core, "act", forbidden)
    monkeypatch.setattr(horizon, "plan", forbidden)
    monkeypatch.setattr(odyssey, "plan", forbidden)

    checked = verify_receipt(receipt)
    assert checked["verified"] is True
    assert checked["episodes_replayed"] == len(ARMS)
    assert checked["transitions_replayed"] == len(ARMS) * 8
    assert checked["counterfactual_branches_replayed"] == len(ARMS) * 8 * 6

    path = write_receipt(receipt, tmp_path / "odyssey_receipts")
    assert load_and_verify(path) == checked


def test_replay_rejects_altered_receipt(fitted):
    core, legacy, contrast, horizon, odyssey = fitted
    spec = ExperimentSpec(seed=101, episodes=1, max_steps=8)
    receipt = run_experiment(core, legacy, contrast, horizon, odyssey, spec)
    tampered = copy.deepcopy(receipt)
    tampered["summary"]["odyssey"]["mean_reward"] += 1.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_receipt(tampered)


def artifacts_for(fitted):
    return {name: value.artifact for name, value in zip(("atlas_v2", "contrast", "horizon", "odyssey"), fitted[1:])}


def signed(value):
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = digest(value)
    return value


@pytest.fixture(scope="module")
def strict_parents(fitted):
    from poseidon.horizon_experiments import run_experiment as run_horizon
    spec = ExperimentSpec(seed=101, episodes=1, max_steps=8)
    return {
        "horizon": run_horizon(*fitted[:-1], spec),
        "odyssey": run_experiment(*fitted, spec),
    }


@pytest.mark.parametrize("schema, missing", [
    (schema, missing)
    for schema, required in (
        ("horizon", ("atlas_v2", "contrast", "horizon")),
        ("odyssey", ("atlas_v2", "contrast", "horizon", "odyssey")),
    )
    for missing in (*required, "all")
])
def test_strict_evidence_requires_every_sidecar_binding(fitted, strict_parents, schema, missing):
    from poseidon.trajectory_audit import verify_parent
    changed = copy.deepcopy(strict_parents[schema])
    for name in ("atlas_v2", "contrast", "horizon", "odyssey"):
        if missing in (name, "all"):
            changed["protocol"].pop(name + "_sha256", None)
    changed["experiment_id"] = digest(changed["protocol"])
    with pytest.raises(ValueError, match="Missing required artifact binding"):
        verify_parent(signed(changed), artifacts_for(fitted))


@pytest.mark.parametrize("schema", ["horizon", "odyssey"])
def test_strict_evidence_checks_all_required_sidecars(fitted, strict_parents, schema):
    from poseidon.trajectory_audit import verify_parent
    checked = verify_parent(strict_parents[schema], artifacts_for(fitted))
    expected = {"atlas_v2", "contrast", "horizon"} | ({"odyssey"} if schema == "odyssey" else set())
    assert checked["verified"] is True
    assert set(checked["partitions"]) == expected


@pytest.mark.parametrize("schema", ["horizon", "odyssey"])
def test_strict_evidence_rejects_resigned_horizon_mismatch(fitted, strict_parents, schema):
    from poseidon.trajectory_audit import verify_parent
    changed = copy.deepcopy(strict_parents[schema])
    changed["protocol"]["horizon"] += 1
    changed["experiment_id"] = digest(changed["protocol"])
    with pytest.raises(ValueError, match="Parent horizon differs"):
        verify_parent(signed(changed), artifacts_for(fitted))


@pytest.mark.parametrize("kind", ["duplicate_episode", "duplicate_row", "terminal", "forecast", "map", "waypoint", "fit_identity", "fit_partition"])
def test_strict_evidence_rejects_resigned_legacy_tampering(fitted, kind):
    from poseidon.trajectory_audit import verify_parent
    receipt = run_experiment(*fitted, ExperimentSpec(seed=101, episodes=1, max_steps=8))
    changed = json.loads(json.dumps(receipt))
    episode = next(item for item in changed["episodes"] if item["controller"] == "odyssey")
    event = episode["trajectory"][0]
    if kind == "duplicate_episode":
        changed["episodes"].append(copy.deepcopy(episode))
    elif kind == "duplicate_row":
        changed["rows"][-1] = copy.deepcopy(changed["rows"][0])
    elif kind == "terminal":
        episode["final_snapshot"]["tick"] = 0
    elif kind == "forecast":
        event["prediction_squared_errors"][0] += .1
    elif kind == "map":
        event["decision"]["cognitive_map"]["discovered_patches"] = 999
    elif kind == "waypoint":
        event["decision"]["best_waypoint"] = {"patch_id": "invented"}
    elif kind == "fit_identity":
        changed["odyssey"]["artifact_sha256"] = "0" * 64
    else:
        changed["odyssey"]["fit_receipt"]["partition"]["train_seeds"] = [101]
    with pytest.raises(ValueError):
        verify_parent(signed(changed), artifacts_for(fitted))


def test_multistep_audit_replays_without_any_model_calls(fitted, tmp_path, monkeypatch):
    from poseidon.trajectory_audit import collect, verify_parent, verify_audit, save_audit, load_json
    parent = run_experiment(*fitted, ExperimentSpec(seed=101, episodes=1, max_steps=8, scarcity=3.5))
    artifacts = artifacts_for(fitted)
    bundle = collect(fitted[0], parent, artifacts, stride=4, horizon=4)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evidence replay invoked a neural model")
    monkeypatch.setattr(fitted[0].model, "forward", forbidden)
    assert verify_parent(parent, artifacts)["map_scarcity_mismatch_episodes"] == 0
    checked = verify_audit(bundle, parent, artifacts)
    assert checked["branches_replayed"] == checked["anchors_replayed"] * 6
    assert checked["branch_transitions_replayed"] > checked["branches_replayed"]
    assert checked["horizon"] == 4
    path = save_audit(bundle, tmp_path / "audits")
    assert verify_audit(load_json(path), parent, artifacts) == checked


def test_runtime_odyssey_preserves_each_requested_world_scarcity(fitted, tmp_path, monkeypatch):
    from poseidon.runtime import Poseidon
    runtime = Poseidon(tmp_path)
    monkeypatch.setattr(runtime, "core", lambda: fitted[0])
    monkeypatch.setattr(runtime, "odyssey", lambda: fitted[-1])
    for scarcity in (4.0, 1.0):
        result = runtime.respond("Survive", "world", planner="odyssey", scarcity=scarcity, max_steps=3)
        trace = result["episode"]["trajectory"]
        assert all(event["decision"]["cognitive_map"]["scarcity"] == scarcity for event in trace)
        assert trace[0]["decision"]["cognitive_map"]["discovered_patches"] == 1
        assert trace[0]["decision"]["cognitive_map"]["current_tick"] == 1


@pytest.mark.parametrize("kind", ["return", "continuation", "missing_anchor", "summary", "parent"])
def test_multistep_audit_rejects_resigned_tampering(fitted, kind):
    from poseidon.trajectory_audit import collect, verify_audit
    parent = run_experiment(*fitted, ExperimentSpec(seed=101, episodes=1, max_steps=8))
    artifacts = artifacts_for(fitted)
    bundle = json.loads(json.dumps(collect(fitted[0], parent, artifacts, stride=4, horizon=4)))
    if kind == "return":
        bundle["anchors"][0]["branches"][0]["return"] += .1
    elif kind == "continuation":
        bundle["anchors"][0]["branches"][0]["actions"].pop()
    elif kind == "missing_anchor":
        bundle["anchors"].pop()
    elif kind == "summary":
        bundle["summary"]["policy"]["anchors"] += 1
    else:
        bundle["parent_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_audit(signed(bundle), parent, artifacts)
