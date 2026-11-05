import copy
import json

import pytest
import torch

from poseidon.atlas import AtlasConfig, CounterfactualAtlas
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.experiments import ExperimentSpec, digest, load_and_verify, run_experiment, verify_receipt, write_receipt


@pytest.fixture
def fitted(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    checkpoint = tmp_path / "core.pt"
    save_checkpoint(checkpoint, model, receipt={})
    core = CoreRuntime(checkpoint)
    atlas, _ = CounterfactualAtlas.fit(core, train_seeds=[1001, 1002], calibration_seeds=[2001, 2002],
        anchors_per_episode=3, max_steps=8, config=AtlasConfig(scarcity_levels=(1.0,)))
    return core, atlas


def sign(receipt):
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def test_experiment_replays_without_model_and_has_raw_paired_controls(fitted, tmp_path, monkeypatch):
    core, atlas = fitted
    result = run_experiment(core, atlas, ExperimentSpec(seed=3001, episodes=2, max_steps=8))
    assert len(result["rows"]) == len(result["episodes"]) == 12
    assert set(result["summary"]) == {"policy", "neural_mpc", "atlas", "atlas_no_memory", "heuristic", "random"}
    assert len(result["paired"]["atlas"]["per_seed"]) == 2
    def forbidden(*args, **kwargs):
        raise AssertionError("replay cannot invoke model inference")
    monkeypatch.setattr(core, "act", forbidden)
    monkeypatch.setattr(atlas, "plan", forbidden)
    verified = verify_receipt(result)
    assert verified["episodes_replayed"] == 12
    assert verified["transitions_replayed"] == 96
    path = write_receipt(result, tmp_path / "receipts")
    assert load_and_verify(path) == verified


def test_replay_rejects_changed_transition_even_after_checksum_is_recomputed(fitted):
    core, atlas = fitted
    result = run_experiment(core, atlas, ExperimentSpec(seed=3001, episodes=1, max_steps=8))
    changed = copy.deepcopy(result)
    changed["episodes"][0]["trajectory"][0]["reward"] += .1
    with pytest.raises(ValueError, match="checksum"):
        verify_receipt(changed)
    with pytest.raises(ValueError, match="transition"):
        verify_receipt(sign(changed))
    changed = copy.deepcopy(result)
    changed["summary"]["atlas"]["survival_rate"] = .987
    with pytest.raises(ValueError, match="Summary"):
        verify_receipt(sign(changed))


def test_replay_rejects_missing_or_duplicated_paired_episode(fitted):
    core, atlas = fitted
    result = run_experiment(core, atlas, ExperimentSpec(seed=3001, episodes=1, max_steps=8))
    changed = copy.deepcopy(result)
    changed["episodes"][0] = copy.deepcopy(changed["episodes"][1])
    with pytest.raises(ValueError, match="Duplicate"):
        verify_receipt(sign(changed))


def test_held_out_overlap_and_budget_validation(fitted):
    core, atlas = fitted
    with pytest.raises(ValueError, match="overlap"):
        run_experiment(core, atlas, ExperimentSpec(seed=1001, episodes=1, max_steps=8))
    for values in ({"episodes": True}, {"episodes": 0}, {"scarcity": float("nan")}, {"max_steps": 10000}, {"seed": -1}):
        with pytest.raises(ValueError):
            ExperimentSpec(**values)


def test_protocol_id_and_outcomes_are_repeatable(fitted):
    core, atlas = fitted
    spec = ExperimentSpec(seed=3001, episodes=1, max_steps=8)
    first, second = run_experiment(core, atlas, spec), run_experiment(core, atlas, spec)
    assert first["experiment_id"] == second["experiment_id"]
    assert first["episodes"] == second["episodes"]
    assert first["paired"] == second["paired"]


def test_legacy_metrics_and_planner_configuration_have_real_contracts():
    from poseidon.beyond_benchmark import catastrophic_death, wilson_interval
    from poseidon.planning import PlanningConfig
    low, high = wilson_interval(10, 10)
    assert .70 < low < 1 and high == pytest.approx(1)
    assert catastrophic_death({"death": True, "death_reason": "dehydration+exposure"})
    assert not catastrophic_death({"death": False, "death_reason": None})
    with pytest.raises(ValueError):
        PlanningConfig(horizon=3)
    with pytest.raises(ValueError):
        PlanningConfig(gamma=float("nan"))


def test_replay_checks_contract_flags_and_missing_nullable_outcome(fitted):
    core, atlas = fitted
    receipt = run_experiment(core, atlas, ExperimentSpec(seed=3001, episodes=1, max_steps=8))
    for field, value in (("alive", False), ("death", True), ("world_version", "unknown"), ("scarcity", 4.0), ("max_steps", 99)):
        altered = copy.deepcopy(receipt)
        altered["episodes"][0][field] = value
        with pytest.raises(ValueError):
            verify_receipt(sign(altered))
    altered = copy.deepcopy(receipt)
    del altered["episodes"][0]["death_reason"]
    with pytest.raises(ValueError, match="Missing"):
        verify_receipt(sign(altered))


def test_replay_rejects_contradictory_partitions_and_malformed_evidence(fitted):
    core, atlas = fitted
    receipt = run_experiment(core, atlas, ExperimentSpec(seed=3001, episodes=1, max_steps=8))
    altered = copy.deepcopy(receipt)
    altered["atlas"]["fit_receipt"]["partition"]["train_seeds"] = [3001]
    with pytest.raises(ValueError, match="overlap"):
        verify_receipt(sign(altered))
    altered = copy.deepcopy(receipt)
    del altered["protocol"]
    with pytest.raises(ValueError, match="Malformed"):
        verify_receipt(sign(altered))
    altered = copy.deepcopy(receipt)
    altered["promoted"] = True
    with pytest.raises(ValueError, match="activation"):
        verify_receipt(sign(altered))
