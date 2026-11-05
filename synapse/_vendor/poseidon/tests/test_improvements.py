import json

import pytest
import torch

from poseidon.calibrate import expected_calibration_error, fit_temperature
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, active_core_path, save_checkpoint
from poseidon.train_dagger import aggregate, better, collect_on_policy, greedy_policy
from poseidon.world import teacher_action


def small_model():
    torch.manual_seed(3)
    return TidalCore(CoreConfig(hash_buckets=64, hidden_size=24))


def test_dagger_collection_labels_learner_states_with_teacher_and_is_deterministic():
    policy = greedy_policy(small_model())
    a = collect_on_policy(policy, episodes=2, round_index=1, beta=0.0, max_steps=12)
    b = collect_on_policy(policy, episodes=2, round_index=1, beta=0.0, max_steps=12)
    assert a == b and len(a) > 0
    for record in a:
        assert record["teacher_action"] == teacher_action(record["obs"])
        assert record["action"] == policy(record["obs"])  # beta=0: learner drives
    teacher_driven = collect_on_policy(policy, episodes=1, round_index=2, beta=1.0, max_steps=12)
    assert all(r["action"] == r["teacher_action"] for r in teacher_driven)
    assert {r["episode_seed"] for r in a}.isdisjoint(range(91000001, 91001001))


def test_dagger_aggregation_appends_consistent_world_tensors():
    base = {"features": torch.zeros(2, 64), "labels": torch.zeros(2, 5, dtype=torch.long),
            "obs": torch.zeros(3, 16), "delta": torch.zeros(3, 16),
            "actions": torch.zeros(3, dtype=torch.long), "teacher": torch.zeros(3, dtype=torch.long)}
    records = [{"obs": [0.5] * 16, "next_obs": [0.75] * 16, "action": 4, "teacher_action": 2}]
    merged = aggregate(base, records)
    assert len(merged["obs"]) == len(merged["delta"]) == len(merged["actions"]) == len(merged["teacher"]) == 4
    assert torch.allclose(merged["delta"][-1], torch.full((16,), 0.25))
    assert merged["actions"][-1] == 4 and merged["teacher"][-1] == 2
    assert merged["features"] is base["features"]


def test_promotion_requires_strict_validation_improvement():
    assert better({"score": 0.9, "mean_steps": 200}, {"score": 0.8, "mean_steps": 250})
    assert better({"score": 0.8, "mean_steps": 251}, {"score": 0.8, "mean_steps": 250})
    assert not better({"score": 0.8, "mean_steps": 250}, {"score": 0.8, "mean_steps": 250})


def test_temperature_scaling_fits_overconfident_logits_and_refuses_degenerate_case():
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 3) * 1.5
    labels = torch.multinomial(true_logits.softmax(-1), 1).squeeze(-1)  # calibrated at T=1
    logits = true_logits * 4  # overconfident by a known factor
    value, status = fit_temperature(logits, labels)
    assert status == "fitted" and 3.0 < value < 5.0
    assert expected_calibration_error((logits / value).softmax(-1), labels) < expected_calibration_error(logits.softmax(-1), labels)
    perfect = torch.eye(3)[labels] * 5
    assert fit_temperature(perfect, labels) == (1.0, fit_temperature(perfect, labels)[1])
    assert fit_temperature(perfect, labels)[1].startswith("degenerate")


def test_runtime_applies_checkpoint_temperatures(tmp_path):
    model = small_model()
    path = tmp_path / "core.pt"
    save_checkpoint(path, model, receipt={})
    raw = CoreRuntime(path).scene("two small cyan spheres orbit")
    save_checkpoint(path, model, receipt={}, calibration={"scene_temperature": {"shape": 4.0}})
    scaled = CoreRuntime(path).scene("two small cyan spheres orbit")
    assert scaled["shape"] == raw["shape"]
    assert scaled["confidence"]["shape"] < raw["confidence"]["shape"]
    assert scaled["confidence"]["color"] == pytest.approx(raw["confidence"]["color"])
    save_checkpoint(path, model, receipt={}, calibration={"scene_temperature": {"shape": 0.0}})
    with pytest.raises(ValueError, match="temperature"):
        CoreRuntime(path)


def test_active_core_pointer_selects_and_is_confined_to_root(tmp_path):
    assert active_core_path(tmp_path) == tmp_path / "runs/tidal/core.pt"
    candidate = tmp_path / "runs/tidal_dagger/core.pt"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"x")
    (tmp_path / "runs/active_core.json").write_text(json.dumps({"checkpoint": "runs/tidal_dagger/core.pt"}))
    assert active_core_path(tmp_path) == candidate.resolve()
    (tmp_path / "runs/active_core.json").write_text(json.dumps({"checkpoint": "../outside.pt"}))
    with pytest.raises(ValueError, match="inside"):
        active_core_path(tmp_path)
