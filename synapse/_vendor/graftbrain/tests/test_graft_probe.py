"""Mechanism evidence for the real supervised graft-current recovery probe."""
from copy import deepcopy
from importlib import import_module
import json

import numpy as np
import pytest
import torch


def probe_api():
    return import_module("graftbrain.graft_probe")


def artifacts(path):
    return {name: json.loads((path / f"{name}.json").read_text())
            for name in ("manifest", "curves", "summary")}


def assert_same_nested(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_nested(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_same_nested(first, second)
    else:
        assert left == right


def test_probe_trains_real_masked_current_and_collects_dormant_credit(tmp_path):
    probe = probe_api()
    output = probe.run_graft_probe(tmp_path / "actual", seeds=(7,), train_steps=12, recovery_steps=12)
    data = artifacts(output)
    result = data["summary"]["results"][0]
    source = result["source"]
    assert source["gradient_updates"] == 12
    assert source["gradient_positive_candidates"] == 24
    assert source["max_dormant_raw_gradient"] == 0.0
    assert source["weights"] != data["manifest"]["graph"]["initial_weights"]
    assert source["error_before_switch"] < source["error_before_training"]
    assert data["manifest"]["targets"]["before"] != data["manifest"]["targets"]["after"]
    assert len(data["manifest"]["targets"]["before"]) == 24
    disabled = result["branches"]["disrupted_no_rewire"]
    assert disabled["gradient_updates"] == 24
    assert disabled["weights"] != source["weights"]
    assert disabled["learned_swaps"] == 0
    assert disabled["final_active_indices"] == disabled["initial_active_indices"]


def test_interventions_match_budgets_and_leave_trained_source_unchanged(tmp_path, monkeypatch):
    probe = probe_api()
    original_clone = probe._clone_state
    captured = []

    def capture_source(source):
        captured.append((source, deepcopy((source.weights.detach(), source.engine.state_dict(),
                                           source.optimizer.state_dict(), source.elig, source.heb))))
        return original_clone(source)

    monkeypatch.setattr(probe, "_clone_state", capture_source)
    output = probe.run_graft_probe(tmp_path / "paired", seeds=(2,), train_steps=12, recovery_steps=12)
    result = artifacts(output)["summary"]["results"][0]
    assert len(captured) == 3
    for source, snapshot in captured:
        assert_same_nested((source.weights.detach(), source.engine.state_dict(),
                            source.optimizer.state_dict(), source.elig, source.heb), snapshot)
    assert result["source_state_before_branches"] == result["source_state_after_branches"]
    sham, adaptive, fixed = (result["branches"][name] for name in
                             ("sham", "disrupted_rewire", "disrupted_no_rewire"))
    assert sham["intervention"] is None
    assert adaptive["intervention"] == fixed["intervention"]
    assert adaptive["intervention"]["swaps"] == 6
    assert adaptive["initial_active_indices"] == fixed["initial_active_indices"]
    assert adaptive["immediate_error"] == fixed["immediate_error"]
    assert adaptive["newborn_max_abs_weight"] == fixed["newborn_max_abs_weight"] == 0.0
    assert sham["immediate_error"] == result["source"]["error_after_target_switch"]
    for branch in result["branches"].values():
        assert branch["initial_bucket_active"] == branch["final_bucket_active"] == [6, 6]
        assert len(branch["final_active_indices"]) == 12
    assert fixed["final_edge_age"] != fixed["initial_edge_age"]


def test_complete_curves_replay_exactly_without_advancing_shared_randomness(tmp_path):
    probe = probe_api()
    torch.manual_seed(24)
    np.random.seed(24)
    torch_before, numpy_before = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
    first = probe.run_graft_probe(tmp_path / "one", seeds=(0, 1), train_steps=6, recovery_steps=7)
    second = probe.run_graft_probe(tmp_path / "two", seeds=(0, 1), train_steps=6, recovery_steps=7)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert_same_nested(np.random.get_state()[:1], numpy_before[:1])
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    for name in ("manifest", "curves", "summary"):
        assert (first / f"{name}.json").read_bytes() == (second / f"{name}.json").read_bytes()
    data = artifacts(first)
    assert [row["seed"] for row in data["summary"]["results"]] == [0, 1]
    rows = data["curves"]["rows"]
    assert len(rows) == 2 * (7 + 3 * 8)
    for seed in (0, 1):
        assert [row["step"] for row in rows if row["seed"] == seed and row["phase"] == "training"] == list(range(7))
        for branch in ("sham", "disrupted_rewire", "disrupted_no_rewire"):
            selected = [row for row in rows if row["seed"] == seed and row["branch"] == branch]
            assert [row["step"] for row in selected] == list(range(8))
            assert all(row["bucket_active"] == [6, 6] for row in selected)
            assert all(np.isfinite(row["mse"]) for row in selected)
    manifest = data["manifest"]
    assert manifest["versions"]["torch"] and manifest["versions"]["numpy"]
    assert set(manifest["source_sha256"]) >= {"graftbrain/graft_probe.py", "graftbrain/plasticity.py"}
    assert all(len(value) == 64 for value in manifest["source_sha256"].values())
    assert all(len(row["recovery_data_sha256"]) == 64 for row in data["summary"]["results"])


def test_existing_output_is_refused_before_constructing_any_training_state(tmp_path, monkeypatch):
    probe = probe_api()
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "preserve.txt"
    marker.write_text("original")

    def must_not_construct(*args, **kwargs):
        pytest.fail("existing output must fail before model construction")

    monkeypatch.setattr(probe, "_make_state", must_not_construct)
    with pytest.raises(FileExistsError):
        probe.run_graft_probe(existing)
    assert marker.read_text() == "original"
    assert list(existing.iterdir()) == [marker]


@pytest.mark.parametrize("kwargs", [
    {"seeds": ()}, {"seeds": (0, 0)}, {"seeds": (-1,)}, {"seeds": (True,)},
    {"seeds": (1.5,)}, {"train_steps": 0}, {"train_steps": True}, {"train_steps": 1.5},
    {"recovery_steps": -1}, {"recovery_steps": True}, {"recovery_steps": 1.5},
])
def test_invalid_probe_parameters_fail_without_creating_output(tmp_path, kwargs):
    probe = probe_api()
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        probe.run_graft_probe(output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("values, auc_mean, tail_mean, tail_count", [
    ([100.0, 4.0, 2.0, 6.0], 59.0 / 3.0, 4.0, 3),
    ([100.0, *range(1, 13)], 122.0 / 12.0, 7.5, 10),
    ([10.0, 2.0], 6.0, 2.0, 1),
])
def test_trajectory_metrics_include_both_auc_endpoints_and_exclude_step_zero_from_tail(
        values, auc_mean, tail_mean, tail_count):
    # Catches endpoint-only AUC, off-by-one duration, and a tail containing step 0.
    rows = [{"step": step, "mse": value} for step, value in enumerate(values)]
    metrics = probe_api()._trajectory_metrics(rows)
    assert metrics["recovery_auc_mean"] == pytest.approx(auc_mean)
    assert metrics["tail_mean_error"] == pytest.approx(tail_mean)
    assert metrics["tail_update_count"] == tail_count
    assert metrics["recovery_steps"] == len(values) - 1
    assert metrics["final_error"] == values[-1]


def test_rewire_error_measurements_bracket_support_edit_after_optimizer_step(tmp_path, monkeypatch):
    # Catches using the previous curve point, which includes a weight update.
    probe = probe_api()
    graph = probe._graph()
    _, target = probe._targets()
    evaluation = probe._data(2, 2, 103).reshape(128, graph.n)
    original_after = probe.GraftPlasticity.after_update
    original_step = torch.optim.Adam.step
    observed, optimizer_steps = [], []

    def record_optimizer(optimizer, *args, **kwargs):
        answer = original_step(optimizer, *args, **kwargs)
        optimizer_steps.append(len(optimizer_steps) + 1)
        return answer

    def bracket_edit(engine, weights, optimizer, elig, heb, allow_rewire=True):
        state = probe._ProbeState(engine, weights, optimizer, elig, heb)
        before = probe._error(state, graph, evaluation, target)
        step_count = len(optimizer_steps)
        event = original_after(engine, weights, optimizer, elig, heb, allow_rewire)
        after = probe._error(state, graph, evaluation, target)
        if event is not None:
            observed.append((step_count, before, after))
        return event

    monkeypatch.setattr(torch.optim.Adam, "step", record_optimizer)
    monkeypatch.setattr(probe.GraftPlasticity, "after_update", bracket_edit)
    state = probe._make_state(graph, 2)
    rows, events, _ = probe._fit(state, graph, probe._data(2, 10, 102), target, evaluation,
                                seed=2, phase="recovery", branch="sham", allow_rewire=True)
    assert len(events) == 1
    event = events[0]
    assert event["step"] == 10
    assert observed[0][0] == 10
    assert event["pre_rewire_mse"] == observed[0][1]
    assert event["post_rewire_mse"] == observed[0][2] == rows[-1]["mse"]
    assert event["rewire_mse_delta"] == pytest.approx(observed[0][2] - observed[0][1])
    assert event["pre_rewire_mse"] != rows[-2]["mse"]


def test_old_target_measurement_keeps_training_objective_and_learned_state_identical():
    # Catches accidentally including the old labels in the loss or gradient hooks.
    probe = probe_api()
    graph = probe._graph()
    before, after = probe._targets()
    source = probe._make_state(graph, 4)
    source_hash = probe._state_digest(source)
    control, measured = probe._clone_state(source), probe._clone_state(source)
    batches = probe._data(4, 12, 102)
    evaluation = probe._data(4, 2, 103).reshape(128, graph.n)
    options = dict(seed=4, phase="recovery", branch="sham", allow_rewire=True)
    control_rows, control_events, _ = probe._fit(control, graph, batches, after, evaluation, **options)
    measured_rows, measured_events, _ = probe._fit(measured, graph, batches, after, evaluation,
                                                 old_target=before, **options)
    assert probe._state_digest(control) == probe._state_digest(measured)
    assert probe._state_digest(source) == source_hash
    assert control_events == measured_events
    assert [{key: value for key, value in row.items() if key != "old_target_mse"}
            for row in measured_rows] == control_rows
    assert measured_rows[0]["old_target_mse"] == probe._error(source, graph, evaluation, before)
    assert measured_rows[-1]["old_target_mse"] == probe._error(measured, graph, evaluation, before)
    assert all(np.isfinite(row["old_target_mse"]) for row in measured_rows)


def test_probe_v2_records_primary_paired_auc_and_old_target_tradeoffs(tmp_path):
    probe = probe_api()
    output = probe.run_graft_probe(tmp_path / "metrics", seeds=(3, 5), train_steps=10, recovery_steps=12)
    data = artifacts(output)
    manifest, summary = data["manifest"], data["summary"]
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert manifest["source_sha256_end"] == manifest["source_sha256"]
    assert manifest["source_changed_during_run"] is False
    assert manifest["metrics"]["primary"] == "recovery_auc_mean"
    comparison = summary["comparison"]
    assert comparison["primary"]["metric"] == "recovery_auc_mean"
    assert set(comparison["secondary"]) == {"tail_mean_error", "final_error"}
    for result in summary["results"]:
        seed = result["seed"]
        for branch, branch_result in result["branches"].items():
            rows = [row for row in data["curves"]["rows"]
                    if row["seed"] == seed and row["branch"] == branch]
            values = [row["mse"] for row in rows]
            expected_auc = sum((first + second) / 2.0
                               for first, second in zip(values, values[1:])) / 12
            assert branch_result["recovery_auc_mean"] == pytest.approx(expected_auc)
            assert branch_result["tail_mean_error"] == pytest.approx(sum(values[-10:]) / 10)
            assert branch_result["tail_update_count"] == 10
            tradeoff = branch_result["old_target_tradeoff"]
            assert tradeoff["initial_error"] == rows[0]["old_target_mse"]
            assert tradeoff["final_error"] == rows[-1]["old_target_mse"]
            assert tradeoff["change_from_recovery_start"] == pytest.approx(
                rows[-1]["old_target_mse"] - rows[0]["old_target_mse"])
            assert tradeoff["change_from_source_end"] == pytest.approx(
                rows[-1]["old_target_mse"] - result["source"]["error_before_switch"])
        fixed, adaptive = (result["branches"][name] for name in
                           ("disrupted_no_rewire", "disrupted_rewire"))
        for paired, metric in ((comparison["primary"], "recovery_auc_mean"),
                               (comparison["secondary"]["tail_mean_error"], "tail_mean_error"),
                               (comparison["secondary"]["final_error"], "final_error")):
            row = next(row for row in paired["per_seed"] if row["seed"] == seed)
            assert row["difference"] == pytest.approx(fixed[metric] - adaptive[metric])
        old_row = next(row for row in comparison["per_seed"] if row["seed"] == seed)
        new_row = next(row for row in comparison["secondary"]["final_error"]["per_seed"]
                       if row["seed"] == seed)
        assert old_row == new_row


def test_probe_failure_records_status_without_overwriting_existing_artifacts(tmp_path, monkeypatch):
    probe = probe_api()

    def fail_seed(*args, **kwargs):
        raise RuntimeError("controlled probe failure")

    monkeypatch.setattr(probe, "_run_seed", fail_seed)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="controlled probe failure"):
        probe.run_graft_probe(output, seeds=(4,), train_steps=2, recovery_steps=2)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "RuntimeError: controlled probe failure"
    assert manifest["source_sha256_end"] == manifest["source_sha256"]
    assert manifest["source_changed_during_run"] is False
