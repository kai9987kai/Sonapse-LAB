"""Train-only affine anchors and selection-only residual shrinkage contracts."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from poseidon.research import oracle_training as training
from poseidon.research.candidate_evaluation import evaluate_oracle, verify_evaluation_receipt
from poseidon.research.contracts import EpisodeSplit, canonical_bytes
from poseidon.research.oracle_model import (
    LINEAR_RESIDUAL_ARCHITECTURE, MLP_ARCHITECTURE, RESIDUAL_SCALE_GRID,
    OracleConfig, OracleEnsemble, linear_delta_features,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fit(path, **overrides):
    options = dict(base_seed=654_000_000, train_episodes=4, selection_episodes=2,
                   calibration_episodes=4, max_steps=10, training_updates=3,
                   batch_size=8, hidden_size=16, ensemble_size=2, coverage=0.8,
                   architecture=LINEAR_RESIDUAL_ARCHITECTURE)
    options.update(overrides)
    return training.fit_oracle(path, **options)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    return _fit(tmp_path_factory.mktemp("linear-residual"))


def _untrained_model():
    split = training.collect_split("train", [654_200_000, 654_200_001], max_steps=8,
                                    data_protocol=training.MIXTURE_DATA_PROTOCOL)
    rows = training._flatten(split)
    coefficients = torch.tensor(training._fit_linear_delta(rows)["coefficients"], dtype=torch.float64)
    model = OracleEnsemble(OracleConfig(hidden_size=16, ensemble_size=2,
                                        architecture=LINEAR_RESIDUAL_ARCHITECTURE),
                           seed=654_000_071, linear_coefficients=coefficients)
    return model, rows, coefficients


def _rewrite(tmp_path, fitted, payload):
    payload["model_sha256"] = training.model_state_sha256(payload["model_state_dict"])
    artifact = tmp_path / "candidate.pt"
    torch.save(payload, artifact)
    receipt = copy.deepcopy(fitted)
    for key in ("config", "training", "training_manifest", "calibration", "model_sha256"):
        receipt[key] = copy.deepcopy(payload[key])
    receipt["artifact_path"] = str(artifact)
    receipt["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    path = tmp_path / "fit_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return artifact, path


def test_initial_residual_is_exact_affine_anchor_without_rng_side_effects():
    rng = torch.random.get_rng_state().clone()
    model, rows, coefficients = _untrained_model()
    assert torch.equal(rng, torch.random.get_rng_state())
    obs, actions, *_ = training._batch(rows)
    anchor = linear_delta_features(obs.double(), actions) @ coefficients
    output = model(obs, actions)
    assert output["delta_mean"].dtype == torch.float64
    for member in range(model.config.ensemble_size):
        assert torch.equal(output["delta_mean"][member], anchor)
        assert torch.count_nonzero(output["delta_residual_mean"][member]) == 0
    assert torch.equal(model.summarize(output)["delta_mean"], anchor)
    assert torch.count_nonzero(output["delta_log_variance"]) > 0
    assert torch.count_nonzero(output["reward_mean"]) > 0


def test_anchor_and_scale_are_frozen_buffers_while_residual_learns():
    model, rows, _ = _untrained_model()
    before = {name: value.clone() for name, value in model.named_buffers()}
    assert all("linear_coefficients" not in name and "residual_scale" not in name
               for name, _ in model.named_parameters())
    obs, actions, deltas, rewards, deaths = training._batch(rows)
    member = model.members[0]
    optimizer = torch.optim.AdamW(member.parameters(), lr=0.01)
    for _ in range(3):
        loss = training._loss(member(obs, actions), deltas, rewards, deaths, 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert all(torch.equal(value, before[name]) for name, value in model.named_buffers())
    assert torch.count_nonzero(member(obs, actions)["delta_residual_mean"]) > 0


def test_scale_changes_only_state_residual_and_replay_reads_raw_residual(fitted):
    payload = training.load_oracle_artifact(fitted["artifact_path"])
    model = training._load_model(payload)
    manifest = fitted["training_manifest"]
    split = training.collect_split("selection", manifest["seed_sets"]["selection"],
                                    max_steps=manifest["max_steps"], data_protocol=training.MIXTURE_DATA_PROTOCOL)
    obs, actions, *_ = training._batch(training._flatten(split))
    anchor, full = model(obs, actions, residual_scale=0.0), model(obs, actions, residual_scale=1.0)
    for key in ("delta_log_variance", "reward_mean", "reward_log_variance", "death_logit"):
        assert torch.equal(anchor[key], full[key])
    assert not torch.equal(anchor["delta_mean"], full["delta_mean"])
    assert torch.count_nonzero(model.summarize(anchor)["delta_epistemic_variance"]) == 0
    expected = training._residual_selection(model, split)
    model.set_residual_scale(0.0)
    assert training._residual_selection(model, split) == expected
    assert all(member.residual_scale.item() == 0.0 for member in model.members)


def test_shrink_grid_uses_equal_episode_weight_and_smaller_scale_for_ties(fitted):
    model = training._load_model(training.load_oracle_artifact(fitted["artifact_path"]))
    split = EpisodeSplit("selection", (
        training.collect_episode(654_300_000, 1.25, max_steps=2),
        training.collect_episode(654_300_001, 1.25, max_steps=20),
    ))
    selection = training._residual_selection(model, split)
    for row in selection["scores"]:
        per_episode = []
        for episode in split.episodes:
            obs, actions, deltas, *_ = training._batch(episode.transitions)
            prediction = model.summarize(model(obs, actions, residual_scale=row["scale"]))["delta_mean"]
            per_episode.append(float((prediction - deltas.double()).square().mean().detach()))
        assert row["state_mse"] == sum(per_episode) / 2
    initial, _, _ = _untrained_model()
    ties = training._residual_selection(initial, split)
    assert len({row["state_mse"] for row in ties["scores"]}) == 1
    assert ties["selected_scale"] == 0.0
    with pytest.raises(ValueError, match="selection partition"):
        training._residual_selection(model, EpisodeSplit("id_test", split.episodes))


def test_fit_partitions_anchor_shrink_and_calibration_without_test_access(monkeypatch, tmp_path):
    events = []
    collected = []
    original_collect = training.collect_split
    original_anchor = training._fit_linear_delta
    original_shrink = training._residual_selection
    original_calibration = training.episode_calibration_scores

    def collect(name, seeds, **kwargs):
        assert name in {"train", "selection", "calibration"}
        collected.append(name)
        return original_collect(name, seeds, **kwargs)

    def anchor(rows):
        assert {row.episode_seed for row in rows} == set(range(654_000_000, 654_000_004))
        events.append("anchor")
        return original_anchor(rows)

    def shrink(model, split):
        assert {episode.seed for episode in split.episodes} == {655_000_000, 655_000_001}
        events.append("shrink")
        return original_shrink(model, split)

    def calibration(model, episode):
        assert events[:2] == ["anchor", "shrink"]
        assert 656_000_000 <= episode.seed < 656_000_004
        assert all(member.residual_scale.item() in RESIDUAL_SCALE_GRID for member in model.members)
        events.append("calibration")
        return original_calibration(model, episode)

    monkeypatch.setattr(training, "collect_split", collect)
    monkeypatch.setattr(training, "_fit_linear_delta", anchor)
    monkeypatch.setattr(training, "_residual_selection", shrink)
    monkeypatch.setattr(training, "episode_calibration_scores", calibration)
    before = torch.random.get_rng_state().clone()
    fit = _fit(tmp_path / "isolated-fit")
    assert torch.equal(before, torch.random.get_rng_state())
    assert collected == ["train", "selection", "calibration"]
    assert events == ["anchor", "shrink"] + ["calibration"] * 4
    selection = fit["training"]["residual_selection"]
    chosen = next(row for row in selection["scores"] if row["scale"] == selection["selected_scale"])
    assert chosen["state_mse"] <= selection["scores"][0]["state_mse"]


def test_new_artifact_roundtrip_and_both_fit_and_evaluation_replay(fitted, tmp_path):
    artifact_path = Path(fitted["artifact_path"])
    assert training.verify_fit_receipt(artifact_path.with_name("fit_receipt.json"))["valid"]
    payload = training.load_oracle_artifact(artifact_path)
    model = training._load_model(payload)
    assert model.config.architecture == LINEAR_RESIDUAL_ARCHITECTURE
    copied = tmp_path / "roundtrip.pt"
    torch.save(payload, copied)
    restored = training._load_model(training.load_oracle_artifact(copied))
    obs, actions, *_ = training._batch(training.collect_episode(654_400_000, 1.25, 4).transitions)
    assert all(torch.equal(value, restored(obs, actions)[key]) for key, value in model(obs, actions).items())
    evaluate_oracle(copied, tmp_path / "evaluation", base_seed=654_000_000,
                    episodes_per_condition=2, bootstrap_reps=100)
    assert verify_evaluation_receipt(tmp_path / "evaluation" / "evaluation_receipt.json", replay=True)["valid"]


@pytest.mark.parametrize("mutation", [
    "coefficient_metadata", "coefficient_buffer", "scale_buffer", "scale_claim", "scale_grid",
    "architecture", "extra_selection_field", "nonfinite_anchor", "double_neural_weight", "anchor_shape",
])
def test_resealed_artifact_rejects_anchor_scale_and_architecture_contradictions(fitted, tmp_path, mutation):
    payload = torch.load(fitted["artifact_path"], weights_only=True)
    state = payload["model_state_dict"]
    selection = payload["training"]["residual_selection"]
    if mutation == "coefficient_metadata":
        payload["training"]["linear_delta_baseline"]["coefficients"][0][0] += 0.25
    elif mutation == "coefficient_buffer":
        state["members.0.linear_coefficients"][0, 0] += 0.25
    elif mutation == "scale_buffer":
        state["members.0.residual_scale"].fill_(0.75)
    elif mutation == "scale_claim":
        selection["selected_scale"] = 0.75
    elif mutation == "scale_grid":
        selection["scores"][1]["scale"] = 0.125
    elif mutation == "architecture":
        payload["config"]["architecture"] = "linear-residual-v999"
    elif mutation == "extra_selection_field":
        selection["test_selected"] = False
    elif mutation == "nonfinite_anchor":
        state["members.0.linear_coefficients"][0, 0] = float("nan")
    elif mutation == "double_neural_weight":
        state["members.0.delta_head.weight"] = state["members.0.delta_head.weight"].double()
    else:
        state["members.0.linear_coefficients"] = torch.zeros(1, dtype=torch.float64)
    artifact, _ = _rewrite(tmp_path, fitted, payload)
    with pytest.raises(ValueError):
        training.load_oracle_artifact(artifact)


def test_self_consistent_fabricated_anchor_fails_train_data_replay(fitted, tmp_path):
    payload = torch.load(fitted["artifact_path"], weights_only=True)
    payload["training"]["linear_delta_baseline"]["coefficients"][0][0] += 0.25
    for index in range(payload["config"]["ensemble_size"]):
        payload["model_state_dict"][f"members.{index}.linear_coefficients"][0, 0] += 0.25
    artifact, receipt = _rewrite(tmp_path, fitted, payload)
    training.load_oracle_artifact(artifact)
    assert training.verify_fit_receipt(receipt, replay_data=False)["valid"]
    with pytest.raises(ValueError, match="replayed linear baseline"):
        training.verify_fit_receipt(receipt, replay_data=True)


@pytest.mark.parametrize("change_scale", [False, True])
def test_self_consistent_fabricated_shrink_scores_fail_selection_replay(fitted, tmp_path, change_scale):
    payload = torch.load(fitted["artifact_path"], weights_only=True)
    selection = payload["training"]["residual_selection"]
    if change_scale:
        other = next(scale for scale in RESIDUAL_SCALE_GRID if scale != selection["selected_scale"])
        selection["selected_scale"] = other
        for row in selection["scores"]:
            row["state_mse"] = 0.0 if row["scale"] == other else 1.0
        for index in range(payload["config"]["ensemble_size"]):
            payload["model_state_dict"][f"members.{index}.residual_scale"].fill_(other)
    else:
        for row in selection["scores"]:
            row["state_mse"] += 1.0
    artifact, receipt = _rewrite(tmp_path, fitted, payload)
    training.load_oracle_artifact(artifact)
    assert training.verify_fit_receipt(receipt, replay_data=False)["valid"]
    with pytest.raises(ValueError, match="replayed residual selection scores"):
        training.verify_fit_receipt(receipt, replay_data=True)


def test_receipt_scale_claim_is_bound_to_artifact_even_without_replay(fitted, tmp_path):
    receipt = copy.deepcopy(fitted)
    receipt["training"]["residual_selection"]["selected_scale"] = 0.75
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    path = tmp_path / "contradiction.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="training contradicts"):
        training.verify_fit_receipt(path, replay_data=False)


def test_historical_mlp_configuration_loads_without_architecture_field(tmp_path):
    fit = _fit(tmp_path / "legacy", architecture=MLP_ARCHITECTURE,
                data_protocol=training.LEGACY_DATA_PROTOCOL, training_updates=1)
    assert "architecture" not in fit["config"]
    assert "residual_selection" not in fit["training"]
    assert training.verify_fit_receipt(tmp_path / "legacy" / "fit_receipt.json")["valid"]
    payload = training.load_oracle_artifact(fit["artifact_path"])
    historical = training._load_model(payload)
    assert all("linear_coefficients" not in key and "residual_scale" not in key
               for key in historical.state_dict())
    payload["config"]["architecture"] = MLP_ARCHITECTURE
    target = tmp_path / "explicit-mlp.pt"
    torch.save(payload, target)
    explicit = training._load_model(training.load_oracle_artifact(target))
    assert all(torch.equal(value, explicit.state_dict()[key]) for key, value in historical.state_dict().items())


def test_new_architecture_rejects_legacy_data_protocol_before_writing(tmp_path):
    destination = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="requires episode-mixture-v2"):
        _fit(destination, data_protocol=training.LEGACY_DATA_PROTOCOL)
    assert not destination.exists()


def test_residual_adapter_keeps_prediction_parity_and_recursive_input_dtype(fitted):
    from poseidon.research.oracle_adapter import ResearchOracleAdapter
    adapter = ResearchOracleAdapter(fitted["artifact_path"])
    obs = torch.full((1, 16), 0.5)
    action = torch.tensor([2], dtype=torch.long)
    with torch.inference_mode():
        expected = adapter.model.summarize(adapter.model(obs, action))
        actual = adapter(None, obs, action)
    assert torch.equal(actual["mean_delta"], expected["delta_mean"])
    assert actual["mean_delta"].dtype == torch.float64
    rollout = adapter.imagine_trajectory(None, obs, [2, 1])
    assert len(rollout["trajectory"]) == 2
    assert rollout["calibration_applies_to_rollout"] is False
    status = adapter.status()
    assert status["architecture"] == LINEAR_RESIDUAL_ARCHITECTURE
    status["residual_selection"]["selected_scale"] = -1
    assert adapter.status()["residual_selection"]["selected_scale"] >= 0
