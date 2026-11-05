"""Tests for TetherReliabilityGate, TetherOracleAdapter, and Causal 4-View Training."""

import pytest
import torch
from poseidon.research.tether_gate import TetherReliabilityGate, TetherOracleAdapter
from poseidon.research.tether_training import (
    TETHER_ARMS,
    create_synthetic_tether_batch,
    tether_training_step,
    evaluate_gate_resilience,
)
from poseidon.research.oracle_adapter import ResearchOracleAdapter
from poseidon.research.oracle_model import OracleConfig, OracleEnsemble
from poseidon.research.oracle_training import fit_oracle
from poseidon.world import N_ACTIONS, OBS_SIZE


@pytest.fixture(scope="module")
def tether_oracle_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tether-adapter-fit")
    fit_oracle(directory, base_seed=731000000, train_episodes=16, selection_episodes=2,
               calibration_episodes=4, max_steps=32, training_updates=2, batch_size=8,
               hidden_size=16, ensemble_size=3, coverage=0.8)
    return directory / "oracle.pt"


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_tether_adapter_reward_widths_use_member_weights_for_each_batch_row(
    tether_oracle_artifact, batch_size,
):
    adapter = TetherOracleAdapter(tether_oracle_artifact)
    observations = torch.linspace(0.1, 0.9, batch_size * OBS_SIZE).reshape(batch_size, OBS_SIZE)
    actions = torch.arange(batch_size, dtype=torch.long)
    with torch.inference_mode():
        raw = adapter.model(observations, actions)
        actual = adapter(None, observations, actions)

    # Hand-apply the weighted moment identity, independently for each batch row.
    means = []
    variances = []
    for b in range(batch_size):
        weights = actual["tether_weights"][b]
        mean = sum(weights[k] * raw["reward_mean"][k, b] for k in range(3))
        variance = sum(weights[k] * (
            raw["reward_log_variance"][k, b].exp()
            + (raw["reward_mean"][k, b] - mean) ** 2
        ) for k in range(3))
        means.append(mean)
        variances.append(variance)
    expected_half_width = adapter.reward_quantile * torch.stack(variances).sqrt()
    assert actual["mean_reward"].shape == (batch_size,)
    assert actual["reward_interval_half_width"].shape == (batch_size,)
    assert actual["state_interval_half_width"].shape == (batch_size, OBS_SIZE)
    assert torch.allclose(actual["mean_reward"], torch.stack(means), atol=1e-6)
    assert torch.allclose(actual["reward_interval_half_width"], expected_half_width, atol=1e-6)


def test_tether_adapter_preserves_fallback_without_weighted_calibration(tether_oracle_artifact):
    adapter = TetherOracleAdapter(tether_oracle_artifact, epistemic_threshold=0.0)
    supported, reason = adapter.planning_support([0.5] * OBS_SIZE)
    assert supported is False
    assert reason == "tether_weighted_calibration_unavailable"
    assert adapter.status()["calibration_available"] is False
    assert bool(adapter.is_calibrated) is False
    assert adapter.status()["promoted"] is False


def test_tether_adapter_rejects_nonfinite_widths_before_success_counters(tether_oracle_artifact):
    adapter = TetherOracleAdapter(tether_oracle_artifact)
    adapter.state_quantile = 1e308
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter(None, torch.full((1, OBS_SIZE), 0.5), torch.tensor([0]))
    assert adapter.prediction_rows == 0


def test_tether_reliability_gate_forward():
    batch_size = 4
    K = 5
    gate = TetherReliabilityGate(ensemble_size=K, obs_size=OBS_SIZE, num_actions=N_ACTIONS)

    obs = torch.rand(batch_size, OBS_SIZE)
    action = torch.randint(0, N_ACTIONS, (batch_size,))
    deltas = torch.rand(K, batch_size, OBS_SIZE)

    out = gate(obs, action, deltas)

    assert "weights" in out
    assert out["weights"].shape == (batch_size, K)
    # Check weights sum to 1
    assert torch.allclose(out["weights"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    assert out["gated_delta"].shape == (batch_size, OBS_SIZE)
    assert out["epistemic_variance"].shape == (batch_size, OBS_SIZE)
    assert out["total_variance"].shape == (batch_size, OBS_SIZE)
    assert out["epistemic_trust"].shape == (batch_size,)
    assert (out["epistemic_trust"] >= 0.0).all() and (out["epistemic_trust"] <= 1.0).all()
    assert out["abstain"].dtype == torch.bool


def test_tether_reliability_gate_attribution():
    batch_size = 2
    K = 3
    gate = TetherReliabilityGate(ensemble_size=K, obs_size=OBS_SIZE, num_actions=N_ACTIONS)

    obs = torch.rand(batch_size, OBS_SIZE)
    action = torch.randint(0, N_ACTIONS, (batch_size,))
    deltas = torch.rand(K, batch_size, OBS_SIZE)

    attr = gate.attribute(obs, action, deltas)
    assert "saliency" in attr
    assert attr["saliency"].shape == (K, batch_size, OBS_SIZE)
    assert attr["member_saliency"].shape == (K, batch_size)
    assert attr["feature_saliency"].shape == (batch_size, OBS_SIZE)


def test_tether_seven_arms_training_steps():
    batch, clean_deltas = create_synthetic_tether_batch(batch_size=8, ensemble_size=5, seed=123)
    gate = TetherReliabilityGate(ensemble_size=5, obs_size=OBS_SIZE, num_actions=N_ACTIONS)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)

    for arm in TETHER_ARMS:
        optimizer.zero_grad()
        loss, diag = tether_training_step(gate, batch, clean_deltas, arm=arm)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        assert diag["loss"] >= 0.0
        assert "mean_trust" in diag


def test_tether_gate_resilience():
    batch, clean_deltas = create_synthetic_tether_batch(batch_size=16, ensemble_size=5, seed=999)
    gate = TetherReliabilityGate(ensemble_size=5, obs_size=OBS_SIZE, num_actions=N_ACTIONS)

    # Train for a few steps on tether arm to learn suppression
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)
    for _ in range(25):
        optimizer.zero_grad()
        loss, _ = tether_training_step(gate, batch, clean_deltas, arm="tether")
        loss.backward()
        optimizer.step()

    metrics = evaluate_gate_resilience(gate, batch, clean_deltas)
    assert metrics["clean_mse"] >= 0.0
    assert 0.0 <= metrics["clean_trust"] <= 1.0
    assert metrics["one_corrupted_member_weight"] <= 0.35
