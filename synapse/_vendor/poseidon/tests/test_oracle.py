import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.oracle import OracleConfig, OracleDynamicsHead, OracleEnsemble, OracleRuntime
from poseidon.world import N_ACTIONS, OBS_SIZE


def test_oracle_dynamics_head_shapes():
    head = OracleDynamicsHead(hidden_size=64, latent_size=64, obs_size=OBS_SIZE, action_count=N_ACTIONS, seed=42)
    latent = torch.randn(2, 64)
    obs = torch.randn(2, OBS_SIZE)
    actions = torch.tensor([0, 3], dtype=torch.long)

    out = head(latent, obs, actions)
    assert out["delta"].shape == (2, OBS_SIZE)
    assert out["reward"].shape == (2,)
    assert out["depletion"].shape == (2, 2)
    assert out["failure_prob"].shape == (2,)
    assert (out["failure_prob"] >= 0.0).all() and (out["failure_prob"] <= 1.0).all()


def test_oracle_ensemble_forward_and_uncertainty():
    config = OracleConfig(ensemble_size=3, hidden_size=64, latent_size=64, obs_size=OBS_SIZE)
    ensemble = OracleEnsemble(config)

    latent = torch.randn(4, 64)
    obs = torch.randn(4, OBS_SIZE)
    actions = torch.tensor([1, 2, 4, 5], dtype=torch.long)

    res = ensemble(latent, obs, actions)
    assert res["mean_delta"].shape == (4, OBS_SIZE)
    assert res["mean_reward"].shape == (4,)
    assert res["mean_depletion"].shape == (4, 2)
    assert res["mean_failure"].shape == (4,)
    assert res["raw_disagreement"].shape == (4,)
    assert res["calibrated_uncertainty"].shape == (4,)
    assert (res["raw_disagreement"] >= 0.0).all()
    assert (res["calibrated_uncertainty"] > 0.0).all()
    assert res["deltas"].shape == (3, 4, OBS_SIZE)


def test_oracle_ensemble_calibration():
    ensemble = OracleEnsemble(OracleConfig(ensemble_size=3, hidden_size=32, latent_size=32))
    assert not ensemble.is_calibrated.item()

    disagreements = [0.01 * (i + 1) for i in range(20)]
    errors = [0.02 * (i + 1) + 0.005 for i in range(20)]

    report = ensemble.calibrate(disagreements, errors)
    assert ensemble.is_calibrated.item()
    assert "calib_a" in report
    assert "calib_b" in report
    assert "calib_c" in report
    assert "ece" in report
    assert report["sample_count"] == 20
    assert report["ece"] >= 0.0


def test_oracle_trajectory_imagination():
    ensemble = OracleEnsemble(OracleConfig(ensemble_size=2, hidden_size=32, latent_size=32))
    latent = torch.randn(32)
    obs = torch.full((OBS_SIZE,), 0.5)
    actions = [0, 1, 2, 3]

    res = ensemble.imagine_trajectory(latent, obs, actions)
    assert res["horizon"] == 4
    assert len(res["trajectory"]) == 4
    assert 0.0 <= res["survival_probability"] <= 1.0
    assert len(res["final_predicted_obs"]) == OBS_SIZE
    assert isinstance(res["total_predicted_reward"], float)
    assert isinstance(res["total_uncertainty"], float)


def test_oracle_runtime_plan_and_io(tmp_path):
    core = CoreRuntime(active_core_path())
    runtime = OracleRuntime(core=core)
    obs = [0.5] * OBS_SIZE

    decision = runtime.plan_with_oracle(obs, lookahead_horizon=3)
    assert "selected_action" in decision
    assert 0 <= decision["selected_action"] < N_ACTIONS
    assert "base_action" in decision
    assert "risk_score" in decision
    assert len(decision["evaluations"]) == N_ACTIONS

    # Save & reload test
    ckpt_path = tmp_path / "oracle_test.pt"
    runtime.save(ckpt_path)
    assert ckpt_path.is_file()

    runtime2 = OracleRuntime(core=core, checkpoint_path=ckpt_path)
    assert runtime2.ensemble is not None
