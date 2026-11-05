"""Tests for FaultlineOracleAdapter."""

import pytest
import torch

from poseidon.research.faultline_adapter import FaultlineOracleAdapter
from poseidon.research.oracle_adapter import ResearchOracleAdapter
from poseidon.research.oracle_training import fit_oracle
from poseidon.research.tether_gate import TetherReliabilityGate
from poseidon.world import N_ACTIONS, OBS_SIZE


@pytest.fixture(scope="module")
def faultline_oracle_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("faultline-adapter-fit")
    fit_oracle(directory, base_seed=732000000, train_episodes=16, selection_episodes=2,
               calibration_episodes=4, max_steps=32, training_updates=2, batch_size=8,
               hidden_size=16, ensemble_size=3, coverage=0.8)
    return directory / "oracle.pt"


def test_faultline_oracle_adapter_initialization():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    assert adapter.panel_size == 3
    assert adapter.primary_policy == "joint"
    assert len(adapter.active_panel) == 3
    assert len(adapter.active_panel_ids) == 3

    receipt = adapter.audit_receipt
    assert receipt is not None
    assert receipt["schema"] == "poseidon-faultline-oracle-audit-v1"
    assert "sha256" in receipt
    assert receipt["joint_advantage"] >= 0.0


def test_faultline_oracle_adapter_set_active_panel():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    adapter.set_active_panel([0, 1, 4])
    assert adapter.active_panel == [0, 1, 4]
    assert adapter.active_panel_ids == ["m1", "m2", "m5"]

    with pytest.raises(ValueError, match="Invalid member index"):
        adapter.set_active_panel([0, 1, 99])

    with pytest.raises(ValueError, match="Panel size mismatch"):
        adapter.set_active_panel([0, 1])


def test_faultline_oracle_adapter_status():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    st = adapter.status()
    assert st["faultline_enabled"] is True
    assert st["faultline_protocol"] == "faultline-oracle-panel-v1"
    assert st["panel_size"] == 3
    assert st["n_available_members"] == 5
    assert len(st["active_panel"]) == 3
    assert st["sha256_receipt"] is not None


def test_faultline_oracle_adapter_forward():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    batch_size = 4
    obs = torch.randn(batch_size, OBS_SIZE)
    actions = torch.randint(0, N_ACTIONS, (batch_size,))

    out = adapter(None, obs, actions)
    assert "mean_delta" in out
    assert out["mean_delta"].shape == (batch_size, OBS_SIZE)
    assert "mean_reward" in out
    assert "mean_failure" in out
    assert "raw_disagreement" in out
    assert "faultline_active_panel" in out
    assert torch.equal(out["faultline_active_panel"], torch.tensor(adapter.active_panel))


def test_faultline_oracle_adapter_planning_support():
    adapter = FaultlineOracleAdapter(panel_size=3, primary_policy="joint", seed=42)
    obs = [0.1] * OBS_SIZE
    ok, reason = adapter.planning_support(obs)
    assert ok is False
    assert reason == "faultline_oracle_unavailable"

    bad_obs = [0.1] * 5
    ok_bad, reason_bad = adapter.planning_support(bad_obs)
    assert ok_bad is False
    assert "invalid_obs_size" in reason_bad


def test_faultline_synthetic_selection_does_not_calibrate_real_oracle_panel(faultline_oracle_artifact):
    oracle = ResearchOracleAdapter(faultline_oracle_artifact)
    assert oracle.status()["calibration_available"] is True
    adapter = FaultlineOracleAdapter(oracle_adapter=oracle, panel_size=1)
    status = adapter.status()
    assert status["panel_selection_scope"] == "canonical_synthetic_fixture_not_oracle"
    assert status["calibration_available"] is False
    assert status["artifact_bound"] is False
    assert bool(adapter.is_calibrated) is False
    assert oracle.status()["calibration_available"] is True
    assert bool(oracle.is_calibrated) is True
    assert all(0 <= member < 3 for member in adapter.active_panel)
    assert adapter.audit_receipt["applied_to_oracle"] is False
    assert adapter.planning_support([0.5] * OBS_SIZE) == (
        False, "faultline_panel_calibration_unavailable",
    )


@pytest.mark.parametrize("panel", [[0], [2]])
def test_faultline_changed_panel_does_not_inherit_all_member_intervals(faultline_oracle_artifact, panel):
    oracle = ResearchOracleAdapter(faultline_oracle_artifact)
    adapter = FaultlineOracleAdapter(oracle_adapter=oracle, panel_size=1)
    adapter.set_active_panel(panel)
    observations = torch.full((2, OBS_SIZE), 0.5)
    actions = torch.tensor([0, 1])
    raw = oracle(None, observations, actions)
    assert "state_interval_half_width" in raw
    actual = adapter(None, observations, actions)
    assert torch.allclose(actual["mean_delta"], raw["deltas"][panel].mean(dim=0))
    assert torch.allclose(actual["mean_reward"], raw["rewards"][panel].mean(dim=0))
    for inherited_interval in ("calibrated_uncertainty", "state_interval_half_width",
                               "reward_interval_half_width"):
        assert inherited_interval not in actual
    assert adapter.status()["calibration_available"] is False
    assert adapter.planning_support([0.5] * OBS_SIZE) == (
        False, "faultline_panel_calibration_unavailable",
    )
