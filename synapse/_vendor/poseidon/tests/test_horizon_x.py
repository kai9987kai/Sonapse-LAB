import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.horizon_x import HorizonXConfig, HorizonXPlanner
from poseidon.oracle import OracleConfig, OracleEnsemble
from poseidon.world import N_ACTIONS, OBS_SIZE


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


@pytest.fixture(scope="module")
def oracle():
    cfg = OracleConfig(ensemble_size=2, hidden_size=64)
    return OracleEnsemble(cfg)


def test_dynamic_reserve_modulation(core, oracle):
    planner = HorizonXPlanner(core=core, oracle=oracle, config=HorizonXConfig(base_lambda=1.0, min_risk_scale=0.1))

    # Well-fed observation: health=1.0, energy=0.9, hydration=0.8
    healthy_obs = [0.0] * OBS_SIZE
    healthy_obs[0] = 1.0
    healthy_obs[1] = 0.9
    healthy_obs[2] = 0.8
    rho_high = planner.compute_reserve_index(healthy_obs)
    assert abs(rho_high - 0.8) < 1e-5
    lambda_high = planner.compute_dynamic_lambda(healthy_obs)
    assert abs(lambda_high - 0.8) < 1e-5

    # Starving observation: health=0.5, energy=0.02, hydration=0.03
    starving_obs = [0.0] * OBS_SIZE
    starving_obs[0] = 0.5
    starving_obs[1] = 0.02
    starving_obs[2] = 0.03
    rho_low = planner.compute_reserve_index(starving_obs)
    assert abs(rho_low - 0.01) < 1e-5
    # Should be clamped at min_risk_scale (0.1 * 1.0 = 0.1)
    lambda_low = planner.compute_dynamic_lambda(starving_obs)
    assert abs(lambda_low - 0.1) < 1e-5
    assert lambda_low < lambda_high


def test_multi_horizon_planning_and_ledger(core, oracle):
    cfg = HorizonXConfig(horizons=(1, 4), override_threshold=0.01)
    planner = HorizonXPlanner(core=core, oracle=oracle, config=cfg)

    obs = [0.5] * OBS_SIZE
    res = planner.plan(obs)

    assert "selected_action" in res
    assert 0 <= res["selected_action"] < N_ACTIONS
    assert "base_action" in res
    assert "override" in res
    assert "margin" in res
    assert "dynamic_lambda" in res
    assert len(res["candidate_evaluations"]) == N_ACTIONS

    assert res["planning_enabled"] is False
    assert res["selected_action"] == core.act(obs)
    for cand in res["candidate_evaluations"]:
        assert cand["horizon_scores"] == {}
        assert cand["evaluated"] is False

    summary = planner.summary()
    assert summary["total_steps"] == 1
    assert "override_count" in summary
    assert "override_rate" in summary
