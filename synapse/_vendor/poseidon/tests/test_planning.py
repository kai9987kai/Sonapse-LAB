import pytest
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.planning import ModelPredictivePlanner, PlanningConfig, evaluate_predicted_observation
from poseidon.world import rollout


def test_evaluate_predicted_observation_penalizes_hazards_and_rewards_vitality():
    safe_obs = [1.0, 0.8, 0.8, 0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.0, 0.1]
    perilous_obs = [0.2, 0.05, 0.05, 0.02, 0.9, 0.8, 0.1, 0.1, 0.1, 0.9, 0.8, 0.1, 0.8, 0.1, 0.0, 0.9]

    safe_score = evaluate_predicted_observation(safe_obs, action=0)
    peril_score = evaluate_predicted_observation(perilous_obs, action=1)
    assert safe_score > peril_score
    assert peril_score < 0.0


def test_model_predictive_planner_outputs_valid_actions_and_futures(tmp_path):
    model = TidalCore(CoreConfig(hash_buckets=64, hidden_size=24))
    ckpt = tmp_path / "core.pt"
    save_checkpoint(ckpt, model, receipt={})
    core = CoreRuntime(ckpt)

    planner = ModelPredictivePlanner(core, PlanningConfig(horizon=2))
    obs = [0.9, 0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5, 0.0, 0.1]
    decision = planner.plan(obs)

    assert 0 <= decision["action"] < 6
    assert 0 <= decision["mpc_action"] < 6
    assert 0 <= decision["policy_action"] < 6
    assert len(decision["predicted_futures"]) == 6
    assert planner.act(obs) == decision["action"]


def test_mpc_rollout_runs_and_survives(tmp_path):
    from poseidon.core import active_core_path
    core = CoreRuntime(active_core_path("."))
    planner = ModelPredictivePlanner(core, PlanningConfig(horizon=2))
    ep = rollout(planner, seed=123, max_steps=64)
    assert ep["steps"] == 64
    assert ep["survived"]
