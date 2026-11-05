from pathlib import Path

import pytest
import torch

from poseidon.research.oracle_adapter import ResearchOracleAdapter
from poseidon.research.oracle_training import _load_model, fit_oracle, load_oracle_artifact
from poseidon.core import CoreRuntime, active_core_path
from poseidon.cognitive_agent import CognitiveAgent
from poseidon.hyper_agent import HyperAgent
from poseidon.world import TidePool


@pytest.fixture(scope="module")
def oracle_artifact(tmp_path_factory):
    directory = tmp_path_factory.mktemp("adapter-fit")
    fit_oracle(directory, base_seed=631000000, train_episodes=16, selection_episodes=2,
               calibration_episodes=4, max_steps=32, training_updates=2, batch_size=8,
               hidden_size=16, ensemble_size=3, coverage=0.8)
    return directory / "oracle.pt"


def test_adapter_predictions_equal_loaded_research_model(oracle_artifact):
    adapter = ResearchOracleAdapter(oracle_artifact)
    model = _load_model(load_oracle_artifact(oracle_artifact))
    observations = torch.full((2, 16), 0.5)
    actions = torch.tensor([0, 2])
    with torch.inference_mode():
        expected = model.summarize(model(observations, actions))
        actual = adapter(torch.zeros(2, 128), observations, actions)
    assert torch.equal(actual["mean_delta"], expected["delta_mean"])
    assert torch.equal(actual["mean_reward"], expected["reward_mean"])
    assert torch.equal(actual["mean_failure"], expected["death_probability"])
    status = adapter.status()
    assert status["trained"] is True
    assert status["failure_probability_calibrated"] is False
    assert len(status["artifact_sha256"]) == 64


def test_adapter_rejects_invalid_observations_and_missing_artifact(oracle_artifact, tmp_path):
    adapter = ResearchOracleAdapter(oracle_artifact)
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        obs = torch.full((1, 16), 0.5)
        obs[0, 0] = invalid
        with pytest.raises(ValueError, match="observation"):
            adapter(None, obs, torch.tensor([0]))
    with pytest.raises((ValueError, FileNotFoundError)):
        ResearchOracleAdapter(tmp_path / "absent.pt")


def test_imagination_has_no_fabricated_survival_certificate(oracle_artifact):
    adapter = ResearchOracleAdapter(oracle_artifact)
    result = adapter.imagine_trajectory(torch.zeros(128), torch.full((16,), 0.5), [0, 1])
    assert result["survival_probability"] is None
    assert result["failure_probability_calibrated"] is False
    assert result["calibration_applies_to_rollout"] is False
    assert len(result["trajectory"]) == 2
    assert adapter.status()["member_forward_calls"] == 6
    with pytest.raises(ValueError):
        adapter.imagine_trajectory(None, torch.full((16,), 0.5), [True])


@pytest.mark.parametrize("agent_type", [CognitiveAgent, HyperAgent])
def test_default_controller_exactly_preserves_frozen_policy(agent_type):
    core = CoreRuntime(active_core_path())
    agent = agent_type(core=core)
    env = TidePool(seed=631010000, max_steps=20, scarcity=2.5)
    obs = env.reset()
    for _ in range(20):
        decision = agent.act(obs).to_dict()
        assert decision["action"] == core.act(obs)
        assert decision["override"] is False
        assert decision["planning_enabled"] is False
        assert decision["oracle"]["trained"] is False
        next_obs, reward, done, info = env.step(decision["action"])
        agent.observe_transition(obs, decision["action"], reward, next_obs, info)
        obs = next_obs
        if done:
            break


@pytest.mark.parametrize("agent_type", [CognitiveAgent, HyperAgent])
def test_controller_explicitly_loads_artifact(agent_type, oracle_artifact):
    agent = agent_type(core=CoreRuntime(active_core_path()), oracle_checkpoint=oracle_artifact)
    decision = agent.act([0.5] * 16).to_dict()
    assert decision["oracle"]["artifact_sha256"] == agent.oracle.artifact_sha256
    assert decision["oracle"]["trained"] is True
    assert decision["oracle"]["promoted"] is False
    assert decision["planning_enabled"] is True
    assert decision["oracle"]["prediction_rows"] > 0


def test_adapter_source_contract_mismatch_rejected(oracle_artifact, monkeypatch):
    import poseidon.research.oracle_adapter as module
    original = module.source_hashes()
    monkeypatch.setattr(module, "source_hashes", lambda: {**original, "world.py": "0" * 64})
    with pytest.raises(ValueError, match="source mismatch"):
        ResearchOracleAdapter(oracle_artifact)


def test_adapter_limits_planning_to_one_step_and_checks_support(oracle_artifact):
    from poseidon.horizon_x import HorizonXPlanner
    core = CoreRuntime(active_core_path())
    adapter = ResearchOracleAdapter(oracle_artifact)
    planner = HorizonXPlanner(core, adapter)
    result = planner.plan([0.5] * 16)
    assert result["planning_enabled"] is True
    assert all(set(row["horizon_scores"]) == {1} for row in result["candidate_evaluations"])
    adapter.action_counts[4] = 0
    result = planner.plan([0.5] * 16)
    assert result["planning_enabled"] is False
    assert result["fallback_reason"] == "insufficient_training_action_exposure"
    assert result["selected_action"] == core.act([0.5] * 16)


def test_selected_artifact_cannot_claim_3d_transfer(oracle_artifact):
    from poseidon.genesis_3d import ContinuousTidePool3D
    agent = HyperAgent(core=CoreRuntime(active_core_path()), oracle_checkpoint=oracle_artifact)
    with pytest.raises(ValueError, match="continuous 3D"):
        agent.rollout_3d(ContinuousTidePool3D(seed=20), max_steps=2)


def test_runtime_resolves_explicit_artifact_and_defaults_to_baseline(oracle_artifact):
    from poseidon.runtime import Poseidon
    runtime = Poseidon(root=oracle_artifact.parent, oracle_checkpoint=oracle_artifact.name)
    assert runtime.oracle_status()["trained"] is True
    assert Poseidon().oracle_status()["trained"] is False
    with pytest.raises(ValueError, match="missing"):
        Poseidon(oracle_checkpoint="outputs/nonexistent-oracle.pt")


def test_trained_agent_resets_inference_counters(oracle_artifact):
    agent = CognitiveAgent(core=CoreRuntime(active_core_path()), oracle_checkpoint=oracle_artifact)
    agent.act([0.5] * 16)
    assert agent.oracle.prediction_rows > 0
    agent.reset()
    assert agent.oracle.prediction_rows == 0


def test_unrepresentable_interval_fails_before_success_counters(oracle_artifact):
    adapter = ResearchOracleAdapter(oracle_artifact)
    adapter.state_quantile = 1e308
    with pytest.raises(FloatingPointError, match="nonfinite scaled"):
        adapter(None, torch.full((1, 16), 0.5), torch.tensor([0]))
    assert adapter.prediction_rows == 0
