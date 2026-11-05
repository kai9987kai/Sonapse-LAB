import pytest
import torch

from poseidon.cognitive_agent import CognitiveAgent, CognitiveDecision
from poseidon.core import CoreRuntime, active_core_path
from poseidon.mnemosyne import MNEMOSYNE_MODES
from poseidon.world import N_ACTIONS, OBS_SIZE, TidePool
from poseidon.worldlab import POMDPTidePool


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_cognitive_agent_act_and_observe(core):
    agent = CognitiveAgent(core=core, mode="intact")
    agent.reset()

    obs = [0.5] * OBS_SIZE
    decision = agent.act(obs)

    assert isinstance(decision, CognitiveDecision)
    assert 0 <= decision.action < N_ACTIONS
    assert isinstance(decision.override, bool)
    assert isinstance(decision.margin, float)
    assert decision.mode == "intact"
    assert 0.0 <= decision.active_sparsity <= 1.0
    assert 0.5 <= decision.neuromodulatory_gain <= 2.0

    d_dict = decision.to_dict()
    assert d_dict["action"] == decision.action

    # Observe transition
    next_obs = [0.52] * OBS_SIZE
    agent.observe_transition(obs, decision.action, 0.5, next_obs, done=False)
    assert agent.predicted_next_obs is None
    assert agent.metacognition.step_counter == 0
    assert len(agent.adapter.replay) == 1

    # An untrained fallback must not invent forecast error or synthetic replay.
    decision2 = agent.act(next_obs)
    assert decision2.surprise == 0.0
    assert decision2.action == core.act(next_obs)


def test_cognitive_agent_rollout_tidepool(core):
    agent = CognitiveAgent(core=core, mode="intact")
    env = TidePool(seed=42, max_steps=8)
    res = agent.rollout(env, max_steps=8)

    assert res["schema"] == "poseidon-cognitive-agent-v2"
    assert res["steps"] > 0
    assert isinstance(res["survived"], bool)
    assert isinstance(res["total_reward"], float)
    assert "override_rate" in res
    assert len(res["decisions"]) == res["steps"]


def test_cognitive_agent_rollout_pomdp(core):
    agent = CognitiveAgent(core=core, mode="intact")
    env = POMDPTidePool(max_steps=8)
    res = agent.rollout(env, max_steps=8)

    assert res["steps"] > 0
    assert isinstance(res["total_reward"], float)


def test_cognitive_agent_matched_lesion_modes(core):
    agent = CognitiveAgent(core=core)
    for mode in MNEMOSYNE_MODES:
        env = TidePool(seed=101, max_steps=4)
        res = agent.rollout(env, max_steps=4, mode=mode)
        assert res["mode"] == mode
        assert res["steps"] > 0


def test_prediction_matches_current_transition_not_previous(core):
    from poseidon.research.oracle_adapter import ResearchOracleAdapter

    class FixedPredictor(ResearchOracleAdapter):
        """Deterministic test forecast with the trained adapter interface."""
        def __init__(self):
            torch.nn.Module.__init__(self)

        def status(self):
            return {"trained": True}

        def forward(self, latent, observation, actions):
            return {"mean_delta": torch.ones_like(observation) * actions[:, None] * 0.01,
                    "calibrated_uncertainty": torch.full((len(observation),), 0.05)}

    agent = CognitiveAgent(core=core, mode="memory_free")
    agent.oracle = FixedPredictor()
    agent.reset()
    # observe_transition remains valid for callers that do not call act first.
    for action, value in ((2, 0.4), (4, 0.7)):
        obs = [value] * OBS_SIZE
        obs_t = torch.tensor([obs])
        with torch.no_grad():
            latent = core.model(["Survive"], obs_t)["memory"]
            out = agent.oracle(latent, obs_t, torch.tensor([action]))
            successor = (obs_t + out["mean_delta"]).clamp(0, 1)[0].tolist()
        agent.observe_transition(obs, action, 0.0, successor)
        assert len(agent.metacognition.surprise_history) == agent.step_count
        assert agent.metacognition.surprise_history[-1] == 0.0


def test_episode_reset_clears_planner_replay_and_dream_state(core):
    agent = CognitiveAgent(core=core)
    agent.act([0.5] * OBS_SIZE)
    agent.observe_transition([0.5] * OBS_SIZE, 0, 0.0, [0.5] * OBS_SIZE)
    assert len(agent.adapter.replay) > 0
    agent.reset()
    assert agent.planner.step_counter == 0
    assert agent.planner.override_history == []
    assert len(agent.adapter.replay) == 0
    assert agent.dream.consolidated_count == 0
    assert agent.dream.dream_history == []


def test_rollout_snapshot_is_after_reset_and_final_health_is_current(core):
    agent = CognitiveAgent(core=core)
    env = TidePool(seed=42, max_steps=2)
    env.step(4)
    result = agent.rollout(env, max_steps=2)
    assert result["initial_snapshot"]["tick"] == 0
    assert result["final_health"] == env.health
    assert result["truncated"] is True
    assert result["death"] is False


@pytest.mark.parametrize("dead, noisy_health", [(True, 0.7), (False, 0.0)])
def test_rollout_uses_true_terminal_status_not_noisy_health(core, dead, noisy_health):
    from poseidon.worldlab import StepResult

    class MisleadingSensor(POMDPTidePool):
        def step(self, action):
            result = super().step(action)
            self.underlying.health = 0.0 if dead else 0.4
            self.underlying.death_reason = "test-death" if dead else None
            self.underlying.done = True
            result.observation[0] = noisy_health
            return StepResult(result.observation, result.reward, True, not dead,
                              {**result.info, "alive": not dead, "death": dead,
                               "death_reason": self.death_reason, "truncated": not dead,
                               "terminal_reason": "death" if dead else "horizon"})

    env = MisleadingSensor(max_steps=1)
    result = CognitiveAgent(core=core).rollout(env, max_steps=1)
    assert result["survived"] is (not dead)
    assert result["death"] is dead
    assert result["truncated"] is (not dead)
    assert result["final_health"] == env.health
