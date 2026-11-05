import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.genesis_3d import ContinuousTidePool3D
from poseidon.hyper_agent import HyperAgent, HyperDecision
from poseidon.world import N_ACTIONS, OBS_SIZE


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_hyper_agent_decision_cycle(core):
    agent = HyperAgent(core=core)
    agent.reset()

    obs = [0.5] * OBS_SIZE
    obs[0] = 1.0  # health
    obs[1] = 0.8  # energy
    obs[2] = 0.8  # hydration

    decision = agent.act(obs)
    assert isinstance(decision, HyperDecision)
    assert 0 <= decision.action < N_ACTIONS
    assert isinstance(decision.override, bool)
    assert isinstance(decision.margin, float)
    assert 0.0 <= decision.quantum_coherence <= 1.0
    assert decision.active_neurons >= 16

    decision_dict = decision.to_dict()
    assert decision_dict["action"] == decision.action


def test_hyper_agent_rollout_3d(core):
    agent = HyperAgent(core=core)
    env = ContinuousTidePool3D(seed=42)
    res = agent.rollout_3d(env, max_steps=8)

    assert res["schema"] == "poseidon-hyper-agent-v3"
    assert res["steps"] > 0
    assert isinstance(res["survived"], bool)
    assert isinstance(res["total_reward"], float)
    assert len(res["decisions"]) == res["steps"]
