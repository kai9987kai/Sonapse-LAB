import pytest
from poseidon.metamorph import MetamorphController
from poseidon.world import rollout


class DummyCore:
    def act(self, obs):
        return 0  # Rest


def test_metamorph_controller_plan_telemetry():
    core = DummyCore()
    controller = MetamorphController(core=core)
    controller.reset(scarcity=2.0)

    # Standard observation
    obs = [0.9, 0.9, 0.9, 0.8, 0.1, 0.2, 0.3, 0.0, 0.8, 0.2, 0.5, 0.5, 0, 0.1, 0.8, 1]
    decision = controller.plan(obs)

    assert "action" in decision
    assert 0 <= decision["action"] <= 5
    assert "action_source" in decision
    assert "instar_stage" in decision
    assert decision["instar_stage"] in ("larval", "pupa", "imago")
    assert "beacon_phase" in decision
    assert "spectral_divergence" in decision
    assert "causal_flux_magnitude" in decision
    assert "wave_interference" in decision


def test_metamorph_controller_rollout():
    core = DummyCore()
    controller = MetamorphController(core=core)
    controller.reset(scarcity=1.0)

    ep = rollout(controller, seed=42, max_steps=16, scarcity=1.0)
    assert 1 <= ep["steps"] <= 16
    assert isinstance(ep["survived"], bool)
    assert ep["reward"] is not None
