"""Tests for CHIMERA Unified Super-Controller."""
import math
from poseidon.chimera import ChimeraController
from poseidon.core import CoreRuntime, active_core_path
from poseidon.world import TidePool, rollout


def test_chimera_initialization_and_reset():
    core = CoreRuntime(active_core_path())
    controller = ChimeraController(core=core)
    controller.reset(scarcity=2.0)
    assert controller.molt.stage.value == "larval"
    assert controller.chronos.tick == 0
    assert len(controller.flow.waypoints) == 0
    assert controller.lattice.estimated_pos == [0.0, 0.0, 0.0]


def test_chimera_plan_telemetry():
    core = CoreRuntime(active_core_path())
    controller = ChimeraController(core=core)
    controller.reset(scarcity=2.5)

    obs = TidePool(seed=123, max_steps=32).observe()
    plan = controller.plan(obs, step=1)

    assert 0 <= plan["action"] <= 5
    assert "action_source" in plan
    assert plan["instar_stage"] in ("larval", "pupa", "imago")
    assert 1 <= plan["beacon_phase"] <= 8
    assert 0.0 <= plan["pva_coherence"] <= 1.0
    assert 0.0 <= plan["lattice_coherence"] <= 1.0
    assert plan["quantum_entropy"] >= 0.0
    assert 0.0 <= plan["confidence"] <= 1.0
    assert math.isfinite(plan["spectral_divergence"])


def test_chimera_rollout():
    core = CoreRuntime(active_core_path())
    controller = ChimeraController(core=core)
    controller.reset(scarcity=2.5)

    result = rollout(controller, seed=99000001, max_steps=32)
    assert 1 <= result["steps"] <= 32
    assert isinstance(result["survived"], bool)
    assert math.isfinite(result["reward"])
