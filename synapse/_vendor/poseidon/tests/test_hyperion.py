"""Tests for HYPERION Unified Frontier Super-Controller."""
import math
from poseidon.core import CoreRuntime, active_core_path
from poseidon.hyperion import HyperionController
from poseidon.world import TidePool, rollout


def test_hyperion_initialization_and_reset():
    core = CoreRuntime(active_core_path())
    controller = HyperionController(core=core)
    controller.reset(scarcity=2.5)

    assert controller.molt.stage.value == "larval"
    assert controller.lattice.estimated_pos == [0.0, 0.0, 0.0]
    assert controller.morpheus.current_state.value == "wake"
    assert controller.intermittent.virtual_capacitor >= 0.10
    assert controller.nexus_search.stats()["indexed_count"] == 0


def test_hyperion_plan_telemetry():
    core = CoreRuntime(active_core_path())
    controller = HyperionController(core=core)
    controller.reset(scarcity=2.5)

    obs = TidePool(seed=123, max_steps=32).observe()
    plan = controller.plan(obs, step=1)

    assert 0 <= plan["action"] <= 5
    assert "hyperion" in plan["action_source"]
    assert plan["instar_stage"] in ("larval", "pupa", "imago")
    assert 0.0 <= plan["pva_coherence"] <= 1.0
    assert 0.0 <= plan["lattice_coherence"] <= 1.0
    assert plan["quantum_entropy"] >= 0.0
    assert 0.0 <= plan["confidence"] <= 1.0
    assert math.isfinite(plan["spectral_divergence"])

    # Morpheus telemetry
    assert plan["sleep_state"] in ("wake", "sws", "rem")
    assert plan["total_sleep_cycles"] >= 0

    # Prometheus telemetry
    assert plan["scarcity_entropy"] >= 0.0
    assert plan["adapted_gain"] >= 0.10
    assert plan["dominant_drive"] in ("hunger", "thirst", "fatigue", "exposure")

    # Intermittent telemetry
    assert plan["virtual_capacitor"] >= 0.0
    assert plan["total_resuscitations"] >= 0

    # NexusSearch telemetry
    assert plan["search_hits_count"] >= 0


def test_hyperion_rollout():
    core = CoreRuntime(active_core_path())
    controller = HyperionController(core=core)
    controller.reset(scarcity=2.5)

    result = rollout(controller, seed=99000001, max_steps=32)
    assert 1 <= result["steps"] <= 32
    assert isinstance(result["survived"], bool)
    assert math.isfinite(result["reward"])
