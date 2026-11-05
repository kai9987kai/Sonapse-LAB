"""Tests for TITAN Sovereign Frontier Super-Controller."""
import math
from poseidon.core import CoreRuntime, active_core_path
from poseidon.titan import TitanController
from poseidon.world import TidePool
from poseidon.world_controls import rollout_with_observed_transitions


def test_titan_initialization_and_reset():
    core = CoreRuntime(active_core_path())
    controller = TitanController(core=core)
    controller.reset(scarcity=2.5)

    assert controller.molt.stage.value == "larval"
    assert controller.lattice.estimated_pos == [0.0, 0.0, 0.0]
    assert controller.morpheus.current_state.value == "wake"
    assert controller.intermittent.virtual_capacitor >= 0.10
    assert controller.genesis.generation == 1
    assert controller.archimedes.vesicle_inflation == 0.5


def test_titan_plan_telemetry():
    core = CoreRuntime(active_core_path())
    controller = TitanController(core=core)
    controller.reset(scarcity=2.5)

    obs = TidePool(seed=123, max_steps=32).observe()
    plan = controller.plan(obs, step=1)

    assert 0 <= plan["action"] <= 5
    assert "titan" in plan["action_source"]
    assert plan["instar_stage"] in ("larval", "pupa", "imago")
    assert 0.0 <= plan["pva_coherence"] <= 1.0
    assert 0.0 <= plan["lattice_coherence"] <= 1.0
    assert 0.0 <= plan["titan_coherence"] <= 1.0
    assert plan["quantum_entropy"] >= 0.0
    assert math.isfinite(plan["spectral_divergence"])

    # Archimedes telemetry
    assert "buoyancy_force" in plan
    assert "fluid_density" in plan

    # Genesis telemetry
    assert "trophic_richness" in plan
    assert "lineage_digest" in plan

    # QuantumBot telemetry
    assert "bell_fidelity" in plan
    assert plan["total_teleportations"] >= 1

    # S-Video telemetry
    assert "svideo_line_number" in plan


def test_titan_rollout():
    core = CoreRuntime(active_core_path())
    controller = TitanController(core=core)
    episode = rollout_with_observed_transitions(controller, seed=42, max_steps=16, scarcity=2.5)

    assert episode["steps"] == 16
    assert len(controller.morpheus.memory_buffer) == episode["steps"]
    assert controller.morpheus.memory_buffer[-1].next_observation == episode["trajectory"][-1]["next_observation"]
    assert episode["survived"] is True
    assert episode["reward"] > 0.0
