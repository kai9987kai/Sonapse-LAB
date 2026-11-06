"""Comprehensive tests for Supermix-derived architectural innovations.

Validates:
1. ConnectomeNeurogenesisEngine (Archimedes v93 continuous structural plasticity)
2. HomeostaticEngine (Supermix Beyond / Poseidon continuous physiological drives)
3. MiMoMixCognitiveCore (MiMoMix dual-process System 1 reflex vs System 2 deliberative ACT)
4. ComplexWorld (dynamic moving predators, terrain friction fields, dynamic gates, frontier tasks)
"""
import numpy as np
import pytest
import torch
from synapse import sources
from synapse.homeostasis import HomeostaticEngine, HomeostaticState
from synapse.neurogenesis import ConnectomeNeurogenesisEngine
from synapse.mimomix_reasoner import MiMoMixCognitiveCore
from synapse.complex_world import ComplexWorld, ALL_TASKS, COMPLEX_TASKS
from synapse.runtime import LabRuntime
from synapse.config import SynapseConfig


def test_homeostatic_engine_dynamics_and_governor():
    engine = HomeostaticEngine(seed=42)
    init_state = engine.state
    assert init_state.energy == 0.85
    assert init_state.stamina == 1.00
    assert init_state.health == 1.00

    # Exhaustive sprint action: high thrust and reach
    sprint_act = np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(50):
        engine.step(sprint_act, reward=0.0, hazard_dist=10.0, in_friction_zone=True)

    assert engine.state.stamina < 1.00
    assert engine.state.energy < 0.85
    assert engine.state.hunger > 0.15

    # Max allowed thrust throttling under exhaustion
    max_thrust = engine.get_max_allowed_thrust()
    assert max_thrust <= 1.0

    # Feeding replenishes energy and drops hunger
    engine.step(np.zeros(5, dtype=np.float32), reward=1.0, consumed_food=True)
    assert engine.state.energy > 0.5
    assert engine.total_food_consumed >= 1

    # Neuromodulator coupling
    da, sht, oa, ach = engine.modulate_neuromodulators(base_da=0.1, base_5ht=0.1, base_oa=0.1, base_ach=0.1)
    assert da >= 0.1  # Hunger amplifies DA
    assert sht >= 0.1
    assert oa >= 0.1

    # State serialization roundtrip
    state_export = engine.export_state()
    engine2 = HomeostaticEngine(seed=999)
    engine2.import_state(state_export)
    assert engine2.state.energy == engine.state.energy
    assert engine2.state.stamina == engine.state.stamina
    assert engine2.total_food_consumed == engine.total_food_consumed


def test_mimomix_dual_process_reasoner():
    torch.manual_seed(42)
    core = MiMoMixCognitiveCore(latent_dim=64, action_dim=5, max_ponder_steps=4, deliberation_threshold=0.45)
    latent = torch.randn(64)

    # Low uncertainty -> System 1 Reflex
    delta_s1, trace_s1 = core.deliberate(latent, free_energy=0.1, surprisal=0.1, threat=0.0)
    assert trace_s1.system_engaged == "System 1 (Reflex)"
    assert trace_s1.ponder_steps == 0
    assert torch.allclose(delta_s1, torch.zeros(5))
    assert trace_s1.confidence >= 0.5

    # High uncertainty -> System 2 Deliberative Thinking
    delta_s2, trace_s2 = core.deliberate(latent, free_energy=0.8, surprisal=0.6, threat=0.7)
    assert trace_s2.system_engaged == "System 2 (Deliberation)"
    assert trace_s2.ponder_steps >= 1
    assert trace_s2.ponder_cost > 0.0
    assert delta_s2.shape == (5,)
    assert not torch.allclose(delta_s2, torch.zeros(5))


def test_neurogenesis_structural_remodeling():
    engine = ConnectomeNeurogenesisEngine(num_neurons=16, prune_threshold=0.05, growth_rate=0.1, seed=42)

    # Synthetic co-firing between neuron 0 and neuron 1
    for _ in range(25):
        rates = np.zeros(16, dtype=np.float32)
        rates[0] = 1.0
        rates[1] = 0.95
        rates[5] = 0.8
        engine.record_activity(rates)

    # Mock agent brain & graph
    class MockGraph:
        def __init__(self):
            self.n = 16
            self.graft_pre = np.array([0, 2], dtype=np.int32)
            self.graft_post = np.array([1, 3], dtype=np.int32)
            self.base_pre = np.array([], dtype=np.int32)
            self.base_post = np.array([], dtype=np.int32)
            self.origin = np.zeros(16, dtype=np.int32)

    class MockBrain:
        def __init__(self):
            self.graft_w = torch.nn.Parameter(torch.tensor([0.01, 0.45], dtype=torch.float32))

    class MockAgent:
        def __init__(self):
            self.graph = MockGraph()
            self.brain = MockBrain()

    agent = MockAgent()
    event = engine.evaluate_structural_remodeling(agent, dopamine=1.5, astrocyte_calcium=1.2, threat=0.0)
    assert "pruned" in event
    assert "sprouted" in event
    assert engine.stats.step == 1

    # State roundtrip
    exported = engine.export_state()
    engine2 = ConnectomeNeurogenesisEngine(num_neurons=16, seed=123)
    engine2.import_state(exported)
    assert engine2.activity_steps == engine.activity_steps
    assert np.allclose(engine2.rate_mean, engine.rate_mean)
    assert np.allclose(engine2.rate_cov, engine.rate_cov)


def test_complex_world_frontier_tasks():
    for task_name in COMPLEX_TASKS:
        world = ComplexWorld(seed=42, curriculum=task_name)
        obs = world.reset(task_name)
        assert obs.shape == (155,)

        if task_name == "ecosystem":
            assert len(world.predators) >= 1
            assert len(world.terrain_zones) >= 1
        elif task_name == "predator_evasion":
            assert len(world.predators) >= 2
        elif task_name == "dynamic_arena":
            assert len(world.dynamic_gates) >= 1
        elif task_name == "puzzle_chamber":
            assert len(world.dynamic_gates) >= 1
            assert any(obj.get("kind") == "button" for obj in world.objects)

        # Run several environment steps
        action = np.array([0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(5):
            obs, r, done, info = world.step(action)
            assert obs.shape == (155,)
            assert "terrain_friction" in info


def test_complex_world_runtime_integration(tmp_path):
    for task_name in COMPLEX_TASKS:
        cfg = SynapseConfig(nodes_per_source=16, task=task_name, seed=42)
        runtime = LabRuntime(cfg, tmp_path / task_name)
        snap = runtime.step(2)
        assert "world" in snap
        assert "telemetry" in snap
        assert "homeostasis" in snap["telemetry"]
        assert "mimomix" in snap["telemetry"]
        assert "neurogenesis" in snap["telemetry"]
