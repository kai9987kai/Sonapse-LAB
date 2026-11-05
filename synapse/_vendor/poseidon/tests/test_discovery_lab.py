"""Test Suite for Project PROMETHEUS Discovery Lab & Multimodal Enzyme Generator.

Covers:
- Object physics: material densities, Archimedean buoyancy, payload capacity, assemblages
- Manipulation action space: 11 discrete actions
- Multimodal enzyme generator: Michaelis-Menten kinetics, catalytic triads, barrier reduction
- River world 3D simulation: narrow vs wide river, procedural generation, exact replay hashes
- Observation encoder: 76-dimensional object-centric representation
- Curiosity & controllability engine: CASD composite intrinsic reward
- Learned dynamics model: action-conditioned next-state prediction
- Causal hypothesis engine: Causeway do-calculus interventional experiments
- Skill library and hierarchical composer: narrow river stone crossing vs wide river raft construction
- Developmental curriculum runner: Phase 1 through Phase 5 certification
- Paired statistical evaluator: bootstrap confidence intervals
- Runtime and server endpoints
"""

import json
import pytest
import numpy as np

from poseidon.discovery_lab.object_physics import (
    MaterialType,
    PhysicalObject,
    MATERIAL_DENSITIES,
    FLUID_DENSITY_WATER,
    GRAVITY,
    AGENT_MASS,
)
from poseidon.discovery_lab.manipulation_actions import (
    ManipulationActionType,
    ACTION_NAMES,
    ACTION_DESCRIPTIONS,
)
from poseidon.discovery_lab.enzyme_generator import (
    SynthesizedEnzyme,
    CatalyticActiveSite,
    MultimodalEnzymeGenerator,
    CATALYTIC_SIGNATURES,
)
from poseidon.discovery_lab.river_world import (
    RiverWorld,
    AgentState,
)
from poseidon.discovery_lab.observation_encoder import (
    DiscoveryObservationEncoder,
)
from poseidon.discovery_lab.curiosity import (
    DiscoveryCuriosityEngine,
    CuriosityDecomposition,
)
from poseidon.discovery_lab.learned_dynamics import (
    ActionConditionedDynamicsModel,
)
from poseidon.discovery_lab.causal_experiments import (
    DiscoveryCausalEngine,
    CausalHypothesis,
)
from poseidon.discovery_lab.skill_library import (
    DiscoverySkillLibrary,
    DiscoveredSkill,
)
from poseidon.discovery_lab.skill_composer import (
    DiscoverySkillComposer,
    CompositionPlan,
    StrategyType,
)
from poseidon.discovery_lab.curriculum import (
    DiscoveryCurriculumRunner,
    CurriculumPhaseResult,
)
from poseidon.discovery_lab.evaluator import (
    DiscoveryEvaluator,
)
from poseidon.runtime import Poseidon


def test_object_physics_buoyancy():
    # Verify material densities
    assert MATERIAL_DENSITIES[MaterialType.GRANITE_STONE] > FLUID_DENSITY_WATER
    assert MATERIAL_DENSITIES[MaterialType.PINE_LOG] < FLUID_DENSITY_WATER
    assert MATERIAL_DENSITIES[MaterialType.BALSA_LOG] < MATERIAL_DENSITIES[MaterialType.PINE_LOG]

    # Granite stone: density = 2.70 -> sinks
    stone = PhysicalObject(
        object_id="stone_test_1",
        name="Granite Stepping Stone",
        material=MaterialType.GRANITE_STONE,
        position=[1.0, 1.0, 0.0],
        dimensions=[0.8, 0.8, 0.8],
    )
    assert not stone.is_buoyant
    assert stone.net_buoyant_force < 0.0
    assert stone.max_support_payload == 0.0

    # Pine log: density = 0.52 -> floats
    log = PhysicalObject(
        object_id="log_test_1",
        name="Pine Log",
        material=MaterialType.PINE_LOG,
        position=[1.0, 2.0, 0.0],
        dimensions=[2.5, 0.6, 0.5],
    )
    assert log.is_buoyant
    assert log.net_buoyant_force > 0.0
    assert log.max_support_payload > 0.0

    # Assemblage of 3 connected logs
    log2 = PhysicalObject(
        object_id="log_test_2",
        name="Pine Log 2",
        material=MaterialType.PINE_LOG,
        position=[1.0, 2.5, 0.0],
        dimensions=[2.5, 0.6, 0.5],
        connected_object_ids=["log_test_1"],
    )
    log3 = PhysicalObject(
        object_id="log_test_3",
        name="Pine Log 3",
        material=MaterialType.PINE_LOG,
        position=[1.0, 3.0, 0.0],
        dimensions=[2.5, 0.6, 0.5],
        connected_object_ids=["log_test_1"],
    )
    log.connected_object_ids = ["log_test_2", "log_test_3"]

    all_objs = {log.object_id: log, log2.object_id: log2, log3.object_id: log3}
    tot_mass, support_payload, can_float = log.compute_assemblage_buoyancy(all_objs)
    assert can_float is True
    assert support_payload > AGENT_MASS  # Can support human agent (70 kg)


def test_manipulation_actions():
    assert len(ManipulationActionType) == 11
    assert ManipulationActionType.MOVE_EAST == 2
    assert ManipulationActionType.CATALYZE == 10
    assert "push" in ACTION_NAMES[ManipulationActionType.PUSH]
    assert "Enzymatic" in ACTION_DESCRIPTIONS[ManipulationActionType.CATALYZE]


def test_enzyme_generator():
    generator = MultimodalEnzymeGenerator(seed=42)

    catalyst = generator.generate_catalyst_for_substrate(
        substrate_type="lignocellulose",
        target_shear_force=300.0,
        ambient_temperature_c=22.0,
        water_exposure=0.8,
    )

    assert isinstance(catalyst, SynthesizedEnzyme)
    assert catalyst.v_max > 50.0
    assert catalyst.k_m > 0.0
    assert catalyst.k_cat > 50.0
    # Thermodynamic barrier reduction should be significant (negative kcal/mol)
    assert catalyst.delta_delta_g_barrier < -6.0
    assert catalyst.cured_bond_strength >= 0.90
    assert catalyst.water_proof_rating >= 0.90
    assert len(catalyst.active_site.pocket_coordinates_3d) == 3
    assert len(catalyst.synthesis_receipt_hash) == 64

    # Dict serialization
    d = catalyst.to_dict()
    assert d["enzyme_id"] == catalyst.enzyme_id
    assert d["cured_bond_strength"] >= 0.90


def test_river_world_determinism():
    # Narrow river (width = 3)
    env1 = RiverWorld(seed=101, river_width=3)
    env2 = RiverWorld(seed=101, river_width=3)

    assert env1.get_state_hash() == env2.get_state_hash()

    # Step both identically
    s1, r1, d1, info1 = env1.step(ManipulationActionType.MOVE_EAST)
    s2, r2, d2, info2 = env2.step(ManipulationActionType.MOVE_EAST)
    assert env1.get_state_hash() == env2.get_state_hash()
    assert info1["agent_position"] == info2["agent_position"]
    assert r1 == r2

    # Verify trajectory replay hash
    replay_hash = env1.verify_exact_replay()
    assert len(replay_hash) == 64

    # Wide river (width = 7)
    env_wide = RiverWorld(seed=202, river_width=7)
    assert env_wide.river_width == 7
    assert env_wide.river_end_x > env_wide.river_start_x + 5


def test_observation_encoder():
    env = RiverWorld(seed=42)
    encoder = DiscoveryObservationEncoder(max_tracked_objects=8)

    all_objs = list(env.objects.values())
    vec = encoder.encode(
        agent_pos=[env.agent.x, env.agent.y, env.agent.z],
        agent_health=env.agent.health,
        agent_stamina=env.agent.stamina,
        held_object=None,
        all_objects=all_objs,
        river_bounds=(float(env.river_start_x), float(env.river_end_x), float(env.water_depth)),
    )

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (encoder.total_dim,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_curiosity_and_dynamics():
    curiosity = DiscoveryCuriosityEngine()
    dynamics = ActionConditionedDynamicsModel(state_dim=24, num_actions=11)

    s_t = np.random.randn(24).astype(np.float32)
    s_next = s_t + 0.1 * np.random.randn(24).astype(np.float32)

    # Dynamics prediction
    mean_pred, disagreement = dynamics.predict(s_t, int(ManipulationActionType.MOVE_EAST))
    assert mean_pred.shape == (24,)
    assert disagreement >= 0.0

    # Train dynamics
    loss = dynamics.train_step(s_t, int(ManipulationActionType.MOVE_EAST), s_next)
    assert loss >= 0.0

    # Curious reward decomposition
    decomp = curiosity.compute_reward(
        prev_obs=s_t,
        action=int(ManipulationActionType.MOVE_EAST),
        next_obs=s_next,
        model_epistemic_variance=disagreement,
        model_prediction_error=float(loss),
        env_reward=0.1,
        info={"event": "moved_east", "reached_goal": False},
    )
    assert isinstance(decomp, CuriosityDecomposition)
    assert decomp.action_cost_a > 0.0


def test_causal_hypotheses():
    causal_engine = DiscoveryCausalEngine()
    assert len(causal_engine.hypotheses) >= 5

    # Test evaluating H1 (stone sinks -> falsifies floating hypothesis)
    h1 = causal_engine.hypotheses["H1_stone_buoyancy"]
    h1.evaluate("sunk")
    h1.evaluate("sunk")
    assert h1.verdict == "FALSIFIED"
    assert len(h1.receipt_hash) == 64

    # Test evaluating H2 (log floats)
    h2 = causal_engine.hypotheses["H2_log_buoyancy"]
    h2.evaluate("floating")
    h2.evaluate("floating")
    assert h2.verdict == "SUPPORTED"


def test_skill_library_and_composer():
    library = DiscoverySkillLibrary()
    assert len(library.skills) >= 4
    for skill_id, skill in library.skills.items():
        assert skill.status == "certified"
        assert len(skill.skill_hash) == 64

    composer = DiscoverySkillComposer(skill_library=library)

    # Test narrow river: Stepping stones should be chosen
    plan_narrow = composer.plan_crossing(
        river_width=3,
        water_depth=1.2,
        stones_available=4,
        logs_available=4,
        previous_strategy_failed=False,
    )
    assert plan_narrow.selected_strategy == StrategyType.STEPPING_STONES
    assert "place_stepping_stone_v1" in plan_narrow.skill_pipeline
    assert plan_narrow.feasibility_score >= 0.85

    # Test wide river: Stepping stones should be rejected as infeasible, raft selected
    plan_wide = composer.plan_crossing(
        river_width=8,
        water_depth=3.0,
        stones_available=4,
        logs_available=4,
        previous_strategy_failed=False,
    )
    assert plan_wide.selected_strategy == StrategyType.BUOYANT_RAFT
    assert "construct_raft_assemblage_v1" in plan_wide.skill_pipeline
    assert plan_wide.feasibility_score >= 0.80


def test_curriculum_runner():
    runner = DiscoveryCurriculumRunner(base_seed=777)
    report = runner.run_full_curriculum()

    assert report["overall_status"] == "PASSED"
    assert len(report["receipt_sha256"]) == 64
    for phase_key in ["phase_1", "phase_2", "phase_3", "phase_4", "phase_5"]:
        assert report["phases"][phase_key]["passed"] is True


def test_discovery_evaluator():
    evaluator = DiscoveryEvaluator(bootstrap_reps=100)
    report = evaluator.evaluate_crossing_benchmark(seeds=(2001, 2002), river_width=3)

    assert report["num_seeds"] == 2
    assert "mean_paired_reward_gain" in report
    assert len(report["eval_sha256"]) == 64


def test_runtime_discovery_endpoints():
    runtime = Poseidon()

    # Status
    status = runtime.discovery_lab_status()
    assert status["ready"] is True
    assert "river_world_simulation" in status["subsystems"]

    # Step
    step_res = runtime.discovery_lab_step(action="move_east", river_width=3, seed=123)
    assert step_res["action_executed"] == "move_east"
    assert step_res["state_hash"] is not None

    # Enzyme
    enz_res = runtime.discovery_lab_enzyme(substrate="lignocellulose", target_shear=300.0)
    assert enz_res["enzyme_id"].startswith("ENZ-")
    assert enz_res["cured_bond_strength"] > 0.90
