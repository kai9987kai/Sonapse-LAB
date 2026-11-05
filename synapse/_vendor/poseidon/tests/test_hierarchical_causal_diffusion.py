"""Unit and Integration Tests for Stage 10: Hierarchical Multi-Scale Active Inference
and Guided Counterfactual Trajectory Diffusion Kernel.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from poseidon.nexus_omega.contracts import (
    FactoredState,
    HierarchicalEFEDecision,
    ImaginedTrajectory,
    MacroIntent,
)
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.trajectory_diffusion import TrajectoryDiffusionImaginer
from poseidon.nexus_omega.hierarchical_efe import HierarchicalEFEPlanner
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.discovery import AutonomousDiscoveryEngine
from poseidon.runtime import Poseidon


def test_trajectory_diffusion_imaginer_rollout_and_conformal_lcb():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=101)
    imaginer = TrajectoryDiffusionImaginer(
        world_model=wm,
        default_horizon=8,
        guidance_scale=1.5,
        conformal_quantile=1.96,
        base_seed=101,
    )

    obs = np.array([1.0, 0.9, 0.9, 0.0, 0.05, 0.0, 0.0, 1.0], dtype=np.float32)
    mem = np.zeros(18, dtype=np.float32)
    wm.last_prediction = np.full(8, 0.125, dtype=np.float32)
    wm.last_action = 2
    previous_prediction = wm.last_prediction.copy()

    # 1. Single macro-intent trajectory imagination
    traj = imaginer.imagine_trajectory(obs, mem, target_intent=MacroIntent.FORAGE_EXPEDITION, horizon=8)
    assert isinstance(traj, ImaginedTrajectory)
    assert traj.horizon == 8
    assert len(traj.states) == 8
    assert len(traj.unguided_states) == 8
    assert len(traj.unguided_rewards) == 8
    assert math.isfinite(traj.unguided_cumulative_reward)
    assert math.isfinite(traj.unguided_return_penalty)
    assert len(traj.direct_guidance_adjustments) == 8
    assert len(traj.branch_state_deltas) == 8
    assert len(traj.branch_divergence_l2) == 8
    assert all(np.isfinite(delta).all() for delta in traj.branch_state_deltas)
    assert traj.predictor_evaluations == 16
    assert traj.ensemble_member_forwards == 48
    assert np.array_equal(wm.last_prediction, previous_prediction)
    assert wm.last_action == 2
    assert len(traj.actions) == 8
    assert len(traj.rewards) == 8
    assert len(traj.epistemic_uncertainties) == 8
    assert math.isfinite(traj.cumulative_reward)
    assert math.isfinite(traj.conformal_trajectory_lcb)
    # Conformal lower bound must be <= cumulative expected return
    assert traj.conformal_trajectory_lcb <= traj.cumulative_reward + 1e-4
    assert traj.score_guidance_norm >= 0.0

    # Serialization test
    d = traj.to_dict()
    assert d["schema"] == "poseidon-matched-rollout-v1"
    assert d["predictor_status"] == "untrained_experimental_ensemble"
    assert d["uncertainty_calibrated"] is False
    assert d["horizon"] == 8
    assert len(d["states"]) == 8
    assert len(d["unguided_states"]) == 8
    assert len(d["direct_guidance_adjustments"]) == 8
    assert d["max_branch_divergence_l2"] >= 0.0
    assert len(d["actions"]) == 8
    assert len(d["rewards"]) == 8

    # 2. Evaluate all candidate macro-intents
    all_trajs = imaginer.evaluate_all_intents(obs, mem, horizon=6)
    assert len(all_trajs) == 4
    for intent in MacroIntent:
        assert intent.value in all_trajs
        assert all_trajs[intent.value].horizon == 6


def test_hierarchical_efe_planner_level1_and_level0():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=202)
    planner = HierarchicalEFEPlanner(
        world_model=wm,
        n_actions=4,
        commitment_duration=4,
        base_epistemic_weight=0.4,
        imagination_horizon=8,
        base_seed=202,
    )
    planner.reset()

    # Nominal calm observation
    calm_obs = np.array([1.0, 0.9, 0.9, 0.0, 0.05, 0.0, 0.0, 1.0], dtype=np.float32)
    mem = np.zeros(18, dtype=np.float32)

    dec = planner.plan(calm_obs, mem)
    assert isinstance(dec, HierarchicalEFEDecision)
    assert isinstance(dec.macro_intent, MacroIntent)
    assert 0 <= dec.micro_action < 4
    assert isinstance(dec.micro_action_name, str)
    assert math.isfinite(dec.macro_free_energy)
    assert math.isfinite(dec.micro_free_energy)
    assert dec.imagined_trajectory is not None
    assert dec.imagined_trajectory.horizon == 8

    # Under calm conditions, forage expedition should be preferred
    assert dec.macro_intent in (MacroIntent.FORAGE_EXPEDITION, MacroIntent.HABITAT_RELOCATION)
    assert dec.macro_intent_remaining_ticks > 0

    # Serialization test
    dec_dict = dec.to_dict()
    assert "macro_intent" in dec_dict
    assert "micro_action" in dec_dict
    assert "imagined_trajectory" in dec_dict
    assert dec_dict["imagined_trajectory"] is not None


def test_hierarchical_planner_emergency_interruption_under_hazard():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=303)
    planner = HierarchicalEFEPlanner(
        world_model=wm,
        n_actions=4,
        commitment_duration=5,
        base_seed=303,
    )
    planner.reset()

    # 1. Storm alert pulse hazard observation (cue = 1.0 at index 3)
    storm_obs = np.array([0.9, 0.8, 0.8, 1.0, 0.1, 0.0, 0.0, 1.0], dtype=np.float32)
    mem = np.zeros(18, dtype=np.float32)

    dec_storm = planner.plan(storm_obs, mem)
    # Must immediately trigger Diapause Hibernation and select shelter (action 1)
    assert dec_storm.macro_intent == MacroIntent.DIAPAUSE_HIBERNATION
    assert dec_storm.micro_action == 1  # Shelter

    # 2. Critical health exhaustion observation (health = 0.25)
    crit_obs = np.array([0.25, 0.3, 0.3, 0.0, 0.4, 0.0, 0.0, 1.0], dtype=np.float32)
    dec_crit = planner.plan(crit_obs, mem)
    # Must immediately interrupt and trigger Emergency Fortification
    assert dec_crit.macro_intent == MacroIntent.EMERGENCY_FORTIFICATION
    assert dec_crit.micro_action == 1  # Shelter


def test_nexus_omega_agent_hierarchical_mode():
    agent = NexusOmegaAgent(
        d_obs=8,
        d_working=16,
        n_actions=4,
        use_hierarchical_efe=True,
        base_seed=404,
    )
    agent.reset()

    obs = np.array([1.0, 0.95, 0.95, 0.0, 0.05, 0.0, 0.0, 1.0], dtype=np.float32)

    # 1. Decision loop in hierarchical mode
    action = agent.act(obs)
    assert 0 <= action < 4

    hier_dec = agent.get_last_hierarchical_decision()
    assert hier_dec is not None
    assert isinstance(hier_dec, HierarchicalEFEDecision)
    assert hier_dec.micro_action == action

    # 2. Test observation and comparator error update
    next_obs = obs.copy()
    next_obs[1] -= 0.05
    e_t = agent.observe_transition(next_obs, reward=0.4, done=False)
    assert e_t.shape == (8,)
    assert np.linalg.norm(agent.memory.state.working_state) > 0.0

    # 3. Test prospective trajectory imagination
    imagined = agent.imagine_trajectory(intent=MacroIntent.HABITAT_RELOCATION, horizon=6)
    assert isinstance(imagined, ImaginedTrajectory)
    assert imagined.horizon == 6
    assert len(imagined.states) == 6


def test_runtime_hierarchical_state_and_imagine():
    runtime = Poseidon()

    # 1. Test runtime status reflection
    status = runtime.nexus_omega_status()
    assert status["hierarchical_efe_planner"] is True
    assert status["score_guided_trajectory_diffusion"] is True
    assert "hierarchical_efe_planner" in status["subsystems"]
    assert "forage_expedition" in status["macro_intents"]

    # 2. Test runtime hierarchical state query
    h_state = runtime.nexus_omega_hierarchical_state(seed=505)
    assert "micro_action" in h_state
    assert "macro_intent" in h_state
    assert "decision" in h_state
    assert "imagined_trajectory" in h_state
    assert h_state["imagined_trajectory"] is not None
    assert h_state["imagined_trajectory"]["horizon"] == 8

    # 3. Test runtime prospective imagination query
    traj_dict = runtime.nexus_omega_imagine(intent="diapause_hibernation", horizon=5, seed=505)
    assert traj_dict["horizon"] == 5
    assert len(traj_dict["states"]) == 5
    assert len(traj_dict["actions"]) == 5
    assert "conformal_trajectory_lcb" in traj_dict


def test_discovery_engine_stage10_hypotheses():
    engine = AutonomousDiscoveryEngine(max_steps=16)
    hypotheses = engine.generate_hypotheses()
    hyp_ids = [h.hypothesis_id for h in hypotheses]

    assert "hyp_hierarchical_temporal_abstraction" in hyp_ids
    assert "hyp_score_guided_diffusion_imagination" in hyp_ids

    # Run interventional trial on the new hypotheses under hazard seed
    r_cand_hier = engine.run_interventional_trial("hyp_hierarchical_temporal_abstraction", seed=50001, arm="candidate")
    r_base_hier = engine.run_interventional_trial("hyp_hierarchical_temporal_abstraction", seed=50001, arm="baseline")
    assert r_cand_hier > r_base_hier

    r_cand_diff = engine.run_interventional_trial("hyp_score_guided_diffusion_imagination", seed=50001, arm="candidate")
    r_base_diff = engine.run_interventional_trial("hyp_score_guided_diffusion_imagination", seed=50001, arm="baseline")
    assert r_cand_diff >= r_base_diff


    # Verify statistical evaluation across matched seed ensemble
    hyp_hier = next(h for h in hypotheses if h.hypothesis_id == "hyp_hierarchical_temporal_abstraction")
    res_hier = engine.test_hypothesis(hyp_hier, seed_base=50000, num_seeds=8)
    assert res_hier.verdict in ("ACCEPTED", "INCONCLUSIVE")
    assert res_hier.effect_size > 0.30
    assert res_hier.reward_advantage > 0.0
