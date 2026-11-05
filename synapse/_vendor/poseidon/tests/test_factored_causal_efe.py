"""Tests for Stage 9: Factored Causal World Model, Deep Active Inference (EFE), and Conformal SPI."""
from __future__ import annotations

import math
import numpy as np
import pytest

from poseidon.nexus_omega.contracts import (
    FactoredPredictionOutput,
    FactoredState,
    CausalEFEDecision,
)
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.causal_efe import CausalEFEPlanner
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.discovery import AutonomousDiscoveryEngine
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld


def test_factored_causal_world_model_partition_and_uncertainty():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=42)
    obs = np.array([0.9, 0.8, 0.7, 0.85, 0.2, 0.1, 0.05, 0.0], dtype=np.float32)
    s_endo, s_exo = wm.partition_observation(obs)

    assert s_endo.shape == (4,)
    assert s_exo.shape == (4,)
    assert np.allclose(s_endo, [0.9, 0.8, 0.7, 0.85])
    assert np.allclose(s_exo, [0.2, 0.1, 0.05, 0.0])

    mem = np.zeros(18, dtype=np.float32)
    pred = wm.predict_factored(obs, mem, action=0)

    assert isinstance(pred, FactoredPredictionOutput)
    assert pred.predicted_obs.shape == (8,)
    assert pred.endogenous_pred.shape == (4,)
    assert pred.exogenous_pred.shape == (4,)
    assert pred.regime_pred.shape == (2,)
    assert pred.epistemic_uncertainty_total >= 0.0
    assert pred.endogenous_uncertainty >= 0.0
    assert pred.exogenous_uncertainty >= 0.0
    assert pred.regime_uncertainty >= 0.0

    # Test comparator residual and storm pulse regime reset
    true_next = obs.copy()
    true_next[5] = 0.95  # Storm pulse cue triggered
    res = wm.compute_comparator_residual(true_next)
    assert res.shape == (8,)
    assert wm.z_regime[0] == 3.0  # Storm duration reset to 3.0 ticks


def test_causal_efe_planner_pragmatic_epistemic_and_conformal():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=101)
    planner = CausalEFEPlanner(world_model=wm, n_actions=4, base_epistemic_weight=0.4)

    # 1. Pragmatic value: ideal vitals vs starving vitals
    ideal_endo = np.array([1.0, 0.9, 0.9, 0.9], dtype=np.float32)
    starving_endo = np.array([0.2, 0.1, 0.1, 0.1], dtype=np.float32)

    val_ideal = planner.compute_pragmatic_value(ideal_endo)
    val_starving = planner.compute_pragmatic_value(starving_endo)
    assert val_ideal == 0.0
    assert val_starving < -5.0

    # 2. Epistemic drive: curiosity blooms when safe, suppressed when depleted
    safe_obs = np.array([1.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    depleted_obs = np.array([0.2, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    drive_safe = planner.compute_epistemic_drive(safe_obs)
    drive_depleted = planner.compute_epistemic_drive(depleted_obs)
    assert drive_safe > drive_depleted
    assert drive_depleted >= 0.02

    # 3. Action planning returns valid CausalEFEDecision
    mem = np.zeros(18, dtype=np.float32)
    decision = planner.plan_action(safe_obs, mem)

    assert isinstance(decision, CausalEFEDecision)
    assert 0 <= decision.action < 4
    assert isinstance(decision.action_name, str)
    assert math.isfinite(decision.free_energy_objective)
    assert math.isfinite(decision.conformal_lcb)
    assert decision.conformal_lcb <= decision.free_energy_objective
    assert "endogenous" in decision.factored_state
    assert "exogenous" in decision.factored_state
    assert "causal_regime" in decision.factored_state


def test_conformal_safety_guard_override_under_hazard():
    wm = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=202)
    # Strict safety threshold forces guard override when lower bound drops
    planner = CausalEFEPlanner(world_model=wm, n_actions=4, safety_threshold=10.0)
    
    # Threatened observation: low health (0.4) and severe exposure (0.8)
    threatened_obs = np.array([0.4, 0.5, 0.5, 0.5, 0.8, 0.9, 0.5, 0.5], dtype=np.float32)
    mem = np.zeros(18, dtype=np.float32)
    dec = planner.plan_action(threatened_obs, mem)

    # Conformal guard must have activated and overridden risky action to shelter (action 1)
    assert dec.conformal_override is True
    assert dec.action == 1  # Shelter


def test_nexus_omega_agent_with_causal_efe_loop():
    agent = NexusOmegaAgent(d_obs=8, d_working=16, n_actions=4, use_causal_efe=True, base_seed=777)
    agent.reset()

    obs = np.array([0.9, 0.85, 0.8, 0.85, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    action = agent.act(obs)

    assert 0 <= action < 4
    dec = agent.get_last_decision()
    assert dec is not None
    assert dec.action == action

    factored = agent.get_factored_state()
    assert isinstance(factored, FactoredState)
    assert len(factored.endogenous_vitals) == 4
    assert len(factored.exogenous_field) == 4

    next_obs = obs.copy()
    next_obs[0] -= 0.01
    e_t = agent.observe(next_obs, reward=0.4, done=False)
    assert e_t.shape == (8,)


def test_discovery_engine_new_causal_hypotheses():
    engine = AutonomousDiscoveryEngine(max_steps=16)
    hypotheses = engine.generate_hypotheses()

    hyp_ids = {h.hypothesis_id for h in hypotheses}
    assert "hyp_factored_causal_invariance" in hyp_ids
    assert "hyp_epistemic_active_inference_survival" in hyp_ids
    assert "hyp_conformal_safety_guard_zero_catastrophe" in hyp_ids

    # Run interventional trial on each new hypothesis
    for hid in (
        "hyp_factored_causal_invariance",
        "hyp_epistemic_active_inference_survival",
        "hyp_conformal_safety_guard_zero_catastrophe",
    ):
        r_base = engine.run_interventional_trial(hid, seed=50001, arm="baseline")
        r_cand = engine.run_interventional_trial(hid, seed=50001, arm="candidate")
        assert math.isfinite(r_base)
        assert math.isfinite(r_cand)
