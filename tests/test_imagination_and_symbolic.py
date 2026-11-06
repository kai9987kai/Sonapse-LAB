"""Tests for Budgeted Imagination, Intuitive Physics, Causal Planning, Inverse Dynamics, and Hierarchical Predictive Coding (Items 31-35).
"""
import numpy as np
import torch
from synapse.imagination import (
    BudgetedImaginationEngine,
    LearnablePhysicsIntuition,
    NeuroSymbolicCausalPlanner,
    InverseDynamicsReasoning,
    HierarchicalPredictiveCoding
)


def test_budgeted_imagination_engine():
    engine = BudgetedImaginationEngine(state_dim=8, action_dim=5, compute_penalty=0.1, max_horizon=10)
    init_state = np.ones(8, dtype=np.float32)
    actions = [np.ones(5, dtype=np.float32) * 0.5 for _ in range(10)]
    
    res = engine.simulate_rollout(init_state, actions)
    assert 'horizon_achieved' in res
    assert 'compute_cost' in res
    assert 'net_objective' in res
    assert res['compute_cost'] >= 1.0
    assert len(res['imagined_states']) == res['horizon_achieved'] + 1


def test_learnable_physics_intuition():
    phys = LearnablePhysicsIntuition()
    
    # 1. Forward pass check
    v1 = torch.tensor([2.0, 0.0])
    v2 = torch.tensor([0.0, 0.0])
    v1_p, v2_p = phys(v1, v2, m1=1.0, m2=1.0, e=1.0)
    assert v1_p.shape == (2,)
    assert v2_p.shape == (2,)
    
    # 2. Analytical reference conservation check
    # In perfectly elastic 1D collision with equal masses, velocities exchange
    v1_ref, v2_ref = phys.simulate_1d_elastic_reference(v1=3.0, v2=0.0, m1=1.0, m2=1.0, e=1.0)
    assert abs(v1_ref - 0.0) < 1e-4
    assert abs(v2_ref - 3.0) < 1e-4


def test_neuro_symbolic_causal_planner():
    planner = NeuroSymbolicCausalPlanner()
    init_state = {'has_tool': False, 'at_tool': False, 'hands_free': True}
    goal_state = {'barrier_unlocked': True}
    
    plan = planner.plan(init_state, goal_state)
    assert len(plan) >= 2
    # Causal path must pick tool and move to target before unlocking
    assert 'move_to_tool' in plan
    assert 'pick_tool' in plan
    assert 'unlock_barrier' in plan


def test_inverse_dynamics_reasoning():
    inv = InverseDynamicsReasoning(state_dim=8, action_dim=5)
    s_t = np.zeros(8, dtype=np.float32)
    s_next = np.ones(8, dtype=np.float32) * 0.5
    
    act = inv.infer_action(s_t, s_next)
    assert act.shape == (5,)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)


def test_hierarchical_predictive_coding():
    hpc = HierarchicalPredictiveCoding(dims=[32, 24, 16, 8])
    sensory_in = torch.randn(32)
    
    res = hpc(sensory_in)
    assert 'total_free_energy' in res
    assert res['total_free_energy'] >= 0.0
    assert len(res['prediction_errors']) == 3
    assert len(res['layer_states']) == 4
