"""Budgeted imagination, intuitive physics, causal planning, inverse dynamics, and hierarchical predictive coding for SYNAPSE.

Implements:
31. Budgeted imagination engine (DreamerV3-style rollouts with compute cost penalty lambda * C_plan).
32. Learnable physics intuition (collision, restitution, momentum conservation, friction).
33. Neuro-symbolic causal planning (causal dependency graphs and transition operators).
34. Inverse dynamics reasoning (inferring motor actions reaching desired target states).
35. Hierarchical predictive coding (4-layer top-down prediction and precision-weighted error hierarchy).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 31. Budgeted Imagination Engine
class BudgetedImaginationEngine:
    """DreamerV3-style mental rollout engine with adaptive compute cost budgeting."""
    def __init__(self, state_dim: int = 16, action_dim: int = 5, compute_penalty: float = 0.05, max_horizon: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.compute_penalty = compute_penalty
        self.max_horizon = max_horizon
        
        # Simple learnable transition model: s' = f(s, a)
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        self.reward_head = nn.Linear(state_dim, 1)

    def simulate_rollout(self, initial_state: np.ndarray, action_sequence: List[np.ndarray], gamma: float = 0.95) -> Dict[str, Any]:
        """Rolls out imagined future trajectory with explicit compute cost tracking and early stopping."""
        s = torch.tensor(initial_state[:self.state_dim], dtype=torch.float32)
        total_discounted_reward = 0.0
        trajectory = [s.detach().numpy().copy()]
        rewards = []
        compute_cost = 0.0
        
        step = 0
        for step, a in enumerate(action_sequence[:self.max_horizon]):
            # Compute cost increases with each rollout step (FLOPs / latency surrogate)
            compute_cost += 1.0
            
            act = torch.tensor(a[:self.action_dim], dtype=torch.float32)
            sa = torch.cat([s, act])
            s = s + 0.2 * self.transition(sa)
            r = float(self.reward_head(s).item())
            
            discount = gamma ** step
            total_discounted_reward += discount * r
            trajectory.append(s.detach().numpy().copy())
            rewards.append(r)
            
            # Adaptive halting condition: if marginal expected gain < compute_penalty, stop early
            marginal_gain = discount * abs(r)
            if step > 2 and marginal_gain < self.compute_penalty:
                break
                
        net_objective = total_discounted_reward - self.compute_penalty * compute_cost
        return {
            'horizon_achieved': step + 1,
            'compute_cost': compute_cost,
            'discounted_return': total_discounted_reward,
            'net_objective': net_objective,
            'imagined_states': trajectory,
            'imagined_rewards': rewards
        }


# 32. Learnable Physics Intuition
class LearnablePhysicsIntuition(nn.Module):
    """Predicts rigid-body collisions, momentum transfer, restitution, and friction."""
    def __init__(self):
        super().__init__()
        # Predicts post-collision velocity (v1', v2') given (v1, v2, m1, m2, restitution e)
        self.collision_net = nn.Sequential(
            nn.Linear(2 + 2 + 1 + 1 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 4) # (v1x', v1y', v2x', v2y')
        )

    def forward(self, v1: torch.Tensor, v2: torch.Tensor, m1: float, m2: float, e: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        b = v1.shape[0] if v1.ndim > 1 else 1
        m1_t = torch.full((b, 1), m1, device=v1.device)
        m2_t = torch.full((b, 1), m2, device=v1.device)
        e_t = torch.full((b, 1), e, device=v1.device)
        
        inp = torch.cat([v1.view(b, 2), v2.view(b, 2), m1_t, m2_t, e_t], dim=-1)
        out = self.collision_net(inp)
        return out[:, :2].squeeze(0), out[:, 2:].squeeze(0)

    def simulate_1d_elastic_reference(self, v1: float, v2: float, m1: float, m2: float, e: float = 0.8) -> Tuple[float, float]:
        """Analytical conservation of momentum & restitution reference for testing."""
        # v1' = (m1 - e*m2)/(m1+m2)*v1 + (1+e)*m2/(m1+m2)*v2
        # v2' = (1+e)*m1/(m1+m2)*v1 + (m2 - e*m1)/(m1+m2)*v2
        denom = m1 + m2
        v1_prime = ((m1 - e * m2) * v1 + (1.0 + e) * m2 * v2) / denom
        v2_prime = ((1.0 + e) * m1 * v1 + (m2 - e * m1) * v2) / denom
        return float(v1_prime), float(v2_prime)


# 33. Neuro-Symbolic Causal Planning
@dataclass
class CausalOperator:
    name: str
    preconditions: Dict[str, Any]
    effects: Dict[str, Any]
    cost: float = 1.0


class NeuroSymbolicCausalPlanner:
    """Plans high-level action sequences by resolving symbolic causal dependency graphs."""
    def __init__(self):
        self.operators: List[CausalOperator] = []
        self._init_default_operators()

    def _init_default_operators(self):
        self.operators.append(CausalOperator(
            name="pick_tool",
            preconditions={'at_tool': True, 'hands_free': True},
            effects={'has_tool': True, 'hands_free': False}
        ))
        self.operators.append(CausalOperator(
            name="move_to_tool",
            preconditions={'has_tool': False},
            effects={'at_tool': True, 'at_target': False}
        ))
        self.operators.append(CausalOperator(
            name="move_to_target",
            preconditions={'has_tool': True},
            effects={'at_target': True, 'at_tool': False}
        ))
        self.operators.append(CausalOperator(
            name="unlock_barrier",
            preconditions={'at_target': True, 'has_tool': True},
            effects={'barrier_unlocked': True}
        ))

    def plan(self, initial_state: Dict[str, Any], goal_state: Dict[str, Any]) -> List[str]:
        """Backward-chaining / search to find causal sequence achieving goal."""
        current = initial_state.copy()
        plan_steps = []
        
        # Max search iterations
        for _ in range(10):
            # Check if all goal conditions satisfied
            if all(current.get(k) == v for k, v in goal_state.items()):
                break
                
            # Find an applicable operator whose effects advance state towards goal
            applied = False
            for op in self.operators:
                # Check preconditions and ensure operator produces actual state transition
                if all(current.get(k) == v for k, v in op.preconditions.items()):
                    if any(current.get(k) != v for k, v in op.effects.items()):
                        current.update(op.effects)
                        plan_steps.append(op.name)
                        applied = True
                        break
            if not applied:
                break
                
        return plan_steps


# 34. Inverse Dynamics Reasoning
class InverseDynamicsReasoning(nn.Module):
    """Infers motor actions required to transition from state s_t to target state s_{t+1}."""
    def __init__(self, state_dim: int = 16, action_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

    def infer_action(self, current_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        s_curr = torch.tensor(current_state[:16], dtype=torch.float32)
        s_targ = torch.tensor(target_state[:16], dtype=torch.float32)
        inp = torch.cat([s_curr, s_targ], dim=-1)
        action = self.net(inp).detach().numpy()
        return action


# 35. Hierarchical Predictive Coding
class HierarchicalPredictiveCoding(nn.Module):
    """4-layer predictive coding hierarchy (L0: Sensory, L1: Perceptual, L2: Object, L3: Goal)."""
    def __init__(self, dims: List[int] = [32, 24, 16, 8]):
        super().__init__()
        self.dims = dims
        self.num_layers = len(dims)
        
        # Top-down generative predictors: L_{i+1} -> L_i
        self.top_down_predictors = nn.ModuleList([
            nn.Linear(dims[i+1], dims[i]) for i in range(self.num_layers - 1)
        ])
        
        # Learnable precision weights per layer
        self.precisions = nn.ParameterList([
            nn.Parameter(torch.ones(dims[i])) for i in range(self.num_layers)
        ])

    def forward(self, bottom_up_sensory: torch.Tensor, top_down_goal: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Runs bidirectional prediction and computes precision-weighted prediction errors."""
        # Initialize layer representations
        states = []
        curr = bottom_up_sensory
        states.append(curr)
        
        # Ascending bottom-up pass
        for i in range(self.num_layers - 1):
            curr = F.adaptive_avg_pool1d(curr.unsqueeze(0).unsqueeze(0), self.dims[i+1]).squeeze()
            states.append(curr)
            
        if top_down_goal is not None:
            states[-1] = top_down_goal
            
        # Top-down predictions and prediction errors
        errors = []
        predictions = []
        total_free_energy = 0.0
        
        for i in range(self.num_layers - 2, -1, -1):
            pred = self.top_down_predictors[i](states[i+1])
            predictions.append(pred)
            
            # Raw error
            raw_err = states[i] - pred
            # Precision-weighted error: eps = Pi * (x - mu)
            prec = F.softplus(self.precisions[i])
            weighted_err = raw_err * prec
            errors.append(weighted_err)
            
            # Layer free energy component: 0.5 * eps^T * Pi * eps
            layer_fe = 0.5 * torch.sum(weighted_err ** 2)
            total_free_energy = total_free_energy + float(layer_fe.item())
            
        return {
            'total_free_energy': total_free_energy,
            'prediction_errors': [e.detach().numpy() for e in errors],
            'layer_states': [s.detach().numpy() for s in states]
        }
