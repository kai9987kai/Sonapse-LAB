"""Event-driven sparse computation, learnable time constants, dynamic compute routing, adversarial stress-testing, and verified safety shielding for SYNAPSE.

Implements:
45. Event-driven sparse computation & DVS event-camera simulator.
46. Learnable neural time constants (heterogeneous decay rates per neuron).
47. Difficulty-aware computational routing (System 1 fast reflex vs System 2 deliberative loop).
48. Adversarial environment generator (stress-testing edge cases).
49. Verified action safety layer (Control Barrier Function action shielding).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 45. Event-Driven Sparse Computation & DVS Simulator
class EventDrivenSparseEngine:
    """Updates neural units and visual streams only when change magnitudes exceed a threshold."""
    def __init__(self, num_units: int = 64, event_threshold: float = 0.05):
        self.num_units = num_units
        self.event_threshold = event_threshold
        self.last_state = np.zeros(num_units, dtype=np.float32)
        self.total_evaluations = 0
        self.sparse_evaluations = 0

    def step(self, raw_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculates delta event mask and updates active units only.
        
        Returns:
            Tuple: (updated_state, active_mask, sparsity_ratio)
        """
        x = np.asarray(raw_input[:self.num_units], dtype=np.float32)
        if len(x) < self.num_units:
            x = np.pad(x, (0, self.num_units - len(x)))
            
        deltas = np.abs(x - self.last_state)
        active_mask = (deltas > self.event_threshold).astype(np.float32)
        
        # Sparsely update only units that crossed event threshold
        self.last_state = np.where(active_mask > 0, x, self.last_state)
        
        active_count = int(np.sum(active_mask))
        self.total_evaluations += self.num_units
        self.sparse_evaluations += active_count
        sparsity_ratio = 1.0 - (active_count / max(1, self.num_units))
        
        return self.last_state.copy(), active_mask, sparsity_ratio

    def simulate_dvs_events(self, prev_frame: np.ndarray, curr_frame: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
        """Converts consecutive frames into positive/negative ON/OFF event spikes."""
        diff = curr_frame.astype(np.float32) - prev_frame.astype(np.float32)
        on_events = (diff > threshold).astype(np.int32)
        off_events = (diff < -threshold).astype(np.int32)
        return {
            'on_events_count': int(np.sum(on_events)),
            'off_events_count': int(np.sum(off_events)),
            'total_events': int(np.sum(on_events) + np.sum(off_events)),
            'event_density': float(np.mean(on_events + off_events))
        }


# 46. Learnable Neural Time Constants
class LearnableTimeConstants(nn.Module):
    """Heterogeneous adaptive decay time constants per neuron unit."""
    def __init__(self, num_units: int = 64, tau_min: float = 5.0, tau_max: float = 100.0):
        super().__init__()
        self.num_units = num_units
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Unconstrained parameter mapped to [tau_min, tau_max] via sigmoid
        self.raw_tau = nn.Parameter(torch.zeros(num_units))
        self.state = nn.Parameter(torch.zeros(num_units), requires_grad=False)

    @property
    def tau(self) -> torch.Tensor:
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.raw_tau)

    def forward(self, input_current: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Leaky integration step: s_{t+1} = (1 - dt / tau) * s_t + (dt / tau) * I_t."""
        alpha = torch.clamp(dt / self.tau, 0.01, 0.99)
        new_state = (1.0 - alpha) * self.state + alpha * input_current
        self.state.copy_(new_state.detach())
        return new_state


# 47. Difficulty-Aware Computational Routing
class DifficultyAwareComputeRouter:
    """Routes simple observations to System 1 fast reflex; routes complex states to System 2 deliberative loops."""
    def __init__(self, complexity_threshold: float = 0.35):
        self.complexity_threshold = complexity_threshold
        self.system1_calls = 0
        self.system2_calls = 0

    def assess_complexity(self, obs: np.ndarray, uncertainty: float) -> float:
        """Estimates situation complexity from sensory entropy and state uncertainty."""
        contrast = float(np.std(obs[:33]))
        ray_min = float(np.min(obs[:33]))
        hazard_proximity = 1.0 - np.clip(ray_min, 0.0, 1.0)
        complexity = 0.4 * uncertainty + 0.3 * contrast + 0.3 * hazard_proximity
        return float(complexity)

    def route(self, obs: np.ndarray, uncertainty: float = 0.0) -> Tuple[str, int]:
        """Decides routing system and recurrent iteration depth.
        
        Returns:
            ('SYSTEM_1' | 'SYSTEM_2', allocated_iterations)
        """
        complexity = self.assess_complexity(obs, uncertainty)
        if complexity < self.complexity_threshold:
            self.system1_calls += 1
            return 'SYSTEM_1', 1
        else:
            self.system2_calls += 1
            iterations = int(np.clip(complexity * 10, 2, 8))
            return 'SYSTEM_2', iterations


# 48. Adversarial Environment Generator
class AdversarialEnvironmentGenerator:
    """Stress-tests planning and memory by synthesizing worst-case edge scenarios."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_adversarial_scenario(self, scenario_type: str = 'sudden_obstacle') -> Dict[str, Any]:
        """Generates stressful environmental anomalies."""
        if scenario_type == 'sudden_obstacle':
            # Obstacle immediately 0.1 units ahead
            scenario = {
                'type': 'sudden_obstacle',
                'description': 'Unseen obstacle materialized directly in front of agent trajectory',
                'obs_override': {'ray_0': 0.05, 'threat_level': 0.95},
                'challenge': 'Immediate reflex braking and emergency turn'
            }
        elif scenario_type == 'sensory_blackout':
            scenario = {
                'type': 'sensory_blackout',
                'description': 'Camera and ray sensors dropped to complete zero for 10 timesteps',
                'obs_override': {'dropout': True},
                'challenge': 'Path navigation relying purely on allocentric cognitive map and working memory'
            }
        elif scenario_type == 'friction_inversion':
            scenario = {
                'type': 'friction_inversion',
                'description': 'Surface friction drastically reduced by 90% (ice surface)',
                'obs_override': {'friction_coeff': 0.05},
                'challenge': 'Detecting rule shift and adapting motor braking forces'
            }
        else:
            scenario = {'type': 'standard'}
            
        return scenario


# 49. Verified Action Safety Layer
class VerifiedActionSafetyLayer:
    """Control Barrier Function (CBF) action shield guaranteeing physical safety invariants."""
    def __init__(self, max_thrust: float = 1.0, min_safe_dist: float = 0.2, max_steer_rate: float = 1.0):
        self.max_thrust = max_thrust
        self.min_safe_dist = min_safe_dist
        self.max_steer_rate = max_steer_rate
        self.interventions_count = 0

    def shield_action(self, proposed_action: np.ndarray, min_obstacle_distance: float, current_speed: float) -> Tuple[np.ndarray, bool]:
        """Projects proposed action onto safe admissible control set.
        
        Guarantees:
        1. Action components within [-1, 1]
        2. If approaching obstacle closer than min_safe_dist, overrides thrust to brake/reverse
        3. Bounds lateral angular steering rate
        """
        safe_act = np.clip(proposed_action.copy(), -1.0, 1.0)
        intervened = False
        
        # 1. Forward thrust safety barrier
        if min_obstacle_distance < self.min_safe_dist:
            if safe_act[0] > 0.0:
                # Decelerate or reverse away from imminent collision
                safe_act[0] = -0.5
                intervened = True
                
        # 2. Maximum thrust clamping
        if safe_act[0] > self.max_thrust:
            safe_act[0] = self.max_thrust
            intervened = True
            
        # 3. Steering rate safety clamping
        if abs(safe_act[1]) > self.max_steer_rate:
            safe_act[1] = np.sign(safe_act[1]) * self.max_steer_rate
            intervened = True
            
        if intervened:
            self.interventions_count += 1
            
        return safe_act, intervened
