"""Learning-progress curiosity, competence goal selection, active sensory attention, and emergent communication for SYNAPSE.

Implements:
41. Learning-progress curiosity (intrinsic reward based on derivative of error reduction).
42. Competence-driven goal selection (selecting practice goals in the zone of proximal development).
43. Active sensory attention (foveated gaze allocation toward maximum information gain).
44. Emergent agent communication (bandwidth-constrained symbolic channel with message ablation test).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# 41. Learning-Progress Curiosity
class LearningProgressCuriosity:
    """Computes intrinsic reward as the positive derivative of prediction error reduction."""
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.error_history: List[float] = []

    def compute_intrinsic_reward(self, current_prediction_error: float) -> float:
        self.error_history.append(float(current_prediction_error))
        if len(self.error_history) < 2 * self.window_size:
            return 0.1 # Baseline exploration
            
        # Recent window vs previous window
        recent_window = self.error_history[-self.window_size:]
        prev_window = self.error_history[-2 * self.window_size : -self.window_size]
        
        mean_prev = float(np.mean(prev_window))
        mean_recent = float(np.mean(recent_window))
        
        # Learning Progress = max(0, previous_error - recent_error)
        # Higher when agent is successfully learning to predict!
        # If error is zero (already learned) or error is unchanging (unlearnable random noise), LP = 0.
        learning_progress = max(0.0, mean_prev - mean_recent)
        return float(learning_progress)


# 42. Competence-Driven Goal Selection
@dataclass
class PracticeGoal:
    goal_id: str
    description: str
    target_state: np.ndarray
    competence_history: List[float] = field(default_factory=list)

    @property
    def current_competence(self) -> float:
        return self.competence_history[-1] if self.competence_history else 0.0

    @property
    def learning_rate(self) -> float:
        if len(self.competence_history) < 4:
            return 0.5 # High initial curiosity
        return max(0.0, float(self.competence_history[-1] - self.competence_history[-4]))


class CompetenceGoalSelector:
    """Self-curriculum selector picking goals with highest learning velocity (zone of proximal development)."""
    def __init__(self):
        self.goals: Dict[str, PracticeGoal] = {}

    def register_goal(self, goal_id: str, desc: str, target: np.ndarray):
        self.goals[goal_id] = PracticeGoal(goal_id=goal_id, description=desc, target_state=target)

    def record_attempt(self, goal_id: str, success_score: float):
        if goal_id in self.goals:
            self.goals[goal_id].competence_history.append(float(success_score))

    def select_next_goal(self) -> PracticeGoal:
        """Picks goal with maximum learning rate, avoiding fully mastered (score ~ 1.0) and impossible tasks."""
        if not self.goals:
            raise ValueError("No goals registered")
            
        best_goal = None
        best_priority = -1.0
        
        for g in self.goals.values():
            # Priority combines learning velocity with non-mastery
            mastery_penalty = 1.0 - g.current_competence
            priority = g.learning_rate * mastery_penalty + 0.1 # Epsilon exploration
            if priority > best_priority:
                best_priority = priority
                best_goal = g
                
        return best_goal


# 43. Active Sensory Attention
class ActiveSensoryAttention:
    """Foveated gaze allocation: directs high-resolution focus toward regions of high expected information gain."""
    def __init__(self, visual_grid_size: int = 8):
        self.grid_size = visual_grid_size
        # Saliency / uncertainty map over visual field
        self.uncertainty_map = np.ones((visual_grid_size, visual_grid_size), dtype=np.float32)
        self.current_gaze = np.array([visual_grid_size / 2.0, visual_grid_size / 2.0])

    def update_with_frame(self, frame_intensity: np.ndarray):
        """Updates uncertainty map using visual contrast and motion."""
        # Higher contrast / edges increase uncertainty until inspected
        grad_x = np.diff(frame_intensity, axis=1, append=0)
        grad_y = np.diff(frame_intensity, axis=0, append=0)
        edge_energy = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Adaptive pooling to grid_size
        h, w = frame_intensity.shape
        stride_h = max(1, h // self.grid_size)
        stride_w = max(1, w // self.grid_size)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = edge_energy[i*stride_h : (i+1)*stride_h, j*stride_w : (j+1)*stride_w]
                if patch.size > 0:
                    self.uncertainty_map[i, j] = 0.8 * self.uncertainty_map[i, j] + 0.2 * float(patch.mean())

    def saccade_to_highest_information(self) -> Tuple[int, int]:
        """Directs gaze to the location with maximum expected information gain."""
        idx = int(np.argmax(self.uncertainty_map))
        gx = idx // self.grid_size
        gy = idx % self.grid_size
        self.current_gaze = np.array([float(gx), float(gy)])
        # Inspection resolves uncertainty
        self.uncertainty_map[gx, gy] *= 0.2
        return gx, gy


# 44. Emergent Agent Communication
class EmergentCommunicationChannel:
    """Bandwidth-constrained symbolic communication channel between peer SYNAPSE agents."""
    def __init__(self, vocab_size: int = 8, message_len: int = 3):
        self.vocab_size = vocab_size
        self.message_len = message_len
        self.history: List[Dict[str, Any]] = []

    def transmit(self, sender_id: str, receiver_id: str, message: np.ndarray, ablate: bool = False, scramble: bool = False) -> np.ndarray:
        """Transmits discrete/continuous message with optional causal ablation or scrambling."""
        msg = message.copy()
        if ablate:
            msg = np.zeros_like(msg)
        elif scramble:
            msg = np.random.permutation(msg)
            
        self.history.append({
            'sender': sender_id,
            'receiver': receiver_id,
            'original_message': message.tolist(),
            'transmitted_message': msg.tolist(),
            'ablated': ablate,
            'scrambled': scramble
        })
        return msg

    def evaluate_communication_utility(self, success_with_message: float, success_without_message: float) -> Dict[str, Any]:
        """Causal validation: confirms that communicating agents outperform ablated agents."""
        comm_advantage = success_with_message - success_without_message
        has_emergent_utility = comm_advantage > 0.15
        return {
            'success_with_message': float(success_with_message),
            'success_without_message': float(success_without_message),
            'communication_advantage': float(comm_advantage),
            'has_emergent_utility': has_emergent_utility
        }
