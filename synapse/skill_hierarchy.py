"""Hierarchical reasoning and skill discovery engine for SYNAPSE.

Implements reusable behavioral options with explicit preconditions, sub-controllers,
and termination predicates. Enables multi-stage task assembly and zero-shot recombination.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch import nn


@dataclass
class BehaviorSkill:
    """A modular reusable skill primitive."""
    name: str
    description: str
    precondition_fn: Callable[[np.ndarray, Dict[str, Any]], bool]
    policy_fn: Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
    termination_fn: Callable[[np.ndarray, Dict[str, Any], int], bool]
    success_metric: Callable[[np.ndarray, Dict[str, Any]], float]
    max_steps: int = 40
    execution_count: int = 0
    success_count: int = 0

    def can_activate(self, obs: np.ndarray, context: Dict[str, Any]) -> bool:
        return self.precondition_fn(obs, context)

    def step(self, obs: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        return self.policy_fn(obs, context)

    def is_finished(self, obs: np.ndarray, context: Dict[str, Any], elapsed_steps: int) -> bool:
        if elapsed_steps >= self.max_steps:
            return True
        return self.termination_fn(obs, context, elapsed_steps)

    def record_outcome(self, success: bool):
        self.execution_count += 1
        if success:
            self.success_count += 1

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.execution_count)


class HierarchicalSkillController:
    """Higher-level controller orchestrating discovered skills into multi-stage plans."""
    def __init__(self):
        self.skills: Dict[str, BehaviorSkill] = {}
        self._register_default_skills()
        self.active_skill_name: Optional[str] = None
        self.active_step: int = 0
        self.execution_log: List[Dict[str, Any]] = []

    def register_skill(self, skill: BehaviorSkill):
        self.skills[skill.name] = skill

    def _register_default_skills(self):
        # 1. Navigation / Approach skill
        def nav_precondition(obs: np.ndarray, ctx: Dict[str, Any]) -> bool:
            return 'target_pos' in ctx

        def nav_policy(obs: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
            target = np.array(ctx.get('target_pos', [0.0, 0.0]))
            agent_pos = np.array(ctx.get('agent_pos', [0.0, 0.0]))
            direction = target - agent_pos
            dist = np.linalg.norm(direction)
            if dist < 1e-4:
                return np.zeros(5, dtype=np.float32)
            unit_dir = direction / dist
            heading = float(ctx.get('agent_heading', 0.0))
            desired_angle = np.arctan2(unit_dir[1], unit_dir[0])
            angle_err = (desired_angle - heading + np.pi) % (2 * np.pi) - np.pi
            
            # Action: [thrust, steer, lateral, pitch, interact]
            action = np.zeros(5, dtype=np.float32)
            action[0] = float(np.clip(1.0 - abs(angle_err) / np.pi, 0.2, 1.0))
            action[1] = float(np.clip(angle_err * 2.0, -1.0, 1.0))
            return action

        def nav_termination(obs: np.ndarray, ctx: Dict[str, Any], steps: int) -> bool:
            target = np.array(ctx.get('target_pos', [0.0, 0.0]))
            agent_pos = np.array(ctx.get('agent_pos', [0.0, 0.0]))
            return bool(np.linalg.norm(target - agent_pos) < 0.25)

        self.register_skill(BehaviorSkill(
            name="navigate_to",
            description="Navigates to spatial coordinates while orienting towards target",
            precondition_fn=nav_precondition,
            policy_fn=nav_policy,
            termination_fn=nav_termination,
            success_metric=lambda obs, ctx: 1.0 if nav_termination(obs, ctx, 0) else 0.0
        ))

        # 2. Push / Manipulate skill
        def push_precondition(obs: np.ndarray, ctx: Dict[str, Any]) -> bool:
            return 'box_pos' in ctx and 'target_dest' in ctx

        def push_policy(obs: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
            box_pos = np.array(ctx['box_pos'])
            dest_pos = np.array(ctx['target_dest'])
            push_dir = dest_pos - box_pos
            push_norm = np.linalg.norm(push_dir)
            if push_norm > 1e-4:
                push_dir = push_dir / push_norm
            action = np.zeros(5, dtype=np.float32)
            action[0] = 0.8  # forward thrust
            action[4] = 1.0  # interaction / grip
            return action

        def push_termination(obs: np.ndarray, ctx: Dict[str, Any], steps: int) -> bool:
            box_pos = np.array(ctx.get('box_pos', [0.0, 0.0]))
            dest_pos = np.array(ctx.get('target_dest', [10.0, 10.0]))
            return bool(np.linalg.norm(box_pos - dest_pos) < 0.3)

        self.register_skill(BehaviorSkill(
            name="push_object",
            description="Pushes a movable entity toward desired destination",
            precondition_fn=push_precondition,
            policy_fn=push_policy,
            termination_fn=push_termination,
            success_metric=lambda obs, ctx: 1.0 if push_termination(obs, ctx, 0) else 0.0
        ))

        # 3. Hazard Avoidance skill
        def avoid_precondition(obs: np.ndarray, ctx: Dict[str, Any]) -> bool:
            rays = obs[:33]
            if np.all(rays == 0.0):
                return False
            return bool(rays.min() < 0.25)

        def avoid_policy(obs: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
            rays = obs[:33]
            left_clear = float(rays[:16].mean())
            right_clear = float(rays[17:].mean())
            action = np.zeros(5, dtype=np.float32)
            action[0] = 0.2 # decelerate
            action[1] = -1.0 if left_clear < right_clear else 1.0 # steer towards clearer side
            return action

        def avoid_termination(obs: np.ndarray, ctx: Dict[str, Any], steps: int) -> bool:
            return bool(obs[:33].min() > 0.45)

        self.register_skill(BehaviorSkill(
            name="avoid_hazard",
            description="Evades immediate obstacles or proximity hazards",
            precondition_fn=avoid_precondition,
            policy_fn=avoid_policy,
            termination_fn=avoid_termination,
            success_metric=lambda obs, ctx: 1.0 if avoid_termination(obs, ctx, 0) else 0.0
        ))

    def plan_sequence(self, goal_spec: Dict[str, Any], initial_context: Dict[str, Any]) -> List[str]:
        """Synthesizes a multi-stage skill plan for an unfamiliar complex task."""
        plan = []
        task_type = goal_spec.get('type', 'deliver')
        
        if task_type == 'deliver':
            # Multi-stage: navigate to box -> push to destination
            plan = ['navigate_to', 'push_object']
        elif task_type == 'patrol':
            plan = ['navigate_to', 'navigate_to']
        elif task_type == 'escape_and_reach':
            plan = ['avoid_hazard', 'navigate_to']
        else:
            plan = ['navigate_to']
            
        return plan

    def step(self, obs: np.ndarray, context: Dict[str, Any], plan: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Executes one control step using the active hierarchical skill.
        
        Automatically transitions when skill termination conditions are met.
        """
        # Hazard override check
        if self.skills['avoid_hazard'].can_activate(obs, context) and self.active_skill_name != 'avoid_hazard':
            self.active_skill_name = 'avoid_hazard'
            self.active_step = 0
            
        if self.active_skill_name is None:
            if plan and len(plan) > 0:
                self.active_skill_name = plan[0]
                self.active_step = 0
            else:
                self.active_skill_name = 'navigate_to'
                self.active_step = 0
                
        active_skill = self.skills[self.active_skill_name]
        action = active_skill.step(obs, context)
        self.active_step += 1
        
        # Check termination
        finished = active_skill.is_finished(obs, context, self.active_step)
        transitioned = False
        next_skill_name = self.active_skill_name
        
        if finished:
            success = active_skill.success_metric(obs, context) > 0.5
            active_skill.record_outcome(success)
            self.execution_log.append({
                'skill': active_skill.name,
                'steps': self.active_step,
                'success': success
            })
            if plan and active_skill.name in plan:
                idx = plan.index(active_skill.name)
                if idx + 1 < len(plan):
                    next_skill_name = plan[idx + 1]
                    transitioned = True
                    self.active_skill_name = next_skill_name
                    self.active_step = 0
                else:
                    self.active_skill_name = None
            else:
                self.active_skill_name = None
                
        telemetry = {
            'active_skill': active_skill.name,
            'skill_step': self.active_step,
            'finished': finished,
            'transitioned': transitioned,
            'next_skill': next_skill_name
        }
        return action, telemetry
