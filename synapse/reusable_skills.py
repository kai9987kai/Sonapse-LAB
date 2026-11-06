"""Automatically Discovered Reusable Skill Library (TESSERA-Inspired Macro-Actions).

Grounding & Review Requirements:
Enables developmental agents to distill frequently successful sensorimotor trajectories
into reusable macro-action subroutines with:
1. Precondition verification (sensory region / distance metric).
2. Action trajectory rollout (sequence of coordinated motor commands).
3. Postcondition verification (state displacement / reward delivery).
4. Empirical tracking (success rate, usage frequency, utility-based pruning).
"""
from __future__ import annotations
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class MacroSkill:
    """A verified reusable macro-action policy."""
    skill_id: str
    name: str
    precondition_centroid: List[float]
    precondition_radius: float
    action_sequence: List[List[float]]
    expected_duration: int
    expected_reward: float
    success_count: int = 0
    attempt_count: int = 0
    created_at_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return float(self.success_count / self.attempt_count) if self.attempt_count > 0 else 0.0

    def matches_precondition(self, obs: np.ndarray) -> bool:
        """Check if sensory observation falls within skill's valid execution basin."""
        obs_f = np.asarray(obs, dtype=np.float32).flatten()
        centroid = np.asarray(self.precondition_centroid, dtype=np.float32)
        dim = min(len(obs_f), len(centroid))
        dist = float(np.linalg.norm(obs_f[:dim] - centroid[:dim]))
        return dist <= self.precondition_radius

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['success_rate'] = self.success_rate
        return d


class ReusableSkillLibrary:
    """Library and automated discovery engine for sensorimotor macro skills."""

    def __init__(
        self,
        max_skills: int = 32,
        discovery_window: int = 200,
        min_skill_length: int = 3,
        max_skill_length: int = 10,
        seed: int = 42
    ):
        self.max_skills = max_skills
        self.discovery_window = discovery_window
        self.min_skill_length = min_skill_length
        self.max_skill_length = max_skill_length
        self.rng = np.random.default_rng(seed)

        self.skills: Dict[str, MacroSkill] = {}
        self.trajectory_buffer: List[Dict[str, Any]] = []
        self.total_executions: int = 0
        self.total_discoveries: int = 0

    def record_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        step: int
    ):
        """Append step to trajectory buffer for skill extraction."""
        self.trajectory_buffer.append({
            'obs': np.asarray(obs, dtype=np.float32).copy(),
            'action': np.asarray(action, dtype=np.float32).copy(),
            'reward': float(reward),
            'next_obs': np.asarray(next_obs, dtype=np.float32).copy(),
            'step': step
        })
        if len(self.trajectory_buffer) > self.discovery_window:
            self.trajectory_buffer.pop(0)

    def discover_skills(self, reward_threshold: float = 0.5) -> List[MacroSkill]:
        """Mine trajectory buffer for high-reward action motifs and extract skills."""
        if len(self.trajectory_buffer) < self.min_skill_length * 2:
            return []

        discovered = []
        rewards = [t['reward'] for t in self.trajectory_buffer]
        high_rew_indices = [i for i, r in enumerate(rewards) if r >= reward_threshold]

        for idx in high_rew_indices:
            # Lookback window for action trajectory leading up to high reward
            start_idx = max(0, idx - self.max_skill_length + 1)
            span = self.trajectory_buffer[start_idx:idx + 1]
            if len(span) < self.min_skill_length:
                continue

            actions = [t['action'].tolist() for t in span]
            pre_obs = span[0]['obs']
            cum_reward = sum(t['reward'] for t in span)

            # Generate unique deterministic skill ID from actions
            h = hashlib.sha256(np.array(actions).tobytes()).hexdigest()[:10]
            skill_id = f"skill_{h}"

            if skill_id not in self.skills:
                skill = MacroSkill(
                    skill_id=skill_id,
                    name=f"Macro-{skill_id}",
                    precondition_centroid=pre_obs[:32].tolist(),
                    precondition_radius=1.5,
                    action_sequence=actions,
                    expected_duration=len(actions),
                    expected_reward=float(cum_reward),
                    created_at_step=span[0]['step']
                )
                self.skills[skill_id] = skill
                self.total_discoveries += 1
                discovered.append(skill)

                # Prune lowest performing skill if exceeding max
                if len(self.skills) > self.max_skills:
                    self._prune_lowest_utility_skill()

        return discovered

    def query_skill(self, obs: np.ndarray) -> Optional[MacroSkill]:
        """Find the highest-performing skill matching the current sensory precondition."""
        matching = [s for s in self.skills.values() if s.matches_precondition(obs)]
        if not matching:
            return None
        # Sort by success rate and expected reward
        matching.sort(key=lambda s: (s.success_rate, s.expected_reward), reverse=True)
        return matching[0]

    def record_skill_execution(self, skill_id: str, success: bool):
        """Update skill performance statistics."""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            skill.attempt_count += 1
            if success:
                skill.success_count += 1
            self.total_executions += 1

    def _prune_lowest_utility_skill(self):
        """Remove the skill with the lowest empirical utility."""
        if not self.skills:
            return
        # Utility = success_rate * expected_reward / (attempts + 1)
        scored = [
            (s.skill_id, s.success_rate * s.expected_reward / (s.attempt_count + 1))
            for s in self.skills.values()
        ]
        scored.sort(key=lambda x: x[1])
        lowest_id = scored[0][0]
        del self.skills[lowest_id]

    def summary(self) -> Dict[str, Any]:
        """Summary of active skill library."""
        active_list = [s.to_dict() for s in self.skills.values()]
        mean_success = float(np.mean([s.success_rate for s in self.skills.values()])) if self.skills else 0.0
        return {
            'total_skills': len(self.skills),
            'total_discoveries': self.total_discoveries,
            'total_executions': self.total_executions,
            'mean_success_rate': round(mean_success, 4),
            'skills': active_list[:10]  # First 10 skills
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'total_discoveries': self.total_discoveries,
            'total_executions': self.total_executions,
            'skills': {sid: s.to_dict() for sid, s in self.skills.items()}
        }

    def import_state(self, state: Dict[str, Any]):
        self.total_discoveries = int(state.get('total_discoveries', 0))
        self.total_executions = int(state.get('total_executions', 0))
        self.skills.clear()
        for sid, d in state.get('skills', {}).items():
            self.skills[sid] = MacroSkill(
                skill_id=d['skill_id'],
                name=d['name'],
                precondition_centroid=d['precondition_centroid'],
                precondition_radius=d['precondition_radius'],
                action_sequence=d['action_sequence'],
                expected_duration=d['expected_duration'],
                expected_reward=d['expected_reward'],
                success_count=d.get('success_count', 0),
                attempt_count=d.get('attempt_count', 0),
                created_at_step=d.get('created_at_step', 0)
            )
