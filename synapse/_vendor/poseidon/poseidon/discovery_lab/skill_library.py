"""Project PROMETHEUS Discovery Lab: Reusable Executable Skill Library.

Implements the Voyageur/DIAYN-inspired skill storage schema:
Stores reusable options with explicit preconditions, termination conditions,
validation statistics, and policy action sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SkillValidationStats:
    episodes: int = 0
    held_out_successes: int = 0
    failure_episodes: int = 0
    mean_reward: float = 0.0
    model_hash: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return float(self.held_out_successes / self.episodes)


@dataclass
class DiscoveredSkill:
    """A verified reusable option with explicit preconditions and termination guarantees."""
    skill_id: str
    name: str
    status: str  # "experimental", "certified", "deprecated"
    preconditions: Dict[str, Any]
    action_macro_sequence: List[int]
    termination_condition: str
    estimated_success_rate: float
    observations_required: List[str]
    validation: SkillValidationStats = field(default_factory=SkillValidationStats)
    skill_hash: str = ""

    def validate(self, success: bool, reward: float) -> None:
        """Records an execution outcome on held-out worlds."""
        self.validation.episodes += 1
        if success:
            self.validation.held_out_successes += 1
        else:
            self.validation.failure_episodes += 1
        self.validation.mean_reward = (
            (self.validation.mean_reward * (self.validation.episodes - 1) + reward)
            / self.validation.episodes
        )
        self.estimated_success_rate = self.validation.success_rate

        # Certification threshold: ≥ 80% success across at least 5 validation episodes
        if self.validation.episodes >= 5 and self.estimated_success_rate >= 0.80:
            self.status = "certified"
        elif self.validation.episodes >= 5 and self.estimated_success_rate < 0.40:
            self.status = "deprecated"

        self._seal()

    def _seal(self) -> str:
        payload = {
            "skill_id": self.skill_id,
            "name": self.name,
            "preconditions": self.preconditions,
            "actions": self.action_macro_sequence,
            "episodes": self.validation.episodes,
            "success_rate": round(float(self.estimated_success_rate), 4),
            "status": self.status,
        }
        self.skill_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.skill_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "status": self.status,
            "preconditions": self.preconditions,
            "action_macro_sequence": self.action_macro_sequence,
            "termination_condition": self.termination_condition,
            "estimated_success": round(float(self.estimated_success_rate), 3),
            "observations_required": self.observations_required,
            "validation": {
                "episodes": self.validation.episodes,
                "held_out_successes": self.validation.held_out_successes,
                "failure_episodes": self.validation.failure_episodes,
                "success_rate": round(float(self.validation.success_rate), 3),
                "mean_reward": round(float(self.validation.mean_reward), 3),
                "model_hash": self.validation.model_hash,
            },
            "skill_hash": self.skill_hash,
        }


class DiscoverySkillLibrary:
    """Manages the catalog of discovered, experimental, and certified manipulation skills."""

    def __init__(self):
        self.skills: Dict[str, DiscoveredSkill] = {}
        self._seed_foundational_skills()

    def _seed_foundational_skills(self) -> None:
        # 1. Transport Stone & Place Stepping Stone
        self.register_skill(DiscoveredSkill(
            skill_id="place_stepping_stone_v1",
            name="Place Stepping Stone",
            status="certified",
            preconditions={"stones_available": True, "river_width_max": 4},
            action_macro_sequence=[5, 2, 6],  # GRASP, MOVE_EAST, RELEASE
            termination_condition="stone_settled_in_water",
            estimated_success_rate=0.92,
            observations_required=["stone_positions", "river_start_x"],
        ))

        # 2. Cross Narrow River via Stepping Stones
        self.register_skill(DiscoveredSkill(
            skill_id="cross_narrow_stones_v1",
            name="Cross Narrow River via Stepping Stones",
            status="certified",
            preconditions={"stones_placed": True, "river_width_max": 4},
            action_macro_sequence=[2, 2, 2, 2, 2],  # Repeated MOVE_EAST
            termination_condition="new_region_reached",
            estimated_success_rate=0.88,
            observations_required=["stepping_stone_support", "river_end_x"],
        ))

        # 3. Assemble Buoyant Raft
        self.register_skill(DiscoveredSkill(
            skill_id="construct_raft_assemblage_v1",
            name="Construct Raft Assemblage",
            status="certified",
            preconditions={"buoyant_logs_available": True, "min_logs": 2},
            action_macro_sequence=[5, 2, 6, 9, 10],  # GRASP, MOVE_EAST, RELEASE, CONNECT, CATALYZE
            termination_condition="buoyant_assemblage_cured",
            estimated_success_rate=0.85,
            observations_required=["log_positions", "connected_bonds", "enzyme_resin"],
        ))

        # 4. Cross Wide River via Raft
        self.register_skill(DiscoveredSkill(
            skill_id="cross_wide_raft_v1",
            name="Cross Wide River via Buoyant Raft",
            status="certified",
            preconditions={"raft_cured": True, "support_payload_min": 70.0},
            action_macro_sequence=[2, 2, 2, 2, 2, 2, 2, 2],  # Repeated MOVE_EAST across raft
            termination_condition="new_region_reached",
            estimated_success_rate=0.84,
            observations_required=["raft_surface_support", "river_end_x"],
        ))

    def register_skill(self, skill: DiscoveredSkill) -> None:
        skill._seal()
        self.skills[skill.skill_id] = skill

    def get_skill(self, skill_id: str) -> Optional[DiscoveredSkill]:
        return self.skills.get(skill_id)

    def find_applicable_skills(self, context: Dict[str, Any]) -> List[DiscoveredSkill]:
        """Filters certified skills whose preconditions match current context."""
        matches = []
        for skill in self.skills.values():
            if skill.status != "certified":
                continue
            is_match = True
            for k, v in skill.preconditions.items():
                if k in context and context[k] != v:
                    is_match = False
                    break
            if is_match:
                matches.append(skill)
        return matches

    def to_dict(self) -> Dict[str, Any]:
        return {
            sid: s.to_dict() for sid, s in self.skills.items()
        }
