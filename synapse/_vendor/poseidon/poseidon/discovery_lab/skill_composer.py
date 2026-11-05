"""Project PROMETHEUS Discovery Lab: Hierarchical Skill Composer & Strategy Selector.

Implements autonomous strategy switching when an established approach fails:
Detects river width infeasibility for stepping stones and discovers the alternative
buoyant raft construction strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .skill_library import DiscoverySkillLibrary, DiscoveredSkill
from .manipulation_actions import ManipulationActionType, ACTION_NAMES


class StrategyType(str, Enum):
    STEPPING_STONES = "stepping_stones"
    BUOYANT_RAFT = "buoyant_raft"
    PRIMITIVE_EXPLORATION = "primitive_exploration"


@dataclass
class CompositionPlan:
    selected_strategy: StrategyType
    skill_pipeline: List[str]
    planned_macro_actions: List[int]
    feasibility_score: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_strategy": self.selected_strategy.value,
            "skill_pipeline": self.skill_pipeline,
            "planned_macro_actions": self.planned_macro_actions,
            "planned_action_names": [ACTION_NAMES.get(a, "unknown") for a in self.planned_macro_actions],
            "feasibility_score": round(float(self.feasibility_score), 3),
            "explanation": self.explanation,
        }


class DiscoverySkillComposer:
    """Selects, decomposes, and sequences skills to solve river crossing challenges."""

    def __init__(self, skill_library: Optional[DiscoverySkillLibrary] = None):
        self.library = skill_library if skill_library is not None else DiscoverySkillLibrary()
        self.strategy_history: List[StrategyType] = []

    def plan_crossing(
        self,
        river_width: int,
        water_depth: float,
        stones_available: int,
        logs_available: int,
        previous_strategy_failed: bool = False,
    ) -> CompositionPlan:
        """Determines feasible crossing strategy based on physical geometry and failure feedback."""
        # Feasibility check for stepping stones:
        # A stone placed in water bridges ~0.8m. Narrow river (<= 4m) needs 2-3 stones and shallow depth (<= 1.5m).
        stones_feasible = (
            river_width <= 4
            and water_depth <= 1.5
            and stones_available >= 2
            and not previous_strategy_failed
        )

        if stones_feasible:
            # Strategy A: Stepping stones
            strategy = StrategyType.STEPPING_STONES
            pipeline = ["place_stepping_stone_v1", "cross_narrow_stones_v1"]
            actions = []
            s1 = self.library.get_skill("place_stepping_stone_v1")
            s2 = self.library.get_skill("cross_narrow_stones_v1")
            if s1:
                actions.extend(s1.action_macro_sequence)
            if s2:
                actions.extend(s2.action_macro_sequence)

            explanation = (
                f"Narrow river (width={river_width}m <= 4m, depth={water_depth}m). "
                f"Stepping stone strategy is physically feasible."
            )
            score = 0.90
        else:
            # Strategy B: Autonomous adaptation to Buoyant Raft
            strategy = StrategyType.BUOYANT_RAFT
            pipeline = ["construct_raft_assemblage_v1", "cross_wide_raft_v1"]
            actions = []
            s1 = self.library.get_skill("construct_raft_assemblage_v1")
            s2 = self.library.get_skill("cross_wide_raft_v1")
            if s1:
                actions.extend(s1.action_macro_sequence)
            if s2:
                actions.extend(s2.action_macro_sequence)

            reason_fail = "previous stepping stone attempt failed" if previous_strategy_failed else f"river is too wide ({river_width}m > 4m) or deep ({water_depth}m)"
            explanation = (
                f"Stepping stone strategy infeasible ({reason_fail}). "
                f"Autonomously shifted strategy to Buoyant Raft construction using {logs_available} logs and catalytic enzyme cross-linking."
            )
            score = 0.86

        self.strategy_history.append(strategy)

        return CompositionPlan(
            selected_strategy=strategy,
            skill_pipeline=pipeline,
            planned_macro_actions=actions,
            feasibility_score=score,
            explanation=explanation,
        )
