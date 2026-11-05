"""Project PROMETHEUS Discovery Lab: Manipulation Action Specification.

Separates object manipulation actions (move, inspect, grasp, release, push, rotate,
connect, catalyze) from the baseline survival interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple


class ManipulationActionType(IntEnum):
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    INSPECT = 4
    GRASP = 5
    RELEASE = 6
    PUSH = 7
    ROTATE = 8
    CONNECT = 9
    CATALYZE = 10


ACTION_NAMES: Dict[int, str] = {
    ManipulationActionType.MOVE_NORTH: "move_north",
    ManipulationActionType.MOVE_SOUTH: "move_south",
    ManipulationActionType.MOVE_EAST: "move_east",
    ManipulationActionType.MOVE_WEST: "move_west",
    ManipulationActionType.INSPECT: "inspect",
    ManipulationActionType.GRASP: "grasp",
    ManipulationActionType.RELEASE: "release",
    ManipulationActionType.PUSH: "push",
    ManipulationActionType.ROTATE: "rotate",
    ManipulationActionType.CONNECT: "connect",
    ManipulationActionType.CATALYZE: "catalyze",
}

ACTION_DESCRIPTIONS: Dict[int, str] = {
    ManipulationActionType.MOVE_NORTH: "Translate agent north along Y-axis",
    ManipulationActionType.MOVE_SOUTH: "Translate agent south along Y-axis",
    ManipulationActionType.MOVE_EAST: "Translate agent east across river toward goal",
    ManipulationActionType.MOVE_WEST: "Translate agent west back toward left bank",
    ManipulationActionType.INSPECT: "Probe physical dimensions, density, and buoyancy of adjacent object",
    ManipulationActionType.GRASP: "Pick up adjacent physical object into agent inventory",
    ManipulationActionType.RELEASE: "Place currently held object in front of agent on land or water",
    ManipulationActionType.PUSH: "Apply lateral interventional force impulse to adjacent object",
    ManipulationActionType.ROTATE: "Orient object yaw axis by 90 degrees",
    ManipulationActionType.CONNECT: "Structurally couple two adjacent buoyant objects into an assemblage",
    ManipulationActionType.CATALYZE: "Enzymatic biochemical curing to cross-link assemblage bonds into high-strength raft",
}

NUM_MANIPULATION_ACTIONS = len(ManipulationActionType)


@dataclass
class ManipulationCommand:
    """Parametrized manipulation command for higher-level skill execution."""
    action_type: ManipulationActionType
    target_object_id: Optional[str] = None
    target_position: Optional[Tuple[float, float, float]] = None
    force_magnitude: float = 1.0
    relative_yaw_delta: float = 0.0

    @classmethod
    def from_int(cls, action_int: int) -> ManipulationCommand:
        valid_type = ManipulationActionType(int(action_int) % NUM_MANIPULATION_ACTIONS)
        return cls(action_type=valid_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": int(self.action_type),
            "action_name": ACTION_NAMES.get(self.action_type, "unknown"),
            "target_object_id": self.target_object_id,
            "target_position": list(self.target_position) if self.target_position else None,
            "force_magnitude": round(float(self.force_magnitude), 3),
            "relative_yaw_delta": round(float(self.relative_yaw_delta), 3),
        }
