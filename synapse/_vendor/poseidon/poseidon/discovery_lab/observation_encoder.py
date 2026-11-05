"""Project PROMETHEUS Discovery Lab: Observation Encoder.

Encodes raw 3D scene graphs and object spatial arrangements into structured,
compact representation tensors for world models, curiosity engines, and skill policies.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .object_physics import PhysicalObject


class DiscoveryObservationEncoder:
    """Encodes physical object arrangements and relational graphs into flat feature vectors."""

    def __init__(self, max_tracked_objects: int = 8, d_obj_feature: int = 8):
        self.max_tracked_objects = int(max_tracked_objects)
        self.d_obj_feature = int(d_obj_feature)
        # Total dim: 8 (agent) + max_tracked_objects * 8 (objects) + 4 (terrain) = 76
        self.total_dim = 8 + (self.max_tracked_objects * self.d_obj_feature) + 4

    def encode(
        self,
        agent_pos: List[float],
        agent_health: float,
        agent_stamina: float,
        held_object: Optional[PhysicalObject],
        all_objects: List[PhysicalObject],
        river_bounds: Tuple[float, float, float],  # (start_x, end_x, depth)
    ) -> np.ndarray:
        """Constructs an invariant, object-centric observation tensor."""
        vec = np.zeros(self.total_dim, dtype=np.float32)

        # 0..7: Agent proprioception
        vec[0] = float(agent_pos[0])
        vec[1] = float(agent_pos[1])
        vec[2] = float(agent_pos[2])
        vec[3] = float(agent_health)
        vec[4] = float(agent_stamina)
        vec[5] = float(1.0 if held_object is not None else 0.0)
        vec[6] = float(held_object.density if held_object else 0.0)
        vec[7] = float(held_object.is_buoyant if held_object else 0.0)

        # 8..71: Tracked objects sorted by Euclidean proximity to agent
        sorted_objs = sorted(
            all_objects,
            key=lambda o: math.hypot(o.position[0] - agent_pos[0], o.position[1] - agent_pos[1])
        )[:self.max_tracked_objects]

        offset = 8
        for obj in sorted_objs:
            # 8 features per object: [rel_x, rel_y, rel_z, mass, density, buoyant, connected_count, bond_strength]
            vec[offset + 0] = float(obj.position[0] - agent_pos[0])
            vec[offset + 1] = float(obj.position[1] - agent_pos[1])
            vec[offset + 2] = float(obj.position[2] - agent_pos[2])
            vec[offset + 3] = float(obj.mass / 100.0)
            vec[offset + 4] = float(obj.density)
            vec[offset + 5] = float(1.0 if obj.is_buoyant else 0.0)
            vec[offset + 6] = float(len(obj.connected_object_ids))
            vec[offset + 7] = float(obj.bond_strength)
            offset += self.d_obj_feature

        # 72..75: River terrain parameters
        t_offset = 8 + (self.max_tracked_objects * self.d_obj_feature)
        vec[t_offset + 0] = float(river_bounds[0])  # start_x
        vec[t_offset + 1] = float(river_bounds[1])  # end_x
        vec[t_offset + 2] = float(river_bounds[1] - river_bounds[0])  # width
        vec[t_offset + 3] = float(river_bounds[2])  # depth

        return vec
