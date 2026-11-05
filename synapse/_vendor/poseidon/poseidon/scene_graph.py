"""Object-Centric 3D Scene Graph and Interactive Environment Simulation.

Implements Experiment F of Supermix Beyond:
1. Object-Centric Scene Graph:
   - Nodes: 3D entities with explicit geometry, transforms, materials, physical
     mass/velocity, and semantic resource/hazard properties.
   - Edges: Spatial and physical relations ("on_top_of", "orbiting", "adjacent_to").
2. Interactive Physics Evolution:
   - Predicts and updates scene state under agent action impulses and environmental dynamics.
3. Multi-Object Wavefront OBJ & JSON Export:
   - Produces inspectable 3D environments suitable for local inspection and Unity WorldLab importing.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .media import (
    _primitive, COLORS, _SCALES
)


@dataclass
class PhysicsProperties:
    mass: float = 1.0
    is_static: bool = False
    velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    interactable: bool = True
    resource_type: str = "none"  # "food", "water", "shelter", "hazard", "none"


@dataclass
class SceneObject:
    id: str
    name: str
    shape: str  # "cube", "sphere", "pyramid", "cylinder"
    position: list[float]  # [x, y, z]
    rotation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [rx, ry, rz]
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color_name: str = "blue"
    rgb: list[float] = field(default_factory=lambda: [0.2, 0.5, 0.9])
    material: str = "matte"
    physics: PhysicsProperties = field(default_factory=PhysicsProperties)


@dataclass
class SceneRelation:
    source_id: str
    relation: str  # "on_top_of", "orbiting", "adjacent_to", "sheltering"
    target_id: str


class ObjectCentricSceneGraph:
    """Structured, interactable 3D scene representation."""

    def __init__(self, scene_id: str = "world_01"):
        self.scene_id = scene_id
        self.objects: dict[str, SceneObject] = {}
        self.relations: list[SceneRelation] = []
        self.time: float = 0.0

    def add_object(self, obj: SceneObject) -> None:
        self.objects[obj.id] = obj

    def add_relation(self, source_id: str, relation: str, target_id: str) -> None:
        if source_id in self.objects and target_id in self.objects:
            self.relations.append(SceneRelation(source_id, relation, target_id))

    def step_simulation(self, dt: float = 0.1) -> None:
        """Evolves object transforms and orbital/relational dynamics."""
        self.time += dt

        for rel in self.relations:
            if rel.relation == "orbiting":
                src = self.objects.get(rel.source_id)
                tgt = self.objects.get(rel.target_id)
                if src and tgt and not src.physics.is_static:
                    # Circular orbit around target
                    radius = 2.0
                    angle = self.time * 1.5
                    src.position[0] = tgt.position[0] + radius * math.cos(angle)
                    src.position[2] = tgt.position[2] + radius * math.sin(angle)
                    src.rotation[1] += dt * 45.0

        for obj in self.objects.values():
            if not obj.physics.is_static:
                # Apply velocity with damping
                for i in range(3):
                    obj.position[i] += obj.physics.velocity[i] * dt
                    obj.physics.velocity[i] *= 0.95

    def apply_agent_impulse(self, agent_id: str, action: str) -> None:
        """Applies interactive force from agent decisions."""
        agent = self.objects.get(agent_id)
        if not agent:
            return

        if action == "explore":
            agent.physics.velocity[0] += 0.8
            agent.physics.velocity[2] += 0.5
        elif action == "flee":
            agent.physics.velocity[0] -= 1.2
            agent.physics.velocity[2] -= 1.0
        elif action == "rest":
            agent.physics.velocity = [0.0, 0.0, 0.0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "time": round(self.time, 3),
            "objects": {k: asdict(v) for k, v in self.objects.items()},
            "relations": [asdict(r) for r in self.relations],
        }

    def export_wavefront_obj(self, filepath: str | Path) -> str:
        """Merges all objects into an indexed Wavefront OBJ file."""
        lines = [f"# Supermix Beyond Scene Graph Export: {self.scene_id}"]
        vertex_offset = 1

        for obj in self.objects.values():
            # Get base mesh
            shape_name = obj.shape if obj.shape in ("cube", "sphere", "pyramid", "cylinder") else "cube"
            vertices, faces = _primitive(shape_name)

            lines.append(f"o {obj.id}_{obj.shape}")

            # Transform vertices
            sx, sy, sz = obj.scale
            px, py, pz = obj.position
            for vx, vy, vz in vertices:
                tx = vx * sx + px
                ty = vy * sy + py
                tz = vz * sz + pz
                lines.append(f"v {tx:.4f} {ty:.4f} {tz:.4f}")

            # Offset faces
            for face in faces:
                adjusted = [idx + vertex_offset for idx in face]
                lines.append("f " + " ".join(str(idx) for idx in adjusted))

            vertex_offset += len(vertices)

        content = "\n".join(lines) + "\n"
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)


def create_procedural_beyond_world() -> ObjectCentricSceneGraph:
    """Builds a rich multi-entity 3D interactive world."""
    world = ObjectCentricSceneGraph("supermix_worldlab_01")

    # 1. Agent entity
    world.add_object(SceneObject(
        id="agent", name="BeyondAgent", shape="sphere",
        position=[0.0, 0.5, 0.0], scale=[0.5, 0.5, 0.5],
        color_name="cyan", rgb=[c / 255.0 for c in COLORS["cyan"]],
        physics=PhysicsProperties(mass=1.0, is_static=False)
    ))

    # 2. Water oasis resource
    world.add_object(SceneObject(
        id="oasis", name="FreshWaterOasis", shape="cylinder",
        position=[3.0, 0.1, 2.0], scale=[1.5, 0.2, 1.5],
        color_name="blue", rgb=[c / 255.0 for c in COLORS["blue"]],
        physics=PhysicsProperties(is_static=True, resource_type="water")
    ))

    # 3. Food foraging patch
    world.add_object(SceneObject(
        id="forage_patch", name="TidalFloraCache", shape="cube",
        position=[-2.5, 0.3, 1.5], scale=[0.8, 0.6, 0.8],
        color_name="green", rgb=[c / 255.0 for c in COLORS["green"]],
        physics=PhysicsProperties(is_static=True, resource_type="food")
    ))

    # 4. Rocky shelter
    world.add_object(SceneObject(
        id="shelter_rock", name="GraniteShelter", shape="pyramid",
        position=[0.0, 1.0, -3.5], scale=[2.0, 2.0, 2.0],
        color_name="yellow", rgb=[c / 255.0 for c in COLORS["yellow"]],
        physics=PhysicsProperties(is_static=True, resource_type="shelter")
    ))

    # 5. Satellite drone / sentinel
    world.add_object(SceneObject(
        id="sentinel", name="OrbitalSentinel", shape="sphere",
        position=[0.0, 3.0, 0.0], scale=[0.3, 0.3, 0.3],
        color_name="purple", rgb=[c / 255.0 for c in COLORS["purple"]],
        physics=PhysicsProperties(is_static=False, resource_type="hazard")
    ))
    world.add_relation("sentinel", "orbiting", "shelter_rock")

    return world
