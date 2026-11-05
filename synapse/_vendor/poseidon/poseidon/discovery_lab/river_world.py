"""Project PROMETHEUS Discovery Lab: Deterministic Procedural River World.

Implements Phase 1-5 of the Discovery Lab Roadmap:
- Two riverbanks (Left bank origin, Right bank goal territory)
- Procedurally generated river channel with configurable width (Narrow vs Wide)
- Interactive physical objects: Movable stones (sinking) and Buoyant logs (floating, connectable)
- Walkable connectivity verification across fluid obstacles
- Deterministic seeded execution and exact replay verification
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple

from .object_physics import (
    PhysicalObject, MaterialType, MATERIAL_DENSITIES,
    FLUID_DENSITY_WATER, GRAVITY
)
from .manipulation_actions import ManipulationActionType, ACTION_NAMES
from .enzyme_generator import MultimodalEnzymeGenerator, SynthesizedEnzyme


AGENT_MASS = 70.0  # kg / standard mass units


@dataclass
class AgentState:
    x: float
    y: float
    z: float = 0.0
    held_object_id: Optional[str] = None
    stamina: float = 1.0
    health: float = 1.0
    alive: bool = True
    reached_goal: bool = False
    synthesized_enzymes: List[str] = field(default_factory=list)


class RiverWorld:
    """Deterministic, seed-reproducible 3D manipulation environment."""

    VERSION = "prometheus-river-world-v1"

    def __init__(
        self,
        seed: int = 42,
        grid_width: int = 16,
        grid_height: int = 10,
        river_width: int = 3,       # 3 for Narrow (stepping stones viable), 7 for Wide (raft required)
        water_depth: float = 1.5,
        max_steps: int = 100,
        num_stones: int = 6,
        num_logs: int = 6,
    ):
        self.base_seed = int(seed)
        self.rng = np.random.RandomState(seed)
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.river_width = int(river_width)
        self.water_depth = float(water_depth)
        self.max_steps = int(max_steps)
        self.num_stones = int(num_stones)
        self.num_logs = int(num_logs)

        # River channel placement
        center_x = self.grid_width // 2
        half_w = self.river_width // 2
        self.river_start_x = max(2, center_x - half_w)
        self.river_end_x = min(self.grid_width - 3, self.river_start_x + self.river_width - 1)
        self.river_width_actual = self.river_end_x - self.river_start_x + 1

        self.step_idx = 0
        self.agent = AgentState(x=1.0, y=self.grid_height / 2.0, z=0.0)
        self.objects: Dict[str, PhysicalObject] = {}
        self.enzyme_generator = MultimodalEnzymeGenerator(seed=self.base_seed + 100)

        # Transition history for replay verification
        self.action_history: List[int] = []
        self.reward_history: List[float] = []

        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Resets world to deterministic initial state."""
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.RandomState(seed)
            self.enzyme_generator = MultimodalEnzymeGenerator(seed=self.base_seed + 100)

        self.step_idx = 0
        self.agent = AgentState(
            x=1.0,
            y=float(self.grid_height // 2),
            z=0.0,
            held_object_id=None,
            stamina=1.0,
            health=1.0,
            alive=True,
            reached_goal=False,
            synthesized_enzymes=[]
        )
        self.objects.clear()
        self.action_history.clear()
        self.reward_history.clear()

        # 1. Spawn Movable Granite Stones on Left Bank
        for i in range(self.num_stones):
            sx = float(self.rng.uniform(0.5, self.river_start_x - 0.5))
            sy = float(self.rng.uniform(0.5, self.grid_height - 0.5))
            stone_id = f"stone_{i+1:02d}"
            self.objects[stone_id] = PhysicalObject(
                object_id=stone_id,
                name=f"Granite Stone {i+1}",
                material=MaterialType.GRANITE_STONE,
                position=[sx, sy, 0.0],
                dimensions=[0.8, 0.8, 0.8],
            )

        # 2. Spawn Buoyant Pine Logs on Left Bank
        for j in range(self.num_logs):
            lx = float(self.rng.uniform(0.5, self.river_start_x - 0.5))
            ly = float(self.rng.uniform(0.5, self.grid_height - 0.5))
            log_id = f"log_{j+1:02d}"
            self.objects[log_id] = PhysicalObject(
                object_id=log_id,
                name=f"Pine Log {j+1}",
                material=MaterialType.PINE_LOG,
                position=[lx, ly, 0.0],
                dimensions=[1.8, 0.6, 0.6],  # Length = 1.8m, radius/width = 0.6m
            )

        return self.get_observation()

    def is_in_river(self, x: float) -> bool:
        """Checks if coordinate falls inside the river channel."""
        return self.river_start_x <= x <= (self.river_end_x + 1.0)

    def is_on_right_bank(self, x: float) -> bool:
        """Checks if coordinate reaches the opposite goal territory."""
        return x > (self.river_end_x + 1.0)

    def get_supported_surface_at(self, x: float, y: float) -> Tuple[bool, float, Optional[str]]:
        """Determines if a location is walkable and what object/ground supports it.

        Returns: (is_supported, support_height_z, support_type)
        """
        # Ground banks are naturally solid
        if not self.is_in_river(x):
            return True, 0.0, "ground_bank"

        # In river: check if any stone or buoyant raft supports this position
        for obj in self.objects.values():
            if obj.is_held:
                continue
            ox, oy, oz = obj.position
            dx, dy, dz = obj.dimensions
            # Bounding box support check
            if (ox - dx/2.0 <= x <= ox + dx/2.0) and (oy - dy/2.0 <= y <= oy + dy/2.0):
                if obj.material in (MaterialType.GRANITE_STONE, MaterialType.BASALT_STONE):
                    # Stones sink unless water depth is shallow enough to reach surface
                    if oz >= -0.2:  # Reaches surface
                        return True, oz + dz/2.0, "stepping_stone"
                elif obj.is_buoyant:
                    # Floating log or raft assemblage
                    total_mass, support_payload, can_float = obj.compute_assemblage_buoyancy(self.objects)
                    if can_float and support_payload >= AGENT_MASS:
                        return True, 0.0, "buoyant_raft" if len(obj.connected_object_ids) > 0 else "floating_log"

        # Unsupported open water
        return False, -self.water_depth, "deep_water"

    def get_adjacent_objects(self, radius: float = 1.8) -> List[PhysicalObject]:
        """Finds all objects within physical reach of the agent."""
        reachable = []
        for obj in self.objects.values():
            if obj.is_held:
                continue
            dist = math.hypot(obj.position[0] - self.agent.x, obj.position[1] - self.agent.y)
            if dist <= radius:
                reachable.append(obj)
        return reachable

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Executes one discrete manipulation action."""
        if not self.agent.alive:
            return self.get_observation(), 0.0, True, {"alive": False, "reason": "agent_dead"}

        self.step_idx += 1
        if isinstance(action, str):
            if action in ManipulationActionType.__members__:
                action_type = ManipulationActionType[action]
            else:
                matched = None
                for k, v in ACTION_NAMES.items():
                    if v.lower() == action.lower():
                        matched = k
                        break
                action_type = matched if matched is not None else ManipulationActionType.MOVE_EAST
        else:
            action_type = ManipulationActionType(int(action) % len(ManipulationActionType))
        self.action_history.append(int(action_type))

        reward = -0.01  # Small metabolic time cost
        done = False
        info: Dict[str, Any] = {
            "step": self.step_idx,
            "action": ACTION_NAMES.get(action_type, "unknown"),
            "event": "none",
            "connected_assemblages": 0,
        }

        step_dist = 0.8

        # --- 1. Movement Actions ---
        if action_type in (
            ManipulationActionType.MOVE_NORTH,
            ManipulationActionType.MOVE_SOUTH,
            ManipulationActionType.MOVE_EAST,
            ManipulationActionType.MOVE_WEST,
        ):
            dx, dy = 0.0, 0.0
            if action_type == ManipulationActionType.MOVE_NORTH:
                dy = step_dist
            elif action_type == ManipulationActionType.MOVE_SOUTH:
                dy = -step_dist
            elif action_type == ManipulationActionType.MOVE_EAST:
                dx = step_dist
            elif action_type == ManipulationActionType.MOVE_WEST:
                dx = -step_dist

            target_x = max(0.0, min(float(self.grid_width), self.agent.x + dx))
            target_y = max(0.0, min(float(self.grid_height), self.agent.y + dy))

            supported, surf_z, surf_type = self.get_supported_surface_at(target_x, target_y)

            if not supported:
                # Agent falls into deep river
                self.agent.health -= 0.50
                reward -= 2.0
                info["event"] = "fell_in_deep_water"
                if self.agent.health <= 0.0:
                    self.agent.alive = False
                    done = True
                    info["event"] = "drowned"
            else:
                self.agent.x = target_x
                self.agent.y = target_y
                self.agent.z = surf_z
                info["event"] = f"moved_onto_{surf_type}"

                # Update held object position
                if self.agent.held_object_id and self.agent.held_object_id in self.objects:
                    held = self.objects[self.agent.held_object_id]
                    held.position[0] = self.agent.x
                    held.position[1] = self.agent.y
                    held.position[2] = self.agent.z + 0.5

                # Goal Check: Reaching Right Bank
                if self.is_on_right_bank(self.agent.x):
                    self.agent.reached_goal = True
                    reward += 10.0
                    done = True
                    info["event"] = "reached_opposite_bank_success"

        # --- 2. Inspect Action ---
        elif action_type == ManipulationActionType.INSPECT:
            adj = self.get_adjacent_objects(radius=2.0)
            if adj:
                inspected = adj[0]
                info["event"] = "inspected_object"
                info["inspected_object"] = inspected.to_dict()
                reward += 0.05
            else:
                info["event"] = "inspect_nothing_nearby"

        # --- 3. Grasp Action ---
        elif action_type == ManipulationActionType.GRASP:
            if self.agent.held_object_id is None:
                adj = self.get_adjacent_objects(radius=1.5)
                if adj:
                    # Pick up nearest unheld object
                    chosen = adj[0]
                    chosen.is_held = True
                    chosen.holder_id = "agent"
                    self.agent.held_object_id = chosen.object_id
                    info["event"] = f"grasped_{chosen.object_id}"
                    reward += 0.05
                else:
                    info["event"] = "grasp_no_object"
            else:
                info["event"] = "already_holding_object"

        # --- 4. Release Action ---
        elif action_type == ManipulationActionType.RELEASE:
            if self.agent.held_object_id and self.agent.held_object_id in self.objects:
                held = self.objects[self.agent.held_object_id]
                held.is_held = False
                held.holder_id = None
                # Place slightly ahead of agent
                place_x = self.agent.x + 0.8
                place_y = self.agent.y
                held.position[0] = place_x
                held.position[1] = place_y

                # Physics settle in water or land
                if self.is_in_river(place_x):
                    if held.is_buoyant:
                        held.position[2] = 0.0  # Floats on water surface
                        held.submerged_fraction = held.density / FLUID_DENSITY_WATER
                        info["event"] = f"released_{held.object_id}_floating"
                    else:
                        # Heavy stone settles to water depth (if shallow, reaches surface)
                        held.position[2] = max(-self.water_depth, -self.water_depth + held.dimensions[2])
                        info["event"] = f"released_{held.object_id}_settled"
                else:
                    held.position[2] = 0.0
                    info["event"] = f"released_{held.object_id}_on_ground"

                self.agent.held_object_id = None
                reward += 0.10
            else:
                info["event"] = "release_nothing_held"

        # --- 5. Push Action ---
        elif action_type == ManipulationActionType.PUSH:
            adj = self.get_adjacent_objects(radius=1.8)
            if adj:
                pushed = adj[0]
                # Push toward river (+x)
                pushed.position[0] += 1.2
                if self.is_in_river(pushed.position[0]):
                    if pushed.is_buoyant:
                        pushed.position[2] = 0.0
                    else:
                        pushed.position[2] = -self.water_depth + pushed.dimensions[2]
                info["event"] = f"pushed_{pushed.object_id}"
                reward += 0.05
            else:
                info["event"] = "push_no_target"

        # --- 6. Rotate Action ---
        elif action_type == ManipulationActionType.ROTATE:
            if self.agent.held_object_id and self.agent.held_object_id in self.objects:
                held = self.objects[self.agent.held_object_id]
                held.orientation_yaw = (held.orientation_yaw + math.pi / 2.0) % (2.0 * math.pi)
                info["event"] = f"rotated_held_{held.object_id}"
            else:
                adj = self.get_adjacent_objects(radius=1.8)
                if adj:
                    adj[0].orientation_yaw = (adj[0].orientation_yaw + math.pi / 2.0) % (2.0 * math.pi)
                    info["event"] = f"rotated_{adj[0].object_id}"

        # --- 7. Connect Action (Assemble logs) ---
        elif action_type == ManipulationActionType.CONNECT:
            logs = [
                obj for obj in self.objects.values()
                if obj.is_buoyant and not obj.is_held
                and math.hypot(obj.position[0] - self.agent.x, obj.position[1] - self.agent.y) <= 2.5
            ]
            if len(logs) >= 2:
                # Connect the two closest logs
                log_a, log_b = logs[0], logs[1]
                if log_b.object_id not in log_a.connected_object_ids:
                    log_a.connected_object_ids.append(log_b.object_id)
                if log_a.object_id not in log_b.connected_object_ids:
                    log_b.connected_object_ids.append(log_a.object_id)

                # Snap together side-by-side
                avg_y = (log_a.position[1] + log_b.position[1]) / 2.0
                log_a.position[1] = avg_y - 0.35
                log_b.position[1] = avg_y + 0.35
                log_a.position[0] = (log_a.position[0] + log_b.position[0]) / 2.0
                log_b.position[0] = log_a.position[0]

                info["event"] = f"connected_{log_a.object_id}_to_{log_b.object_id}"
                reward += 0.50
            else:
                info["event"] = "connect_insufficient_logs"

        # --- 8. Catalyze Action (Enzyme Cross-Linking) ---
        elif action_type == ManipulationActionType.CATALYZE:
            # Synthesize tailored bio-resin enzyme for wood/lignin cross-linking
            enzyme = self.enzyme_generator.generate_catalyst_for_substrate(
                substrate_type="lignocellulose",
                target_shear_force=350.0,
            )
            self.agent.synthesized_enzymes.append(enzyme.enzyme_id)

            # Apply catalytic curing to all connected assemblages
            cured_count = 0
            for obj in self.objects.values():
                if len(obj.connected_object_ids) > 0:
                    obj.bond_strength = max(obj.bond_strength, enzyme.cured_bond_strength)
                    cured_count += 1

            if cured_count > 0:
                reward += 1.0
                info["event"] = f"catalytic_curing_applied_{enzyme.enzyme_id}"
                info["cured_bonds"] = cured_count
                info["enzyme"] = enzyme.to_dict()
            else:
                info["event"] = "catalyze_no_assemblage_nearby"

        # Check step timeout
        if self.step_idx >= self.max_steps:
            done = True
            info["terminal_reason"] = "timeout"

        self.reward_history.append(float(reward))
        info["alive"] = self.agent.alive
        info["reached_goal"] = self.agent.reached_goal
        info["agent_position"] = [round(self.agent.x, 3), round(self.agent.y, 3), round(self.agent.z, 3)]

        return self.get_observation(), reward, done, info

    def get_observation(self) -> np.ndarray:
        """Returns compact 24-dimensional observation vector suitable for learning and planning."""
        obs = np.zeros(24, dtype=np.float32)
        # 0..3: Agent state
        obs[0] = float(self.agent.x / self.grid_width)
        obs[1] = float(self.agent.y / self.grid_height)
        obs[2] = float(self.agent.z)
        obs[3] = float(1.0 if self.agent.held_object_id is not None else 0.0)

        # 4..7: River geometry
        obs[4] = float(self.river_start_x / self.grid_width)
        obs[5] = float(self.river_end_x / self.grid_width)
        obs[6] = float(self.river_width_actual / self.grid_width)
        obs[7] = float(self.water_depth)

        # 8..11: Agent vitality & stamina
        obs[8] = float(self.agent.health)
        obs[9] = float(self.agent.stamina)
        obs[10] = float(1.0 if self.agent.alive else 0.0)
        obs[11] = float(1.0 if self.agent.reached_goal else 0.0)

        # 12..15: Nearest stone features
        stones = [o for o in self.objects.values() if "stone" in o.material and not o.is_held]
        if stones:
            stones.sort(key=lambda s: math.hypot(s.position[0] - self.agent.x, s.position[1] - self.agent.y))
            nearest_s = stones[0]
            obs[12] = float(nearest_s.position[0] / self.grid_width)
            obs[13] = float(nearest_s.position[1] / self.grid_height)
            obs[14] = float(nearest_s.density)
            obs[15] = float(1.0 if nearest_s.position[0] >= self.river_start_x else 0.0)

        # 16..19: Nearest log / raft features
        logs = [o for o in self.objects.values() if o.is_buoyant and not o.is_held]
        if logs:
            logs.sort(key=lambda l: math.hypot(l.position[0] - self.agent.x, l.position[1] - self.agent.y))
            nearest_l = logs[0]
            obs[16] = float(nearest_l.position[0] / self.grid_width)
            obs[17] = float(nearest_l.position[1] / self.grid_height)
            obs[18] = float(nearest_l.max_support_payload / 200.0)
            obs[19] = float(len(nearest_l.connected_object_ids) / 4.0)

        # 20..23: World state & enzymatic affordances
        obs[20] = float(len(self.agent.synthesized_enzymes) / 3.0)
        obs[21] = float(self.step_idx / self.max_steps)
        # Check current standing tile support
        supported, _, _ = self.get_supported_surface_at(self.agent.x, self.agent.y)
        obs[22] = float(1.0 if supported else 0.0)
        obs[23] = float(1.0 if self.is_in_river(self.agent.x) else 0.0)

        return obs

    def get_state_hash(self) -> str:
        """Computes deterministic hash of complete current world state."""
        state_dict = {
            "step": self.step_idx,
            "agent_x": round(float(self.agent.x), 3),
            "agent_y": round(float(self.agent.y), 3),
            "agent_z": round(float(self.agent.z), 3),
            "held": self.agent.held_object_id,
            "health": round(float(self.agent.health), 3),
            "stamina": round(float(self.agent.stamina), 3),
            "objects": {oid: o.to_dict() for oid, o in sorted(self.objects.items())},
        }
        return hashlib.sha256(json.dumps(state_dict, sort_keys=True).encode("utf-8")).hexdigest()

    def verify_exact_replay(self) -> str:
        """Computes deterministic replay hash of the episode trajectory."""
        payload = {
            "version": self.VERSION,
            "seed": self.base_seed,
            "river_width": self.river_width,
            "actions": self.action_history,
            "rewards": [round(float(r), 4) for r in self.reward_history],
            "reached_goal": self.agent.reached_goal,
            "final_health": round(float(self.agent.health), 3),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
