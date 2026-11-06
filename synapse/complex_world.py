"""Advanced Dynamic Complex Environments: Ecosystems, Moving Predators, Terrain Fields & Shifting Gates.

Extends the native embodied World with rich ecological and physical complexity:
1. **Dynamic Moving Predators & Hazards**: Active autonomous drones that patrol or track the agent.
2. **Ecological Resource Patches**: Food clusters that deplete and regrow dynamically.
3. **Terrain Zones & Variable Friction Fields**:
   - Sand patches: High friction, increased metabolic burn.
   - Ice fields: Low friction, inertial momentum slip.
   - Mud pits: Viscous damping on locomotion.
4. **Dynamic Barrier Gates**: Walls that open/close in response to button events or cycle on a clock.
5. **Day/Night Ambient Cycle**: Variable illumination altering retinal sensory signal-to-noise ratio.
6. **New Frontier Tasks**: `ecosystem`, `predator_evasion`, `dynamic_arena`, `puzzle_chamber`.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from . import sources
from graftbrain.world import World as BaseWorld, TASKS as BASE_TASKS, COLORS, HALF, RANGE, FOV, N_RAYS


COMPLEX_TASKS = [
    "ecosystem",
    "predator_evasion",
    "dynamic_arena",
    "puzzle_chamber",
]

ALL_TASKS = BASE_TASKS + COMPLEX_TASKS


class ComplexWorld(BaseWorld):
    """Frontier virtual 3D environment with dynamic multi-agent ecology and shifting terrain physics."""

    def __init__(self, seed: int = 0, curriculum: str = "auto"):
        self.complex_seed = seed
        self.rng_complex = np.random.default_rng(seed)
        self.predators: List[Dict[str, Any]] = []
        self.terrain_zones: List[Dict[str, Any]] = []
        self.dynamic_gates: List[Dict[str, Any]] = []
        self.day_night_clock = 0.0
        self.ambient_light = 1.0
        self.food_respawn_timer = 0
        self.base_walls: List[Tuple[float, float, float, float]] = []
        super().__init__(seed=seed, curriculum=curriculum)

    def reset(self, task: Optional[str] = None):
        """Initializes both standard baseline and advanced complex tasks."""
        self.predators.clear()
        self.terrain_zones.clear()
        self.dynamic_gates.clear()

        # Check if the requested task is one of our new frontier tasks
        chosen_task = task or self.curriculum
        if chosen_task == "auto":
            # Rotate across all tasks
            chosen_task = ALL_TASKS[self.episode % len(ALL_TASKS)]

        # Run base reset
        obs = super().reset(chosen_task if chosen_task in BASE_TASKS else "forage")
        self.task = chosen_task

        # Setup specialized frontier mechanics
        if self.task == "ecosystem":
            self._setup_ecosystem()
        elif self.task == "predator_evasion":
            self._setup_predator_evasion()
        elif self.task == "dynamic_arena":
            self._setup_dynamic_arena()
        elif self.task == "puzzle_chamber":
            self._setup_puzzle_chamber()

        self.base_walls = self.walls.copy()
        self.history.setdefault(self.task,deque(maxlen=50))
        # A frontier task cannot inherit the temporary forage potential.
        self._prev = self._progress()
        return self.observe()

    # ------------------------------------------------------------------
    # Frontier Task Setups
    # ------------------------------------------------------------------
    def _setup_ecosystem(self):
        """Food patches with regrowth, 1 stalking predator, and terrain zones."""
        self.objects.clear()
        self.walls = self._border()

        # Terrain zones
        self.terrain_zones = [
            {"kind": "ice", "x": -2.5, "y": 0.0, "r": 1.6, "friction": 0.25, "color": "cyan"},
            {"kind": "sand", "x": 2.5, "y": 0.0, "r": 1.6, "friction": 2.40, "color": "yellow"},
            {"kind": "mud", "x": 0.0, "y": -2.5, "r": 1.4, "friction": 1.80, "color": "grey"},
        ]

        # Regenerating food objects
        for _ in range(6):
            spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
            self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

        # Autonomous moving predator
        self.predators.append({
            "id": 100,
            "x": 3.0,
            "y": 3.0,
            "th": float(self.rng_complex.uniform(0, math.pi * 2)),
            "speed": 0.12,
            "kind": "stalker",
            "r": 0.40,
            "color": "red",
        })

    def _setup_predator_evasion(self):
        """Intense evasion: 2 agile predator drones pursuing the agent while agent collects power orbs."""
        self.objects.clear()
        self.walls = self._border()

        # Central shelter obstacle
        self.walls.extend([
            (-1.2, -1.2, 1.2, -1.2),
            (1.2, -1.2, 1.2, 1.2),
            (1.2, 1.2, -1.2, 1.2),
            (-1.2, 1.2, -1.2, -1.2),
        ])

        # Survival power orbs
        for _ in range(4):
            spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
            self._obj("food", spot[0], spot[1], "gold", r=0.28, movable=False)

        # 2 Fast pursuit predators
        self.predators.append({
            "id": 101,
            "x": -3.5,
            "y": 3.5,
            "th": 0.0,
            "speed": 0.16,
            "kind": "pursuer",
            "r": 0.38,
            "color": "red",
        })
        self.predators.append({
            "id": 102,
            "x": 3.5,
            "y": -3.5,
            "th": math.pi,
            "speed": 0.14,
            "kind": "pursuer",
            "r": 0.38,
            "color": "purple",
        })

    def _setup_dynamic_arena(self):
        """Dynamic sliding gates and shifting corridors."""
        self.objects.clear()
        self.walls = self._border()

        # Dynamic gates that oscillate
        self.dynamic_gates = [
            {"id": "gate1", "x1": 0.0, "y1": -4.0, "x2": 0.0, "y2": -1.0, "open": False, "timer": 0},
            {"id": "gate2", "x1": 0.0, "y1": 1.0, "x2": 0.0, "y2": 4.0, "open": True, "timer": 20},
        ]

        # Target reward in far room
        self._obj("target", 3.5, 0.0, "green", r=0.35, movable=False)
        self._obj("food", -3.5, 0.0, "yellow", r=0.25, movable=False)

    def _setup_puzzle_chamber(self):
        """Button unlock puzzle: Press yellow button to open locked gate to gold goal."""
        self.objects.clear()
        self.walls = self._border()

        # Dividing wall with locked gate
        self.walls.extend([
            (0.0, -5.0, 0.0, -0.8),
            (0.0, 0.8, 0.0, 5.0),
        ])
        self.dynamic_gates = [
            {"id": "puzzle_gate", "x1": 0.0, "y1": -0.8, "x2": 0.0, "y2": 0.8, "open": False, "locked": True}
        ]

        # Button on left side
        self._obj("button", -2.5, 2.5, "yellow", r=0.35, shape="button", movable=False, solid=False, pressed=False)
        # Goal on right side
        self._obj("target", 3.0, 0.0, "gold", r=0.40, movable=False)

    # ------------------------------------------------------------------
    # Step Physics & Environmental Updates
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        from .actions import DRIVE, validate_action
        action = validate_action(action)
        # 1. Terrain zone physics (modulate thrust/turn)
        current_friction = 1.0
        in_friction_zone = False
        ax, ay = self.a["x"], self.a["y"]

        for zone in self.terrain_zones:
            dist = math.hypot(ax - zone["x"], ay - zone["y"])
            if dist < zone["r"]:
                current_friction = zone["friction"]
                in_friction_zone = True
                break

        # Adjust effective action under terrain friction
        eff_action = action.copy()
        eff_action[DRIVE] = np.clip(eff_action[DRIVE] / current_friction, -1, 1)

        # 2. Step base environment
        _, reward, done, info = super().step(eff_action)
        base_reward = reward
        ax, ay = self.a['x'], self.a['y']
        info["in_friction_zone"] = in_friction_zone
        info["terrain_friction"] = current_friction

        # 3. Dynamic Gates update
        for gate in self.dynamic_gates:
            if gate.get("locked"):
                # Check if puzzle button was pressed
                for obj in self.objects:
                    if obj.get("kind") == "button" and obj.get("pressed"):
                        gate["open"] = True
            else:
                gate["timer"] = (gate.get("timer", 0) + 1) % 50
                gate["open"] = gate["timer"] < 25

        # Rebuild walls including currently closed gates
        if self.dynamic_gates:
            active_walls = self.base_walls.copy()
            for gate in self.dynamic_gates:
                if not gate.get("open", False):
                    active_walls.append((gate["x1"], gate["y1"], gate["x2"], gate["y2"]))
            self.walls = active_walls

        # 4. Moving Predators update
        min_hazard_dist = 10.0
        for pred in self.predators:
            # Vector to agent
            p_dx = ax - pred["x"]
            p_dy = ay - pred["y"]
            dist_to_agent = math.hypot(p_dx, p_dy)

            if pred["kind"] == "pursuer" and dist_to_agent < 4.0:
                # Direct chase steering
                desired_th = math.atan2(p_dy, p_dx)
                pred["th"] += 0.12 * math.sin(desired_th - pred["th"])
            else:
                # Patrol bounce
                pred["th"] += float(self.rng_complex.normal(0.0, 0.08))

            # Move predator
            pred["x"] += math.cos(pred["th"]) * pred["speed"]
            pred["y"] += math.sin(pred["th"]) * pred["speed"]

            # Arena boundary bounce
            h = HALF - 0.5
            if pred["x"] < -h or pred["x"] > h:
                pred["th"] = math.pi - pred["th"]
                pred["x"] = float(np.clip(pred["x"], -h, h))
            if pred["y"] < -h or pred["y"] > h:
                pred["th"] = -pred["th"]
                pred["y"] = float(np.clip(pred["y"], -h, h))

            # Collision with agent
            dist_to_agent = math.hypot(ax-pred['x'],ay-pred['y'])
            min_hazard_dist = min(min_hazard_dist,dist_to_agent)
            if dist_to_agent < (pred["r"] + 0.35):
                reward -= 0.45  # Penalty for getting hit by predator
                info["predator_impact"] = True

        info["min_hazard_dist"] = min_hazard_dist

        # 5. Ecological Food Respawn
        self.food_respawn_timer += 1
        if self.task == "ecosystem" and self.food_respawn_timer % 40 == 0:
            current_food = sum(1 for o in self.objects if o.get("kind") == "food" and o.get("visible", True))
            if current_food < 6:
                spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
                self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

        # 6. Day/Night ambient illumination cycle
        self.day_night_clock += 0.015
        self.ambient_light = float(0.65 + 0.35 * math.sin(self.day_night_clock))
        info["ambient_light"] = self.ambient_light

        # BaseWorld recorded its reward before the ecological penalties. Keep
        # transition observations and episode accounting bound to the final reward.
        self.ep_return += reward-base_reward
        self.last_reward = reward
        if done:
            info['ret'] = self.ep_return
        return self.observe(), reward, done, info

    def observe(self):
        """Generates 155-dim sensory observation with ambient lighting modulation."""
        a = self.a
        prop = [
            (a["reach"] - 0.25) / 0.65,
            a["height"] / 0.75,
            a["grip"],
            float(a["held"] is not None),
            a["v"] / 1.4,
            a["w"] / 2.5,
            math.sin(a["th"]),
            math.cos(a["th"]),
        ]
        if self.task in BASE_TASKS:
            onehot = np.eye(len(BASE_TASKS))[BASE_TASKS.index(self.task)]
        else:
            idx = (BASE_TASKS.index("forage") + COMPLEX_TASKS.index(self.task)) % len(BASE_TASKS)
            onehot = np.eye(len(BASE_TASKS))[idx]

        ret = self.retina()
        if hasattr(self, "ambient_light") and self.ambient_light != 1.0:
            ret = ret * float(self.ambient_light)

        return np.concatenate([ret, prop, onehot, self.cue, [self.last_reward, self.t / self.max_t]]).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        """Extends snapshot with predators, terrain zones, and dynamic gate states."""
        snap = super().snapshot()
        snap["predators"] = [
            {"id": p["id"], "x": round(float(p["x"]), 3), "y": round(float(p["y"]), 3),
             "th": round(float(p["th"]), 3), "r": p["r"], "kind": p["kind"], "color": p["color"]}
            for p in self.predators
        ]
        snap["terrain_zones"] = self.terrain_zones
        snap["dynamic_gates"] = self.dynamic_gates
        snap["ambient_light"] = round(self.ambient_light, 3)
        return snap
