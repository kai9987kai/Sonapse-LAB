"""Advanced Dynamic Complex Environments: Ecosystems, Moving Predators, Terrain Fields & Shifting Gates.

Extends the native embodied World with rich ecological and physical complexity:
1. **Dynamic Moving Predators & Coordinated Packs**: Autonomous drones that patrol, track, and flank.
2. **Dynamic Vector Flow Fields (River Currents & Conveyors)**: Physical current displacement fields.
3. **Topographic Elevation Hill Fields**: Continuous gravity-gradient forces that accelerate/retard motion.
4. **Sentinel Searchlights & Acoustic Tripwires**: Stealth zones, rotating light cones, and sound barriers.
5. **Multi-Switch Sequential Logic & Oscillating Gates**: Timed and pressure-actuated barrier bulkheads.
6. **Frontier Tasks**:
   - `ecosystem`: Foraging amidst regenerating resource patches and stalking predators.
   - `predator_evasion`: High-speed evasion of dual pursuit drones.
   - `dynamic_arena`: Timing passage through oscillating barrier gates.
   - `puzzle_chamber`: Button unlock mechanics and gate traversal.
   - `apex_megasite`: Cross-river traversal against vector currents, elevation hills, and pack hunters.
   - `stealth_labyrinth`: Corridor navigation evading rotating searchlights and tripwires via shadow cloaks.
   - `kinetic_foundry`: Navigating opposing conveyor belts and placing cargo blocks on pressure pads.
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
    "apex_megasite",
    "stealth_labyrinth",
    "kinetic_foundry",
]

ALL_TASKS = BASE_TASKS + COMPLEX_TASKS


class ComplexWorld(BaseWorld):
    """Frontier virtual 3D environment with dynamic multi-agent ecology, vector fields, and shifting terrain physics."""

    def __init__(self, seed: int = 0, curriculum: str = "auto"):
        self.complex_seed = seed
        self.rng_complex = np.random.default_rng(seed)
        self.predators: List[Dict[str, Any]] = []
        self.terrain_zones: List[Dict[str, Any]] = []
        self.dynamic_gates: List[Dict[str, Any]] = []
        self.elevation_hills: List[Dict[str, Any]] = []
        self.sentinels: List[Dict[str, Any]] = []
        self.sound_tripwires: List[Dict[str, Any]] = []
        self.switches: List[Dict[str, Any]] = []
        self.last_alarm_tripped: bool = False
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
        self.elevation_hills.clear()
        self.sentinels.clear()
        self.sound_tripwires.clear()
        self.switches.clear()
        self.last_alarm_tripped = False

        # Check if the requested task is one of our new frontier tasks
        chosen_task = task or self.curriculum
        if chosen_task == "auto":
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
        elif self.task == "apex_megasite":
            self._setup_apex_megasite()
        elif self.task == "stealth_labyrinth":
            self._setup_stealth_labyrinth()
        elif self.task == "kinetic_foundry":
            self._setup_kinetic_foundry()

        self.base_walls = self.walls.copy()
        self.history.setdefault(self.task, deque(maxlen=50))
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
            "alerted": False,
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
            "alerted": False,
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
            "alerted": False,
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

    def _setup_apex_megasite(self):
        """Apex frontier mega-environment: vector river current, 2 elevation hills, coordinated pack predators, and locked apex core."""
        self.objects.clear()
        self.walls = self._border()

        # Central river vector current running north-to-south
        self.terrain_zones = [
            {"kind": "river", "x": 0.0, "y": 0.0, "r": 2.2, "friction": 1.10, "vx": 0.0, "vy": -0.15, "color": "blue"},
            {"kind": "sand", "x": -2.8, "y": -2.5, "r": 1.5, "friction": 2.20, "color": "yellow"},
            {"kind": "ice", "x": 2.8, "y": 2.5, "r": 1.5, "friction": 0.30, "color": "cyan"},
        ]

        # Topographic elevation hills
        self.elevation_hills = [
            {"x": -2.6, "y": 2.0, "height": 1.8, "sigma": 1.4},   # West Plateau
            {"x": 2.6, "y": -2.0, "height": -1.5, "sigma": 1.3},  # East Basin
        ]

        # Coordinated predator pack (Leader + Flanker)
        self.predators.append({
            "id": 201,
            "x": -3.2,
            "y": -3.2,
            "th": float(self.rng_complex.uniform(0, math.pi * 2)),
            "speed": 0.13,
            "kind": "pack_leader",
            "r": 0.38,
            "color": "red",
            "alerted": False,
        })
        self.predators.append({
            "id": 202,
            "x": -2.2,
            "y": -3.6,
            "th": float(self.rng_complex.uniform(0, math.pi * 2)),
            "speed": 0.12,
            "kind": "pack_hunter",
            "r": 0.36,
            "color": "darkred",
            "alerted": False,
        })

        # Barrier separating the apex sanctuary on the east
        self.walls.extend([
            (2.2, -5.0, 2.2, -1.0),
            (2.2, 1.0, 2.2, 5.0),
        ])
        self.dynamic_gates = [
            {"id": "apex_gate", "x1": 2.2, "y1": -1.0, "x2": 2.2, "y2": 1.0, "open": False, "locked": True}
        ]

        # Puzzle switch on the west plateau
        self._obj("button", -3.2, 2.6, "yellow", r=0.35, shape="button", movable=False, solid=False, pressed=False)
        # Apex Energy Core inside sanctuary
        self._obj("target", 3.8, 0.0, "gold", r=0.45, movable=False)

        # Regenerating forage crystals
        for _ in range(4):
            spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
            self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

    def _setup_stealth_labyrinth(self):
        """Stealth labyrinth: rotating sentinel searchlights, acoustic sound tripwires, cloaking shadow zones."""
        self.objects.clear()
        self.walls = self._border()

        # Labyrinth dividing corridor walls
        self.walls.extend([
            (-2.2, -4.5, -2.2, 0.8),
            (2.2, -0.8, 2.2, 4.5),
            (-4.5, 0.0, -1.2, 0.0),
            (1.2, 0.0, 4.5, 0.0),
        ])

        # Cloaking shadow zones (prevent sentinel detection)
        self.terrain_zones = [
            {"kind": "shadow", "x": -1.2, "y": -2.0, "r": 1.3, "friction": 1.0, "color": "purple"},
            {"kind": "shadow", "x": 1.2, "y": 2.0, "r": 1.3, "friction": 1.0, "color": "purple"},
        ]

        # Rotating sentinel searchlights
        self.sentinels = [
            {"id": 301, "x": 0.0, "y": 2.2, "angle": 0.0, "angular_vel": 0.045, "range": 3.0, "cone_half_angle": 0.42, "alerted": False, "color": "orange"},
            {"id": 302, "x": 0.0, "y": -2.2, "angle": math.pi, "angular_vel": -0.045, "range": 3.0, "cone_half_angle": 0.42, "alerted": False, "color": "orange"},
        ]

        # Acoustic sound tripwire across central crossing
        self.sound_tripwires = [
            {"id": "trip1", "x1": -1.0, "y1": 0.0, "x2": 1.0, "y2": 0.0, "max_safe_speed": 0.28}
        ]

        # Extraction beacon
        self._obj("target", 3.8, -3.8, "cyan", r=0.42, movable=False)
        # Stealth decoy orbs
        self._obj("food", -3.6, 3.6, "gold", r=0.26, movable=False)

    def _setup_kinetic_foundry(self):
        """Kinetic foundry: opposing directional conveyor belts, movable cargo box, and pressure plate reactor unlock."""
        self.objects.clear()
        self.walls = self._border()

        # Opposing conveyor belts
        self.terrain_zones = [
            {"kind": "conveyor", "x": -2.2, "y": 0.0, "r": 1.7, "friction": 0.9, "vx": 0.16, "vy": 0.0, "color": "orange"},
            {"kind": "conveyor", "x": 2.2, "y": 0.0, "r": 1.7, "friction": 0.9, "vx": -0.16, "vy": 0.0, "color": "orange"},
            {"kind": "conveyor", "x": 0.0, "y": -2.4, "r": 1.5, "friction": 0.9, "vx": 0.0, "vy": 0.16, "color": "orange"},
        ]

        # Industrial blast barrier
        self.walls.extend([
            (-5.0, 2.0, -1.2, 2.0),
            (1.2, 2.0, 5.0, 2.0),
        ])
        self.dynamic_gates = [
            {"id": "blast_door", "x1": -1.2, "y1": 2.0, "x2": 1.2, "y2": 2.0, "open": False, "locked": True}
        ]

        # Heavy cargo box
        self._obj("box", -2.2, 0.0, "grey", r=0.45, movable=True, solid=True)
        # Pressure sensor pad
        self._obj("pad", 2.2, 0.0, "yellow", r=0.60, shape="pad", movable=False, solid=False)
        # Foundry reactor core target
        self._obj("target", 0.0, 3.8, "purple", r=0.45, movable=False)

    # ------------------------------------------------------------------
    # Step Physics & Environmental Updates
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        from .actions import DRIVE, validate_action
        action = validate_action(action)

        # 1. Terrain zone physics (modulate thrust/turn and compute flow displacement)
        current_friction = 1.0
        in_friction_zone = False
        flow_dx, flow_dy = 0.0, 0.0
        ax, ay = self.a["x"], self.a["y"]

        for zone in self.terrain_zones:
            dist = math.hypot(ax - zone["x"], ay - zone["y"])
            if dist < zone["r"]:
                current_friction = zone.get("friction", 1.0)
                flow_dx += zone.get("vx", 0.0)
                flow_dy += zone.get("vy", 0.0)
                in_friction_zone = True

        # Adjust effective action under terrain friction
        eff_action = action.copy()
        eff_action[DRIVE] = np.clip(eff_action[DRIVE] / current_friction, -1, 1)

        # 2. Step base environment
        _, reward, done, info = super().step(eff_action)
        base_reward = reward
        ax, ay = self.a['x'], self.a['y']
        info["in_friction_zone"] = in_friction_zone
        info["terrain_friction"] = current_friction

        # 3. Vector Flow Displacement & Object Current Physics
        if flow_dx != 0.0 or flow_dy != 0.0:
            h = HALF - 0.35
            self.a["x"] = float(np.clip(self.a["x"] + flow_dx, -h, h))
            self.a["y"] = float(np.clip(self.a["y"] + flow_dy, -h, h))
            ax, ay = self.a["x"], self.a["y"]

        # Move movable objects caught in vector flow zones
        for obj in self.objects:
            if obj.get("movable") and not obj.get("held"):
                for zone in self.terrain_zones:
                    if math.hypot(obj["x"] - zone["x"], obj["y"] - zone["y"]) < zone["r"]:
                        obj["x"] = float(np.clip(obj["x"] + zone.get("vx", 0.0), -HALF + 0.35, HALF - 0.35))
                        obj["y"] = float(np.clip(obj["y"] + zone.get("vy", 0.0), -HALF + 0.35, HALF - 0.35))

        # 4. Topographic Elevation Gravity Gradient
        if self.elevation_hills:
            gx, gy = 0.0, 0.0
            for hill in self.elevation_hills:
                hx, hy = hill["x"], hill["y"]
                H, sigma = hill["height"], hill["sigma"]
                dx = ax - hx
                dy = ay - hy
                d2 = dx * dx + dy * dy
                val = H * math.exp(-d2 / (2.0 * sigma * sigma))
                gx += - (dx / (sigma * sigma)) * val
                gy += - (dy / (sigma * sigma)) * val
            # Gravitational drift acceleration applied to agent
            h = HALF - 0.35
            self.a["x"] = float(np.clip(self.a["x"] - 0.04 * gx, -h, h))
            self.a["y"] = float(np.clip(self.a["y"] - 0.04 * gy, -h, h))
            ax, ay = self.a["x"], self.a["y"]
            info["elevation_gradient"] = [round(float(gx), 4), round(float(gy), 4)]

        # 5. Dynamic Gates & Puzzle Locks
        for gate in self.dynamic_gates:
            if gate.get("locked"):
                # Check if puzzle button was pressed or box is placed on pad
                button_pressed = any(obj.get("kind") == "button" and obj.get("pressed") for obj in self.objects)
                box_on_pad = False
                box = next((o for o in self.objects if o.get("kind") == "box"), None)
                pad = next((o for o in self.objects if o.get("kind") == "pad"), None)
                if box and pad and math.hypot(box["x"] - pad["x"], box["y"] - pad["y"]) < 0.65:
                    box_on_pad = True
                if button_pressed or box_on_pad:
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

        # 6. Coordinated Predator Pack Dynamics
        min_hazard_dist = 10.0
        pack_alerted = any(p.get("alerted", False) for p in self.predators)

        # Check if any predator spots agent
        for pred in self.predators:
            p_dist = math.hypot(ax - pred["x"], ay - pred["y"])
            if p_dist < 3.5:
                pred["alerted"] = True
                pack_alerted = True

        for pred in self.predators:
            if pack_alerted:
                pred["alerted"] = True

            p_dx = ax - pred["x"]
            p_dy = ay - pred["y"]
            dist_to_agent = math.hypot(p_dx, p_dy)

            if pred["kind"] == "pursuer" or (pred.get("alerted") and pred["kind"] in ("pack_leader", "stalker")):
                # Direct chase steering
                desired_th = math.atan2(p_dy, p_dx)
                pred["th"] += 0.14 * math.sin(desired_th - pred["th"])
            elif pred.get("alerted") and pred["kind"] == "pack_hunter":
                # Flanking pursuit: steer toward predicted lead position
                lead_x = ax + math.cos(self.a["th"]) * self.a["v"] * 2.0
                lead_y = ay + math.sin(self.a["th"]) * self.a["v"] * 2.0
                desired_th = math.atan2(lead_y - pred["y"], lead_x - pred["x"])
                pred["th"] += 0.16 * math.sin(desired_th - pred["th"])
            else:
                # Patrol bounce
                pred["th"] += float(self.rng_complex.normal(0.0, 0.08))

            # Move predator
            pred_speed = pred["speed"] * (1.3 if pred.get("alerted") else 1.0)
            pred["x"] += math.cos(pred["th"]) * pred_speed
            pred["y"] += math.sin(pred["th"]) * pred_speed

            # Arena boundary bounce
            h = HALF - 0.5
            if pred["x"] < -h or pred["x"] > h:
                pred["th"] = math.pi - pred["th"]
                pred["x"] = float(np.clip(pred["x"], -h, h))
            if pred["y"] < -h or pred["y"] > h:
                pred["th"] = -pred["th"]
                pred["y"] = float(np.clip(pred["y"], -h, h))

            # Collision with agent
            dist_to_agent = math.hypot(ax - pred['x'], ay - pred['y'])
            min_hazard_dist = min(min_hazard_dist, dist_to_agent)
            if dist_to_agent < (pred["r"] + 0.35):
                reward -= 0.45
                info["predator_impact"] = True

        info["min_hazard_dist"] = min_hazard_dist

        # 7. Rotating Sentinel Searchlights & Stealth Cloak Mechanics
        alarm_tripped = False
        in_shadow = any(
            tz.get("kind") == "shadow" and math.hypot(ax - tz["x"], ay - tz["y"]) < tz["r"]
            for tz in self.terrain_zones
        )
        info["in_shadow"] = in_shadow

        for sentinel in self.sentinels:
            sentinel["angle"] = (sentinel["angle"] + sentinel.get("angular_vel", 0.04)) % (math.pi * 2)
            if not in_shadow:
                s_dx = ax - sentinel["x"]
                s_dy = ay - sentinel["y"]
                s_dist = math.hypot(s_dx, s_dy)
                if s_dist < sentinel["range"]:
                    rel_angle = (math.atan2(s_dy, s_dx) - sentinel["angle"] + math.pi) % (math.pi * 2) - math.pi
                    if abs(rel_angle) < sentinel["cone_half_angle"]:
                        sentinel["alerted"] = True
                        alarm_tripped = True
                        reward -= 0.30
                        sentinel["color"] = "red"

        # 8. Acoustic Sound Tripwires
        agent_speed = abs(self.a["v"])
        for trip in self.sound_tripwires:
            # Check segment proximity
            t_x1, t_y1, t_x2, t_y2 = trip["x1"], trip["y1"], trip["x2"], trip["y2"]
            seg_dist = abs((t_y2 - t_y1) * ax - (t_x2 - t_x1) * ay + t_x2 * t_y1 - t_y2 * t_x1) / max(1e-5, math.hypot(t_y2 - t_y1, t_x2 - t_x1))
            if seg_dist < 0.4 and min(t_x1, t_x2) - 0.2 <= ax <= max(t_x1, t_x2) + 0.2 and min(t_y1, t_y2) - 0.2 <= ay <= max(t_y1, t_y2) + 0.2:
                if agent_speed > trip.get("max_safe_speed", 0.30):
                    alarm_tripped = True
                    reward -= 0.25

        self.last_alarm_tripped = alarm_tripped
        info["alarm_tripped"] = alarm_tripped

        # 9. Ecological Food Respawn
        self.food_respawn_timer += 1
        if self.task in ("ecosystem", "apex_megasite") and self.food_respawn_timer % 40 == 0:
            current_food = sum(1 for o in self.objects if o.get("kind") == "food" and o.get("visible", True))
            max_food = 6 if self.task == "ecosystem" else 4
            if current_food < max_food:
                spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
                self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

        # 10. Ambient Illumination Cycle
        self.day_night_clock += 0.015
        self.ambient_light = float(0.65 + 0.35 * math.sin(self.day_night_clock))
        info["ambient_light"] = self.ambient_light

        # 11. Complex Task Completion and Terminal Goal Rewards
        target_obj = next((o for o in self.objects if o.get("kind") == "target"), None)
        if target_obj is not None:
            dist_to_target = math.hypot(ax - target_obj["x"], ay - target_obj["y"])
            if dist_to_target < (target_obj.get("r", 0.4) + 0.25):
                if self.task == "puzzle_chamber":
                    btn = next((o for o in self.objects if o.get("kind") == "button"), None)
                    if btn and btn.get("pressed"):
                        done = True
                        reward += 6.0
                        info["outcome"] = "success"
                elif self.task in ("dynamic_arena", "apex_megasite", "stealth_labyrinth", "kinetic_foundry"):
                    done = True
                    reward += 8.0
                    info["outcome"] = "success"

        # Update episode accounting
        self.ep_return += (reward - base_reward)
        self.last_reward = reward
        if done:
            info['ret'] = self.ep_return
        return self.observe(), reward, done, info

    def _progress(self) -> float:
        """Task potential progress function across both base and frontier tasks."""
        t = self.task
        ax, ay = self.a["x"], self.a["y"]

        if t in ("ecosystem", "predator_evasion"):
            visible_food = [o for o in self.objects if o.get("kind") == "food" and o.get("visible", True)]
            if visible_food:
                min_f = min(math.hypot(ax - f["x"], ay - f["y"]) for f in visible_food)
                pot = -min_f
            else:
                pot = 0.0
            if t == "predator_evasion" and self.predators:
                min_p = min(math.hypot(ax - p["x"], ay - p["y"]) for p in self.predators)
                pot += 0.25 * min_p
            return pot

        if t == "dynamic_arena":
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            return -math.hypot(ax - target["x"], ay - target["y"]) if target else 0.0

        if t == "puzzle_chamber":
            btn = next((o for o in self.objects if o.get("kind") == "button"), None)
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            if btn and not btn.get("pressed"):
                return -math.hypot(ax - btn["x"], ay - btn["y"])
            if target:
                return 2.0 - math.hypot(ax - target["x"], ay - target["y"])
            return 0.0

        if t == "apex_megasite":
            btn = next((o for o in self.objects if o.get("kind") == "button"), None)
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            if btn and not btn.get("pressed"):
                return -math.hypot(ax - btn["x"], ay - btn["y"])
            if target:
                return 3.0 - math.hypot(ax - target["x"], ay - target["y"])
            return 0.0

        if t == "stealth_labyrinth":
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            alarm_pen = -1.0 if getattr(self, "last_alarm_tripped", False) else 0.0
            return (-math.hypot(ax - target["x"], ay - target["y"]) + alarm_pen) if target else 0.0

        if t == "kinetic_foundry":
            box = next((o for o in self.objects if o.get("kind") == "box"), None)
            pad = next((o for o in self.objects if o.get("kind") == "pad"), None)
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            if box and pad:
                box_pad_dist = math.hypot(box["x"] - pad["x"], box["y"] - pad["y"])
                if box_pad_dist > 0.65:
                    return -math.hypot(ax - box["x"], ay - box["y"]) - 1.5 * box_pad_dist
            if target:
                return 3.0 - math.hypot(ax - target["x"], ay - target["y"])
            return 0.0

        # Delegate base tasks to BaseWorld._progress
        return super()._progress()

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
        """Extends snapshot with predators, terrain zones, vector fields, sentinels, and elevation hills."""
        snap = super().snapshot()
        snap["predators"] = [
            {
                "id": p["id"],
                "x": round(float(p["x"]), 3),
                "y": round(float(p["y"]), 3),
                "th": round(float(p["th"]), 3),
                "r": p["r"],
                "kind": p["kind"],
                "color": p["color"],
                "alerted": p.get("alerted", False),
            }
            for p in self.predators
        ]
        snap["terrain_zones"] = self.terrain_zones
        snap["dynamic_gates"] = self.dynamic_gates
        snap["elevation_hills"] = self.elevation_hills
        snap["sentinels"] = self.sentinels
        snap["sound_tripwires"] = self.sound_tripwires
        snap["ambient_light"] = round(self.ambient_light, 3)
        snap["alarm_tripped"] = self.last_alarm_tripped
        return snap
