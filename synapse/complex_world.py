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
    "quantum_bifurcation",
    "cybernetic_sanctuary",
    "continental_biosphere",
    "chrono_tidepool_citadel",
    "hyper_boreal_tundra",
    "subterranean_abyss",
    "vortex_maelstrom",
]

ALL_TASKS = BASE_TASKS + COMPLEX_TASKS


class ComplexWorld(BaseWorld):
    """Frontier virtual 3D environment with dynamic multi-agent ecology, vector fields, and shifting terrain physics."""

    def __init__(self, seed: int = 0, curriculum: str = "auto"):
        self.complex_seed = seed
        self.rng_complex = np.random.default_rng(seed)
        self.arena_half: float = float(HALF)
        self.predators: List[Dict[str, Any]] = []
        self.helper_drones: List[Dict[str, Any]] = []
        self.terrain_zones: List[Dict[str, Any]] = []
        self.dynamic_gates: List[Dict[str, Any]] = []
        self.elevation_hills: List[Dict[str, Any]] = []
        self.thermal_geysers: List[Dict[str, Any]] = []
        self.vortices: List[Dict[str, Any]] = []
        self.bioluminescent_flora: List[Dict[str, Any]] = []
        self.sentinels: List[Dict[str, Any]] = []
        self.sound_tripwires: List[Dict[str, Any]] = []
        self.switches: List[Dict[str, Any]] = []
        self.wormholes: List[Dict[str, Any]] = []
        self.weather_fronts: List[Dict[str, Any]] = []
        self.wormhole_cooldown: int = 0
        self.last_alarm_tripped: bool = False
        self.day_night_clock = 0.0
        self.celestial_angle = 0.0
        self.ambient_light = 1.0
        self.food_respawn_timer = 0
        self.base_walls: List[Tuple[float, float, float, float]] = []
        super().__init__(seed=seed, curriculum=curriculum)

    def _border(self, h: Optional[float] = None):
        bound = float(h if h is not None else getattr(self, "arena_half", HALF))
        return [(-bound, -bound, bound, -bound), (bound, -bound, bound, bound), (bound, bound, -bound, bound), (-bound, bound, -bound, -bound)]

    def reset(self, task: Optional[str] = None):
        """Initializes both standard baseline and advanced complex tasks."""
        self.arena_half = float(HALF)
        self.predators.clear()
        self.helper_drones.clear()
        self.terrain_zones.clear()
        self.dynamic_gates.clear()
        self.elevation_hills.clear()
        self.thermal_geysers.clear()
        self.vortices.clear()
        self.bioluminescent_flora.clear()
        self.sentinels.clear()
        self.sound_tripwires.clear()
        self.switches.clear()
        self.wormholes.clear()
        self.weather_fronts.clear()
        self.wormhole_cooldown = 0
        self.last_alarm_tripped = False
        self.celestial_angle = 0.0

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
        elif self.task == "quantum_bifurcation":
            self._setup_quantum_bifurcation()
        elif self.task == "cybernetic_sanctuary":
            self._setup_cybernetic_sanctuary()
        elif self.task == "continental_biosphere":
            self._setup_continental_biosphere()
        elif self.task == "chrono_tidepool_citadel":
            self._setup_chrono_tidepool_citadel()
        elif self.task == "hyper_boreal_tundra":
            self._setup_hyper_boreal_tundra()
        elif self.task == "subterranean_abyss":
            self._setup_subterranean_abyss()
        elif self.task == "vortex_maelstrom":
            self._setup_vortex_maelstrom()

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

    def _setup_quantum_bifurcation(self):
        """Quantum bifurcation: dual phase-shifting bridge gates over a cross-current, entangled cue switch, and decoy vault."""
        self.objects.clear()
        self.walls = self._border()

        # Bifurcating corridor walls
        self.walls.extend([
            (-1.0, -5.0, -1.0, -1.6),
            (-1.0, 1.6, -1.0, 5.0),
            (1.6, -1.2, 5.0, -1.2),
            (1.6, 1.2, 5.0, 1.2),
        ])

        # River current + ice slip zone + shadow cloak pocket
        self.terrain_zones = [
            {"kind": "river", "x": 0.2, "y": 0.0, "r": 1.9, "friction": 1.05, "vx": 0.0, "vy": 0.14, "color": "blue"},
            {"kind": "ice", "x": -2.8, "y": 0.0, "r": 1.4, "friction": 0.28, "color": "cyan"},
            {"kind": "shadow", "x": 2.4, "y": 0.0, "r": 1.2, "friction": 1.0, "color": "purple"},
        ]

        # Elevation basin & plateau
        self.elevation_hills = [
            {"x": -2.5, "y": -2.5, "height": 1.4, "sigma": 1.3},
            {"x": 2.5, "y": 2.8, "height": -1.3, "sigma": 1.3},
        ]

        # Dual phase-shifting gates (one locked by entanglement button, one oscillating)
        self.dynamic_gates = [
            {"id": "phase_bridge_north", "x1": -1.0, "y1": 0.0, "x2": -1.0, "y2": 1.6, "open": True, "timer": 10},
            {"id": "phase_bridge_south", "x1": -1.0, "y1": -1.6, "x2": -1.0, "y2": 0.0, "open": False, "locked": True},
        ]

        # Rotating sentinel guarding the bifurcation nexus
        self.sentinels = [
            {"id": 401, "x": 1.4, "y": 0.0, "angle": 0.0, "angular_vel": 0.05, "range": 2.6, "cone_half_angle": 0.40, "alerted": False, "color": "orange"}
        ]

        # Entanglement calibration switch & quantum target core
        self._obj("button", -3.2, -2.5, "yellow", r=0.35, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 3.8, 0.0, "cyan", r=0.44, movable=False)
        self._obj("food", -3.0, 2.5, "gold", r=0.28, movable=False)
        self._obj("food", 0.3, -2.8, "green", r=0.25, movable=False)

    def _setup_cybernetic_sanctuary(self):
        """Cybernetic sanctuary: autonomous cooperative helper drone with protective shield field, bio-reactor valves, and pack hunters."""
        self.objects.clear()
        self.walls = self._border()

        # Sanctuary inner ring walls
        self.walls.extend([
            (1.8, -5.0, 1.8, -1.1),
            (1.8, 1.1, 1.8, 5.0),
        ])
        self.dynamic_gates = [
            {"id": "sanctuary_bulkhead", "x1": 1.8, "y1": -1.1, "x2": 1.8, "y2": 1.1, "open": False, "locked": True}
        ]

        self.terrain_zones = [
            {"kind": "conveyor", "x": -1.8, "y": 1.8, "r": 1.5, "friction": 0.9, "vx": 0.14, "vy": -0.06, "color": "orange"},
            {"kind": "shadow", "x": -0.5, "y": -2.2, "r": 1.4, "friction": 1.0, "color": "purple"},
        ]

        # Cooperative Helper Drone (orbits and projects a stabilizing shield aura)
        self.helper_drones = [
            {
                "id": 501,
                "x": 1.0,
                "y": 0.0,
                "th": 0.0,
                "orbit_angle": 0.0,
                "orbit_center": [-0.6, 0.0],
                "orbit_radius": 1.6,
                "shield_radius": 1.6,
                "r": 0.32,
                "kind": "guardian_drone",
                "color": "cyan",
                "active": True,
            }
        ]

        # Perimeter stalker drone
        self.predators.append({
            "id": 205,
            "x": -3.2,
            "y": -3.0,
            "th": 0.5,
            "speed": 0.12,
            "kind": "pack_leader",
            "r": 0.36,
            "color": "red",
            "alerted": False,
        })

        # Bio-reactor valve button & sanctuary core target
        self._obj("button", -2.8, 2.4, "yellow", r=0.36, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 3.6, 0.0, "gold", r=0.45, movable=False)
        for _ in range(3):
            spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
            self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

    def _setup_continental_biosphere(self):
        """Mega-scale 17x17 (arena_half=8.5) multi-biome continental ecosystem with wormhole relays, weather storms, 3 predator packs, and 2 guardian drones."""
        self.arena_half = 8.5
        self.max_t = 700
        self.objects.clear()
        self.walls = self._border(self.arena_half)
        self.a.update(x=-6.5, y=-6.0, th=0.4)

        # Continental mountain ridge barriers with central canyon gates
        self.walls.extend([
            (-1.5, -8.5, -1.5, -2.5),
            (-1.5, 2.5, -1.5, 8.5),
            (3.5, -8.5, 3.5, -1.8),
            (3.5, 1.8, 3.5, 8.5),
        ])
        self.dynamic_gates = [
            {"id": "canyon_west_gate", "x1": -1.5, "y1": -2.5, "x2": -1.5, "y2": 2.5, "open": True, "timer": 10},
            {"id": "citadel_east_gate", "x1": 3.5, "y1": -1.8, "x2": 3.5, "y2": 1.8, "open": False, "locked": True},
        ]

        # 5 Continental Biomes
        self.terrain_zones = [
            {"kind": "ice", "x": -5.2, "y": 4.8, "r": 2.6, "friction": 0.25, "color": "cyan"},
            {"kind": "sand", "x": -4.8, "y": -4.5, "r": 2.5, "friction": 2.35, "color": "yellow"},
            {"kind": "river", "x": 1.0, "y": 0.0, "r": 2.8, "friction": 1.10, "vx": 0.0, "vy": -0.16, "color": "blue"},
            {"kind": "mud", "x": 1.2, "y": -5.5, "r": 2.2, "friction": 1.85, "color": "grey"},
            {"kind": "shadow", "x": 1.0, "y": 5.2, "r": 2.2, "friction": 1.0, "color": "purple"},
        ]

        # 3 Topographic Mountain / Basin Elevation Hills
        self.elevation_hills = [
            {"x": -5.0, "y": 0.5, "height": 2.2, "sigma": 2.0},
            {"x": 1.0, "y": 3.5, "height": -1.8, "sigma": 1.8},
            {"x": 6.0, "y": -4.0, "height": 2.0, "sigma": 1.9},
        ]

        # Paired Quantum Wormhole Relay (transports agent from (-6.0, 6.0) <-> (2.2, -6.0))
        self.wormholes = [
            {"id": "wormhole_alpha", "label": "ALPHA RELAY", "x": -6.0, "y": 6.0, "x1": -6.0, "y1": 6.0, "x2": 2.2, "y2": -6.0, "r": 0.75, "target_x": 2.2, "target_y": -6.0, "color": "#00e5ff"},
            {"id": "wormhole_omega", "label": "OMEGA RELAY", "x": 2.2, "y": -6.0, "x1": 2.2, "y1": -6.0, "x2": -6.0, "y2": 6.0, "r": 0.75, "target_x": -6.0, "target_y": 6.0, "color": "#c084fc"},
        ]

        # Dynamic Electromagnetic Weather Storm Fronts
        self.weather_fronts = [
            {"id": "ion_storm_1", "kind": "ion_storm", "x": -2.0, "y": 0.0, "r": 2.4, "vx": 0.05, "vy": 0.04, "wind_x": -0.05, "wind_y": 0.04, "sensor_noise": 0.03},
            {"id": "monsoon_squall_2", "kind": "monsoon_squall", "x": 4.2, "y": -2.5, "r": 2.1, "vx": -0.04, "vy": 0.05, "wind_x": 0.04, "wind_y": -0.03, "sensor_noise": 0.025},
        ]

        # 2 Cooperative Guardian Drones
        self.helper_drones = [
            {
                "id": 601, "x": -1.4, "y": 0.0, "th": 0.0,
                "orbit_angle": 0.0, "orbit_center": [-3.4, 0.0],
                "orbit_radius": 2.0, "shield_radius": 1.9, "r": 0.34,
                "kind": "guardian_drone", "color": "cyan", "active": True,
            },
            {
                "id": 602, "x": 3.0, "y": 0.0, "th": math.pi,
                "orbit_angle": math.pi, "orbit_center": [1.0, 0.0],
                "orbit_radius": 2.0, "shield_radius": 1.9, "r": 0.34,
                "kind": "guardian_drone", "color": "cyan", "active": True,
            },
        ]

        # 3 Coordinated Predator Drones
        self.predators.extend([
            {"id": 301, "x": -4.5, "y": 3.5, "th": 0.0, "speed": 0.14, "kind": "pack_leader", "r": 0.38, "color": "red", "alerted": False},
            {"id": 302, "x": 0.8, "y": -3.8, "th": 1.2, "speed": 0.15, "kind": "pack_hunter", "r": 0.38, "color": "purple", "alerted": False},
            {"id": 303, "x": 5.5, "y": 4.2, "th": -1.0, "speed": 0.13, "kind": "stalker", "r": 0.38, "color": "red", "alerted": False},
        ])

        self._obj("button", 1.2, 5.2, "yellow", r=0.40, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 6.8, 0.0, "gold", r=0.50, movable=False)
        for fx, fy in [(-6.2, -1.5), (-3.0, 5.5), (0.0, -6.2), (2.2, 2.8), (5.8, -5.5), (6.0, 5.0)]:
            self._obj("food", fx, fy, "green", r=0.28, movable=False)

    def _setup_chrono_tidepool_citadel(self):
        """Poseidon Tidepool v2 inspired 15x15 (arena_half=7.5) observation-aliased tidal arena with periodic surge waves and cue beacons."""
        self.arena_half = 7.5
        self.max_t = 650
        self.objects.clear()
        self.walls = self._border(self.arena_half)
        self.a.update(x=-5.8, y=0.0, th=0.0)
        self.cue = np.array([1.0, 0.0, 0.5, -0.5], dtype=np.float32)

        # Citadel inner sanctum walls
        self.walls.extend([
            (2.8, -7.5, 2.8, -1.4),
            (2.8, 1.4, 2.8, 7.5),
        ])
        self.dynamic_gates = [
            {"id": "tidepool_sluice", "x1": 2.8, "y1": -1.4, "x2": 2.8, "y2": 1.4, "open": False, "locked": True},
        ]

        # Tidal surge vector channels & stealth kelp shadow zones
        self.terrain_zones = [
            {"kind": "river", "x": -1.5, "y": 2.8, "r": 2.4, "friction": 1.15, "vx": -0.14, "vy": 0.08, "color": "blue"},
            {"kind": "river", "x": -1.5, "y": -2.8, "r": 2.4, "friction": 1.15, "vx": 0.14, "vy": -0.08, "color": "cyan"},
            {"kind": "shadow", "x": -4.0, "y": 0.0, "r": 1.8, "friction": 1.0, "color": "purple"},
            {"kind": "sand", "x": 1.0, "y": 0.0, "r": 2.0, "friction": 2.10, "color": "yellow"},
        ]

        self.wormholes = [
            {"id": "tide_relay_north", "label": "TIDE RELAY N", "x": -5.0, "y": 5.2, "x1": -5.0, "y1": 5.2, "x2": 1.2, "y2": -4.8, "r": 0.70, "target_x": 1.2, "target_y": -4.8, "color": "#00e5ff"},
            {"id": "tide_relay_south", "label": "TIDE RELAY S", "x": 1.2, "y": -4.8, "x1": 1.2, "y1": -4.8, "x2": -5.0, "y2": 5.2, "r": 0.70, "target_x": -5.0, "target_y": 5.2, "color": "#c084fc"},
        ]

        self.weather_fronts = [
            {"id": "tidal_squall", "kind": "tidal_squall", "x": 0.0, "y": 0.0, "r": 2.6, "vx": 0.04, "vy": -0.05, "wind_x": -0.06, "wind_y": 0.03, "sensor_noise": 0.04}
        ]

        self.sentinels = [
            {"id": 410, "x": 0.5, "y": 3.5, "angle": 0.0, "angular_vel": 0.045, "range": 3.8, "cone_half_angle": 0.38, "color": "yellow", "alerted": False},
            {"id": 411, "x": 0.5, "y": -3.5, "angle": math.pi, "angular_vel": -0.045, "range": 3.8, "cone_half_angle": 0.38, "color": "yellow", "alerted": False},
        ]

        self.helper_drones = [
            {
                "id": 610, "x": 0.3, "y": 0.0, "th": 0.0,
                "orbit_angle": 0.0, "orbit_center": [-1.5, 0.0],
                "orbit_radius": 1.8, "shield_radius": 1.75, "r": 0.32,
                "kind": "guardian_drone", "color": "cyan", "active": True,
            }
        ]

        self.predators.append({
            "id": 310, "x": 4.8, "y": 3.2, "th": -1.5, "speed": 0.14, "kind": "pursuer", "r": 0.38, "color": "red", "alerted": False
        })

        self._obj("button", -1.2, -4.8, "yellow", r=0.38, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 5.8, 0.0, "gold", r=0.48, movable=False)
        for fx, fy in [(-4.5, -3.5), (-2.0, 4.8), (1.5, 4.2), (5.0, -3.5)]:
            self._obj("food", fx, fy, "green", r=0.26, movable=False)

    def _setup_hyper_boreal_tundra(self):
        """Hyper-boreal tundra: 18x18 mega-biome with blizzard fronts, thermal geysers, low-friction ice sheets, wolf pack, and aerial frost raptor."""
        self.arena_half = 9.0
        self.max_t = 750
        self.objects.clear()
        self.walls = self._border(self.arena_half)
        self.a.update(x=-7.0, y=-7.0, th=0.5)

        # Mountain crevasse ridges
        self.walls.extend([
            (-2.5, -9.0, -2.5, -3.0),
            (-2.5, 3.0, -2.5, 9.0),
            (2.5, -9.0, 2.5, -2.0),
            (2.5, 2.0, 2.5, 9.0),
        ])
        self.dynamic_gates = [
            {"id": "tundra_crevasse_gate", "x1": -2.5, "y1": -3.0, "x2": -2.5, "y2": 3.0, "open": True, "timer": 15},
            {"id": "citadel_frost_gate", "x1": 2.5, "y1": -2.0, "x2": 2.5, "y2": 2.0, "open": False, "locked": True},
        ]

        # Glacial ice sheet & tundra snow
        self.terrain_zones = [
            {"kind": "ice", "x": -4.5, "y": 0.0, "r": 3.2, "friction": 0.15, "color": "cyan"},
            {"kind": "sand", "x": 0.0, "y": -5.0, "r": 2.4, "friction": 2.2, "color": "yellow"},
            {"kind": "river", "x": 0.0, "y": 4.5, "r": 2.5, "friction": 1.1, "vx": 0.15, "vy": 0.0, "color": "blue"},
            {"kind": "shadow", "x": 5.0, "y": 5.0, "r": 2.2, "friction": 1.0, "color": "purple"},
        ]

        # Topographic peak and crevasse
        self.elevation_hills = [
            {"x": -5.0, "y": 5.0, "height": 2.5, "sigma": 2.0},
            {"x": 5.0, "y": -5.0, "height": -2.2, "sigma": 1.8},
        ]

        # 2 Geothermal steam vents
        self.thermal_geysers = [
            {"id": "geyser_north", "x": 0.0, "y": 1.5, "r": 1.8, "thrust": 0.16, "temp": "hot"},
            {"id": "geyser_south", "x": -4.0, "y": -4.0, "r": 1.6, "thrust": 0.14, "temp": "hot"},
        ]

        # Blizzard front
        self.weather_fronts = [
            {"id": "boreal_blizzard", "kind": "blizzard", "x": -2.0, "y": 0.0, "r": 3.5, "vx": 0.05, "vy": -0.04, "wind_x": -0.07, "wind_y": 0.04, "sensor_noise": 0.05}
        ]

        # Wolf pack (leader + hunter + ground stalker) + Aerial raptor
        self.predators.extend([
            {"id": 701, "x": 4.5, "y": -4.5, "th": 0.0, "speed": 0.14, "kind": "pack_leader", "r": 0.38, "color": "red", "state": "patrol", "stamina": 100},
            {"id": 702, "x": 5.5, "y": -3.5, "th": 1.0, "speed": 0.13, "kind": "pack_hunter", "r": 0.36, "color": "darkred", "state": "patrol", "stamina": 100},
            {"id": 703, "x": -4.0, "y": 4.0, "th": -1.0, "speed": 0.12, "kind": "ground_stalker", "r": 0.35, "color": "purple", "state": "stalk", "stamina": 80},
            {"id": 704, "x": 0.0, "y": 0.0, "th": 0.0, "speed": 0.18, "kind": "aerial_raptor", "r": 0.42, "color": "#ec4899", "state": "patrol", "altitude": 1.5, "dive_timer": 0},
        ])

        # 2 Guardian helper drones
        self.helper_drones = [
            {
                "id": 710, "x": -2.0, "y": -2.0, "th": 0.0,
                "orbit_angle": 0.0, "orbit_center": [-4.0, -4.0],
                "orbit_radius": 1.8, "shield_radius": 1.8, "r": 0.32,
                "kind": "guardian_drone", "color": "cyan", "active": True,
            }
        ]

        # Geothermal button & Arctic target core
        self._obj("button", -5.5, 4.5, "yellow", r=0.40, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 6.8, 0.0, "gold", r=0.50, movable=False)
        for fx, fy in [(-6.5, -4.0), (-3.5, 6.0), (1.5, -6.5), (6.0, 4.5)]:
            self._obj("food", fx, fy, "green", r=0.28, movable=False)

    def _setup_subterranean_abyss(self):
        """Subterranean abyss: dark cavern labyrinth, bioluminescent flora clusters, acoustic tripwires, and blind abyssal leviathan."""
        self.arena_half = 8.0
        self.max_t = 700
        self.objects.clear()
        self.walls = self._border(self.arena_half)
        self.a.update(x=-6.0, y=0.0, th=0.0)

        # Stalactite pillar corridor walls
        self.walls.extend([
            (-3.0, -8.0, -3.0, -1.8),
            (-3.0, 1.8, -3.0, 8.0),
            (3.0, -8.0, 3.0, -1.8),
            (3.0, 1.8, 3.0, 8.0),
            (-1.5, 0.0, 1.5, 0.0),
        ])
        self.dynamic_gates = [
            {"id": "abyss_chasm_gate", "x1": -3.0, "y1": -1.8, "x2": -3.0, "y2": 1.8, "open": True, "timer": 20},
            {"id": "vault_gate", "x1": 3.0, "y1": -1.8, "x2": 3.0, "y2": 1.8, "open": False, "locked": True},
        ]

        # Low ambient light with shadow pockets
        self.terrain_zones = [
            {"kind": "shadow", "x": -4.5, "y": 4.5, "r": 2.2, "friction": 1.0, "color": "purple"},
            {"kind": "shadow", "x": 4.5, "y": -4.5, "r": 2.2, "friction": 1.0, "color": "purple"},
            {"kind": "mud", "x": 0.0, "y": -4.0, "r": 2.0, "friction": 1.9, "color": "grey"},
        ]

        # 4 Bioluminescent flora clusters
        self.bioluminescent_flora = [
            {"id": "flora_1", "x": -5.0, "y": -5.0, "r": 1.6, "color": "#39ff14", "pulse_rate": 0.04},
            {"id": "flora_2", "x": 5.0, "y": 5.0, "r": 1.6, "color": "#00f0ff", "pulse_rate": 0.05},
            {"id": "flora_3", "x": -4.0, "y": 4.0, "r": 1.5, "color": "#bf00ff", "pulse_rate": 0.035},
            {"id": "flora_4", "x": 4.0, "y": -4.0, "r": 1.5, "color": "#ff007f", "pulse_rate": 0.045},
        ]

        # Acoustic sound tripwires guarding the corridors
        self.sound_tripwires = [
            {"id": "abyss_trip_1", "x1": -3.0, "y1": -1.8, "x2": -3.0, "y2": 1.8, "max_safe_speed": 0.22},
            {"id": "abyss_trip_2", "x1": 3.0, "y1": -1.8, "x2": 3.0, "y2": 1.8, "max_safe_speed": 0.22},
        ]

        # Blind Abyssal Leviathan (hunts by acoustic vibration)
        self.predators.append({
            "id": 801, "x": 0.0, "y": 0.0, "th": 0.0, "speed": 0.16,
            "kind": "abyssal_leviathan", "r": 0.48, "color": "#ff0055",
            "state": "dormant", "alerted": False,
        })

        self._obj("button", -5.0, 5.0, "yellow", r=0.40, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 6.0, 0.0, "gold", r=0.48, movable=False)
        for fx, fy in [(-5.0, -5.0), (5.0, 5.0), (-4.0, 4.0), (4.0, -4.0)]:
            self._obj("food", fx, fy, "green", r=0.28, movable=False)

    def _setup_vortex_maelstrom(self):
        """Vortex maelstrom: central whirlpool with Coriolis rotational forces and radial suction, 4 sanctuary atolls, and paired wormholes."""
        self.arena_half = 8.0
        self.max_t = 650
        self.objects.clear()
        self.walls = self._border(self.arena_half)
        self.a.update(x=-5.5, y=-5.5, th=0.8)

        # Central whirlpool vortex
        self.vortices = [
            {"id": "central_maelstrom", "x": 0.0, "y": 0.0, "r": 4.8, "pull": 0.11, "spin": 0.16, "depth": 2.5}
        ]

        # 4 Atoll elevation islands
        self.elevation_hills = [
            {"x": -5.0, "y": -5.0, "height": 1.8, "sigma": 1.6},
            {"x": 5.0, "y": 5.0, "height": 1.8, "sigma": 1.6},
            {"x": -5.0, "y": 5.0, "height": 1.6, "sigma": 1.5},
            {"x": 5.0, "y": -5.0, "height": 1.6, "sigma": 1.5},
        ]

        # Tidal current rings
        self.terrain_zones = [
            {"kind": "river", "x": 0.0, "y": 3.0, "r": 2.2, "friction": 1.1, "vx": 0.16, "vy": 0.0, "color": "blue"},
            {"kind": "river", "x": 0.0, "y": -3.0, "r": 2.2, "friction": 1.1, "vx": -0.16, "vy": 0.0, "color": "blue"},
            {"kind": "ice", "x": 4.5, "y": 0.0, "r": 1.8, "friction": 0.28, "color": "cyan"},
        ]

        # Paired tidal wormhole relays
        self.wormholes = [
            {"id": "maelstrom_portal_1", "label": "VORTEX RELAY A", "x": -5.0, "y": 5.0, "x1": -5.0, "y1": 5.0, "x2": 5.0, "y2": -5.0, "r": 0.75, "target_x": 5.0, "target_y": -5.0, "color": "#00e5ff"},
            {"id": "maelstrom_portal_2", "label": "VORTEX RELAY B", "x": 5.0, "y": -5.0, "x1": 5.0, "y1": -5.0, "x2": -5.0, "y2": 5.0, "r": 0.75, "target_x": -5.0, "target_y": 5.0, "color": "#c084fc"},
        ]

        self.predators.append({
            "id": 901, "x": 2.0, "y": 2.0, "th": 0.0, "speed": 0.14,
            "kind": "pursuer", "r": 0.38, "color": "red", "alerted": False,
        })

        self._obj("button", -5.0, 5.0, "yellow", r=0.38, shape="button", movable=False, solid=False, pressed=False)
        self._obj("target", 5.0, 5.0, "gold", r=0.48, movable=False)
        for fx, fy in [(-5.0, -3.0), (3.0, 5.0), (5.0, -3.0)]:
            self._obj("food", fx, fy, "green", r=0.26, movable=False)

    def get_elevation(self, x: float, y: float) -> float:
        """Computes continuous elevation Z(x, y) across all topographic hills and basins."""
        z = 0.0
        for hill in self.elevation_hills:
            hx, hy = hill["x"], hill["y"]
            H, sigma = hill["height"], hill.get("sigma", 1.5)
            dx = x - hx
            dy = y - hy
            d2 = dx * dx + dy * dy
            z += H * math.exp(-d2 / (2.0 * sigma * sigma))
        return float(z)

    def get_elevation_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Computes slope gradient vector grad Z = [dZ/dx, dZ/dy]."""
        gx, gy = 0.0, 0.0
        for hill in self.elevation_hills:
            hx, hy = hill["x"], hill["y"]
            H, sigma = hill["height"], hill.get("sigma", 1.5)
            dx = x - hx
            dy = y - hy
            d2 = dx * dx + dy * dy
            val = H * math.exp(-d2 / (2.0 * sigma * sigma))
            gx += -(dx / (sigma * sigma)) * val
            gy += -(dy / (sigma * sigma)) * val
        return float(gx), float(gy)

    def get_wind(self, x: float, y: float) -> Tuple[float, float]:
        """Computes convective wind and vortex swirl vector field W(x, y, t)."""
        wx, wy = 0.0, 0.0
        t_phase = self.day_night_clock * 2.5
        wx += 0.025 * math.sin(t_phase + y * 0.25)
        wy += 0.020 * math.cos(t_phase + x * 0.25)

        for storm in getattr(self, "weather_fronts", []):
            dist = math.hypot(x - storm["x"], y - storm["y"])
            sr = storm.get("r", 2.4)
            if dist < sr:
                factor = 1.0 - (dist / sr)
                wx += storm.get("wind_x", 0.0) * factor
                wy += storm.get("wind_y", 0.0) * factor

        for vortex in getattr(self, "vortices", []):
            dx = x - vortex["x"]
            dy = y - vortex["y"]
            dist = math.hypot(dx, dy)
            vr = vortex.get("r", 4.5)
            if 0.15 < dist < vr:
                intensity = 1.0 - dist / vr
                pull = vortex.get("pull", 0.08) * intensity
                spin = vortex.get("spin", 0.14) * intensity
                rx, ry = dx / dist, dy / dist
                tx, ty = -ry, rx
                wx += -pull * rx + spin * tx
                wy += -pull * ry + spin * ty

        for vent in getattr(self, "thermal_geysers", []):
            dx = x - vent["x"]
            dy = y - vent["y"]
            dist = math.hypot(dx, dy)
            vr = vent.get("r", 1.8)
            if 0.05 < dist < vr:
                thrust = vent.get("thrust", 0.12) * (1.0 - dist / vr)
                wx += thrust * (dx / dist)
                wy += thrust * (dy / dist)

        return float(wx), float(wy)

    def _get_elevation_grid(self, resolution: int = 11) -> List[List[float]]:
        half = float(getattr(self, "arena_half", HALF))
        coords = np.linspace(-half, half, resolution)
        grid = []
        for y in coords:
            row = []
            for x in coords:
                row.append(round(self.get_elevation(float(x), float(y)), 3))
            grid.append(row)
        return grid

    def _sample_wind_streamlines(self, count: int = 16) -> List[Dict[str, Any]]:
        half = float(getattr(self, "arena_half", HALF))
        streamlines = []
        for i in range(count):
            ang = (i / count) * math.pi * 2
            r = (0.2 + 0.65 * ((i * 5) % count) / count) * half
            sx = math.cos(ang) * r
            sy = math.sin(ang) * r
            wx, wy = self.get_wind(sx, sy)
            streamlines.append({
                "x": round(float(sx), 2),
                "y": round(float(sy), 2),
                "vx": round(float(wx), 3),
                "vy": round(float(wy), 3),
                "speed": round(float(math.hypot(wx, wy)), 3),
            })
        return streamlines

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

        # Adjust effective action under terrain friction and uphill slope resistance
        eff_action = action.copy()
        if self.elevation_hills:
            gx, gy = self.get_elevation_gradient(ax, ay)
            av = float(self.a.get("v", 0.0))
            ath = float(self.a.get("th", 0.0))
            vx_agent = av * math.cos(ath)
            vy_agent = av * math.sin(ath)
            uphill = gx * vx_agent + gy * vy_agent
            if uphill > 0:
                current_friction += min(2.0, uphill * 1.5)
        eff_action[DRIVE] = np.clip(eff_action[DRIVE] / current_friction, -1, 1)

        # 2. Step base environment
        _, reward, done, info = super().step(eff_action)
        base_reward = reward
        ax, ay = self.a['x'], self.a['y']
        info["in_friction_zone"] = in_friction_zone
        info["terrain_friction"] = current_friction

        # 3. Vector Flow Displacement & Convective Wind Dynamics
        arena_bound = float(getattr(self, "arena_half", HALF))
        wx, wy = self.get_wind(ax, ay)
        tot_dx = flow_dx + wx * 0.25
        tot_dy = flow_dy + wy * 0.25
        if tot_dx != 0.0 or tot_dy != 0.0:
            h = arena_bound - 0.35
            self.a["x"] = float(np.clip(self.a["x"] + tot_dx, -h, h))
            self.a["y"] = float(np.clip(self.a["y"] + tot_dy, -h, h))
            ax, ay = self.a["x"], self.a["y"]

        # Move movable objects caught in vector flow zones and convective winds
        for obj in self.objects:
            if obj.get("movable") and not obj.get("held"):
                owx, owy = self.get_wind(obj["x"], obj["y"])
                for zone in self.terrain_zones:
                    if math.hypot(obj["x"] - zone["x"], obj["y"] - zone["y"]) < zone["r"]:
                        owx += zone.get("vx", 0.0)
                        owy += zone.get("vy", 0.0)
                if owx != 0.0 or owy != 0.0:
                    obj["x"] = float(np.clip(obj["x"] + owx * 0.2, -arena_bound + 0.35, arena_bound - 0.35))
                    obj["y"] = float(np.clip(obj["y"] + owy * 0.2, -arena_bound + 0.35, arena_bound - 0.35))

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
            h = arena_bound - 0.35
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

        # 6. Rotating Sentinel Searchlights & Stealth Cloak Mechanics
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

        # 7. Acoustic Sound Tripwires
        agent_speed = abs(self.a["v"])
        for trip in self.sound_tripwires:
            t_x1, t_y1, t_x2, t_y2 = trip["x1"], trip["y1"], trip["x2"], trip["y2"]
            seg_dist = abs((t_y2 - t_y1) * ax - (t_x2 - t_x1) * ay + t_x2 * t_y1 - t_y2 * t_x1) / max(1e-5, math.hypot(t_y2 - t_y1, t_x2 - t_x1))
            if seg_dist < 0.4 and min(t_x1, t_x2) - 0.2 <= ax <= max(t_x1, t_x2) + 0.2 and min(t_y1, t_y2) - 0.2 <= ay <= max(t_y1, t_y2) + 0.2:
                if agent_speed > trip.get("max_safe_speed", 0.30):
                    alarm_tripped = True
                    reward -= 0.25

        self.last_alarm_tripped = alarm_tripped
        info["alarm_tripped"] = alarm_tripped

        # 8. Hierarchical Multi-Agent Predator Ecology & State Machines
        min_hazard_dist = 10.0
        pack_alerted = any(p.get("alerted", False) for p in self.predators) or alarm_tripped

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
            p_kind = pred.get("kind", "stalker")

            if p_kind == "aerial_raptor":
                pred["altitude"] = max(0.2, float(pred.get("altitude", 1.5)))
                if pred.get("state") == "dive":
                    desired_th = math.atan2(p_dy, p_dx)
                    pred["th"] += 0.22 * math.sin(desired_th - pred["th"])
                    pred["speed"] = 0.24
                    pred["altitude"] = max(0.1, pred["altitude"] - 0.08)
                    pred["dive_timer"] = pred.get("dive_timer", 0) + 1
                    if pred["dive_timer"] > 20 or pred["altitude"] <= 0.15:
                        pred["state"] = "recovery"
                        pred["dive_timer"] = 0
                elif pred.get("state") == "recovery":
                    pred["altitude"] += 0.06
                    pred["speed"] = 0.12
                    if pred["altitude"] >= 1.5:
                        pred["state"] = "patrol"
                else:
                    if dist_to_agent < 4.2 and not in_shadow and abs(self.a["v"]) > 0.12:
                        pred["state"] = "dive"
                        pred["dive_timer"] = 0
                        pred["alerted"] = True
                    else:
                        pred["th"] += 0.04
                        pred["speed"] = 0.13
            elif p_kind == "ground_stalker":
                rel_ang = math.atan2(pred["y"] - ay, pred["x"] - ax)
                diff_ang = abs((rel_ang - self.a["th"] + math.pi) % (math.pi * 2) - math.pi)
                agent_looking = diff_ang < 0.65
                if pred.get("state") == "pursuit":
                    desired_th = math.atan2(p_dy, p_dx)
                    pred["th"] += 0.20 * math.sin(desired_th - pred["th"])
                    pred["speed"] = 0.22
                    pred["stamina"] = pred.get("stamina", 80) - 2
                    if pred["stamina"] <= 0 or dist_to_agent > 4.5:
                        pred["state"] = "fatigue"
                elif pred.get("state") == "fatigue":
                    pred["speed"] = 0.06
                    pred["stamina"] = min(80, pred.get("stamina", 0) + 1)
                    if pred["stamina"] >= 75:
                        pred["state"] = "stalk"
                else:
                    if agent_looking and dist_to_agent > 1.8:
                        pred["speed"] = 0.01
                    else:
                        desired_th = math.atan2(p_dy, p_dx)
                        pred["th"] += 0.12 * math.sin(desired_th - pred["th"])
                        pred["speed"] = 0.11
                        if dist_to_agent < 1.8:
                            pred["state"] = "pursuit"
                            pred["stamina"] = 80
                            pred["alerted"] = True
            elif p_kind == "abyssal_leviathan":
                if pred.get("state") == "dormant":
                    pred["speed"] = 0.02
                    pred["th"] += 0.02
                    if abs(self.a["v"]) > 0.22 or alarm_tripped:
                        pred["state"] = "pursuit"
                        pred["alerted"] = True
                else:
                    desired_th = math.atan2(p_dy, p_dx)
                    pred["th"] += 0.15 * math.sin(desired_th - pred["th"])
                    pred["speed"] = 0.18
                    if abs(self.a["v"]) < 0.05 and not alarm_tripped:
                        pred["state"] = "dormant"
            elif pred["kind"] == "pursuer" or (pred.get("alerted") and pred["kind"] in ("pack_leader", "stalker")):
                desired_th = math.atan2(p_dy, p_dx)
                pred["th"] += 0.14 * math.sin(desired_th - pred["th"])
            elif pred.get("alerted") and pred["kind"] == "pack_hunter":
                lead_x = ax + math.cos(self.a["th"]) * self.a["v"] * 2.0
                lead_y = ay + math.sin(self.a["th"]) * self.a["v"] * 2.0
                desired_th = math.atan2(lead_y - pred["y"], lead_x - pred["x"])
                pred["th"] += 0.16 * math.sin(desired_th - pred["th"])
            else:
                pred["th"] += float(self.rng_complex.normal(0.0, 0.08))

            pred_speed = pred["speed"] * (1.3 if pred.get("alerted") else 1.0)
            pred["x"] += math.cos(pred["th"]) * pred_speed
            pred["y"] += math.sin(pred["th"]) * pred_speed

            h = arena_bound - 0.5
            if pred["x"] < -h or pred["x"] > h:
                pred["th"] = math.pi - pred["th"]
                pred["x"] = float(np.clip(pred["x"], -h, h))
            if pred["y"] < -h or pred["y"] > h:
                pred["th"] = -pred["th"]
                pred["y"] = float(np.clip(pred["y"], -h, h))

            dist_to_agent = math.hypot(ax - pred['x'], ay - pred['y'])
            min_hazard_dist = min(min_hazard_dist, dist_to_agent)
            if dist_to_agent < (pred["r"] + 0.35):
                reward -= 0.45
                info["predator_impact"] = True

        info["min_hazard_dist"] = min_hazard_dist

        # 9. Ecological Food Respawn
        self.food_respawn_timer += 1
        if self.task in ("ecosystem", "apex_megasite") and self.food_respawn_timer % 40 == 0:
            current_food = sum(1 for o in self.objects if o.get("kind") == "food" and o.get("visible", True))
            max_food = 6 if self.task == "ecosystem" else 4
            if current_food < max_food:
                spot = self._free_spot(avoid=[(o['x'], o['y']) for o in self.objects])
                self._obj("food", spot[0], spot[1], "green", r=0.25, movable=False)

        # 10. Ambient Illumination Cycle & Cooperative Helper Drones
        self.day_night_clock += 0.015
        self.ambient_light = float(0.65 + 0.35 * math.sin(self.day_night_clock))
        info["ambient_light"] = self.ambient_light

        helper_shield_active = False
        for drone in self.helper_drones:
            drone["orbit_angle"] = float(drone.get("orbit_angle", 0.0) + 0.065) % (math.pi * 2)
            cx, cy = drone.get("orbit_center", [0.0, 0.0])
            rad = float(drone.get("orbit_radius", 1.5))
            nx = float(cx + math.cos(drone["orbit_angle"]) * rad)
            ny = float(cy + math.sin(drone["orbit_angle"]) * rad)
            drone["th"] = float(math.atan2(ny - drone["y"], nx - drone["x"]))
            drone["x"], drone["y"] = nx, ny
            if math.hypot(ax - nx, ay - ny) <= float(drone.get("shield_radius", 1.6)):
                helper_shield_active = True
                reward += 0.04
                info["min_hazard_dist"] = max(float(info.get("min_hazard_dist", 10.0)), 6.0)
        info["helper_shield_active"] = helper_shield_active

        # 10b. Wormhole Teleportation Relays & Weather Storm Fronts
        if getattr(self, "wormhole_cooldown", 0) > 0:
            self.wormhole_cooldown -= 1
        wormhole_teleported = False
        for wh in getattr(self, "wormholes", []):
            if self.wormhole_cooldown == 0 and math.hypot(ax - wh["x"], ay - wh["y"]) <= float(wh.get("r", 0.75)):
                self.a["x"] = float(wh["target_x"])
                self.a["y"] = float(wh["target_y"])
                ax, ay = self.a["x"], self.a["y"]
                self.wormhole_cooldown = 18
                wormhole_teleported = True
                reward += 0.15
                break
        info["wormhole_teleported"] = wormhole_teleported

        in_weather_storm = False
        for storm in getattr(self, "weather_fronts", []):
            storm["x"] = float(storm["x"] + storm.get("vx", 0.03))
            storm["y"] = float(storm["y"] + storm.get("vy", 0.02))
            sb = arena_bound - 1.2
            if abs(storm["x"]) > sb:
                storm["vx"] = -storm.get("vx", 0.03)
                storm["x"] = float(np.clip(storm["x"], -sb, sb))
            if abs(storm["y"]) > sb:
                storm["vy"] = -storm.get("vy", 0.02)
                storm["y"] = float(np.clip(storm["y"], -sb, sb))
            if math.hypot(ax - storm["x"], ay - storm["y"]) <= float(storm.get("r", 2.4)):
                in_weather_storm = True
                h = arena_bound - 0.35
                self.a["x"] = float(np.clip(self.a["x"] + storm.get("wind_x", 0.0), -h, h))
                self.a["y"] = float(np.clip(self.a["y"] + storm.get("wind_y", 0.0), -h, h))
                ax, ay = self.a["x"], self.a["y"]
        info["in_weather_storm"] = in_weather_storm

        # 11. Complex Task Completion and Terminal Goal Rewards
        target_obj = next((o for o in self.objects if o.get("kind") == "target"), None)
        if target_obj is not None:
            dist_to_target = math.hypot(ax - target_obj["x"], ay - target_obj["y"])
            if dist_to_target < (target_obj.get("r", 0.4) + 0.25):
                if self.task in ("puzzle_chamber", "cybernetic_sanctuary", "continental_biosphere", "chrono_tidepool_citadel"):
                    btn = next((o for o in self.objects if o.get("kind") == "button"), None)
                    if btn and btn.get("pressed"):
                        done = True
                        reward += 6.0 if self.task == "puzzle_chamber" else 9.0
                        info["outcome"] = "success"
                elif self.task in ("dynamic_arena", "apex_megasite", "stealth_labyrinth", "kinetic_foundry", "quantum_bifurcation", "hyper_boreal_tundra", "subterranean_abyss", "vortex_maelstrom"):
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

        if t in ("quantum_bifurcation", "cybernetic_sanctuary", "continental_biosphere", "chrono_tidepool_citadel", "hyper_boreal_tundra", "subterranean_abyss", "vortex_maelstrom"):
            btn = next((o for o in self.objects if o.get("kind") == "button"), None)
            target = next((o for o in self.objects if o.get("kind") == "target"), None)
            if btn and not btn.get("pressed"):
                return -math.hypot(ax - btn["x"], ay - btn["y"])
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
        """Extends snapshot with predators, terrain zones, vector fields, sentinels, elevation hills, and dynamic flora/geysers."""
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
                "state": p.get("state", "patrol" if not p.get("alerted") else "pursuit"),
                "altitude": round(float(p.get("altitude", 0.0)), 2),
            }
            for p in self.predators
        ]
        snap["terrain_zones"] = self.terrain_zones
        snap["dynamic_gates"] = self.dynamic_gates
        snap["elevation_hills"] = self.elevation_hills
        snap["thermal_geysers"] = self.thermal_geysers
        snap["vortices"] = self.vortices
        snap["bioluminescent_flora"] = self.bioluminescent_flora
        snap["elevation_grid"] = self._get_elevation_grid(resolution=11)
        snap["wind_streamlines"] = self._sample_wind_streamlines(count=16)
        snap["sentinels"] = self.sentinels
        snap["sound_tripwires"] = self.sound_tripwires
        snap["helper_drones"] = [
            {
                "id": d["id"],
                "x": round(float(d["x"]), 3),
                "y": round(float(d["y"]), 3),
                "th": round(float(d.get("th", 0.0)), 3),
                "r": d.get("r", 0.32),
                "shield_radius": d.get("shield_radius", 1.6),
                "kind": d.get("kind", "guardian_drone"),
                "color": d.get("color", "cyan"),
            }
            for d in self.helper_drones
        ]
        snap["ambient_light"] = round(self.ambient_light, 3)
        snap["celestial_angle"] = round(float(self.day_night_clock % (2 * math.pi)), 4)
        snap["alarm_tripped"] = self.last_alarm_tripped
        snap["half"] = round(float(getattr(self, "arena_half", HALF)), 3)
        snap["wormholes"] = [
            {
                "id": wh["id"],
                "x": round(float(wh["x"]), 3),
                "y": round(float(wh["y"]), 3),
                "r": float(wh.get("r", 0.75)),
                "target_x": round(float(wh["target_x"]), 3),
                "target_y": round(float(wh["target_y"]), 3),
                "color": wh.get("color", "purple"),
            }
            for wh in getattr(self, "wormholes", [])
        ]
        snap["weather_fronts"] = [
            {
                "id": wf["id"],
                "x": round(float(wf["x"]), 3),
                "y": round(float(wf["y"]), 3),
                "r": float(wf.get("r", 2.4)),
                "wind_x": round(float(wf.get("wind_x", 0.0)), 3),
                "wind_y": round(float(wf.get("wind_y", 0.0)), 3),
            }
            for wf in getattr(self, "weather_fronts", [])
        ]
        return snap
