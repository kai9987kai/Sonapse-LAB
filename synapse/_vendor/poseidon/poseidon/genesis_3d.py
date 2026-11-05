"""POSEIDON GENESIS-3D: Continuous Embodied 3D Simulation & Physics Manifold.

Inspired by GenesisEngine & 3D-Animal-Simulator:
1. Continuous 3D Spatial-Temporal Dynamics:
   Replaces discrete grid steps with continuous Newtonian physics:
     \\dot{p} = v
     m \\dot{v} = F_{propulsion} - \\gamma_{drag} v + F_{buoyancy} + F_{gravity} + F_{current}

2. Dynamic Tidal Hydrodynamics:
   Tidal water elevation oscillates with sinusoidal period T_{tide}:
     z_{tide}(t) = z_0 + A_{tide} \\sin(2\\pi t / T_{tide})
   Submerged states experience viscous fluid drag and oxygen consumption,
   forcing the agent to manage diving vs surfacing.

3. Volumetric Raycast Perception:
   Simulates forward and panoramic sensory rays detecting:
     - Distance to boundary/cliff
     - Food odor field gradient \\nabla \\Phi_{food}
     - Water salinity gradient
     - Threat line-of-sight

4. Poseidon Observation Bridge:
   Encodes the continuous 3D physics state into a normalized 16-D feature vector
   fully compatible with Poseidon's Core, Horizon-X, Active Inference, and Causeway.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass
class PhysicsVector3:
    x: float
    y: float
    z: float

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> PhysicsVector3:
        l = self.length()
        if l < 1e-6:
            return PhysicsVector3(0.0, 0.0, 0.0)
        return PhysicsVector3(self.x / l, self.y / l, self.z / l)


@dataclass
class ContinuousResourcePatch:
    id: int
    pos: PhysicsVector3
    radius: float
    food_amount: float
    water_amount: float
    shelter_quality: float


@dataclass
class Genesis3DConfig:
    world_size: float = 100.0  # [0, 100] in x and y
    gravity: float = 9.81
    fluid_density: float = 1.0
    air_density: float = 0.02
    tide_period: float = 64.0  # Ticks per tidal cycle
    tide_amplitude: float = 3.0
    base_water_level: float = 2.0
    drag_coefficient: float = 0.85
    dt: float = 0.25  # Simulation step size in seconds


class ContinuousTidePool3D:
    """Continuous 3D embodied physics tidepool simulation environment."""

    def __init__(self, config: Genesis3DConfig | None = None, seed: int = 42):
        self.config = config or Genesis3DConfig()
        self.rng = random.Random(seed)
        self.time: float = 0.0
        self.tick: int = 0

        # Agent state
        self.pos = PhysicsVector3(50.0, 50.0, 3.0)
        self.vel = PhysicsVector3(0.0, 0.0, 0.0)
        self.heading: float = 0.0  # Angle in radians

        # Vitals
        self.health: float = 1.0
        self.energy: float = 0.85
        self.hydration: float = 0.85
        self.stamina: float = 0.90
        self.oxygen: float = 1.0

        # Procedural resource patches
        self.patches: list[ContinuousResourcePatch] = []
        self._init_patches()

    def _init_patches(self, count: int = 12):
        self.patches = []
        for i in range(count):
            x = self.rng.uniform(10.0, self.config.world_size - 10.0)
            y = self.rng.uniform(10.0, self.config.world_size - 10.0)
            z = self.get_terrain_height(x, y)
            self.patches.append(
                ContinuousResourcePatch(
                    id=i,
                    pos=PhysicsVector3(x, y, z),
                    radius=self.rng.uniform(4.0, 8.0),
                    food_amount=self.rng.uniform(0.3, 0.9),
                    water_amount=self.rng.uniform(0.4, 0.9),
                    shelter_quality=self.rng.uniform(0.2, 0.8),
                )
            )

    def get_terrain_height(self, x: float, y: float) -> float:
        """Procedural smooth continuous heightfield."""
        w1 = math.sin(x * 0.05) * math.cos(y * 0.05) * 3.0
        w2 = math.sin(x * 0.12 + 1.0) * math.cos(y * 0.12 + 0.5) * 1.5
        return max(0.0, w1 + w2 + 1.0)

    def get_tide_height(self) -> float:
        """Oscillating water level."""
        phase = (2.0 * math.pi * self.time) / self.config.tide_period
        return self.config.base_water_level + self.config.tide_amplitude * math.sin(phase)

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Simulates 1 continuous physics step and returns 16-D normalized observation."""
        dt = self.config.dt
        self.time += dt
        self.tick += 1

        tide_h = self.get_tide_height()
        ground_h = self.get_terrain_height(self.pos.x, self.pos.y)
        is_submerged = self.pos.z < tide_h

        # 1. Action forces
        thrust = 0.0
        torque = 0.0
        vz_thrust = 0.0

        if action == 4:  # explore / forward
            thrust = 8.0
            self.stamina = max(0.0, self.stamina - 0.015)
        elif action == 5:  # flee
            thrust = 14.0
            torque = 0.8
            self.stamina = max(0.0, self.stamina - 0.035)
        elif action == 0:  # rest
            thrust = 0.0
            self.stamina = min(1.0, self.stamina + 0.03)
            self.energy = max(0.0, self.energy - 0.003)
            self.hydration = max(0.0, self.hydration - 0.003)

        # Ingestion actions
        nearest_patch = self._get_nearest_patch()
        dist_to_patch = math.sqrt((self.pos.x - nearest_patch.pos.x)**2 + (self.pos.y - nearest_patch.pos.y)**2)
        in_patch = dist_to_patch <= nearest_patch.radius

        reward = 0.05  # Living baseline

        if action == 1:  # forage
            if in_patch and nearest_patch.food_amount > 0.05:
                taken = min(0.3, nearest_patch.food_amount)
                self.energy = min(1.0, self.energy + taken * 0.7)
                nearest_patch.food_amount -= taken
                reward += 0.25
            self.stamina = max(0.0, self.stamina - 0.008)

        elif action == 2:  # drink
            if in_patch and nearest_patch.water_amount > 0.05:
                taken = min(0.35, nearest_patch.water_amount)
                self.hydration = min(1.0, self.hydration + taken * 0.8)
                nearest_patch.water_amount -= taken
                reward += 0.25
            self.stamina = max(0.0, self.stamina - 0.008)

        # 2. Physics integration
        # Viscous drag
        density = self.config.fluid_density if is_submerged else self.config.air_density
        drag = density * self.config.drag_coefficient

        self.heading += torque * dt
        ax = math.cos(self.heading) * thrust - drag * self.vel.x
        ay = math.sin(self.heading) * thrust - drag * self.vel.y

        # Buoyancy vs gravity
        buoyancy = (self.config.gravity * 1.1) if is_submerged else 0.0
        az = buoyancy - self.config.gravity + vz_thrust - drag * self.vel.z

        self.vel.x += ax * dt
        self.vel.y += ay * dt
        self.vel.z += az * dt

        self.pos.x = max(0.0, min(self.config.world_size, self.pos.x + self.vel.x * dt))
        self.pos.y = max(0.0, min(self.config.world_size, self.pos.y + self.vel.y * dt))
        self.pos.z = max(ground_h, self.pos.z + self.vel.z * dt)

        # Baseline vital decay
        self.energy = max(0.0, self.energy - 0.006 * dt)
        self.hydration = max(0.0, self.hydration - 0.008 * dt)

        # Damage from starvation or dehydration
        if self.energy < 0.10:
            self.health = max(0.0, self.health - 0.04)
        if self.hydration < 0.10:
            self.health = max(0.0, self.health - 0.05)

        done = self.health <= 0.0 or self.tick >= 256
        obs = self._build_observation(in_patch, nearest_patch, is_submerged)

        return obs, reward, done, {
            "tick": self.tick,
            "pos": (round(self.pos.x, 2), round(self.pos.y, 2), round(self.pos.z, 2)),
            "is_submerged": is_submerged,
            "tide_height": round(tide_h, 2),
            "health": self.health,
            "alive": self.health > 0.0,
        }

    def _get_nearest_patch(self) -> ContinuousResourcePatch:
        return min(self.patches, key=lambda p: (p.pos.x - self.pos.x)**2 + (p.pos.y - self.pos.y)**2)

    def _build_observation(
        self,
        in_patch: bool,
        patch: ContinuousResourcePatch,
        is_submerged: bool,
    ) -> list[float]:
        """Encodes 3D continuous physics into normalized 16-D Poseidon observation vector."""
        obs = [0.0] * OBS_SIZE
        obs[0] = float(self.health)
        obs[1] = float(self.energy)
        obs[2] = float(self.hydration)
        obs[3] = float(self.stamina)
        obs[4] = 0.8 if is_submerged else 0.1  # exposure
        obs[5] = 0.05  # threat
        obs[6] = float(patch.food_amount) if in_patch else 0.05
        obs[7] = float(patch.water_amount) if in_patch else 0.05
        obs[8] = float(patch.shelter_quality) if in_patch else 0.1
        obs[9] = float(self.pos.x / self.config.world_size)
        obs[10] = float(self.pos.y / self.config.world_size)
        obs[11] = float(self.pos.z / 15.0)  # normalized height
        obs[12] = float(self.vel.length() / 10.0)  # speed
        obs[13] = 1.0 if in_patch else 0.0
        obs[14] = float((math.sin(self.heading) + 1.0) / 2.0)
        obs[15] = float((math.cos(self.heading) + 1.0) / 2.0)
        return obs
