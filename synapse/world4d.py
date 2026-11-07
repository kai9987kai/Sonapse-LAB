"""Four-Spatial-Dimensional Virtual Environment (world4d) & 4D-to-155D Task Adapter.

Grounding & Scientific Scope:
- ``w`` is explicitly a fourth SPATIAL coordinate (meters), NOT time.
- Implements 4D Euclidean obstacles: 4D axis-aligned bounding boxes (hyperboxes) and
  4D hyperspheres with exact distance metric d = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 + (w1-w2)^2).
- Includes the ``world4d_tesseract_bypass`` puzzle: an obstacle completely blocks all 3D paths
  at w = 0, but the agent can move in the fourth spatial dimension w in [-arena, arena] to bypass
  the barrier and reach the target.
- Adapts the 5-action motor command [drive, turn, reach, height, grip] into 4D spatial navigation:
  [dx, dy, dz, dw, interact], strictly preserving the 5-element action contract and 155-element
  observation schema for seamless IntegratedAgent compatibility.
- Provides 3D slice visualization at current w and 4D-to-3D perspective projection telemetry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HyperBox4D:
    id: str
    min_corner: np.ndarray  # [4]
    max_corner: np.ndarray  # [4]
    kind: str = "barrier"   # 'barrier' | 'hazard' | 'goal'

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.min_corner) and np.all(p <= self.max_corner))

    def distance_to(self, p: np.ndarray) -> float:
        clamped = np.clip(p, self.min_corner, self.max_corner)
        return float(np.linalg.norm(p - clamped))


@dataclass
class HyperSphere4D:
    id: str
    center: np.ndarray  # [4]
    radius: float
    kind: str = "goal"

    def contains(self, p: np.ndarray) -> bool:
        return float(np.linalg.norm(p - self.center)) <= self.radius

    def distance_to(self, p: np.ndarray) -> float:
        return max(0.0, float(np.linalg.norm(p - self.center)) - self.radius)


class World4D:
    """Four-spatial-dimensional environment with 4D Euclidean geometry and 155D observation adapter."""

    def __init__(self, arena_half: float = 6.0, seed: int = 42):
        self.arena_half = float(arena_half)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        # Agent 4D state: position [x, y, z, w], velocity [vx, vy, vz, vw]
        self.pos = np.zeros(4, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        self.target_pos = np.array([3.5, 0.0, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        self.ep_return = 0.0
        self.done = False

        # Build 4D Tesseract Bypass Puzzle:
        # A 3D wall spanning x in [-0.4, 0.4], y in [-6, 6], z in [-6, 6] at w in [-0.5, 0.5]
        # To bypass, the agent MUST travel into w > 0.8 or w < -0.8 in the 4th spatial dimension.
        self.obstacles: List[HyperBox4D] = [
            HyperBox4D(
                id="central_3d_hyperwall",
                min_corner=np.array([-0.4, -self.arena_half, -self.arena_half, -0.6], dtype=np.float32),
                max_corner=np.array([0.4, self.arena_half, self.arena_half, 0.6], dtype=np.float32),
                kind="barrier",
            )
        ]
        self.goals: List[HyperSphere4D] = [
            HyperSphere4D(id="target_sphere", center=self.target_pos.copy(), radius=0.6, kind="goal")
        ]

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = np.array([-3.5, 0.0, 0.0, 0.0], dtype=np.float32)
        self.vel.fill(0.0)
        self.target_pos = np.array([3.5, self.rng.uniform(-1.0, 1.0), 0.0, 0.0], dtype=np.float32)
        self.goals[0].center = self.target_pos.copy()
        self.steps = 0
        self.ep_return = 0.0
        self.done = False
        return self._build_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).flatten()
        if len(act) < 5:
            padded = np.zeros(5, dtype=np.float32)
            padded[: len(act)] = act
            act = padded
        act = np.clip(act[:5], -1.0, 1.0)

        # Map 5-action vector to 4D spatial controls:
        # act[0] (drive)   -> dx
        # act[1] (turn)    -> dy
        # act[2] (reach)   -> dw (Fourth spatial dimension navigation!)
        # act[3] (height)  -> dz
        # act[4] (grip)    -> interaction
        dx, dy, dw, dz, interact = float(act[0]), float(act[1]), float(act[2]), float(act[3]), float(act[4])

        dt = 0.15
        force = np.array([dx, dy, dz, dw], dtype=np.float32) * 1.8
        self.vel = np.clip(self.vel * 0.82 + force * dt, -1.2, 1.2)
        cand_pos = self.pos + self.vel * dt

        # Arena boundary clamping
        cand_pos = np.clip(cand_pos, -self.arena_half, self.arena_half)

        # 4D Collision detection against hyperboxes
        collided = False
        for obs in self.obstacles:
            if obs.contains(cand_pos):
                collided = True
                self.vel *= -0.2
                break

        if not collided:
            self.pos = cand_pos

        dist_to_goal = float(np.linalg.norm(self.pos - self.target_pos))
        reward = -0.01 * dist_to_goal  # Distance shaping penalty

        # Goal attainment check
        reached_goal = dist_to_goal < 0.75 or self.goals[0].contains(self.pos)
        if reached_goal:
            reward += 5.0
            self.done = True

        self.steps += 1
        if self.steps >= 128:
            self.done = True

        self.ep_return += reward
        obs_155 = self._build_observation()
        info = {
            "pos_4d": [round(float(x), 4) for x in self.pos],
            "target_4d": [round(float(x), 4) for x in self.target_pos],
            "dist_to_goal": round(dist_to_goal, 4),
            "collided": collided,
            "bypassed_via_w": bool(abs(self.pos[3]) > 0.6 and abs(self.pos[0]) < 0.5),
            "reached_goal": reached_goal,
            "steps": self.steps,
            "w_coordinate": round(float(self.pos[3]), 4),
        }
        return obs_155, float(reward), bool(self.done), info

    def _build_observation(self) -> np.ndarray:
        """Encodes 4D spatial geometry into standard 155-dimensional observation vector."""
        obs = np.zeros(155, dtype=np.float32)

        # Channels 0..32: 33 pseudo-depth rays sampled across 4D hyperspherical coordinates
        rays = np.zeros(33, dtype=np.float32)
        angles = np.linspace(-math.pi, math.pi, 33)
        for i, ang in enumerate(angles):
            ray_dir = np.array([math.cos(ang), math.sin(ang), 0.0, math.sin(2 * ang) * 0.5], dtype=np.float32)
            ray_dir /= np.linalg.norm(ray_dir) + 1e-6
            # Step along ray
            d = 1.0
            for step_k in np.linspace(0.2, 5.0, 10):
                sample_p = self.pos + ray_dir * step_k
                for obs_box in self.obstacles:
                    if obs_box.contains(sample_p):
                        d = float(step_k / 5.0)
                        break
                if d < 1.0:
                    break
            rays[i] = d
        obs[:33] = rays

        # Channels 33..131: RGB visual retinotopic representation of target in 4D projection
        rel_target = self.target_pos - self.pos
        t_dist = max(0.1, float(np.linalg.norm(rel_target)))
        t_dir = rel_target / t_dist
        rgb = np.zeros((3, 33), dtype=np.float32)
        center_bin = int(np.clip(16 + int(t_dir[1] * 12), 0, 32))
        rgb[0, center_bin] = float(np.clip(1.0 - t_dist / 8.0, 0.2, 1.0))
        rgb[1, center_bin] = float(np.clip(0.5 + 0.5 * t_dir[3], 0.0, 1.0))  # Green encodes w direction
        rgb[2, center_bin] = 0.8
        obs[33:132] = rgb.flatten()

        # Channels 132..154: Proprioception & explicit 4D spatial telemetry
        obs[132] = self.pos[0] / self.arena_half
        obs[133] = self.pos[1] / self.arena_half
        obs[134] = self.pos[2] / self.arena_half
        obs[135] = self.pos[3] / self.arena_half   # Fourth spatial coordinate w
        obs[136] = self.vel[0]
        obs[137] = self.vel[1]
        obs[138] = self.vel[2]
        obs[139] = self.vel[3]
        obs[140] = rel_target[0] / self.arena_half
        obs[141] = rel_target[1] / self.arena_half
        obs[142] = rel_target[2] / self.arena_half
        obs[143] = rel_target[3] / self.arena_half
        obs[144] = float(np.clip(1.0 - t_dist / self.arena_half, 0.0, 1.0))
        obs[154] = float(self.steps) / 128.0
        return obs

    def observe(self) -> np.ndarray:
        return self._build_observation()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task": "world4d_tesseract_bypass",
            "pos_4d": [round(float(x), 4) for x in self.pos],
            "vel_4d": [round(float(x), 4) for x in self.vel],
            "target_4d": [round(float(x), 4) for x in self.target_pos],
            "dist_to_goal": round(float(np.linalg.norm(self.pos - self.target_pos)), 4),
            "arena_half": self.arena_half,
            "steps": self.steps,
            "ep_return": round(float(self.ep_return), 4),
            "obstacles": [
                {
                    "id": o.id,
                    "min": [round(float(x), 2) for x in o.min_corner],
                    "max": [round(float(x), 2) for x in o.max_corner],
                }
                for o in self.obstacles
            ],
            "projection_3d": {
                "x": round(float(self.pos[0]), 3),
                "y": round(float(self.pos[1]), 3),
                "z": round(float(self.pos[2]), 3),
                "w_slice": round(float(self.pos[3]), 3),
            },
        }


# Backwards-compatible alias
World4DEnvironment = World4D
