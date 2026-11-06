"""Allocentric cognitive mapping and probabilistic belief-state tracking for SYNAPSE.

Implements an allocentric spatial memory grid and landmark graph that persists across
cue disappearances, combined with a Bayesian particle filter tracking partially
observable hidden states under noisy observations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import heapq
import numpy as np


@dataclass
class Landmark:
    """A spatial landmark with coordinates and visual descriptor."""
    id: str
    position: np.ndarray        # [x, y]
    feature_vector: np.ndarray  # Visual / sensory signature
    last_observed_step: int
    confidence: float = 1.0


class AllocentricCognitiveMap:
    """Internal spatial coordinate map supporting path planning under sensory loss."""
    def __init__(self, grid_size: int = 40, resolution: float = 0.5):
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin = np.array([grid_size * resolution / 2.0, grid_size * resolution / 2.0])
        
        # 2D Occupancy grid: 0.0 = free, 1.0 = occupied, 0.5 = unexplored
        self.occupancy_grid = np.full((grid_size, grid_size), 0.5, dtype=np.float32)
        self.landmarks: Dict[str, Landmark] = {}
        
        # Agent estimated allocentric pose: [x, y, theta]
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0

    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        gx = int(np.clip((pos[0] + self.origin[0]) / self.resolution, 0, self.grid_size - 1))
        gy = int(np.clip((pos[1] + self.origin[1]) / self.resolution, 0, self.grid_size - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> np.ndarray:
        wx = gx * self.resolution - self.origin[0]
        wy = gy * self.resolution - self.origin[1]
        return np.array([wx, wy], dtype=np.float32)

    def integrate_odometry(self, forward_velocity: float, angular_velocity: float, dt: float = 0.1):
        """Dead reckoning path integration from internal motor commands."""
        theta = self.pose[2] + angular_velocity * dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        dx = forward_velocity * np.cos(theta) * dt
        dy = forward_velocity * np.sin(theta) * dt
        
        self.pose[0] += dx
        self.pose[1] += dy
        self.pose[2] = theta
        self.step_count += 1
        
        # Clear current cell as traversed / free
        gx, gy = self.world_to_grid(self.pose[:2])
        self.occupancy_grid[gx, gy] = 0.0

    def observe_rays(self, ray_distances: np.ndarray, max_range: float = 5.0):
        """Updates occupancy grid using multi-angle distance sensor returns."""
        num_rays = len(ray_distances)
        angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays) + self.pose[2]
        
        px, py = self.pose[0], self.pose[1]
        
        for d, angle in zip(ray_distances, angles):
            d_clamped = min(float(d) * max_range, max_range)
            ox = px + d_clamped * np.cos(angle)
            oy = py + d_clamped * np.sin(angle)
            
            # Ray casting: cells along ray are free, endpoint is occupied if ray hit something
            gx, gy = self.world_to_grid(np.array([ox, oy]))
            if d < 0.95: # Hit obstacle
                self.occupancy_grid[gx, gy] = min(1.0, self.occupancy_grid[gx, gy] + 0.3)
            else:
                self.occupancy_grid[gx, gy] = max(0.0, self.occupancy_grid[gx, gy] - 0.1)

    def register_landmark(self, name: str, relative_pos: np.ndarray, feature: np.ndarray):
        """Registers or updates a spatial landmark in allocentric coordinates."""
        # Transform relative (ego) pos to world pos
        theta = self.pose[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        world_pos = self.pose[:2] + R @ relative_pos[:2]
        
        self.landmarks[name] = Landmark(
            id=name,
            position=world_pos,
            feature_vector=feature.copy(),
            last_observed_step=self.step_count,
            confidence=1.0
        )

    def plan_path_to_target(self, target_world_pos: np.ndarray) -> List[np.ndarray]:
        """A* path planning over the internal occupancy grid.
        
        Succeeds even if sensory cues are currently absent, by navigating through stored map.
        """
        start_grid = self.world_to_grid(self.pose[:2])
        goal_grid = self.world_to_grid(target_world_pos)
        
        # Priority queue for A*: (cost, (gx, gy))
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0.0}
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        found = False
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_grid:
                found = True
                break
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                # Obstacle cost
                occ = self.occupancy_grid[neighbor[0], neighbor[1]]
                if occ > 0.7: # Impassable
                    continue
                step_cost = 1.0 + occ * 3.0
                tentative_g = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        if not found:
            # Fallback direct line
            return [self.pose[:2].copy(), target_world_pos.copy()]
            
        # Reconstruct path
        path_grid = [goal_grid]
        curr = goal_grid
        while curr in came_from:
            curr = came_from[curr]
            path_grid.append(curr)
        path_grid.reverse()
        
        # Convert to world coordinates
        return [self.grid_to_world(gx, gy) for gx, gy in path_grid]


class ParticleFilterBeliefTracker:
    """Maintains probabilistic distribution over hidden environment states under noisy/missing observations."""
    def __init__(self, num_particles: int = 150, state_dim: int = 2, seed: int = 42):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        
        # Initialize particles uniformly in [-5, 5]
        self.particles = self.rng.uniform(-5.0, 5.0, size=(num_particles, state_dim))
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, velocity: np.ndarray, dt: float = 0.1, noise_std: float = 0.05):
        """State transition update with process noise."""
        displacement = velocity[:self.state_dim] * dt
        noise = self.rng.normal(0.0, noise_std, size=self.particles.shape)
        self.particles += displacement + noise

    def update(self, noisy_measurement: np.ndarray, meas_std: float = 0.2):
        """Bayesian observation update weighting particles by measurement likelihood."""
        diff = self.particles - noisy_measurement[:self.state_dim]
        dist_sq = np.sum(diff ** 2, axis=1)
        # Gaussian likelihood
        likelihoods = np.exp(-0.5 * dist_sq / (meas_std ** 2)) + 1e-12
        self.weights *= likelihoods
        sum_w = np.sum(self.weights)
        if sum_w > 1e-12:
            self.weights /= sum_w
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
            
        # Systematic resampling if effective sample size drops
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2.0:
            indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles

    @property
    def estimated_mean(self) -> np.ndarray:
        return np.sum(self.particles * self.weights[:, None], axis=0)

    @property
    def estimated_covariance(self) -> np.ndarray:
        mean = self.estimated_mean
        diff = self.particles - mean
        return (diff.T * self.weights) @ diff
