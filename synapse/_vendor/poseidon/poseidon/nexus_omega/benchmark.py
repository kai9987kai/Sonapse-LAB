"""PROJECT NEXUS-Ω: Controlled POMDP Benchmark Environment.

Implements a partially observable environment with transient hazard alert pulses
where memory retention is mathematically necessary for survival.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class POMDPPhaseWorld:
    """Canonical POMDP: Transient alert pulse at start of storm; state unobservable thereafter."""

    def __init__(self, seed: int = 12345, max_steps: int = 32, storm_interval: int = 6, storm_duration: int = 3):
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self.storm_interval = storm_interval
        self.storm_duration = storm_duration
        self.d_obs = 8
        self.n_actions = 4  # 0: Forage (Optimal during calm), 1: Shelter (Optimal during storm), 2: Explore, 3: Rest

        self.step_idx = 0
        self.cycle_step = 0
        self.energy = 1.0
        self.hydration = 1.0
        self.health = 1.0

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.cycle_step = int(self.rng.randint(0, self.storm_interval))
        self.energy = 1.0
        self.hydration = 1.0
        self.health = 1.0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.d_obs, dtype=np.float32)
        obs[0] = self.health
        obs[1] = self.energy
        obs[2] = self.hydration
        
        # Transient cue: A distinct alert pulse (1.0) appears ONLY at the first step of the storm!
        # For the remaining duration of the storm, the sensory cue drops back to ambient noise (0.0)!
        is_pulse = (self.cycle_step == (self.storm_interval - self.storm_duration))
        obs[3] = float(1.0 if is_pulse else self.rng.normal(0, 0.02))
        
        obs[4] = float(self.step_idx / float(self.max_steps))
        obs[5] = float(self.rng.normal(0, 0.05))
        obs[6] = float(self.rng.normal(0, 0.05))
        obs[7] = 1.0 if self.health > 0.2 else 0.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_idx += 1
        
        # Ground truth storm condition: last storm_duration steps of each cycle
        is_storm = (self.cycle_step >= (self.storm_interval - self.storm_duration))
        
        self.cycle_step = (self.cycle_step + 1) % self.storm_interval

        reward = 0.0
        death = False

        if action == 0:  # Forage
            if is_storm:
                self.health -= 0.40  # Catastrophic storm exposure
                reward = -0.6
            else:
                self.energy = min(1.0, self.energy + 0.25)
                reward = 0.4
        elif action == 1:  # Shelter
            if is_storm:
                reward = 0.35  # Protected during active storm
            else:
                self.energy -= 0.05
                reward = 0.05  # Inefficient sheltering during calm
        elif action == 2:  # Explore
            self.energy -= 0.1
            reward = 0.1
        elif action == 3:  # Rest
            self.energy = min(1.0, self.energy + 0.05)
            reward = 0.05

        # Basal metabolic consumption
        self.energy -= 0.05
        self.hydration -= 0.05
        if self.energy <= 0.0 or self.hydration <= 0.0:
            self.health -= 0.2

        if self.health <= 0.0:
            self.health = 0.0
            death = True
            reward -= 1.0

        done = death or (self.step_idx >= self.max_steps)

        info = {
            "step": self.step_idx,
            "cycle_step": self.cycle_step,
            "is_storm": is_storm,
            "death": death,
            "health": self.health,
        }

        return self._get_obs(), float(reward), done, info
