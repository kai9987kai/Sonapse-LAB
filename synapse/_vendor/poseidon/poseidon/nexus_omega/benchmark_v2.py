"""Versioned viable POMDP phase world for NEXUS-Ω research.

The legacy benchmark has no action that replenishes hydration. This version
preserves its observation and action dimensions while adding bounded hydration
recovery to Rest, allowing survival comparisons without changing old receipts.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from poseidon.nexus_omega.benchmark import POMDPPhaseWorld


class POMDPPhaseWorldV2(POMDPPhaseWorld):
    """Phase world v2; same 8 observations/4 actions with sustainable Rest."""

    VERSION = "nexus-phaseworld-v2"
    REST_HYDRATION_RECOVERY = 0.12

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if type(action) is not int or not 0 <= action < self.n_actions:
            raise ValueError("action must be an integer in [0, 3]")

        self.step_idx += 1
        is_storm = self.cycle_step >= (self.storm_interval - self.storm_duration)
        self.cycle_step = (self.cycle_step + 1) % self.storm_interval

        reward = 0.0
        death = False
        if action == 0:  # Forage
            if is_storm:
                self.health -= 0.40
                reward = -0.60
            else:
                self.energy = min(1.0, self.energy + 0.25)
                reward = 0.40
        elif action == 1:  # Shelter
            if is_storm:
                reward = 0.35
            else:
                self.energy -= 0.05
                reward = 0.05
        elif action == 2:  # Explore
            self.energy -= 0.10
            reward = 0.10
        elif action == 3:  # Rest; versioned hydration recovery is the v2 change
            self.energy = min(1.0, self.energy + 0.05)
            self.hydration = min(1.0, self.hydration + self.REST_HYDRATION_RECOVERY)
            reward = 0.05

        # Keep the legacy metabolic drain and death rule unchanged.
        self.energy -= 0.05
        self.hydration -= 0.05
        if self.energy <= 0.0 or self.hydration <= 0.0:
            self.health -= 0.20

        if self.health <= 0.0:
            self.health = 0.0
            death = True
            reward -= 1.0

        done = death or self.step_idx >= self.max_steps
        info = {
            "version": self.VERSION,
            "step": self.step_idx,
            "cycle_step": self.cycle_step,
            "is_storm": is_storm,
            "action": action,
            "death": death,
            "truncated": bool(not death and self.step_idx >= self.max_steps),
            "health": float(self.health),
            "energy": float(self.energy),
            "hydration": float(self.hydration),
            "rest_hydration_recovery": self.REST_HYDRATION_RECOVERY if action == 3 else 0.0,
        }
        return self._get_obs(), float(reward), bool(done), info
