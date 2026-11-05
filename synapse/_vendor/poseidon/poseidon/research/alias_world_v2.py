"""Viable version of the observation-alias research world.

V1 permits high reward from one resource per hidden regime, but gives no action
that replenishes the other resource. At scarcity 1, every policy therefore
dies from a resource reaching zero by step 15, making long-horizon survival
comparisons uninformative. V2 adds a small maintenance yield to the mismatched
resource action while preserving the high-yield regime-specific reward.
"""
from __future__ import annotations

from typing import Any

from ..world import ACTIONS, N_ACTIONS, _integer
from .alias_world import ObservationAliasTidePool
from .contracts import ALIAS_WORLD_V2_VERSION


class ObservationAliasTidePoolV2(ObservationAliasTidePool):
    """V1 observation contract with viable low-yield resource maintenance."""

    VERSION = ALIAS_WORLD_V2_VERSION
    MAINTENANCE_YIELD = 0.12

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        action = _integer(action, "action", 0, N_ACTIONS - 1)
        if self.done:
            raise RuntimeError("Episode ended; call reset before stepping again")

        self.last_action = action
        reward = 0.0
        info: dict[str, Any] = {
            "version": self.VERSION,
            "tick": self.tick,
            "regime": self.regime,
            "is_aliased": self.is_aliased,
            "action": action,
            "action_name": ACTIONS[action],
            "death": False,
            "truncated": False,
        }

        metabolic_cost = 0.04 * self.scarcity
        self.energy = max(0.0, self.energy - metabolic_cost)
        self.hydration = max(0.0, self.hydration - metabolic_cost)
        self.stamina = min(1.0, self.stamina + 0.05)

        if action == 0:  # Rest
            self.stamina = min(1.0, self.stamina + 0.25)
            self.exposure = max(0.0, self.exposure - 0.05)
            reward = 0.05
        elif action == 1:  # Forage; even a poor patch yields maintenance energy
            self.stamina = max(0.0, self.stamina - 0.10)
            yield_amount = 0.35 if self.regime == 0 else self.MAINTENANCE_YIELD
            self.energy = min(1.0, self.energy + yield_amount)
            reward = 0.80 if self.regime == 0 else 0.0
            info["resource_harvested"] = "food_bloom" if self.regime == 0 else "energy_maintenance"
        elif action == 2:  # Drink; even brackish water yields maintenance hydration
            self.stamina = max(0.0, self.stamina - 0.10)
            yield_amount = 0.35 if self.regime == 1 else self.MAINTENANCE_YIELD
            self.hydration = min(1.0, self.hydration + yield_amount)
            reward = 0.80 if self.regime == 1 else 0.0
            info["resource_harvested"] = "water_spring" if self.regime == 1 else "hydration_maintenance"
        elif action == 3:  # Shelter
            self.exposure = max(0.0, self.exposure - 0.15)
            reward = 0.05
        elif action == 4:  # Explore
            self.stamina = max(0.0, self.stamina - 0.08)
            reward = 0.05
        elif action == 5:  # Flee
            self.threat = max(0.0, self.threat - 0.20)
            self.stamina = max(0.0, self.stamina - 0.15)
            reward = 0.02

        info.update({
            "health": float(self.health),
            "energy": float(self.energy),
            "hydration": float(self.hydration),
            "stamina": float(self.stamina),
            "exposure": float(self.exposure),
            "threat": float(self.threat),
        })

        if self.energy <= 0.0:
            self.health = 0.0
            self.death_reason = "starvation"
        elif self.hydration <= 0.0:
            self.health = 0.0
            self.death_reason = "dehydration"

        self.tick += 1
        self.total_reward += reward
        if self.health <= 0.0:
            self.done = True
            info["death"] = True
            info["death_reason"] = self.death_reason
        elif self.tick >= self.max_steps:
            self.done = True
            info["truncated"] = True

        next_obs = self.observe()
        return next_obs, float(reward), self.done, info
