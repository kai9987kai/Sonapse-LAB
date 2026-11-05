"""POSEIDON OBSERVATION ALIAS WORLD: Controlled POMDP with Delayed Temporal Cues.

An explicitly versioned synthetic research environment adhering strictly to
the 16-dimensional observation contract and 6-action space of TidePool.

Core Scientific Design:
1. Two Latent Environmental Regimes:
   - Regime 0 (Algae Bloom): Foraging (action 1) yields high reward (+0.80) and energy (+0.35);
     Drinking (action 2) yields zero return and wastes energy.
   - Regime 1 (Freshwater Spring): Drinking (action 2) yields high reward (+0.80) and hydration (+0.35);
     Foraging (action 1) yields zero return and wastes energy.
2. Temporal Observation Alias:
   - During the initial cue window (t < cue_horizon, e.g. t < 4), channel 13 ('resource_scent')
     displays an unambiguous regime cue: 0.85 for Regime 0, 0.15 for Regime 1.
   - At t >= cue_horizon, the cue vanishes into ambient neutral reading (0.50), and resource
     channels (channels 6 and 7) display identical ambiguous local traces (0.35).
   - Thus, any purely reactive or Markovian policy receiving only o_t cannot distinguish
     Regime 0 from Regime 1, making optimal action choice impossible from o_t alone.
3. Causal Disambiguation via Episodic Memory:
   - An agent remembering the earlier cue transition can disambiguate the regime.
   - However, naive similarity retrieval matches o_t against other aliased steps, which
     retrieves contradictory actions or uninformative states.
   - Decision-Value-Gated Episodic Memory identifies and admits only the past cue
     that carries positive decision value advantage.
4. Exact Deterministic Branching & Restoration:
   - Fully supports snapshot() and restore() for counterfactual multi-horizon synthetic rollouts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..world import ACTIONS, N_ACTIONS, OBSERVATION_NAMES, OBS_SIZE, _integer, _number, _unit, event_uniform
from .contracts import ALIAS_WORLD_VERSION, CONTRACT_VERSION, EnvironmentContract, alias_tidepool_contract

GRID_SIZE = 6
DEFAULT_CUE_HORIZON = 4


@dataclass(frozen=True)
class AliasWorldConfig:
    scarcity: float = 1.0
    max_steps: int = 64
    cue_horizon: int = DEFAULT_CUE_HORIZON
    sensor_noise_std: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if not 0.5 <= self.scarcity <= 4.0:
            raise ValueError("scarcity must be in [0.5, 4.0]")
        if not 1 <= self.max_steps <= 10000:
            raise ValueError("max_steps must be in [1, 10000]")
        if not 1 <= self.cue_horizon <= self.max_steps:
            raise ValueError("cue_horizon must be in [1, max_steps]")
        if not 0.0 <= self.sensor_noise_std <= 0.5:
            raise ValueError("sensor_noise_std must be in [0.0, 0.5]")


class ObservationAliasTidePool:
    """Versioned research environment featuring temporal observation aliases."""

    VERSION = ALIAS_WORLD_VERSION

    def __init__(
        self,
        seed: int = 0,
        scarcity: float = 1.0,
        max_steps: int = 64,
        cue_horizon: int = DEFAULT_CUE_HORIZON,
        sensor_noise_std: float = 0.0,
        forced_regime: int | None = None,
    ):
        self.config = AliasWorldConfig(
            scarcity=_number(scarcity, "scarcity", 0.5, 4.0),
            max_steps=_integer(max_steps, "max_steps", 1, 10000),
            cue_horizon=_integer(cue_horizon, "cue_horizon", 1, max_steps),
            sensor_noise_std=_number(sensor_noise_std, "sensor_noise_std", 0.0, 0.5),
            seed=_integer(seed, "seed", 0, 2**63 - 1),
        )
        self.scarcity = self.config.scarcity
        self.max_steps = self.config.max_steps
        self.cue_horizon = self.config.cue_horizon
        self.sensor_noise_std = self.config.sensor_noise_std
        self._forced_regime = forced_regime
        self.reset(seed)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        else:
            self.seed = getattr(self, "seed", self.config.seed)

        self.tick = 0
        if self._forced_regime is not None:
            self.regime = _integer(self._forced_regime, "forced_regime", 0, 1)
        else:
            # Deterministic regime derivation from episode seed
            self.regime = 0 if event_uniform(self.seed, 0, "regime", "bloom_vs_spring") < 0.5 else 1

        self.health = 1.0
        self.energy = 0.60
        self.hydration = 0.60
        self.stamina = 0.80
        self.exposure = 0.10
        self.threat = 0.05
        self.last_action = 0
        self.total_reward = 0.0
        self.done = False
        self.death_reason: str | None = None
        self.rng = random.Random(self.seed)
        return self.observe()

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def is_aliased(self) -> bool:
        """True if currently in the observation-aliased window where sensory cues vanish."""
        return self.tick >= self.cue_horizon

    def observe(self) -> list[float]:
        """Constructs 16-D normalized observation vector."""
        # Channels:
        # 0: health, 1: energy, 2: hydration, 3: stamina, 4: exposure, 5: threat
        # 6: food, 7: water, 8: shelter, 9: weather_severity, 10: temperature, 11: daylight
        # 12: terrain_difficulty, 13: resource_scent (CUE CHANNEL), 14: last_action_scaled, 15: progress
        if not self.is_aliased:
            # Clear temporal cue during early ticks
            scent = 0.85 if self.regime == 0 else 0.15
            food_vis = 0.50 if self.regime == 0 else 0.20
            water_vis = 0.20 if self.regime == 0 else 0.50
        else:
            # Scent occluded; identical ambient reading regardless of regime
            scent = 0.50
            food_vis = 0.35
            water_vis = 0.35

        shelter = 0.50
        severity = 0.10
        temperature = 0.50
        daylight = (1.0 + math.sin(self.tick * math.tau / 24)) / 2.0
        terrain = 0.20

        raw_obs = [
            _unit(self.health),
            _unit(self.energy),
            _unit(self.hydration),
            _unit(self.stamina),
            _unit(self.exposure),
            _unit(self.threat),
            _unit(food_vis),
            _unit(water_vis),
            _unit(shelter),
            _unit(severity),
            _unit(temperature),
            _unit(daylight),
            _unit(terrain),
            _unit(scent),
            _unit(self.last_action / 5.0),
            _unit(self.tick / float(self.max_steps)),
        ]

        if self.sensor_noise_std > 0.0:
            noisy_obs = []
            for idx, val in enumerate(raw_obs):
                if idx in (14, 15):
                    noisy_obs.append(val)
                else:
                    noise = self.rng.gauss(0.0, self.sensor_noise_std)
                    noisy_obs.append(_unit(val + noise))
            return noisy_obs

        return raw_obs

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        action = _integer(action, "action", 0, N_ACTIONS - 1)
        if self.done:
            raise RuntimeError("Episode ended; call reset before stepping again")

        self.last_action = action
        reward = 0.0
        info: dict[str, Any] = {
            "tick": self.tick,
            "regime": self.regime,
            "is_aliased": self.is_aliased,
            "action": action,
            "action_name": ACTIONS[action],
            "death": False,
            "truncated": False,
        }

        # Metabolic decay per step
        self.energy = max(0.0, self.energy - 0.04 * self.scarcity)
        self.hydration = max(0.0, self.hydration - 0.04 * self.scarcity)
        self.stamina = min(1.0, self.stamina + 0.05)

        # Action Execution with Causal Regime Divergence
        if action == 0:  # Rest
            self.stamina = min(1.0, self.stamina + 0.25)
            self.exposure = max(0.0, self.exposure - 0.05)
            reward = 0.05
        elif action == 1:  # Forage
            self.stamina = max(0.0, self.stamina - 0.10)
            if self.regime == 0:
                # Algae Bloom: high food yield
                self.energy = min(1.0, self.energy + 0.35)
                reward = 0.80
                info["resource_harvested"] = "food_bloom"
            else:
                # Freshwater Spring: dormant food
                reward = -0.05
                info["resource_harvested"] = "none_brackish"
        elif action == 2:  # Drink
            self.stamina = max(0.0, self.stamina - 0.10)
            if self.regime == 1:
                # Freshwater Spring: high water yield
                self.hydration = min(1.0, self.hydration + 0.35)
                reward = 0.80
                info["resource_harvested"] = "water_spring"
            else:
                # Algae Bloom: saline/dormant water
                reward = -0.05
                info["resource_harvested"] = "none_saline"
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

        # Check Vitals and Survival
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
        return next_obs, reward, self.done, info

    def snapshot(self) -> dict[str, Any]:
        """Captures exact environment state for counterfactual branch replay."""
        return {
            "version": self.VERSION,
            "seed": self.seed,
            "tick": self.tick,
            "regime": self.regime,
            "cue_horizon": self.cue_horizon,
            "health": self.health,
            "energy": self.energy,
            "hydration": self.hydration,
            "stamina": self.stamina,
            "exposure": self.exposure,
            "threat": self.threat,
            "last_action": self.last_action,
            "total_reward": self.total_reward,
            "done": self.done,
            "death_reason": self.death_reason,
            "sensor_noise_std": self.sensor_noise_std,
            "rng_state": copy.deepcopy(self.rng.getstate()),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restores exact environment state from a prior snapshot."""
        if snap.get("version") != self.VERSION:
            raise ValueError(f"Snapshot version mismatch: expected {self.VERSION}, got {snap.get('version')}")
        self.seed = snap["seed"]
        self.tick = snap["tick"]
        self.regime = snap["regime"]
        self.cue_horizon = snap["cue_horizon"]
        self.health = snap["health"]
        self.energy = snap["energy"]
        self.hydration = snap["hydration"]
        self.stamina = snap["stamina"]
        self.exposure = snap["exposure"]
        self.threat = snap["threat"]
        self.last_action = snap["last_action"]
        self.total_reward = snap["total_reward"]
        self.done = snap["done"]
        self.death_reason = snap["death_reason"]
        self.sensor_noise_std = snap["sensor_noise_std"]
        self.rng = random.Random()
        self.rng.setstate(copy.deepcopy(snap["rng_state"]))
