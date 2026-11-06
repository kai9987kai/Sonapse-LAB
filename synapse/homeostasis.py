"""Supermix Beyond Homeostatic Physiological Drive Engine.

Adapted from Kai9987kai's Supermix Beyond / Poseidon WorldLab architecture.
Models continuous bodily drives and interoception:
1. **Energy**: Depletes with locomotion and basic metabolic rate; replenished by consuming food objects.
2. **Stamina**: Dynamic effort reserve; drains rapidly under high thrust/reaching and recharges during rest.
3. **Threat**: Proximity to hostile hazard zones or stalking predators.
4. **Exposure**: Ambient terrain friction and environmental volatility stress.
5. **Health**: Structural integrity; depleted by hazard impacts, regenerates when energy is充足.

Interoceptive states dynamically modulate neuromodulatory baselines (DA, 5-HT, OA, ACh)
and sensory salience, creating genuine survival-oriented goal dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from .actions import DRIVE, REACH, validate_action


@dataclass
class HomeostaticState:
    energy: float = 0.85
    stamina: float = 1.00
    threat: float = 0.00
    exposure: float = 0.10
    health: float = 1.00
    hunger: float = 0.15
    thirst: float = 0.10
    vitality: float = 0.95

    def to_array(self) -> np.ndarray:
        return np.array([
            self.energy,
            self.stamina,
            self.threat,
            self.exposure,
            self.health,
            self.hunger,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class HomeostaticEngine:
    """Manages internal physiology, drive dynamics, and interoceptive feedback."""

    def __init__(
        self,
        base_metabolism: float = 0.0008,
        stamina_burn_rate: float = 0.006,
        stamina_recovery_rate: float = 0.012,
        seed: int = 42,
    ):
        self.base_metabolism = base_metabolism
        self.stamina_burn = stamina_burn_rate
        self.stamina_recovery = stamina_recovery_rate
        self.rng = np.random.default_rng(seed)
        self.state = HomeostaticState()
        self.total_food_consumed = 0
        self.total_hazard_hits = 0

    def reset(self) -> HomeostaticState:
        """Resets homeostatic drives to nominal baseline."""
        self.state = HomeostaticState()
        return self.state

    def step(
        self,
        applied_action: np.ndarray,
        reward: float,
        hazard_dist: float = 10.0,
        consumed_food: bool = False,
        in_friction_zone: bool = False,
    ) -> HomeostaticState:
        """Updates physiological vitals based on actions, environmental stress, and rewards."""
        action = validate_action(applied_action)
        thrust = float(abs(action[DRIVE]))
        reach = float(abs(action[REACH]))

        # 1. Stamina Dynamics
        effort = thrust * 0.7 + reach * 0.3
        if effort > 0.15:
            self.state.stamina = max(0.0, self.state.stamina - effort * self.stamina_burn)
        else:
            self.state.stamina = min(1.0, self.state.stamina + self.stamina_recovery)

        # 2. Energy & Hunger Dynamics
        metabolic_cost = self.base_metabolism + effort * 0.0012
        if in_friction_zone:
            metabolic_cost *= 1.8
        self.state.energy = max(0.0, self.state.energy - metabolic_cost)

        if consumed_food or reward > 0.4:
            self.state.energy = min(1.0, self.state.energy + 0.35)
            self.total_food_consumed += 1

        self.state.hunger = float(np.clip(1.0 - self.state.energy, 0.0, 1.0))

        # 3. Threat Dynamics
        proximity_threat = float(np.clip((3.0 - hazard_dist) / 3.0, 0.0, 1.0)) if hazard_dist < 3.0 else 0.0
        self.state.threat = float(0.85 * self.state.threat + 0.15 * proximity_threat)

        # 4. Exposure & Friction Stress
        target_exposure = 0.6 if in_friction_zone else 0.1
        self.state.exposure = float(0.9 * self.state.exposure + 0.1 * target_exposure)

        # 5. Health Damage & Regeneration
        if reward < -0.2 or hazard_dist < 0.4:
            self.state.health = max(0.0, self.state.health - 0.15)
            self.total_hazard_hits += 1
        elif self.state.energy > 0.5 and self.state.threat < 0.2:
            self.state.health = min(1.0, self.state.health + 0.001)

        # Vitality summary score
        self.state.vitality = float(np.mean([
            self.state.energy,
            self.state.stamina,
            1.0 - self.state.threat,
            self.state.health
        ]))

        return self.state

    def modulate_neuromodulators(self, base_da: float, base_5ht: float, base_oa: float, base_ach: float) -> Tuple[float, float, float, float]:
        """Couples homeostatic drives into neuromodulatory field levels."""
        # Hunger amplifies Dopamine salience to food cues
        da = base_da * (1.0 + self.state.hunger * 0.8)

        # Threat drives Serotonin (threat vigilance) and Octopamine (arousal/escape vigor)
        sht = float(np.clip(base_5ht + self.state.threat * 0.6, 0.0, 2.0))
        oa = float(np.clip(base_oa + self.state.threat * 0.8 + (1.0 - self.state.stamina) * 0.2, 0.0, 2.0))

        # Exposure and surprise elevate Acetylcholine precision
        ach = float(np.clip(base_ach + self.state.exposure * 0.4, 0.0, 2.0))

        return da, sht, oa, ach

    def get_max_allowed_thrust(self) -> float:
        """Stamina depletion acts as a physical governor on maximum motor thrust."""
        if self.state.stamina < 0.1:
            return 0.35  # Exhaustion throttle
        if self.state.stamina < 0.3:
            return 0.65
        return 1.0

    def export_state(self) -> Dict[str, Any]:
        return {
            'state': self.state.to_dict(),
            'rng': self.rng.bit_generator.state,
            'total_food_consumed': self.total_food_consumed,
            'total_hazard_hits': self.total_hazard_hits,
        }

    def import_state(self, d: Dict[str, Any]) -> None:
        if 'state' in d:
            self.state = HomeostaticState(**d['state'])
        if 'rng' in d:
            self.rng.bit_generator.state = d['rng']
        self.total_food_consumed = d.get('total_food_consumed', 0)
        self.total_hazard_hits = d.get('total_hazard_hits', 0)

    def summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable homeostatic vitals telemetry."""
        return {
            'vitals': {k: round(float(v), 4) for k, v in asdict(self.state).items()},
            'food_consumed': self.total_food_consumed,
            'hazard_hits': self.total_hazard_hits,
            'max_allowed_thrust': round(self.get_max_allowed_thrust(), 3),
        }
