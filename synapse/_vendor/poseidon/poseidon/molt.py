"""MOLT: Metamorphic Online Life-Stage Transmutation & Morphogenesis.

Synthesizing bio-inspired developmental transitions, ecdysis, and life-stage
specialization for the TidePool environment without altering frozen core weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class InstarStage(str, Enum):
    LARVAL = "larval"      # High foraging efficiency, acute local sensitivity, exposure vulnerability
    PUPA = "pupa"          # Diapause metabolic arrest, shell hardening, internal remodeling
    IMAGO = "imago"        # Adult dispersal, expanded receptive horizon, high-speed travel


@dataclass(frozen=True)
class MorphologicalProfile:
    stage: InstarStage
    sensory_gain: float          # Perception scaling on local resource signals
    exposure_vulnerability: float # Vulnerability multiplier to environmental hazards
    forage_bonus: float          # Additional energy gain from foraging
    basal_metabolic_cost: float  # Multiplier on per-tick energy expenditure
    mobility_stamina_cost: float # Stamina cost per translation action
    receptive_horizon: float     # Spatial perception radius multiplier


PROFILES: Dict[InstarStage, MorphologicalProfile] = {
    InstarStage.LARVAL: MorphologicalProfile(
        stage=InstarStage.LARVAL,
        sensory_gain=1.35,
        exposure_vulnerability=1.25,
        forage_bonus=0.25,
        basal_metabolic_cost=0.85,
        mobility_stamina_cost=0.90,
        receptive_horizon=1.0,
    ),
    InstarStage.PUPA: MorphologicalProfile(
        stage=InstarStage.PUPA,
        sensory_gain=0.50,
        exposure_vulnerability=0.35,  # Chitinous hardening against exposure
        forage_bonus=0.0,
        basal_metabolic_cost=0.40,   # Deep diapause metabolic reduction
        mobility_stamina_cost=2.00,  # Highly suppressed movement
        receptive_horizon=0.5,
    ),
    InstarStage.IMAGO: MorphologicalProfile(
        stage=InstarStage.IMAGO,
        sensory_gain=1.10,
        exposure_vulnerability=0.85,
        forage_bonus=0.10,
        basal_metabolic_cost=1.20,   # Higher adult dispersal metabolism
        mobility_stamina_cost=0.65,  # Aerodynamic/hydrodynamic locomotion efficiency
        receptive_horizon=1.75,
    ),
}


class ExuviaRecord:
    """Cryptographic record of an ecdysis event shedding an old morphological shell."""

    def __init__(
        self,
        from_stage: InstarStage,
        to_stage: InstarStage,
        step: int,
        cumulative_intake: float,
        health: float,
        stamina: float,
    ):
        self.from_stage = from_stage
        self.to_stage = to_stage
        self.step = step
        self.cumulative_intake = round(float(cumulative_intake), 4)
        self.health = round(float(health), 4)
        self.stamina = round(float(stamina), 4)
        
        payload = json.dumps({
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "step": self.step,
            "cumulative_intake": self.cumulative_intake,
            "health": self.health,
            "stamina": self.stamina,
        }, sort_keys=True)
        self.digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def exuvia_sha256(self) -> str:
        return self.digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "step": self.step,
            "cumulative_intake": self.cumulative_intake,
            "health": self.health,
            "stamina": self.stamina,
            "exuvia_sha256": self.digest,
        }


class MoltEngine:
    """Online morphological transmutation controller for adaptive life-stage transitions."""

    def __init__(
        self,
        larval_intake_threshold: float = 3.0,
        pupa_duration_steps: int = 4,
    ):
        self.larval_intake_threshold = float(larval_intake_threshold)
        self.pupa_duration_steps = int(pupa_duration_steps)
        self.reset()

    def reset(self) -> None:
        self.stage = InstarStage.LARVAL
        self.step_count = 0
        self.pupa_ticks = 0
        self.cumulative_intake = 0.0
        self.exuviae: List[ExuviaRecord] = []
        self.last_action: Optional[int] = None

    @property
    def current_profile(self) -> MorphologicalProfile:
        return PROFILES[self.stage]

    def update_development(self, observation: List[float] | np.ndarray) -> Optional[ExuviaRecord]:
        """Update developmental biomass and check for ecdysis triggers."""
        obs = [float(x) for x in observation]
        self.step_count += 1
        
        # Energy and health indices in standard 16-d TidePool observation:
        # obs[0]: health, obs[1]: energy, obs[2]: hydration, obs[3]: stamina
        health = obs[0] if len(obs) > 0 else 1.0
        energy = obs[1] if len(obs) > 1 else 1.0
        stamina = obs[3] if len(obs) > 3 else 1.0

        if energy > 0.8:
            self.cumulative_intake += (energy - 0.7) * 0.1

        exuvia: Optional[ExuviaRecord] = None

        if self.stage == InstarStage.LARVAL:
            # Trigger pupation when biomass accumulation threshold is fulfilled
            if self.cumulative_intake >= self.larval_intake_threshold and stamina >= 0.5:
                exuvia = ExuviaRecord(
                    from_stage=self.stage,
                    to_stage=InstarStage.PUPA,
                    step=self.step_count,
                    cumulative_intake=self.cumulative_intake,
                    health=health,
                    stamina=stamina,
                )
                self.exuviae.append(exuvia)
                self.stage = InstarStage.PUPA
                self.pupa_ticks = 0

        elif self.stage == InstarStage.PUPA:
            self.pupa_ticks += 1
            if self.pupa_ticks >= self.pupa_duration_steps:
                exuvia = ExuviaRecord(
                    from_stage=self.stage,
                    to_stage=InstarStage.IMAGO,
                    step=self.step_count,
                    cumulative_intake=self.cumulative_intake,
                    health=health,
                    stamina=stamina,
                )
                self.exuviae.append(exuvia)
                self.stage = InstarStage.IMAGO

        return exuvia

    def filter_observation(self, observation: List[float] | np.ndarray) -> np.ndarray:
        """Modulate the observation according to current life-stage sensory profile."""
        obs = np.array(observation, dtype=np.float32).copy()
        profile = self.current_profile

        # Modulate food/water proximity by sensory gain
        if len(obs) >= 7:
            # obs[4]: exposure, obs[5]: food_dist, obs[6]: water_dist
            obs[4] = float(np.clip(obs[4] * profile.exposure_vulnerability, 0.0, 1.0))
            obs[5] = float(np.clip(obs[5] / max(profile.sensory_gain, 1e-4), 0.0, 1.0))
            obs[6] = float(np.clip(obs[6] / max(profile.sensory_gain, 1e-4), 0.0, 1.0))

        return obs

    def modulate_action(
        self,
        candidate_action: int,
        homeostatic_drives: Dict[str, float],
    ) -> Tuple[int, str]:
        """Modulate or override motor outputs based on life-stage constraints."""
        profile = self.current_profile

        if self.stage == InstarStage.PUPA:
            # In pupation diapause, active movement is suppressed in favor of rest/quiescence
            if candidate_action in (2, 3, 4):  # Movement translations: left, right, down
                return 0, "pupa_diapause_rest"  # Rest action
            return candidate_action, "pupa_permissive"

        elif self.stage == InstarStage.LARVAL:
            # In larval instar, acute metabolic priority heavily reinforces foraging
            if homeostatic_drives.get("metabolic", 0.0) > 0.4 and candidate_action not in (1, 5):
                # Favor foraging/gathering
                return 5, "larval_forage_drive"
            return candidate_action, "larval_policy"

        elif self.stage == InstarStage.IMAGO:
            # In adult imago, dispersal mobility is privileged when threat or depletion strikes
            if homeostatic_drives.get("threat", 0.0) > 0.6:
                return 1, "imago_dispersal_flight"
            return candidate_action, "imago_policy"

        return candidate_action, "molt_default"

    def status(self) -> Dict[str, Any]:
        return {
            "current_stage": self.stage.value,
            "step_count": self.step_count,
            "cumulative_intake": round(self.cumulative_intake, 4),
            "pupa_ticks": self.pupa_ticks,
            "exuviae_count": len(self.exuviae),
            "profile": asdict(self.current_profile),
        }
