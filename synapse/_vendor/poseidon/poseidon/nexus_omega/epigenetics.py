"""PROJECT NEXUS-Ω: Epigenetic Regulation & Developmental Morphogenesis.

Bio-inspired epigenetic hyperparameter modulation, developmental life stages,
and verifiable ecdysis (molting) transitions.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from poseidon.nexus_omega.contracts import (
    DevelopmentalStage,
    EpigeneticProfile,
    ExuviaShell,
)


EPIGENETIC_PROFILES: Dict[DevelopmentalStage, EpigeneticProfile] = {
    DevelopmentalStage.PROTOCELL: EpigeneticProfile(
        stage=DevelopmentalStage.PROTOCELL,
        basal_metabolism=1.00,
        sensory_gain=1.00,
        risk_penalty_scale=1.00,
        plasticity_rate=0.10,
        mobility_efficiency=1.00,
        receptive_horizon=1.00,
    ),
    DevelopmentalStage.LARVAL_FORAGER: EpigeneticProfile(
        stage=DevelopmentalStage.LARVAL_FORAGER,
        basal_metabolism=0.85,
        sensory_gain=1.40,
        risk_penalty_scale=0.70,
        plasticity_rate=0.08,
        mobility_efficiency=0.90,
        receptive_horizon=1.00,
    ),
    DevelopmentalStage.PUPAL_DIAPAUSE: EpigeneticProfile(
        stage=DevelopmentalStage.PUPAL_DIAPAUSE,
        basal_metabolism=0.35,  # Metabolic arrest saves energy during storms
        sensory_gain=0.40,
        risk_penalty_scale=1.60,
        plasticity_rate=0.01,
        mobility_efficiency=2.20,
        receptive_horizon=0.50,
    ),
    DevelopmentalStage.IMAGO_NAVIGATOR: EpigeneticProfile(
        stage=DevelopmentalStage.IMAGO_NAVIGATOR,
        basal_metabolism=1.15,
        sensory_gain=1.10,
        risk_penalty_scale=1.00,
        plasticity_rate=0.04,
        mobility_efficiency=0.65,  # Efficient dispersal locomotion
        receptive_horizon=1.75,
    ),
    DevelopmentalStage.SENESCENT_ARCHIVE: EpigeneticProfile(
        stage=DevelopmentalStage.SENESCENT_ARCHIVE,
        basal_metabolism=0.60,
        sensory_gain=0.80,
        risk_penalty_scale=2.00,
        plasticity_rate=0.001,  # Crystallized memories, no catastrophic drift
        mobility_efficiency=1.50,
        receptive_horizon=2.00,
    ),
}


class EpigeneticMorphogenesisSystem:
    """Online Epigenetic Developmental Transition Controller."""

    def __init__(
        self,
        biomass_threshold: float = 2.0,
        pupa_duration_ticks: int = 4,
        imago_duration_ticks: int = 24,
        exposure_index: Optional[int] = None,
    ):
        if exposure_index is not None and (type(exposure_index) is not int or exposure_index < 0):
            raise ValueError("exposure_index must be a non-negative integer or None")
        self.biomass_threshold = biomass_threshold
        self.pupa_duration_ticks = pupa_duration_ticks
        self.imago_duration_ticks = imago_duration_ticks
        self.exposure_index = exposure_index
        self.reset()

    def reset(self) -> None:
        self.stage = DevelopmentalStage.PROTOCELL
        self.step_counter = 0
        self.stage_ticks = 0
        self.biomass_accumulated = 0.0
        self.stress_counter = 0
        self.exuviae: List[ExuviaShell] = []

    @property
    def current_profile(self) -> EpigeneticProfile:
        return EPIGENETIC_PROFILES[self.stage]

    def update_development(self, observation: np.ndarray | Sequence[float]) -> Optional[ExuviaShell]:
        """Processes vital state to advance developmental clocks and trigger ecdysis."""
        obs = np.asarray(observation, dtype=np.float32)
        self.step_counter += 1
        self.stage_ticks += 1

        health = float(obs[0]) if len(obs) > 0 else 1.0
        energy = float(obs[1]) if len(obs) > 1 else 1.0
        hydration = float(obs[2]) if len(obs) > 2 else 1.0
        stamina = float(obs[3]) if len(obs) > 3 else 1.0
        exposure = (
            float(obs[self.exposure_index])
            if self.exposure_index is not None and self.exposure_index < len(obs)
            else 0.0
        )

        if energy > 0.70 and hydration > 0.70:
            self.biomass_accumulated += (energy - 0.50) * 0.35

        if exposure > 0.60:
            self.stress_counter += 1

        ecdysis_event: Optional[ExuviaShell] = None

        # State transition machine
        if self.stage == DevelopmentalStage.PROTOCELL:
            if self.step_counter >= 3 and health > 0.8:
                ecdysis_event = self._transition_to(DevelopmentalStage.LARVAL_FORAGER, health, stamina)

        elif self.stage == DevelopmentalStage.LARVAL_FORAGER:
            # Trigger pupation when biomass reaches threshold OR extreme storm stress
            if (self.biomass_accumulated >= self.biomass_threshold and stamina >= 0.35) or self.stress_counter >= 3:
                ecdysis_event = self._transition_to(DevelopmentalStage.PUPAL_DIAPAUSE, health, stamina)

        elif self.stage == DevelopmentalStage.PUPAL_DIAPAUSE:
            if self.stage_ticks >= self.pupa_duration_ticks:
                ecdysis_event = self._transition_to(DevelopmentalStage.IMAGO_NAVIGATOR, health, stamina)

        elif self.stage == DevelopmentalStage.IMAGO_NAVIGATOR:
            if self.stage_ticks >= self.imago_duration_ticks:
                ecdysis_event = self._transition_to(DevelopmentalStage.SENESCENT_ARCHIVE, health, stamina)

        return ecdysis_event

    def _transition_to(
        self,
        new_stage: DevelopmentalStage,
        health: float,
        stamina: float,
    ) -> ExuviaShell:
        shell = ExuviaShell(
            from_stage=self.stage,
            to_stage=new_stage,
            step=self.step_counter,
            biomass_accumulated=self.biomass_accumulated,
            health_at_ecdysis=health,
            stamina_at_ecdysis=stamina,
        )
        self.exuviae.append(shell)
        self.stage = new_stage
        self.stage_ticks = 0
        return shell

    def filter_observation(self, observation: np.ndarray) -> np.ndarray:
        """Modulates observation vector based on life stage sensory profile."""
        obs = observation.copy()
        profile = self.current_profile
        # Amplify/attenuate local resource scent cues by sensory_gain
        if len(obs) >= 8:
            obs[6] = np.clip(obs[6] * profile.sensory_gain, 0.0, 1.0)
            obs[7] = np.clip(obs[7] * profile.sensory_gain, 0.0, 1.0)
        return obs

    def modulate_action_potentials(self, action_scores: np.ndarray) -> np.ndarray:
        """Biases candidate action scores based on developmental constraints."""
        scores = action_scores.copy()
        profile = self.current_profile

        if self.stage == DevelopmentalStage.PUPAL_DIAPAUSE:
            # Diapause metabolic arrest: heavily boost Rest (action 0), penalize movement (action 4)
            scores[0] += 0.45
            scores[4] -= 0.35
        elif self.stage == DevelopmentalStage.LARVAL_FORAGER:
            # Larval instar: boost Forage (action 1) and Drink (action 2)
            scores[1] += 0.20
            scores[2] += 0.20
        elif self.stage == DevelopmentalStage.IMAGO_NAVIGATOR:
            # Imago adult: boost Explore (action 4)
            scores[4] += 0.15

        return scores
