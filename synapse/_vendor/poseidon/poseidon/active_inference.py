"""POSEIDON ACTIVE INFERENCE: Free Energy Minimization & Homeostatic Agency.

Formulates decision making not as brute reward maximization, but as biological
Active Inference under the Free Energy Principle (FEP / Friston):
1. Target Homeostatic Prior p(s):
   The organism embodies a prior preference distribution over vital physiological states:
     \\mu_{target}: health=1.0, energy=0.9, hydration=0.9, stamina=0.9, threat=0.0, exposure=0.0
   weighted by homeostatic precision matrix \\Lambda_{target}.

2. Expected Free Energy (EFE) Decomposition:
   For any policy / imagined trajectory \\pi:
     G(\\pi) = \\underbrace{D_{KL}[q(s_\\tau | \\pi) || p(s_\\tau)]}_{\\text{Pragmatic / Instrumental Risk}}
              - \\underbrace{\\mathbb{E}_q[I(s_\\tau; o_\\tau | \\pi)]}_{\\text{Epistemic / Information Gain (Curiosity)}}

3. Dynamic Precision & Drive Transitions:
   - Satiated Regime: When physiological error is low, epistemic curiosity naturally
     drives mapping and exploration of novel/ambiguous patches.
   - Depleted / Threatened Regime: Homeostatic divergence explodes, forcing high-precision
     instrumental foraging and protective sheltering.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass
class HomeostaticTarget:
    """Target physiological vitals and precision weighting."""
    health: float = 1.0
    energy: float = 0.90
    hydration: float = 0.90
    stamina: float = 0.90
    exposure: float = 0.05
    threat: float = 0.05

    # Precision weights (inverse variance): how critically deviation is penalized
    w_health: float = 5.0
    w_energy: float = 3.0
    w_hydration: float = 3.5
    w_stamina: float = 1.5
    w_exposure: float = 2.0
    w_threat: float = 4.0


@dataclass(frozen=True)
class ActiveInferenceConfig:
    base_epistemic_weight: float = 0.40
    min_epistemic_weight: float = 0.05
    pragmatic_temperature: float = 1.0
    stamina_depletion_cost: float = 0.05
    active_inference_blend: float = 0.05


class ActiveInferenceEvaluator:
    """Computes Expected Free Energy (EFE) across candidate actions and imagined trajectories."""

    def __init__(
        self,
        config: ActiveInferenceConfig | None = None,
        target: HomeostaticTarget | None = None,
    ):
        self.config = config or ActiveInferenceConfig()
        self.target = target or HomeostaticTarget()

    def compute_pragmatic_value(self, state: Sequence[float]) -> float:
        """Instrumental log-likelihood under homeostatic prior p(s). High = near homeostasis."""
        s = list(state)
        d_health = (s[0] - self.target.health) ** 2 * self.target.w_health
        d_energy = (s[1] - self.target.energy) ** 2 * self.target.w_energy
        d_hydration = (s[2] - self.target.hydration) ** 2 * self.target.w_hydration
        d_stamina = (s[3] - self.target.stamina) ** 2 * self.target.w_stamina
        d_exposure = (s[4] - self.target.exposure) ** 2 * self.target.w_exposure
        d_threat = (s[5] - self.target.threat) ** 2 * self.target.w_threat

        # Negative Mahalanobis-like homeostatic distance
        total_divergence = d_health + d_energy + d_hydration + d_stamina + d_exposure + d_threat
        return -float(total_divergence)

    def compute_epistemic_drive(self, current_obs: Sequence[float]) -> float:
        """Modulates curiosity weight based on vital reserve safety."""
        health = max(0.0, min(1.0, float(current_obs[0])))
        energy = max(0.0, min(1.0, float(current_obs[1])))
        hydration = max(0.0, min(1.0, float(current_obs[2])))

        # When reserves are high, curiosity blooms; when reserves crash, curiosity halts
        vital_safety = health * min(energy, hydration)
        epistemic_weight = max(
            self.config.min_epistemic_weight,
            self.config.base_epistemic_weight * vital_safety,
        )
        return float(epistemic_weight)

    def compute_empowerment(
        self,
        predicted_outcomes: Sequence[Sequence[float]],
    ) -> float:
        """Calculates empowerment: degrees of future behavioral freedom.

        Empowerment is proportional to the dispersion (variance / volume) of
        reachable non-terminal future states across the available action choices.
        """
        valid_outcomes = [list(o) for o in predicted_outcomes if o[0] > 0.0]
        if not valid_outcomes:
            return 0.0

        n = len(valid_outcomes)
        if n == 1:
            return 0.05  # Minimal survival flexibility

        # Compute centroid across vital channels (health, energy, hydration, stamina)
        dim = min(4, len(valid_outcomes[0]))
        centroid = [sum(o[i] for o in valid_outcomes) / n for i in range(dim)]

        # Mean Euclidean dispersion
        total_disp = sum(
            math.sqrt(sum((o[i] - centroid[i]) ** 2 for i in range(dim)))
            for o in valid_outcomes
        ) / n

        # Scaled by fraction of surviving action pathways
        survival_fraction = n / len(predicted_outcomes)
        return float(survival_fraction * total_disp)

    def evaluate_expected_free_energy(
        self,
        current_obs: Sequence[float],
        predicted_obs: Sequence[float],
        calibrated_uncertainty: float,
        step_cost: float = 0.0,
        empowerment_value: float = 0.0,
    ) -> dict[str, float]:
        """Calculates EFE components: pragmatic divergence + epistemic info gain + empowerment."""
        pragmatic_val = self.compute_pragmatic_value(predicted_obs)
        epistemic_weight = self.compute_epistemic_drive(current_obs)

        # Epistemic value: expected information gain ~ model uncertainty U(s, a)
        epistemic_val = calibrated_uncertainty

        # Empowerment: future behavioral freedom weighted when not in immediate crisis
        vital_safety = max(0.0, min(1.0, float(current_obs[0]) * min(float(current_obs[1]), float(current_obs[2]))))
        empowerment_weight = 0.25 * vital_safety

        # Free Energy Objective to be maximized
        free_energy_objective = (
            pragmatic_val
            + epistemic_weight * epistemic_val
            + empowerment_weight * empowerment_value
            - step_cost
        )

        return {
            "free_energy_objective": float(free_energy_objective),
            "pragmatic_value": float(pragmatic_val),
            "epistemic_value": float(epistemic_val),
            "epistemic_weight": float(epistemic_weight),
            "empowerment_value": float(empowerment_value),
            "empowerment_weight": float(empowerment_weight),
            "homeostatic_divergence": float(-pragmatic_val),
        }
