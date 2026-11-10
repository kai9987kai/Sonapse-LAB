"""Allostatic Load and Chronic Physiological Stress Engine.

Scientific Foundations:
1. McEwen (N. Engl. J. Med. 1998; Ann. N.Y. Acad. Sci. 2004):
   Allostasis represents the biological process of maintaining stability (homeostasis)
   through physiological change. Allostatic Load (AL) is the cumulative wear-and-tear
   on body and brain resulting from chronic overactivity or dysregulation of
   physiological stress response systems.
2. Sterling & Eyer (1988): Allostatic mechanisms adjust setpoints dynamically in
   response to environmental demands (predictive regulation).
3. Primary Mediators: Glucocorticoids (cortisol), catecholamines (adrenaline/noradrenaline),
   metabolic hormones. Sustained elevation causes HPA axis exhaustion (blunted acute
   reactivity with elevated baseline wear).
4. Secondary Outcomes: Cellular strain, accelerated metabolic drain, sleep debt.
5. Tertiary Outcomes: Allostatic overload (Type 1: energy deficit vs demand; Type 2: chronic
   psychosocial/predation stress), reduced longevity, and chromatin remodeling.

Governing Invariant:
When allostatic_load_enabled == False:
Ecology agent dynamics and phenotypes match legacy behavior bit-for-bit.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclasses.dataclass
class AllostaticProfile:
    agent_id: str
    allostatic_load_index: float = 0.0  # Normalized [0..1]
    cortisol_auc: float = 0.0           # Cumulative area-under-the-curve
    adrenaline_spikes: int = 0          # Number of acute fright/alarm pulses
    metabolic_strain: float = 0.0       # Cumulative starvation / energy depletion strain
    kinematic_wear: float = 0.0         # Locomotor mechanical exertion
    hpa_exhaustion: float = 0.0         # Blunting of acute cortisol reactivity [0..1]
    is_overloaded: bool = False         # Whether ALI exceeds critical threshold
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "allostatic_load_index": round(float(self.allostatic_load_index), 4),
            "cortisol_auc": round(float(self.cortisol_auc), 4),
            "adrenaline_spikes": int(self.adrenaline_spikes),
            "metabolic_strain": round(float(self.metabolic_strain), 4),
            "kinematic_wear": round(float(self.kinematic_wear), 4),
            "hpa_exhaustion": round(float(self.hpa_exhaustion), 4),
            "is_overloaded": bool(self.is_overloaded),
            "generation": int(self.generation),
        }


@dataclasses.dataclass
class AllostaticPopulationStats:
    mean_ali: float = 0.0
    median_ali: float = 0.0
    max_ali: float = 0.0
    overloaded_count: int = 0
    overloaded_fraction: float = 0.0
    mean_hpa_exhaustion: float = 0.0
    total_tracked_agents: int = 0
    evidence_kind: str = "synthetic_simulation"
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ali": round(float(self.mean_ali), 4),
            "median_ali": round(float(self.median_ali), 4),
            "max_ali": round(float(self.max_ali), 4),
            "overloaded_count": int(self.overloaded_count),
            "overloaded_fraction": round(float(self.overloaded_fraction), 4),
            "mean_hpa_exhaustion": round(float(self.mean_hpa_exhaustion), 4),
            "total_tracked_agents": int(self.total_tracked_agents),
            "evidence_kind": self.evidence_kind,
            "timestamp": round(float(self.timestamp), 3),
        }


class AllostaticLoadEngine:
    """Computes cumulative wear-and-tear and allostatic overload across virtual ecology agents."""

    def __init__(
        self,
        decay_rate: float = 0.002,           # Natural physiological clearance per step
        overload_threshold: float = 0.60,     # ALI threshold for allostatic overload
        cortisol_weight: float = 0.35,
        adrenaline_weight: float = 0.25,
        metabolic_weight: float = 0.25,
        kinematic_weight: float = 0.15,
        seed: int = 42,
    ):
        self.decay_rate = float(decay_rate)
        self.overload_threshold = float(overload_threshold)
        self.cortisol_weight = float(cortisol_weight)
        self.adrenaline_weight = float(adrenaline_weight)
        self.metabolic_weight = float(metabolic_weight)
        self.kinematic_weight = float(kinematic_weight)
        self.rng = np.random.default_rng(seed)

        self.profiles: Dict[str, AllostaticProfile] = {}
        self._last_stats: Optional[AllostaticPopulationStats] = None

    def get_or_create_profile(self, agent_id: str, generation: int = 0) -> AllostaticProfile:
        if agent_id not in self.profiles:
            self.profiles[agent_id] = AllostaticProfile(agent_id=agent_id, generation=generation)
        return self.profiles[agent_id]

    def remove_dead_agent(self, agent_id: str) -> None:
        self.profiles.pop(agent_id, None)

    def update_agent(
        self,
        profile: AllostaticProfile,
        cortisol: float,
        adrenaline: float,
        energy: float,
        speed_magnitude: float,
        dt: float = 0.1,
    ) -> float:
        """Updates allostatic profile for a single agent across acute stressors.

        Returns:
            Calculated instantaneous Allostatic Load Index (ALI) in [0..1].
        """
        # 1. Primary mediator accumulation: Cortisol area-under-the-curve
        effective_cortisol = max(0.0, float(cortisol))
        profile.cortisol_auc = max(0.0, profile.cortisol_auc * (1.0 - self.decay_rate) + effective_cortisol * dt * 0.10)

        # 2. Adrenaline acute spikes
        if adrenaline > 0.6:
            profile.adrenaline_spikes += 1

        # 3. Secondary mediator: Metabolic strain (starvation depletion)
        starvation = max(0.0, (50.0 - float(energy)) / 50.0) if float(energy) < 50.0 else 0.0
        profile.metabolic_strain = max(0.0, profile.metabolic_strain * (1.0 - self.decay_rate) + starvation * dt * 0.12)

        # 4. Secondary mediator: Kinematic mechanical wear (speed squared)
        v_wear = (float(speed_magnitude) ** 2) * 0.02
        profile.kinematic_wear = max(0.0, profile.kinematic_wear * (1.0 - self.decay_rate * 2.0) + v_wear * dt * 0.04)

        # 5. Composite Allostatic Load Index (ALI) via non-linear saturation
        c_term = 1.0 - math.exp(-profile.cortisol_auc * 2.2)
        a_term = 1.0 - math.exp(-profile.adrenaline_spikes * 0.15)
        m_term = 1.0 - math.exp(-profile.metabolic_strain * 2.2)
        k_term = 1.0 - math.exp(-profile.kinematic_wear * 1.5)

        ali = (
            self.cortisol_weight * c_term +
            self.adrenaline_weight * a_term +
            self.metabolic_weight * m_term +
            self.kinematic_weight * k_term
        )
        ali = float(np.clip(ali, 0.0, 1.0))
        profile.allostatic_load_index = ali

        # 6. HPA Axis Exhaustion: Prolonged high cortisol blunts subsequent acute dynamic range
        profile.hpa_exhaustion = float(np.clip(profile.cortisol_auc * 0.5, 0.0, 0.85))

        # 7. Allostatic Overload Determination
        profile.is_overloaded = bool(ali >= self.overload_threshold)
        return ali

    def inherit_allostasis(
        self,
        parent_profile: AllostaticProfile,
        child_id: str,
        heritability: float = 0.25,
    ) -> AllostaticProfile:
        """Transgenerational baseline transmission of maternal/parental stress burden."""
        inherited_auc = parent_profile.cortisol_auc * heritability
        inherited_strain = parent_profile.metabolic_strain * heritability
        child = AllostaticProfile(
            agent_id=child_id,
            cortisol_auc=float(inherited_auc),
            metabolic_strain=float(inherited_strain),
            generation=parent_profile.generation + 1,
        )
        self.profiles[child_id] = child
        return child

    def compute_population_stats(self) -> AllostaticPopulationStats:
        """Aggregates allostatic load telemetry across living population."""
        t0 = time.time()
        if not self.profiles:
            stats = AllostaticPopulationStats(timestamp=t0)
            self._last_stats = stats
            return stats

        alis = [p.allostatic_load_index for p in self.profiles.values()]
        exhausts = [p.hpa_exhaustion for p in self.profiles.values()]
        overloaded = sum(1 for p in self.profiles.values() if p.is_overloaded)

        stats = AllostaticPopulationStats(
            mean_ali=float(np.mean(alis)),
            median_ali=float(np.median(alis)),
            max_ali=float(np.max(alis)),
            overloaded_count=overloaded,
            overloaded_fraction=float(overloaded / len(self.profiles)),
            mean_hpa_exhaustion=float(np.mean(exhausts)),
            total_tracked_agents=len(self.profiles),
            timestamp=t0,
        )
        self._last_stats = stats
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Returns fresh JSON-serializable population allostasis summary."""
        return self.compute_population_stats().to_dict()
