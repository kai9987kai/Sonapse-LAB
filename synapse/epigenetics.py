"""Transgenerational Epigenetic Inheritance & Chromatin Remodeling Engine.

Based on foundational evolutionary and epigenetic biology:
- Heard & Martienssen (Cell 2014): Transgenerational epigenetic inheritance in animals.
- Jablonka & Raz (Q Rev Biol 2009): Transgenerational epigenetic inheritance: prevalence, mechanisms, and implications.
- Baccarelli & Bollati (Curr Opin Pediatr 2009): Epigenetics and environmental exposures.

Governing Invariant:
When epigenetics_enabled == False or lambda_epi == 0.0:
p_k = g_k bit-for-bit identical to baseline Mendelian alleles.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclasses.dataclass
class EpigeneticProfile:
    """Represents the chromatin state (methylation and acetylation) across trait loci."""
    methylation: Dict[str, float]  # [0.0, 1.0]: gene silencing / transcription repression
    acetylation: Dict[str, float]  # [0.0, 1.0]: chromatin opening / transcription promotion
    plasticity_index: float = 0.0
    epigenetic_generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "methylation": {k: round(float(v), 4) for k, v in self.methylation.items()},
            "acetylation": {k: round(float(v), 4) for k, v in self.acetylation.items()},
            "plasticity_index": round(float(self.plasticity_index), 4),
            "epigenetic_generation": self.epigenetic_generation,
        }


class EpigeneticRemodelingEngine:
    """Simulates physiological chromatin remodeling and transgenerational epigenetic inheritance."""

    # Key ecological trait loci
    TRAIT_LOCI = ("speed", "sense", "metabolism", "vocal_freq", "vocal_amp")

    def __init__(
        self,
        heritability: float = 0.65,      # Fraction of parental epigenetic marks transmitted (0.0 to 1.0)
        remodeling_rate: float = 0.05,   # Environmental chromatin plasticity rate per event
        decay_rate: float = 0.01,        # Spontaneous relaxation back to baseline
        modulation_scale: float = 0.25,  # Maximum phenotypic trait shift induced by epigenetics (+/- 25%)
    ):
        self.heritability = min(1.0, max(0.0, float(heritability)))
        self.remodeling_rate = float(remodeling_rate)
        self.decay_rate = float(decay_rate)
        self.modulation_scale = float(modulation_scale)

    def default_profile(self) -> EpigeneticProfile:
        """Returns neutral baseline epigenetic marks (zero net modification)."""
        return EpigeneticProfile(
            methylation={k: 0.1 for k in self.TRAIT_LOCI},
            acetylation={k: 0.1 for k in self.TRAIT_LOCI},
            plasticity_index=0.0,
            epigenetic_generation=0,
        )

    def express_phenotype(
        self,
        genome: Dict[str, float],
        profile: Optional[EpigeneticProfile],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        enabled: bool = True,
    ) -> Dict[str, float]:
        """Calculates expressed phenotype from genome and epigenetic marks.

        p_k = g_k * (1.0 + modulation_scale * (acetylation_k - methylation_k))
        When enabled == False or profile is None: returns exact genome copy.
        """
        if not enabled or profile is None:
            return dict(genome)

        expressed = {}
        for k, g_val in genome.items():
            if k in profile.acetylation and k in profile.methylation:
                a_val = profile.acetylation[k]
                m_val = profile.methylation[k]
                # delta_epi in [-modulation_scale, +modulation_scale]
                delta_epi = self.modulation_scale * (a_val - m_val)
                p_val = g_val * (1.0 + delta_epi)
                if bounds and k in bounds:
                    p_val = max(bounds[k][0], min(bounds[k][1], p_val))
                expressed[k] = float(p_val)
            else:
                expressed[k] = float(g_val)
        return expressed

    def update_from_stress(
        self,
        profile: EpigeneticProfile,
        cortisol: float,
        adrenaline: float,
        starvation_stress: float,
        acoustic_alarm_heard: bool = False,
    ) -> None:
        """Dynamically remodels chromatin marks in response to acute and chronic environmental stressors."""
        r = self.remodeling_rate

        # 1. Acute Predation/Panic Stress: Upregulate speed and sensory acuity, repress non-essential metabolism
        predation_stress = min(1.0, 0.6 * cortisol + 0.4 * adrenaline)
        if predation_stress > 0.3:
            # Acetylate speed and sense
            profile.acetylation["speed"] = min(1.0, profile.acetylation.get("speed", 0.1) + r * predation_stress)
            profile.acetylation["sense"] = min(1.0, profile.acetylation.get("sense", 0.1) + r * predation_stress)
            # Methylate metabolism (conserve energy)
            profile.methylation["metabolism"] = min(1.0, profile.methylation.get("metabolism", 0.1) + r * predation_stress * 0.5)

        # 2. Chronic Starvation / Food Deprivation: Enhance metabolic efficiency (thrift phenotype)
        if starvation_stress > 0.4:
            profile.acetylation["metabolism"] = min(1.0, profile.acetylation.get("metabolism", 0.1) + r * starvation_stress)
            profile.methylation["speed"] = min(1.0, profile.methylation.get("speed", 0.1) + r * starvation_stress * 0.3)

        # 3. Acoustic Sentinel Alarm Exposure: Sensitize acoustic vocal amplitude and pitch
        if acoustic_alarm_heard:
            profile.acetylation["vocal_amp"] = min(1.0, profile.acetylation.get("vocal_amp", 0.1) + r * 0.8)
            profile.acetylation["sense"] = min(1.0, profile.acetylation.get("sense", 0.1) + r * 0.5)

        # 4. Spontaneous chromatin relaxation / decay back toward basal state (0.1)
        for k in self.TRAIT_LOCI:
            profile.acetylation[k] = max(0.0, profile.acetylation.get(k, 0.1) * (1.0 - self.decay_rate) + 0.1 * self.decay_rate)
            profile.methylation[k] = max(0.0, profile.methylation.get(k, 0.1) * (1.0 - self.decay_rate) + 0.1 * self.decay_rate)

        # Compute Phenotypic Plasticity Index (PPI)
        total_shift = sum(abs(profile.acetylation.get(k, 0.1) - profile.methylation.get(k, 0.1)) for k in self.TRAIT_LOCI)
        profile.plasticity_index = float(total_shift / len(self.TRAIT_LOCI))

    def inherit_epigenetics(
        self,
        parent_profile: Optional[EpigeneticProfile],
        rng: np.random.Generator,
    ) -> EpigeneticProfile:
        """Models germline transmission with incomplete reprogramming (heritability h_epi).

        Offspring inherit h_epi of parent chromatin marks, while (1 - h_epi) undergoes demethylation reset.
        """
        if parent_profile is None:
            return self.default_profile()

        child_meth = {}
        child_acet = {}
        h = self.heritability
        for k in self.TRAIT_LOCI:
            p_m = parent_profile.methylation.get(k, 0.1)
            p_a = parent_profile.acetylation.get(k, 0.1)
            # Incomplete epigenetic reset with small stochastic variance
            noise_m = float(rng.normal(0.0, 0.02))
            noise_a = float(rng.normal(0.0, 0.02))
            child_meth[k] = max(0.0, min(1.0, h * p_m + (1.0 - h) * 0.1 + noise_m))
            child_acet[k] = max(0.0, min(1.0, h * p_a + (1.0 - h) * 0.1 + noise_a))

        ppi = float(sum(abs(child_acet[k] - child_meth[k]) for k in self.TRAIT_LOCI) / len(self.TRAIT_LOCI))
        return EpigeneticProfile(
            methylation=child_meth,
            acetylation=child_acet,
            plasticity_index=ppi,
            epigenetic_generation=parent_profile.epigenetic_generation + 1,
        )
