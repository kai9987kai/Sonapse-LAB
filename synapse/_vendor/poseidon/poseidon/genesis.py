"""Genesis Planetary Ecology & Evolutionary Genome Engine.

Synthesizes planetary ecological carrying capacity, multi-trophic population cascades,
and evolutionary phenotypic chromosome dynamics from GenesisEngine and Supermix-expanse:
1. Multi-trophic non-linear differential dynamics (Primary Producers, Grazers, Apex Predators)
   integrated via Runge-Kutta 4th order (RK4) with daylight and weather driving terms.
2. Ecosystem carrying capacity K_eco modulating resource replenishment rates in TidePool.
3. 16-gene evolutionary chromosome encoding phenotypic morphology, behavioral drives,
   and quantum resonance affinities.
4. Generational mutation, crossover, and phylogenetic SHA-256 lineage tracking.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Tuple


class GenesisEcosystemEngine:
    """Coupled multi-trophic planetary ecology and evolutionary genetic engine."""

    # Default trophic parameters
    R_PRODUCER: float = 0.35  # primary producer intrinsic growth rate
    K_BASE: float = 1.0  # baseline carrying capacity
    C1_GRAZING: float = 0.25  # grazing consumption coefficient
    E1_EFFICIENCY: float = 0.30  # grazer biomass conversion efficiency
    D_HERBIVORE: float = 0.10  # grazer natural mortality rate
    C2_PREDATION: float = 0.20  # apex predation rate
    E2_EFFICIENCY: float = 0.25  # predator biomass conversion efficiency
    D_PREDATOR: float = 0.08  # predator natural mortality rate

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.step_count: int = 0
        # Trophic populations normalized in [0.05, 2.0]
        self.producers: float = 0.85
        self.grazers: float = 0.45
        self.predators: float = 0.20
        # 16-gene evolutionary chromosome (normalized in [0.0, 1.0])
        # Genes: [0..3: metabolism, 4..7: hydrodynamics, 8..11: behavior, 12..15: quantum affinity]
        self.chromosome: List[float] = self._init_chromosome(seed)
        self.generation: int = 1
        self.cumulative_fitness: float = 0.0

    def _init_chromosome(self, seed: int) -> List[float]:
        """Derive initial 16-gene chromosome deterministically from seed."""
        genes = []
        state = seed & 0xFFFFFFFF
        for i in range(16):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            genes.append(round((state % 1000) / 1000.0, 4))
        return genes

    def reset(self, seed: int | None = None) -> None:
        """Start a reproducible episode with the selected seed's initial genome."""
        if seed is not None:
            if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError("seed must be an integer in [0, 2^32 - 1]")
            self.seed = seed
        self.step_count = 0
        self.producers = 0.85
        self.grazers = 0.45
        self.predators = 0.20
        self.chromosome = self._init_chromosome(self.seed)
        self.generation = 1
        self.cumulative_fitness = 0.0

    def record_observed_reward(self, reward: float) -> None:
        """Accumulate fitness only from a reward returned by the environment."""
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("reward must be a finite number")
        self.cumulative_fitness += max(0.0, float(reward))

    @property
    def lineage_digest(self) -> str:
        """Compute SHA-256 digest of current generational chromosome."""
        payload = json.dumps({"gen": self.generation, "genes": self.chromosome}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _trophic_derivatives(
        self,
        p: float,
        h: float,
        c: float,
        k_eff: float,
    ) -> Tuple[float, float, float]:
        """Compute coupled Lotka-Volterra derivatives (dp/dt, dh/dt, dc/dt)."""
        dp = self.R_PRODUCER * p * (1.0 - p / max(0.01, k_eff)) - self.C1_GRAZING * h * p
        dh = self.E1_EFFICIENCY * self.C1_GRAZING * h * p - self.D_HERBIVORE * h - self.C2_PREDATION * c * h
        dc = self.E2_EFFICIENCY * self.C2_PREDATION * c * h - self.D_PREDATOR * c
        return (dp, dh, dc)

    def _integrate_rk4(self, k_eff: float, dt: float = 0.1) -> None:
        """Perform Runge-Kutta 4th order numerical integration of trophic populations."""
        p, h, c = self.producers, self.grazers, self.predators

        # k1
        k1_p, k1_h, k1_c = self._trophic_derivatives(p, h, c, k_eff)
        # k2
        k2_p, k2_h, k2_c = self._trophic_derivatives(
            p + 0.5 * dt * k1_p, h + 0.5 * dt * k1_h, c + 0.5 * dt * k1_c, k_eff
        )
        # k3
        k3_p, k3_h, k3_c = self._trophic_derivatives(
            p + 0.5 * dt * k2_p, h + 0.5 * dt * k2_h, c + 0.5 * dt * k2_c, k_eff
        )
        # k4
        k4_p, k4_h, k4_c = self._trophic_derivatives(
            p + dt * k3_p, h + dt * k3_h, c + dt * k3_c, k_eff
        )

        self.producers += (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        self.grazers += (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
        self.predators += (dt / 6.0) * (k1_c + 2.0 * k2_c + 2.0 * k3_c + k4_c)

        # Enforce realistic non-zero biological biomass bounds
        self.producers = max(0.05, min(2.5, self.producers))
        self.grazers = max(0.02, min(2.0, self.grazers))
        self.predators = max(0.01, min(1.5, self.predators))

    def mutate_chromosome(self, mutation_rate: float = 0.05) -> None:
        """Introduce generational adaptive mutations into chromosome."""
        self.generation += 1
        new_genes = []
        for i, g in enumerate(self.chromosome):
            # Deterministic pseudo-random variation based on step and gene index
            h_val = int(hashlib.md5(f"{self.generation}:{i}:{g}".encode()).hexdigest()[:6], 16)
            delta = ((h_val % 100) / 50.0 - 1.0) * mutation_rate
            new_genes.append(round(max(0.0, min(1.0, g + delta)), 4))
        self.chromosome = new_genes

    def step(
        self,
        daylight: float,
        severity: float,
        step_reward: float,
    ) -> Dict[str, float]:
        """Update planetary ecological dynamics and chromosome fitness."""
        self.step_count += 1
        if isinstance(step_reward, bool) or not isinstance(step_reward, (int, float)) or not math.isfinite(step_reward):
            raise ValueError("step_reward must be a finite number")
        self.cumulative_fitness += max(0.0, float(step_reward))

        # Solar energy and storm severity modulate effective carrying capacity
        solar_forcing = 0.4 + 0.8 * max(0.0, min(1.0, daylight))
        storm_suppression = 1.0 - 0.4 * max(0.0, min(1.0, severity))
        effective_k = self.K_BASE * solar_forcing * storm_suppression

        self._integrate_rk4(effective_k)

        # Environmental carrying capacity index (resource availability proxy)
        trophic_richness = (self.producers * 0.5 + self.grazers * 0.3 + self.predators * 0.2)

        # Phenotypic gene properties
        metabolic_efficiency = self.chromosome[0]
        hydrodynamic_adaptation = self.chromosome[4]
        risk_affinity = self.chromosome[8]
        quantum_resonance = self.chromosome[12]

        return {
            "producers_biomass": round(self.producers, 4),
            "grazers_biomass": round(self.grazers, 4),
            "predators_biomass": round(self.predators, 4),
            "trophic_richness": round(trophic_richness, 4),
            "generation": float(self.generation),
            "metabolic_gene": round(metabolic_efficiency, 4),
            "hydrodynamic_gene": round(hydrodynamic_adaptation, 4),
            "risk_gene": round(risk_affinity, 4),
            "quantum_gene": round(quantum_resonance, 4),
            "cumulative_fitness": round(self.cumulative_fitness, 4),
        }
