"""SONAPSE-EVO Virtual Ecology & Multi-Agent Co-Evolutionary Biome.

Provides an open-ended multi-agent virtual biome where autonomous neural connectomes
interact, forage, compete, cooperate, exchange neural pulses, transfer knowledge packets
via the Holographic Knowledge Pool, and co-evolve across multiple ecological niches:
- Foragers / Pollinators: Specialize in resource discovery, energy harvesting, and symbiotic signaling.
- Sentinels / Navigators: Specialize in hazard detection, spatial mapping, and alarm pulse broadcasting.
- Artisans / Synthesizers: Specialize in mineral manipulation, cache building, and environmental engineering.
- Ecosystem Predators: Stalk prey, drive selection pressure, and trigger neuroendocrine stress cascades.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .bioacoustic_soundscape import BioacousticSoundscapeEngine, SoundscapeTelemetry
from .epigenetics import EpigeneticRemodelingEngine, EpigeneticProfile
from .allostatic_load import AllostaticLoadEngine, AllostaticProfile, AllostaticPopulationStats

from .neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class EcologyAgentState:
    agent_id: str
    niche: str  # 'forager' | 'sentinel' | 'artisan' | 'predator'
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    energy: float = 100.0
    health: float = 100.0
    age_steps: int = 0
    generation: int = 0
    valence: float = 0.0
    arousal: float = 0.2
    dominance: float = 0.5
    dopamine: float = 0.5
    adrenaline: float = 0.1
    cortisol: float = 0.1
    oxytocin: float = 0.4
    adenosine: float = 0.1
    sleep_state: str = "awake"
    connectome_rates: Optional[List[float]] = None
    alive: bool = True
    genome: Optional[Dict[str, float]] = None
    epigenetic_profile: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self._legacy_dict()
        if self.genome is not None:
            d["genome"] = {k: round(float(v), 4) for k, v in self.genome.items()}
        if self.epigenetic_profile is not None:
            d["epigenetic_profile"] = (
                self.epigenetic_profile.to_dict()
                if hasattr(self.epigenetic_profile, "to_dict")
                else self.epigenetic_profile
            )
        return d

    def _legacy_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "niche": self.niche,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "vx": round(float(self.vx), 3),
            "vy": round(float(self.vy), 3),
            "heading": round(float(self.heading), 3),
            "energy": round(float(self.energy), 2),
            "health": round(float(self.health), 2),
            "age_steps": self.age_steps,
            "generation": self.generation,
            "valence": round(float(self.valence), 3),
            "arousal": round(float(self.arousal), 3),
            "dominance": round(float(self.dominance), 3),
            "dopamine": round(float(self.dopamine), 3),
            "adrenaline": round(float(self.adrenaline), 3),
            "cortisol": round(float(self.cortisol), 3),
            "oxytocin": round(float(self.oxytocin), 3),
            "adenosine": round(float(self.adenosine), 3),
            "sleep_state": self.sleep_state,
            "alive": self.alive,
        }


@dataclasses.dataclass
class EcologicalResource:
    resource_id: str
    kind: str  # 'flora_bloom' | 'mineral_cache' | 'thermal_vent'
    x: float
    y: float
    amount: float = 1.0
    max_amount: float = 1.0
    regrowth_rate: float = 0.02
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "amount": round(float(self.amount), 3),
            "active": self.active,
        }


@dataclasses.dataclass
class VisualEcologySnapshot:
    step_count: int
    biome_half: float
    population_count: int
    agents: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    active_pulses: List[Dict[str, Any]]
    temperature_c: float
    biomass_energy: float
    swarm_coherence: float
    soundscape: Optional[Dict[str, Any]] = None
    allostasis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "step_count": self.step_count,
            "biome_half": self.biome_half,
            "population_count": self.population_count,
            "agents": self.agents,
            "resources": self.resources,
            "active_pulses": self.active_pulses,
            "temperature_c": round(float(self.temperature_c), 1),
            "biomass_energy": round(float(self.biomass_energy), 2),
            "swarm_coherence": round(float(self.swarm_coherence), 3),
        }
        if self.soundscape is not None:
            d["soundscape"] = self.soundscape
        if self.allostasis is not None:
            d["allostasis"] = self.allostasis
        return d


class VirtualEcologyBiome:
    """Multi-agent virtual ecology environment with continuous physics and co-evolution."""

    def __init__(
        self,
        biome_half: float = 10.0,
        num_foragers: int = 4,
        num_sentinels: int = 2,
        num_artisans: int = 2,
        num_predators: int = 1,
        seed: int = 42,
        evolution_enabled: bool = False,
        mutation_sigma: float = 0.08,
        max_population: int = 18,
        soundscape_enabled: bool = False,
        epigenetics_enabled: bool = False,
        allostatic_load_enabled: bool = False,
    ):
        self.evolution_enabled = bool(evolution_enabled)
        self.mutation_sigma = float(mutation_sigma)
        self.max_population = int(max_population)
        # Separate stream so enabling evolution never shifts the legacy RNG sequence.
        self.evo_rng = np.random.default_rng(int(seed) + 7919)
        self.births = 0
        self.deaths = 0
        self.evolution_history: List[Dict[str, Any]] = []
        self.biome_half = float(biome_half)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.temperature_c = 21.5
        self.seed = seed

        self.agents: List[EcologyAgentState] = []
        self.resources: List[EcologicalResource] = []
        self.active_pulses: List[Dict[str, Any]] = []

        # Bioacoustic soundscape engine (opt-in)
        self.soundscape_enabled = bool(soundscape_enabled)
        self.soundscape: Optional[BioacousticSoundscapeEngine] = None
        if self.soundscape_enabled:
            self.enable_soundscape()

        # Epigenetics & transgenerational chromatin remodeling engine (opt-in)
        self.epigenetics_enabled = bool(epigenetics_enabled)
        self.epigenetics_engine = EpigeneticRemodelingEngine()

        # Allostatic load and chronic stress engine (opt-in)
        self.allostatic_load_enabled = bool(allostatic_load_enabled)
        self.allostatic_engine = AllostaticLoadEngine(seed=seed)
        if self.allostatic_load_enabled:
            self.enable_allostatic_load()

        # Connectome weights for micro-brain simulation across agents
        self._init_resources()
        self._init_agents(num_foragers, num_sentinels, num_artisans, num_predators)
        if self.evolution_enabled:
            self.enable_evolution()
        if self.epigenetics_enabled:
            self.enable_epigenetics()

    # ------------------------------------------------------------------
    # Heritable genome (opt-in). Traits trade performance against energy:
    #   speed  -> locomotion multiplier, cost ~ speed^2
    #   sense  -> perception-radius multiplier, cost ~ (0.75 + 0.25*sense)
    # With evolution disabled every factor below is exactly 1.0.
    # ------------------------------------------------------------------
    GENE_BOUNDS = {"speed": (0.5, 1.8), "sense": (0.5, 2.0)}

    def _random_genome(self) -> Dict[str, float]:
        return {k: float(np.clip(1.0 + self.evo_rng.normal(0.0, 0.15), lo, hi))
                for k, (lo, hi) in self.GENE_BOUNDS.items()}

    def _mutate(self, parent: Dict[str, float]) -> Dict[str, float]:
        return {k: float(np.clip(parent.get(k, 1.0) + self.evo_rng.normal(0.0, self.mutation_sigma), lo, hi))
                for k, (lo, hi) in self.GENE_BOUNDS.items()}

    def enable_allostatic_load(self) -> None:
        """Enables allostatic load tracking and cumulative stress wear across agents."""
        self.allostatic_load_enabled = True
        for ag in self.agents:
            self.allostatic_engine.get_or_create_profile(ag.agent_id, ag.generation)

    def allostasis_stats(self) -> Dict[str, Any]:
        """Returns population allostatic load statistics."""
        return self.allostatic_engine.get_summary()

    def enable_epigenetics(self, heritability: float = 0.65) -> None:
        """Activates transgenerational epigenetic inheritance and chromatin plasticity."""
        self.epigenetics_enabled = True
        self.epigenetics_engine.heritability = min(1.0, max(0.0, float(heritability)))
        for ag in self.agents:
            if ag.epigenetic_profile is None:
                ag.epigenetic_profile = self.epigenetics_engine.default_profile()

    def epigenetics_stats(self) -> Dict[str, Any]:
        """Population chromatin methylation, acetylation, and plasticity statistics."""
        profiles = [a.epigenetic_profile for a in self.agents if a.alive and a.epigenetic_profile is not None]
        if not profiles:
            return {
                "epigenetics_enabled": self.epigenetics_enabled,
                "count": 0,
                "mean_plasticity_index": 0.0,
                "traits": {},
            }
        mean_ppi = float(np.mean([p.plasticity_index for p in profiles]))
        traits = {}
        for locus in self.epigenetics_engine.TRAIT_LOCI:
            m_vals = [p.methylation.get(locus, 0.1) for p in profiles]
            a_vals = [p.acetylation.get(locus, 0.1) for p in profiles]
            traits[locus] = {
                "mean_methylation": round(float(np.mean(m_vals)), 4),
                "mean_acetylation": round(float(np.mean(a_vals)), 4),
            }
        return {
            "epigenetics_enabled": self.epigenetics_enabled,
            "count": len(profiles),
            "mean_plasticity_index": round(mean_ppi, 4),
            "heritability": self.epigenetics_engine.heritability,
            "traits": traits,
        }

    def _gene(self, ag: EcologyAgentState, key: str) -> float:
        if not self.evolution_enabled or ag.genome is None:
            return 1.0
        if self.epigenetics_enabled and ag.epigenetic_profile is not None:
            expressed = self.epigenetics_engine.express_phenotype(
                ag.genome, ag.epigenetic_profile, self.GENE_BOUNDS, enabled=True
            )
            return float(expressed.get(key, 1.0))
        return float(ag.genome.get(key, 1.0))

    def _metabolic_factor(self, ag: EcologyAgentState) -> float:
        if not self.evolution_enabled or ag.genome is None:
            return 1.0
        return self._gene(ag, "speed") ** 2 * (0.75 + 0.25 * self._gene(ag, "sense"))

    def enable_soundscape(self) -> BioacousticSoundscapeEngine:
        """Activates bioacoustic soundscape simulation."""
        if self.soundscape is None:
            self.soundscape = BioacousticSoundscapeEngine(biome_half=self.biome_half, seed=int(self.seed) + 333)
        self.soundscape_enabled = True
        return self.soundscape

    def enable_evolution(self) -> int:
        """Turns on heritable variation; assigns genomes to agents lacking one."""
        self.evolution_enabled = True
        assigned = 0
        for ag in self.agents:
            if ag.genome is None:
                ag.genome = self._random_genome()
                assigned += 1
        return assigned

    def evolution_stats(self) -> Dict[str, Any]:
        """Per-niche trait statistics over the living population (measured, not modelled)."""
        niches: Dict[str, Any] = {}
        for niche in ("forager", "sentinel", "artisan", "predator"):
            members = [a for a in self.agents if a.alive and a.niche == niche and a.genome is not None]
            if not members:
                niches[niche] = {"count": 0}
                continue
            entry: Dict[str, Any] = {"count": len(members),
                                     "max_generation": max(a.generation for a in members),
                                     "mean_energy": round(float(np.mean([a.energy for a in members])), 3)}
            for k in self.GENE_BOUNDS:
                vals = np.array([a.genome[k] for a in members])
                entry[f"{k}_mean"] = round(float(vals.mean()), 4)
                entry[f"{k}_std"] = round(float(vals.std()), 4)
            niches[niche] = entry
        return {
            "evolution_enabled": self.evolution_enabled,
            "step_count": self.step_count,
            "population": len(self.agents),
            "births": self.births,
            "deaths": self.deaths,
            "max_generation": max((a.generation for a in self.agents), default=0),
            "niches": niches,
            "history": self.evolution_history[-200:],
            "evidence_kind": EvidenceKind.SYNTHETIC_SIMULATION.value,
        }

    def _record_evolution_point(self) -> None:
        point: Dict[str, Any] = {"step": self.step_count, "population": len(self.agents)}
        genomes = [a.genome for a in self.agents if a.alive and a.genome is not None]
        for k in self.GENE_BOUNDS:
            point[f"{k}_mean"] = round(float(np.mean([g[k] for g in genomes])), 4) if genomes else None
        self.evolution_history.append(point)
        if len(self.evolution_history) > 1000:
            self.evolution_history = self.evolution_history[-1000:]

    def _init_resources(self):
        """Initializes distributed food flora, mineral caches, and thermal vents."""
        self.resources.clear()
        # Flora bloom clusters (green nutrients)
        for i in range(12):
            x = float(self.rng.uniform(-self.biome_half + 1.5, self.biome_half - 1.5))
            y = float(self.rng.uniform(-self.biome_half + 1.5, self.biome_half - 1.5))
            self.resources.append(EcologicalResource(
                resource_id=f"flora_{i}",
                kind="flora_bloom",
                x=x,
                y=y,
                amount=float(self.rng.uniform(0.6, 1.0)),
                max_amount=1.0,
                regrowth_rate=0.015,
            ))

        # Mineral caches (crafting/tools)
        for i in range(4):
            x = float(self.rng.uniform(-self.biome_half + 2.0, self.biome_half - 2.0))
            y = float(self.rng.uniform(-self.biome_half + 2.0, self.biome_half - 2.0))
            self.resources.append(EcologicalResource(
                resource_id=f"mineral_{i}",
                kind="mineral_cache",
                x=x,
                y=y,
                amount=1.0,
                max_amount=1.0,
                regrowth_rate=0.005,
            ))

        # Geothermal warmth vents (temperature regulation / refuge)
        for i in range(2):
            x = float(self.rng.uniform(-self.biome_half + 3.0, self.biome_half - 3.0))
            y = float(self.rng.uniform(-self.biome_half + 3.0, self.biome_half - 3.0))
            self.resources.append(EcologicalResource(
                resource_id=f"vent_{i}",
                kind="thermal_vent",
                x=x,
                y=y,
                amount=1.0,
                max_amount=1.0,
                regrowth_rate=0.05,
            ))

    def _init_agents(self, foragers: int, sentinels: int, artisans: int, predators: int):
        """Populates the biome with diverse initial agents across niches."""
        self.agents.clear()
        idx = 0

        # 1. Foragers
        for _ in range(foragers):
            idx += 1
            x = float(self.rng.uniform(-self.biome_half * 0.7, self.biome_half * 0.7))
            y = float(self.rng.uniform(-self.biome_half * 0.7, self.biome_half * 0.7))
            th = float(self.rng.uniform(0, math.pi * 2))
            self.agents.append(EcologyAgentState(
                agent_id=f"agent_forager_{idx}",
                niche="forager",
                x=x, y=y, vx=0.0, vy=0.0, heading=th,
                energy=80.0,
                dopamine=0.6, oxytocin=0.5,
            ))

        # 2. Sentinels
        for _ in range(sentinels):
            idx += 1
            x = float(self.rng.uniform(-self.biome_half * 0.5, self.biome_half * 0.5))
            y = float(self.rng.uniform(-self.biome_half * 0.5, self.biome_half * 0.5))
            th = float(self.rng.uniform(0, math.pi * 2))
            self.agents.append(EcologyAgentState(
                agent_id=f"agent_sentinel_{idx}",
                niche="sentinel",
                x=x, y=y, vx=0.0, vy=0.0, heading=th,
                energy=90.0,
                arousal=0.4, adrenaline=0.2,
            ))

        # 3. Artisans
        for _ in range(artisans):
            idx += 1
            x = float(self.rng.uniform(-self.biome_half * 0.6, self.biome_half * 0.6))
            y = float(self.rng.uniform(-self.biome_half * 0.6, self.biome_half * 0.6))
            th = float(self.rng.uniform(0, math.pi * 2))
            self.agents.append(EcologyAgentState(
                agent_id=f"agent_artisan_{idx}",
                niche="artisan",
                x=x, y=y, vx=0.0, vy=0.0, heading=th,
                energy=85.0,
                dominance=0.6, dopamine=0.4,
            ))

        # 4. Predators
        for _ in range(predators):
            idx += 1
            x = float(self.rng.uniform(-self.biome_half * 0.8, self.biome_half * 0.8))
            y = float(self.rng.uniform(-self.biome_half * 0.8, self.biome_half * 0.8))
            th = float(self.rng.uniform(0, math.pi * 2))
            self.agents.append(EcologyAgentState(
                agent_id=f"agent_predator_{idx}",
                niche="predator",
                x=x, y=y, vx=0.0, vy=0.0, heading=th,
                energy=95.0,
                arousal=0.6, dominance=0.8, adrenaline=0.4, cortisol=0.2,
            ))

    def step(self, external_stimulus: Optional[Dict[str, Any]] = None) -> VisualEcologySnapshot:
        """Executes one continuous simulation step across all ecological entities."""
        self.step_count += 1
        dt = 0.1

        if self.soundscape_enabled and self.soundscape is not None:
            self.soundscape.step()

        # Process external interventions / stimulus
        if external_stimulus:
            action_type = external_stimulus.get("type", "")
            if action_type == "seed_resources":
                for r in self.resources:
                    r.amount = min(r.max_amount, r.amount + 0.4)
            elif action_type == "trigger_predator_wave":
                for ag in self.agents:
                    if ag.niche == "predator":
                        ag.arousal = min(1.0, ag.arousal + 0.4)
                        ag.adrenaline = min(1.0, ag.adrenaline + 0.5)
            elif action_type == "broadcast_alarm":
                px = float(external_stimulus.get("x", 0.0))
                py = float(external_stimulus.get("y", 0.0))
                self.active_pulses.append({
                    "pulse_id": f"pulse_{self.step_count}",
                    "x": px,
                    "y": py,
                    "radius": 0.5,
                    "max_radius": 6.0,
                    "amplitude": 1.5,
                    "color": "rgba(239, 68, 68, 0.7)",
                })
            elif action_type == "induce_sleep_cycle":
                for ag in self.agents:
                    ag.sleep_state = "sws_sleep"
                    ag.adenosine = max(0.6, ag.adenosine + 0.3)

        # 1. Regrow ecological resources
        for r in self.resources:
            if r.amount < r.max_amount:
                r.amount = min(r.max_amount, r.amount + r.regrowth_rate)

        # 2. Update traveling wave pulses
        active_remaining = []
        for p in self.active_pulses:
            p["radius"] += 0.35
            p["amplitude"] *= 0.92
            if p["radius"] < p["max_radius"] and p["amplitude"] > 0.05:
                active_remaining.append(p)
        self.active_pulses = active_remaining

        # 3. Simulate agent behavior & micro-connectome dynamics
        predators = [a for a in self.agents if a.niche == "predator" and a.alive]
        prey = [a for a in self.agents if a.niche != "predator" and a.alive]

        for ag in self.agents:
            if not ag.alive:
                continue

            ag.age_steps += 1
            # Metabolic basal cost
            ag.energy = max(0.0, ag.energy - 0.04 * self._metabolic_factor(ag))

            # Sleep homeostat accumulation
            if ag.sleep_state == "awake":
                ag.adenosine = min(1.0, ag.adenosine + 0.001)
                if ag.adenosine > 0.85:
                    ag.sleep_state = "sws_sleep"
            else:
                ag.adenosine = max(0.05, ag.adenosine - 0.005)
                if ag.adenosine < 0.2:
                    ag.sleep_state = "awake"

            # Behavioral steering according to ecological niche
            if ag.sleep_state != "awake":
                # Quiescent sleep deceleration
                ag.vx *= 0.85
                ag.vy *= 0.85
            elif ag.niche == "forager":
                # Seek nearest active flora bloom
                active_food = [r for r in self.resources if r.kind == "flora_bloom" and r.amount > 0.2]
                if active_food:
                    nearest = min(active_food, key=lambda f: math.hypot(f.x - ag.x, f.y - ag.y))
                    dx = nearest.x - ag.x
                    dy = nearest.y - ag.y
                    dist = math.hypot(dx, dy)
                    if dist < 0.6:
                        # Feed
                        consumed = min(nearest.amount, 0.25)
                        nearest.amount -= consumed
                        ag.energy = min(100.0, ag.energy + consumed * 40.0)
                        ag.dopamine = min(1.0, ag.dopamine + 0.1)
                        ag.valence = min(1.0, ag.valence + 0.15)
                        if self.soundscape_enabled and self.soundscape is not None and self.step_count % 10 == 0:
                            self.soundscape.emit_vocalization(ag.agent_id, 'forager', ag.x, ag.y, amplitude=0.8)
                    else:
                        ag.heading = math.atan2(dy, dx)
                        speed = 0.8
                        ag.vx = speed * math.cos(ag.heading)
                        ag.vy = speed * math.sin(ag.heading)

                # Predator evasion if predator nearby
                for p in predators:
                    p_dist = math.hypot(p.x - ag.x, p.y - ag.y)
                    if p_dist < 4.0 * self._gene(ag, "sense"):
                        # Flee away from predator
                        flee_th = math.atan2(ag.y - p.y, ag.x - p.x)
                        ag.heading = flee_th
                        ag.vx = 1.3 * math.cos(flee_th)
                        ag.vy = 1.3 * math.sin(flee_th)
                        ag.arousal = min(1.0, ag.arousal + 0.3)
                        ag.adrenaline = min(1.0, ag.adrenaline + 0.4)
                        ag.cortisol = min(1.0, ag.cortisol + 0.2)
                        ag.valence = max(-1.0, ag.valence - 0.2)
                        break

            elif ag.niche == "sentinel":
                # Patrol around perimeter and scan for predators
                orbit_speed = 0.9
                ag.heading += 0.08
                ag.vx = orbit_speed * math.cos(ag.heading)
                ag.vy = orbit_speed * math.sin(ag.heading)
                if self.evolution_enabled and ag.energy < 55.0:
                    vents = [r for r in self.resources if r.kind == "thermal_vent" and r.amount > 0.2]
                    if vents:
                        v = min(vents, key=lambda r: math.hypot(r.x - ag.x, r.y - ag.y))
                        dx, dy = v.x - ag.x, v.y - ag.y
                        if math.hypot(dx, dy) < 0.6:
                            take = min(v.amount, 0.2)
                            v.amount -= take
                            ag.energy = min(100.0, ag.energy + take * 35.0)
                        else:
                            ag.heading = math.atan2(dy, dx)
                            ag.vx = orbit_speed * math.cos(ag.heading)
                            ag.vy = orbit_speed * math.sin(ag.heading)

                for p in predators:
                    p_dist = math.hypot(p.x - ag.x, p.y - ag.y)
                    if p_dist < 5.0 * self._gene(ag, "sense") and self.step_count % 16 == 0:
                        # Broadcast alarm pulse!
                        self.active_pulses.append({
                            "pulse_id": f"alarm_{ag.agent_id}_{self.step_count}",
                            "x": ag.x,
                            "y": ag.y,
                            "radius": 0.4,
                            "max_radius": 7.0,
                            "amplitude": 1.8,
                            "color": "rgba(234, 179, 8, 0.65)",
                        })
                        ag.oxytocin = min(1.0, ag.oxytocin + 0.1)
                        if self.soundscape_enabled and self.soundscape is not None:
                            self.soundscape.emit_vocalization(ag.agent_id, 'sentinel', ag.x, ag.y, amplitude=1.2)
                        break

            elif ag.niche == "artisan":
                # Seek mineral caches
                minerals = [r for r in self.resources if r.kind == "mineral_cache"]
                if minerals:
                    nearest = min(minerals, key=lambda m: math.hypot(m.x - ag.x, m.y - ag.y))
                    dx = nearest.x - ag.x
                    dy = nearest.y - ag.y
                    ag.heading = math.atan2(dy, dx)
                    ag.vx = 0.7 * math.cos(ag.heading)
                    ag.vy = 0.7 * math.sin(ag.heading)
                    if self.evolution_enabled and math.hypot(dx, dy) < 0.6 and nearest.amount > 0.1:
                        take = min(nearest.amount, 0.15)
                        nearest.amount -= take
                        ag.energy = min(100.0, ag.energy + take * 40.0)
                        ag.vx *= 0.2
                        ag.vy *= 0.2

            elif ag.niche == "predator":
                # Stalk nearest prey
                living_prey = [p for p in prey if p.alive]
                if living_prey:
                    target = min(living_prey, key=lambda p: math.hypot(p.x - ag.x, p.y - ag.y))
                    dx = target.x - ag.x
                    dy = target.y - ag.y
                    dist = math.hypot(dx, dy)
                    if dist < 0.7:
                        # Attack / feed
                        target.health = max(0.0, target.health - 25.0)
                        ag.energy = min(100.0, ag.energy + 30.0)
                        ag.dopamine = min(1.0, ag.dopamine + 0.25)
                        if target.health <= 0.0:
                            target.alive = False
                    else:
                        ag.heading = math.atan2(dy, dx)
                        speed = 1.05
                        ag.vx = speed * math.cos(ag.heading)
                        ag.vy = speed * math.sin(ag.heading)

            # Bioacoustic hearing and vocalization updates (opt-in)
            if self.soundscape_enabled and self.soundscape is not None:
                tot_amp, _, niche_amps = self.soundscape.get_acoustic_at_point(ag.x, ag.y)
                if niche_amps.get("sentinel", 0.0) > 0.2 and ag.niche in ("forager", "artisan"):
                    ag.arousal = min(1.0, ag.arousal + 0.2)
                    ag.adrenaline = min(1.0, ag.adrenaline + 0.25)
                    ag.cortisol = min(1.0, ag.cortisol + 0.15)
                    if ag.sleep_state != "awake":
                        ag.sleep_state = "awake"

            # Update allostatic load from chronic stress mediators (opt-in)
            if self.allostatic_load_enabled:
                prof = self.allostatic_engine.get_or_create_profile(ag.agent_id, ag.generation)
                spd_mag = math.hypot(ag.vx, ag.vy)
                self.allostatic_engine.update_agent(
                    profile=prof,
                    cortisol=ag.cortisol,
                    adrenaline=ag.adrenaline,
                    energy=ag.energy,
                    speed_magnitude=spd_mag,
                    dt=dt,
                )
                if prof.is_overloaded:
                    # Allostatic overload induces accelerated metabolic wear
                    ag.energy = max(0.0, ag.energy - 0.04)

            # Update chromatin remodeling from environmental stress (opt-in)
            if self.epigenetics_enabled and ag.epigenetic_profile is not None:
                starv = max(0.0, (50.0 - ag.energy) / 50.0) if ag.energy < 50.0 else 0.0
                alarm_heard = bool(self.soundscape_enabled and self.soundscape is not None)
                self.epigenetics_engine.update_from_stress(
                    ag.epigenetic_profile,
                    cortisol=ag.cortisol,
                    adrenaline=ag.adrenaline,
                    starvation_stress=starv,
                    acoustic_alarm_heard=alarm_heard,
                )

            # Continuous motion update with biome bounds
            spd = self._gene(ag, "speed")
            ag.x += ag.vx * dt * spd
            ag.y += ag.vy * dt * spd

            # Soft boundary bouncing
            margin = self.biome_half - 0.8
            if abs(ag.x) > margin:
                ag.x = math.copysign(margin, ag.x)
                ag.vx = -ag.vx * 0.8
                ag.heading = math.atan2(ag.vy, ag.vx)
            if abs(ag.y) > margin:
                ag.y = math.copysign(margin, ag.y)
                ag.vy = -ag.vy * 0.8
                ag.heading = math.atan2(ag.vy, ag.vx)

            # Mortality check
            if ag.energy <= 0.0:
                ag.health = max(0.0, ag.health - 2.0)
                if ag.health <= 0.0:
                    ag.alive = False

        # Reproduction check: high energy foragers divide
        new_offspring = []
        for ag in self.agents:
            cap = self.max_population if self.evolution_enabled else 18
            if ag.alive and ag.energy > 92.0 and len(self.agents) + len(new_offspring) < cap:
                ag.energy -= 45.0
                offspring = EcologyAgentState(
                    agent_id=f"{ag.agent_id}_child_{self.step_count}",
                    niche=ag.niche,
                    x=ag.x + float(self.rng.uniform(-0.4, 0.4)),
                    y=ag.y + float(self.rng.uniform(-0.4, 0.4)),
                    vx=float(self.rng.uniform(-0.5, 0.5)),
                    vy=float(self.rng.uniform(-0.5, 0.5)),
                    heading=float(self.rng.uniform(0, math.pi * 2)),
                    energy=50.0,
                    generation=ag.generation + 1,
                    valence=ag.valence,
                    arousal=ag.arousal,
                )
                if self.evolution_enabled and ag.genome is not None:
                    offspring.genome = self._mutate(ag.genome)
                if self.epigenetics_enabled and ag.epigenetic_profile is not None:
                    offspring.epigenetic_profile = self.epigenetics_engine.inherit_epigenetics(
                        ag.epigenetic_profile, self.evo_rng
                    )
                if self.allostatic_load_enabled:
                    p_prof = self.allostatic_engine.get_or_create_profile(ag.agent_id, ag.generation)
                    self.allostatic_engine.inherit_allostasis(p_prof, offspring.agent_id)
                new_offspring.append(offspring)
        self.agents.extend(new_offspring)
        self.births += len(new_offspring)

        # Clean dead agents
        if self.allostatic_load_enabled:
            for a in self.agents:
                if not a.alive:
                    self.allostatic_engine.remove_dead_agent(a.agent_id)
        living_agents = [a for a in self.agents if a.alive]
        self.deaths += len(self.agents) - len(living_agents)
        self.agents = living_agents
        if self.evolution_enabled and self.step_count % 25 == 0:
            self._record_evolution_point()

        return self._build_snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Current state without advancing the simulation (safe for polling)."""
        return self._build_snapshot().to_dict()

    def _build_snapshot(self) -> VisualEcologySnapshot:
        # Compute swarm coherence (Kuramoto-like phase alignment across headings)
        if len(self.agents) > 1:
            mean_sin = np.mean([math.sin(a.heading) for a in self.agents])
            mean_cos = np.mean([math.cos(a.heading) for a in self.agents])
            coherence = float(math.hypot(mean_sin, mean_cos))
        else:
            coherence = 1.0

        total_biomass = float(sum(a.energy for a in self.agents) + sum(r.amount * 20.0 for r in self.resources))

        sc_summary = None
        if self.soundscape_enabled and self.soundscape is not None:
            sc_summary = self.soundscape.get_summary()

        allo_summary = None
        if self.allostatic_load_enabled:
            allo_summary = self.allostatic_engine.get_summary()

        return VisualEcologySnapshot(
            step_count=self.step_count,
            biome_half=self.biome_half,
            population_count=len(self.agents),
            agents=[a.to_dict() for a in self.agents],
            resources=[r.to_dict() for r in self.resources],
            active_pulses=self.active_pulses,
            temperature_c=self.temperature_c,
            biomass_energy=total_biomass,
            swarm_coherence=coherence,
            soundscape=sc_summary,
            allostasis=allo_summary,
        )

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns the current state snapshot dict."""
        return self.step(None).to_dict()
