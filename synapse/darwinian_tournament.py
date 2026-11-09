"""Darwinian Virtual Ecology Tournament & Multi-Generation Speciation Engine.

Orchestrates long-horizon co-evolutionary tournaments across the four ecological niches:
Foragers, Sentinels, Artisans, and Apex Predators.

Tracks:
1. Multi-generational phylogenetic lineage trees and ancestral pedigree tracking.
2. Speciation & niche divergence index via ANOVA F-ratio:
     F_divergence = MS_between / (MS_within + eps)
   quantifying whether phenotypes diverge into specialized ecological niches.
3. Lotka-Volterra predator-prey oscillatory phase dynamics and phase portraits.
4. Cataclysmic environmental shocks (drought, thermal shock, apex surge) to assess
   macro-evolutionary resilience and punctuated equilibrium.
5. Verifiable cryptographic EvidenceReceipt records with SHA-256 digests.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)
from synapse.virtual_ecology import (
    VirtualEcologyBiome,
    EcologyAgentState,
    VisualEcologySnapshot,
)


@dataclasses.dataclass
class LineageNode:
    agent_id: str
    parent_id: Optional[str]
    niche: str
    generation: int
    birth_step: int
    death_step: Optional[int]
    genome: Dict[str, float]
    offspring_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "niche": self.niche,
            "generation": self.generation,
            "birth_step": self.birth_step,
            "death_step": self.death_step,
            "genome": {k: round(float(v), 4) for k, v in self.genome.items()},
            "offspring_count": self.offspring_count,
        }


@dataclasses.dataclass
class SpeciationMetrics:
    divergence_index: float
    trait_f_ratios: Dict[str, float]
    niche_mean_traits: Dict[str, Dict[str, float]]
    predator_prey_phase_rad: float
    predator_prey_correlation: float
    extinction_events: int
    active_lineages: int
    max_lineage_depth: int
    shannon_entropy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divergence_index": round(float(self.divergence_index), 4),
            "trait_f_ratios": {k: round(float(v), 4) for k, v in self.trait_f_ratios.items()},
            "niche_mean_traits": self.niche_mean_traits,
            "predator_prey_phase_rad": round(float(self.predator_prey_phase_rad), 3),
            "predator_prey_correlation": round(float(self.predator_prey_correlation), 3),
            "extinction_events": self.extinction_events,
            "active_lineages": self.active_lineages,
            "max_lineage_depth": self.max_lineage_depth,
            "shannon_entropy": round(float(self.shannon_entropy), 4),
        }


@dataclasses.dataclass
class DarwinianTournamentReceipt:
    receipt_id: str
    timestamp: float
    total_steps: int
    total_generations: int
    total_births: int
    total_deaths: int
    metrics: SpeciationMetrics
    evidence: EvidenceReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "total_steps": self.total_steps,
            "total_generations": self.total_generations,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "metrics": self.metrics.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


class DarwinianEcologyTournamentEngine:
    """Manages multi-generation evolutionary tournaments, phylogenetic tracking, and speciation metrics."""

    def __init__(
        self,
        biome: Optional[VirtualEcologyBiome] = None,
        seed: int = 42,
    ):
        self.biome = biome or VirtualEcologyBiome(
            biome_half=10.0,
            num_foragers=4,
            num_sentinels=2,
            num_artisans=2,
            num_predators=1,
            seed=seed,
            evolution_enabled=True,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Make sure evolution is active
        if not self.biome.evolution_enabled:
            self.biome.enable_evolution()

        # Phylogenetic tracking
        self.lineage_tree: Dict[str, LineageNode] = {}
        self.known_agent_ids: set[str] = set()
        self.extinction_counter = 0

        # Time series history for phase portraits & Lotka-Volterra dynamics
        self.population_series: collections.deque = collections.deque(maxlen=2000)
        self.history_receipts: List[DarwinianTournamentReceipt] = []

        self._receipt_counter = 0
        self._sync_initial_agents()

    def _sync_initial_agents(self) -> None:
        """Registers founding organisms in the phylogenetic lineage tree."""
        for ag in self.biome.agents:
            if ag.agent_id not in self.lineage_tree:
                self.known_agent_ids.add(ag.agent_id)
                self.lineage_tree[ag.agent_id] = LineageNode(
                    agent_id=ag.agent_id,
                    parent_id=None,
                    niche=ag.niche,
                    generation=ag.generation,
                    birth_step=self.biome.step_count,
                    death_step=None,
                    genome=dict(ag.genome or {"speed": 1.0, "sense": 1.0}),
                )

    def _update_lineages(self) -> None:
        """Synchronizes new births and agent deaths into the lineage tree."""
        current_living_ids = {a.agent_id for a in self.biome.agents if a.alive}

        # Check for new births
        for ag in self.biome.agents:
            if ag.agent_id not in self.lineage_tree:
                self.known_agent_ids.add(ag.agent_id)
                parent_id = None
                # Parent inference from agent_id convention "parent_child_step"
                if "_child_" in ag.agent_id:
                    parent_id = ag.agent_id.split("_child_")[0]
                    if parent_id in self.lineage_tree:
                        self.lineage_tree[parent_id].offspring_count += 1

                self.lineage_tree[ag.agent_id] = LineageNode(
                    agent_id=ag.agent_id,
                    parent_id=parent_id,
                    niche=ag.niche,
                    generation=ag.generation,
                    birth_step=self.biome.step_count,
                    death_step=None,
                    genome=dict(ag.genome or {"speed": 1.0, "sense": 1.0}),
                )

        # Check for deaths
        for aid, node in self.lineage_tree.items():
            if node.death_step is None and aid not in current_living_ids:
                node.death_step = self.biome.step_count

        # Check for niche extinction
        living_niches = {a.niche for a in self.biome.agents if a.alive}
        for n in ("forager", "sentinel", "artisan", "predator"):
            if n not in living_niches:
                # If niche has gone extinct, increment counter and seed 1 migrant
                self.extinction_counter += 1
                self._reseed_migrant(n)

    def _reseed_migrant(self, niche: str) -> None:
        """Immigrates a novel explorer organism if a niche goes extinct to maintain ecological tension."""
        th = float(self.rng.uniform(0, math.pi * 2))
        x = float(self.rng.uniform(-self.biome.biome_half * 0.5, self.biome.biome_half * 0.5))
        y = float(self.rng.uniform(-self.biome.biome_half * 0.5, self.biome_half_y(0.5)))
        migrant_id = f"migrant_{niche}_{self.biome.step_count}"
        migrant = EcologyAgentState(
            agent_id=migrant_id,
            niche=niche,
            x=x, y=y, vx=0.0, vy=0.0, heading=th,
            energy=85.0,
            generation=max((a.generation for a in self.biome.agents), default=0) + 1,
            genome=dict(self.biome._random_genome()),
        )
        self.biome.agents.append(migrant)
        self.lineage_tree[migrant_id] = LineageNode(
            agent_id=migrant_id,
            parent_id=None,
            niche=niche,
            generation=migrant.generation,
            birth_step=self.biome.step_count,
            death_step=None,
            genome=dict(migrant.genome),
        )

    def biome_half_y(self, factor: float) -> float:
        return float(self.rng.uniform(-self.biome.biome_half * factor, self.biome.biome_half * factor))

    def step_tournament(self, steps: int = 1) -> VisualEcologySnapshot:
        """Advances the tournament by the specified number of simulation steps."""
        snap = None
        for _ in range(max(1, int(steps))):
            snap = self.biome.step()
            self._update_lineages()

            # Record time series point for phase portrait
            counts = {
                "forager": sum(1 for a in self.biome.agents if a.alive and a.niche == "forager"),
                "sentinel": sum(1 for a in self.biome.agents if a.alive and a.niche == "sentinel"),
                "artisan": sum(1 for a in self.biome.agents if a.alive and a.niche == "artisan"),
                "predator": sum(1 for a in self.biome.agents if a.alive and a.niche == "predator"),
            }
            self.population_series.append({
                "step": self.biome.step_count,
                "counts": counts,
                "total_population": len(self.biome.agents),
                "biomass": snap.biomass_energy,
                "coherence": snap.swarm_coherence,
            })
        return snap or self.biome.step()

    def compute_speciation_divergence(self) -> SpeciationMetrics:
        """Computes statistical speciation divergence (ANOVA F-ratio between vs within niche variance)."""
        traits = ("speed", "sense")
        niches = ("forager", "sentinel", "artisan", "predator")
        f_ratios = {}
        niche_means: Dict[str, Dict[str, float]] = {n: {} for n in niches}

        # Collect genome values per niche
        niche_data: Dict[str, Dict[str, List[float]]] = {n: {t: [] for t in traits} for n in niches}
        for ag in self.biome.agents:
            if ag.alive and ag.genome is not None and ag.niche in niche_data:
                for t in traits:
                    niche_data[ag.niche][t].append(float(ag.genome.get(t, 1.0)))

        for n in niches:
            for t in traits:
                vals = niche_data[n][t]
                niche_means[n][t] = round(float(np.mean(vals)), 4) if vals else 1.0

        for t in traits:
            # Group means and grand mean
            all_vals = []
            group_means = []
            group_sizes = []
            ss_within = 0.0

            for n in niches:
                vals = np.array(niche_data[n][t])
                if len(vals) > 0:
                    all_vals.extend(vals.tolist())
                    mean_val = float(np.mean(vals))
                    group_means.append(mean_val)
                    group_sizes.append(len(vals))
                    ss_within += float(np.sum((vals - mean_val) ** 2))

            if len(all_vals) > len(group_means) and len(group_means) > 1:
                grand_mean = float(np.mean(all_vals))
                ss_between = float(np.sum([
                    size * (gm - grand_mean) ** 2
                    for size, gm in zip(group_sizes, group_means)
                ]))
                df_between = max(1, len(group_means) - 1)
                df_within = max(1, len(all_vals) - len(group_means))
                ms_between = ss_between / df_between
                ms_within = ss_within / df_within
                f_stat = ms_between / (ms_within + 1e-6)
                f_ratios[t] = float(np.clip(f_stat, 0.0, 100.0))
            else:
                f_ratios[t] = 1.0

        divergence_idx = float(np.mean(list(f_ratios.values()))) if f_ratios else 1.0

        # Phase difference & correlation between Foragers and Predators
        forager_ts = [p["counts"]["forager"] for p in self.population_series]
        predator_ts = [p["counts"]["predator"] for p in self.population_series]

        phase_rad = 0.0
        corr = 0.0
        if len(forager_ts) >= 16:
            f_arr = np.array(forager_ts[-64:], dtype=np.float64)
            p_arr = np.array(predator_ts[-64:], dtype=np.float64)
            f_std = np.std(f_arr)
            p_std = np.std(p_arr)
            if f_std > 1e-5 and p_std > 1e-5:
                corr = float(np.corrcoef(f_arr, p_arr)[0, 1])
                # Approximate phase angle via cross-correlation peak
                cross_corr = np.correlate(f_arr - f_arr.mean(), p_arr - p_arr.mean(), mode="full")
                lag = np.argmax(cross_corr) - (len(f_arr) - 1)
                phase_rad = float((lag / 64.0) * (2 * math.pi))

        # Shannon biodiversity entropy
        total_pop = len(self.biome.agents)
        if total_pop > 0:
            probs = [sum(1 for a in self.biome.agents if a.alive and a.niche == n) / total_pop for n in niches]
            shannon = float(-sum(p * math.log2(p) for p in probs if p > 0))
        else:
            shannon = 0.0

        living_ids = {a.agent_id for a in self.biome.agents if a.alive}
        max_depth = max((a.generation for a in self.biome.agents if a.alive), default=0)

        return SpeciationMetrics(
            divergence_index=divergence_idx,
            trait_f_ratios=f_ratios,
            niche_mean_traits=niche_means,
            predator_prey_phase_rad=phase_rad,
            predator_prey_correlation=corr,
            extinction_events=self.extinction_counter,
            active_lineages=len(living_ids),
            max_lineage_depth=max_depth,
            shannon_entropy=shannon,
        )

    def inject_cataclysm(self, cataclysm_type: str = "drought") -> Dict[str, Any]:
        """Triggers punctuated macro-environmental bottleneck / shock."""
        c_type = cataclysm_type.lower()
        impact = {}

        if c_type == "drought":
            # Deplete 80% of flora resources
            depleted = 0
            for r in self.biome.resources:
                if r.kind == "flora_bloom":
                    r.amount *= 0.2
                    depleted += 1
            impact = {"depleted_resources": depleted, "type": "drought", "description": "Severe resource drought"}

        elif c_type == "arctic_freeze":
            # Temperature drops 15 degrees, quadruples basal metabolism
            self.biome.temperature_c = 4.0
            for a in self.biome.agents:
                a.energy = max(10.0, a.energy * 0.7)
            impact = {"temperature_c": 4.0, "type": "arctic_freeze", "description": "Sudden freezing bottleneck"}

        elif c_type == "predator_bloom":
            # Spawns a high-energy apex predator wave
            self.biome.step(external_stimulus={"type": "trigger_predator_wave"})
            # Add an extra apex predator
            th = float(self.rng.uniform(0, math.pi * 2))
            ag_id = f"apex_shock_{self.biome.step_count}"
            pred = EcologyAgentState(
                agent_id=ag_id,
                niche="predator",
                x=0.0, y=0.0, vx=0.0, vy=0.0, heading=th,
                energy=98.0,
                arousal=0.9, dominance=0.95, adrenaline=0.6,
                generation=max((a.generation for a in self.biome.agents), default=0),
                genome=dict(self.biome._random_genome()),
            )
            self.biome.agents.append(pred)
            self.lineage_tree[ag_id] = LineageNode(
                agent_id=ag_id,
                parent_id=None,
                niche="predator",
                generation=pred.generation,
                birth_step=self.biome.step_count,
                death_step=None,
                genome=dict(pred.genome),
            )
            impact = {"spawned_predators": 1, "type": "predator_bloom", "description": "Apex predator surge"}

        # Broadcast shock wave
        self.biome.step(external_stimulus={"type": "broadcast_alarm", "x": 0.0, "y": 0.0})
        return {
            "status": "injected",
            "cataclysm": impact,
            "step": self.biome.step_count,
            "population_after": len(self.biome.agents),
        }

    def run_tournament_epochs(
        self,
        epochs: int = 5,
        steps_per_epoch: int = 50,
        on_epoch: Optional[Callable[[int, SpeciationMetrics], None]] = None,
    ) -> DarwinianTournamentReceipt:
        """Executes a multi-epoch Darwinian evolutionary tournament."""
        self._receipt_counter += 1
        rcpt_id = f"darw_rcpt_{int(time.time()*1000)}_{self._receipt_counter}"

        for ep in range(epochs):
            self.step_tournament(steps=steps_per_epoch)
            metrics = self.compute_speciation_divergence()
            if on_epoch is not None:
                try:
                    on_epoch(ep + 1, metrics)
                except Exception:
                    pass

        final_metrics = self.compute_speciation_divergence()
        receipt = DarwinianTournamentReceipt(
            receipt_id=rcpt_id,
            timestamp=time.time(),
            total_steps=self.biome.step_count,
            total_generations=final_metrics.max_lineage_depth,
            total_births=self.biome.births,
            total_deaths=self.biome.deaths,
            metrics=final_metrics,
            evidence=EvidenceReceipt(
                item_id=rcpt_id,
                kind=EvidenceKind.SYNTHETIC_SIMULATION,
                source="DarwinianEcologyTournamentEngine",
                metrics={
                    "total_steps": self.biome.step_count,
                    "divergence_index": final_metrics.divergence_index,
                    "shannon_entropy": final_metrics.shannon_entropy,
                    "max_lineage_depth": final_metrics.max_lineage_depth,
                    "extinctions": final_metrics.extinction_events,
                    "total_births": self.biome.births,
                    "total_deaths": self.biome.deaths,
                },
            ),
        )
        self.history_receipts.append(receipt)
        return receipt

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive status, phase portrait data, and speciation metrics."""
        metrics = self.compute_speciation_divergence()
        recent_series = list(self.population_series)[-128:]
        return {
            "status": "active",
            "step_count": self.biome.step_count,
            "population_count": len(self.biome.agents),
            "births": self.biome.births,
            "deaths": self.biome.deaths,
            "speciation_metrics": metrics.to_dict(),
            "recent_series": recent_series,
            "history_receipts_count": len(self.history_receipts),
            "last_receipt": self.history_receipts[-1].to_dict() if self.history_receipts else None,
        }
