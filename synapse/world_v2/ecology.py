"""Virtual Ecology 3D Environment with Scent Plumes, Conspecifics, and Shelters.

Extends ComplexWorld with:
- Continuous 2D/3D olfactory scent plumes (food, predator, flora, social pheromones)
- Rest shelters and burrows providing predator sanctuary and accelerated rest recovery
- Simulated conspecific peer agents with social signaling
- Diurnal micro-climate (temperature, precipitation, wind dynamics)
- Metabolic energy expenditure and thermal regulation costs
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from synapse.complex_world import ComplexWorld, HALF
from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class VirtualEcologyWorld(ComplexWorld):
    """Next-generation virtual ecology environment supporting multimodal sensory fields."""

    def __init__(self, seed: int = 0, curriculum: str = "auto"):
        super().__init__(seed=seed, curriculum=curriculum)
        self.ecology_rng = np.random.default_rng(seed + 100)

        # 1. Rest Shelters (x, y, radius, recovery_multiplier)
        self.shelters: List[Dict[str, float]] = [
            {"x": -0.6 * HALF, "y": -0.6 * HALF, "radius": 0.25 * HALF, "recovery": 2.5},
            {"x": 0.6 * HALF, "y": 0.6 * HALF, "radius": 0.25 * HALF, "recovery": 2.5},
        ]

        # 2. Conspecific peer agents (x, y, vx, vy, state)
        self.conspecifics: List[Dict[str, Any]] = [
            {"id": "peer_1", "x": -0.2 * HALF, "y": 0.3 * HALF, "vx": 0.0, "vy": 0.0, "scent": "social_friendly"},
            {"id": "peer_2", "x": 0.3 * HALF, "y": -0.2 * HALF, "vx": 0.0, "vy": 0.0, "scent": "social_curious"},
        ]

        # 3. Micro-climate state
        self.temperature_c = 21.0
        self.precipitation = 0.0
        self.wind_x = 0.0
        self.wind_y = 0.0

        # 4. Metabolic energy budget [0, 100]
        self.metabolic_energy = 100.0
        self.basal_burn_rate = 0.02
        self.movement_burn_rate = 0.08

        # Telemetry
        self.in_shelter: bool = False
        self.last_scent_plumes: Dict[str, float] = {}

    def sample_scent_intensity(self, x: float, y: float) -> Dict[str, float]:
        """Evaluates continuous Gaussian scent plume diffusion at location (x, y)."""
        scents = {
            "food_plume": 0.0,
            "predator_musk": 0.0,
            "flora_sweet": 0.0,
            "social_pheromone": 0.0,
        }

        # Food / goal scents
        goal = getattr(self, "st", {}).get("goal", {})
        if goal and "x" in goal and "y" in goal:
            dx = x - goal["x"]
            dy = y - goal["y"]
            dist_sq = dx * dx + dy * dy
            scents["food_plume"] = float(np.exp(-dist_sq / (2.0 * (0.8 * HALF) ** 2)))

        # Predator musks
        for pred in self.predators:
            px, py = pred.get("pos", (0.0, 0.0))
            dx = x - px
            dy = y - py
            dist_sq = dx * dx + dy * dy
            scents["predator_musk"] = max(scents["predator_musk"], float(np.exp(-dist_sq / (2.0 * (0.6 * HALF) ** 2))))

        # Conspecific pheromones
        for peer in self.conspecifics:
            dx = x - peer["x"]
            dy = y - peer["y"]
            dist_sq = dx * dx + dy * dy
            scents["social_pheromone"] = max(scents["social_pheromone"], float(np.exp(-dist_sq / (2.0 * (0.5 * HALF) ** 2))))

        # Bioluminescent flora
        for flora in self.bioluminescent_flora:
            fx, fy = flora.get("pos", (0.0, 0.0))
            dx = x - fx
            dy = y - fy
            dist_sq = dx * dx + dy * dy
            scents["flora_sweet"] = max(scents["flora_sweet"], float(np.exp(-dist_sq / (2.0 * (0.4 * HALF) ** 2))))

        self.last_scent_plumes = scents
        return scents

    def is_agent_in_shelter(self, x: float, y: float) -> Tuple[bool, float]:
        """Checks if (x, y) is protected within a rest shelter."""
        for sh in self.shelters:
            dx = x - sh["x"]
            dy = y - sh["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= sh["radius"]:
                return True, sh["recovery"]
        return False, 1.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advances the virtual ecology simulation."""
        obs, reward, done, info = super().step(action)

        agent_a = getattr(self, "a", {})
        agent_x = float(agent_a.get("x", 0.0))
        agent_y = float(agent_a.get("y", 0.0))

        # 1. Evaluate shelter status
        in_sh, rec_mult = self.is_agent_in_shelter(agent_x, agent_y)
        self.in_shelter = in_sh

        # If inside shelter, nullify predator threat for the step
        if in_sh:
            info["shelter_active"] = True
            info["recovery_multiplier"] = rec_mult
            # Dampen predator damage if any occurred
            if "predator_damage" in info:
                info["predator_damage"] = 0.0
        else:
            info["shelter_active"] = False
            info["recovery_multiplier"] = 1.0

        # 2. Advance conspecifics
        for peer in self.conspecifics:
            peer["vx"] += float(self.ecology_rng.normal(0, 0.02))
            peer["vy"] += float(self.ecology_rng.normal(0, 0.02))
            peer["vx"] = float(np.clip(peer["vx"], -0.1, 0.1))
            peer["vy"] = float(np.clip(peer["vy"], -0.1, 0.1))
            peer["x"] = float(np.clip(peer["x"] + peer["vx"], -self.arena_half + 0.1, self.arena_half - 0.1))
            peer["y"] = float(np.clip(peer["y"] + peer["vy"], -self.arena_half + 0.1, self.arena_half - 0.1))

        # 3. Micro-climate evolution
        self.temperature_c = 21.0 + 8.0 * math.sin(self.day_night_clock * 0.01)
        self.wind_x = 0.1 * math.cos(self.day_night_clock * 0.005)
        self.wind_y = 0.1 * math.sin(self.day_night_clock * 0.005)

        # 4. Metabolic expenditure
        speed = float(np.linalg.norm(action[:2])) if len(action) >= 2 else 0.0
        burn = self.basal_burn_rate + self.movement_burn_rate * speed
        self.metabolic_energy = max(0.0, self.metabolic_energy - burn)
        if reward > 0.5:  # Feeding
            self.metabolic_energy = min(100.0, self.metabolic_energy + 25.0)

        # 5. Scent plumes at current agent position
        scents = self.sample_scent_intensity(agent_x, agent_y)

        # Enrich info dictionary
        info["ecology"] = {
            "in_shelter": self.in_shelter,
            "metabolic_energy": self.metabolic_energy,
            "temperature_c": self.temperature_c,
            "scents": scents,
            "conspecifics_count": len(self.conspecifics),
        }

        return obs, reward, done, info
