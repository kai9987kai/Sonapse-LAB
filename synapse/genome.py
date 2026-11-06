"""Developmental Neural Genome & Activity-Dependent Neurogenesis.

Encodes morphogenesis, cell-type allocations, axonal growth cone affinity kernels,
and 3D spatial priors that unfold deterministically into executable connectomes.
Implements metabolic neurogenesis gated by ATP budget and localized prediction error,
growing new functional units into high-demand circuits during cognitive surplus.
"""
from __future__ import annotations
import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .stateio import digest


@dataclass
class DevelopmentalGenome:
    """Genetic specification for connectome morphogenesis and cell development."""

    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    excitatory_ratio: float = 0.8
    distance_decay_sigma: float = 0.35
    synapse_density: float = 0.12
    cell_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        'sensory': 0.20,
        'interneuron': 0.55,
        'modulatory': 0.10,
        'motor': 0.15,
    })
    atp_cost_per_neuron: float = 0.15
    atp_cost_per_synapse: float = 0.02
    neurogenesis_threshold_fe: float = 0.5
    atp_neurogenesis_min: float = 0.6

    def mutate(self, rng: np.random.Generator, rate: float = 0.1) -> DevelopmentalGenome:
        """Produces a mutated offspring genome."""
        child = copy.deepcopy(self)
        child.genome_id = uuid.uuid4().hex[:12]
        child.excitatory_ratio = float(np.clip(self.excitatory_ratio + rng.normal(0.0, rate * 0.05), 0.5, 0.95))
        child.distance_decay_sigma = float(np.clip(self.distance_decay_sigma * math.exp(rng.normal(0.0, rate)), 0.1, 1.0))
        child.synapse_density = float(np.clip(self.synapse_density * math.exp(rng.normal(0.0, rate)), 0.02, 0.4))
        return child

    def morphogenesis(self, num_neurons: int = 64, seed: int = 42) -> Dict[str, Any]:
        """Unfolds the genome into a 3D embedded, Dale's-law-compliant connectome."""
        rng = np.random.default_rng(seed)

        # 1. Generate 3D spatial anatomical coordinates
        positions = rng.uniform(-1.0, 1.0, size=(num_neurons, 3)).astype(np.float32)

        # 2. Assign cell types
        types = []
        n_sens = int(num_neurons * self.cell_type_distribution['sensory'])
        n_inter = int(num_neurons * self.cell_type_distribution['interneuron'])
        n_mod = int(num_neurons * self.cell_type_distribution['modulatory'])
        n_mot = num_neurons - (n_sens + n_inter + n_mod)

        types.extend(['sensory'] * n_sens)
        types.extend(['interneuron'] * n_inter)
        types.extend(['modulatory'] * n_mod)
        types.extend(['motor'] * n_mot)
        rng.shuffle(types)

        # 3. Assign excitatory vs inhibitory signs according to Dale's law
        signs = np.array([
            1.0 if rng.random() < self.excitatory_ratio else -1.0
            for _ in range(num_neurons)
        ], dtype=np.float32)

        # 4. Synthesize connectivity using distance affinity kernel:
        # P(i -> j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
        diff = positions[:, None, :] - positions[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        prob = np.exp(-dist_sq / (2.0 * (self.distance_decay_sigma ** 2)))
        np.fill_diagonal(prob, 0.0)  # No autapses

        # Sample edges
        adj = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        target_edges = int(num_neurons * num_neurons * self.synapse_density)
        flat_probs = prob.flatten()
        flat_probs /= flat_probs.sum()

        chosen_indices = rng.choice(num_neurons * num_neurons, size=target_edges, replace=False, p=flat_probs)
        for idx in chosen_indices:
            pre = idx // num_neurons
            post = idx % num_neurons
            w = float(rng.exponential(0.1))
            adj[post, pre] = signs[pre] * w

        return {
            'num_neurons': num_neurons,
            'positions': positions,
            'cell_types': types,
            'signs': signs,
            'adjacency_matrix': adj,
            'genome_id': self.genome_id,
        }


class MetabolicNeurogenesisEngine:
    """Manages energy budget and localized activity-dependent neurogenesis."""

    def __init__(
        self,
        genome: Optional[DevelopmentalGenome] = None,
        max_atp: float = 1.0,
        replenish_rate: float = 0.05,
    ):
        self.genome = genome or DevelopmentalGenome()
        self.max_atp = max_atp
        self.atp = max_atp
        self.replenish_rate = replenish_rate
        self.neurogenesis_events: List[Dict[str, Any]] = []

    def step_metabolism(self, mean_activity: float, total_neurons: int) -> float:
        """Updates ATP reserve based on metabolic expenditure:

        E = E_baseline + activity * E_spikes
        """
        consumption = 0.01 + mean_activity * 0.04 + total_neurons * 0.0005
        self.atp = float(np.clip(self.atp + self.replenish_rate - consumption, 0.0, self.max_atp))
        return self.atp

    def evaluate_neurogenesis(
        self,
        agent,
        local_free_energy: float,
        hotspot_neuropil: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Evaluates whether metabolic and predictive coding conditions warrant neurogenesis.

        If ATP > threshold and Free Energy > threshold, triggers transaction-safe neuron insertion.
        """
        if self.atp < self.genome.atp_neurogenesis_min:
            return None
        if local_free_energy < self.genome.neurogenesis_threshold_fe:
            return None

        # Execute neurogenesis
        from .evolution import TopologyEvolutionEngine
        evo = TopologyEvolutionEngine()

        cost = self.genome.atp_cost_per_neuron + 4 * self.genome.atp_cost_per_synapse
        if self.atp < cost:
            return None

        receipt = evo.add_neuron(agent, neuropil=hotspot_neuropil)
        self.atp -= cost

        event = {
            'event_id': uuid.uuid4().hex[:12],
            'new_neuron_index': receipt.details['new_neuron_index'],
            'neuropil': hotspot_neuropil,
            'local_free_energy': float(local_free_energy),
            'atp_remaining': round(self.atp, 4),
            'atp_cost': round(cost, 4),
            'parent_digest': receipt.parent_digest,
            'resulting_digest': receipt.resulting_digest,
        }
        self.neurogenesis_events.append(event)
        return event
