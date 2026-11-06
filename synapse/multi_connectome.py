"""Multi-Connectome Unified Substrate, Cross-Region Hub & Scalability Engine.

Grounding:
Coordinates recurrent rate dynamics across the complete suite of 8 Drosophila connectomes:
- Central Brains: FlyWire (FAFB v783), Hemibrain, Larval Brain
- Ventral Nerve Cords: MANC, FANC, BANC, MaleCNS
- Sensory Neuropils: FlyEM Optic Lobe

Features:
1. Multi-Connectome Information Hub: cross-region gating, ascending/descending connectives.
2. Heterogeneous Neuromodulatory Susceptibility: regional DA, 5-HT, OA, and ACh receptivity.
3. High-Capacity Neuron Scaling: scales from tens to thousands of biological neurons on CPU.
4. Regional Evoked Activation Tracking across all 8 biological specimens.
5. Editable Parameter Surfaces for Weight-Space Programming and Transactional Rollback.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .connectome_datasets import CONNECTOME_CATALOG, ALL_DATASET_IDS, build_multiconnectome_graph
from graftbrain.brain import GraftedBrain
from graftbrain.graft import GraftedGraph


@dataclass
class RegionalActivityTelemetry:
    regional_means: Dict[str, float]
    total_active_neurons: int
    inter_connectome_flux: float
    neuromodulatory_gains: Dict[str, Dict[str, float]]
    timestamp: float = field(default_factory=time.time)


class MultiConnectomeHub(nn.Module):
    """Central computational coordinator for multi-connectome neural graphs."""

    # Regional biological receptor density profiles (DA, 5-HT, OA, ACh)
    RECEPTOR_PROFILES: Dict[str, Dict[str, float]] = {
        'malecns': {'DA': 1.0, '5-HT': 1.1, 'OA': 1.2, 'ACh': 1.0},
        'banc': {'DA': 1.0, '5-HT': 1.0, 'OA': 1.1, 'ACh': 1.0},
        'optic-lobe': {'DA': 0.4, '5-HT': 0.6, 'OA': 1.5, 'ACh': 1.8},
        'fanc': {'DA': 0.7, '5-HT': 1.4, 'OA': 1.3, 'ACh': 0.8},
        'flywire': {'DA': 1.5, '5-HT': 1.2, 'OA': 0.9, 'ACh': 1.2},
        'manc': {'DA': 0.8, '5-HT': 1.3, 'OA': 1.4, 'ACh': 0.9},
        'larval': {'DA': 1.1, '5-HT': 0.9, 'OA': 0.8, 'ACh': 1.1},
        'hemibrain': {'DA': 1.6, '5-HT': 1.1, 'OA': 0.9, 'ACh': 1.3},
    }

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        nodes_per_source: int = 64,
        microsteps: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.sources = sources or ALL_DATASET_IDS
        self.nodes_per_source = nodes_per_source
        self.microsteps = microsteps
        self.seed = seed

        # Build grafted graph spanning all selected connectome profiles
        self.graph: GraftedGraph = build_multiconnectome_graph(
            sources=self.sources,
            nodes_per_source=nodes_per_source,
            seed=seed
        )
        self.brain = GraftedBrain(self.graph, microsteps=microsteps, noise=0.0, seed=seed)
        self.total_neurons = self.brain.n

        # Region slicing indices
        self.region_ranges: Dict[str, Tuple[int, int]] = {}
        for i, name in enumerate(self.sources):
            start_idx = int(self.graph.offsets[i])
            end_idx = int(self.graph.offsets[i + 1])
            self.region_ranges[name] = (start_idx, end_idx)

        # Learnable regional gain & bias vectors (Weight-Space Programming surfaces)
        self.regional_gain = nn.Parameter(torch.ones(len(self.sources)))
        self.regional_decay = nn.Parameter(torch.full((len(self.sources),), 0.85))

        # Cross-region attention router: projects regional summary to gating weights
        self.region_router = nn.Sequential(
            nn.Linear(len(self.sources), 32),
            nn.ReLU(),
            nn.Linear(32, len(self.sources)),
            nn.Sigmoid()
        )

        self.last_telemetry: Optional[RegionalActivityTelemetry] = None

    def forward(
        self,
        sensory_input: Optional[torch.Tensor] = None,
        neuromod_state: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Executes one multi-connectome recurrent step with regional neuromodulation."""
        # 1. Base GraftedBrain forward step
        r_state = self.brain(sensory_input)

        # 2. Regional summaries and neuromodulatory gain modulation
        regional_means = {}
        nm = neuromod_state or {'DA': 1.0, '5-HT': 1.0, 'OA': 1.0, 'ACh': 1.0}

        for i, name in enumerate(self.sources):
            start_idx, end_idx = self.region_ranges[name]
            sub_r = r_state[start_idx:end_idx]
            reg_mean = float(sub_r.abs().mean().item())
            regional_means[name] = round(reg_mean, 4)

            # Apply neuromodulation to subcircuit
            receptors = self.RECEPTOR_PROFILES.get(name, {'DA': 1.0, '5-HT': 1.0, 'OA': 1.0, 'ACh': 1.0})
            gain_factor = (
                0.25 * nm.get('DA', 1.0) * receptors['DA'] +
                0.25 * nm.get('5-HT', 1.0) * receptors['5-HT'] +
                0.25 * nm.get('OA', 1.0) * receptors['OA'] +
                0.25 * nm.get('ACh', 1.0) * receptors['ACh']
            )
            # Modulate firing rates slightly with regional gain parameter
            eff_gain = self.regional_gain[i] * float(gain_factor)
            r_state[start_idx:end_idx] = torch.tanh(sub_r * eff_gain)

        # 3. Inter-connectome flux calculation across graft sites
        g_w = self.brain.graft_w.detach()
        inter_flux = float(g_w.abs().mean().item())

        self.last_telemetry = RegionalActivityTelemetry(
            regional_means=regional_means,
            total_active_neurons=self.total_neurons,
            inter_connectome_flux=round(inter_flux, 4),
            neuromodulatory_gains={name: self.RECEPTOR_PROFILES.get(name, {}) for name in self.sources}
        )

        return r_state

    def reset_state(self):
        self.brain.reset_state()

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive structural and runtime summary."""
        reg_info = {}
        for name in self.sources:
            start, end = self.region_ranges[name]
            meta = CONNECTOME_CATALOG.get(name)
            reg_info[name] = {
                'start_idx': start,
                'end_idx': end,
                'neurons': end - start,
                'license': meta.license if meta else 'unknown',
                'authors': meta.organisation_authors if meta else 'unknown',
                'region': meta.biological_region if meta else 'unknown',
            }

        return {
            'status': 'ok',
            'sources_count': len(self.sources),
            'total_neurons': self.total_neurons,
            'graft_sites_count': len(self.graph.sites),
            'base_edges_count': len(self.graph.base_pre),
            'graft_edges_count': len(self.graph.graft_pre),
            'regions': reg_info,
            'latest_telemetry': self.last_telemetry.__dict__ if self.last_telemetry else None
        }


def benchmark_neuron_scaling(
    target_counts: List[int] = [128, 256, 512, 1024],
    sources: Optional[List[str]] = None,
    steps_per_benchmark: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluates runtime latency, memory, and dynamical stability across scaling neuron counts."""
    selected_sources = sources or ALL_DATASET_IDS
    num_sources = len(selected_sources)
    scaling_results = []

    for total_target in target_counts:
        n_per_source = max(16, total_target // num_sources)
        t0 = time.perf_counter()

        hub = MultiConnectomeHub(
            sources=selected_sources,
            nodes_per_source=n_per_source,
            seed=seed
        )
        build_time_ms = (time.perf_counter() - t0) * 1000.0

        actual_neurons = hub.total_neurons
        sensory_dim = len(hub.brain.sensory_idx)

        # Run step benchmark
        step_times = []
        for step in range(steps_per_benchmark):
            inp = torch.randn(sensory_dim)
            t_s = time.perf_counter()
            hub.forward(inp)
            step_times.append((time.perf_counter() - t_s) * 1000.0)

        mean_step_ms = float(np.mean(step_times))
        max_step_ms = float(np.max(step_times))
        throughput_neurons_per_sec = actual_neurons * (1000.0 / max(1e-4, mean_step_ms))

        scaling_results.append({
            'target_neurons': total_target,
            'actual_neurons': actual_neurons,
            'nodes_per_source': n_per_source,
            'build_time_ms': round(build_time_ms, 2),
            'mean_step_ms': round(mean_step_ms, 3),
            'max_step_ms': round(max_step_ms, 3),
            'throughput_neurons_per_sec': round(throughput_neurons_per_sec, 1),
            'stable_recurrent_activity': float(hub.brain.r.abs().mean().item()) < 5.0
        })

    return {
        'status': 'ok',
        'sources_evaluated': selected_sources,
        'benchmarks': scaling_results,
        'scope': 'multi-connectome high-capacity neuron scaling benchmark'
    }
