"""Neuromorphic Hardware Efficiency Benchmarking & Pareto Frontiers.

Instruments and measures CPU wall-clock execution time, step latency, memory footprint,
FLOPs/MACs, and event-driven Synaptic Operations (SynOps) across model architectures:
  1. Dense Continuous Rate Model
  2. Sparse Pruned Connectome
  3. Leaky Integrate-and-Fire (LIF) Spiking Connectome
  4. Adaptive Exponential (AdEx) Spiking Connectome
Computes multi-objective Pareto efficiency frontiers for edge and neuromorphic deployment.
"""
from __future__ import annotations
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .spiking import AdExSpikingLayer, ContinuousRateLayer, LIFSpikingLayer


class NeuromorphicEfficiencyBenchmark:
    """Profiles computational throughput, latency, memory, and energy metrics."""

    JOULES_PER_MAC = 4.6e-12    # Standard 32-bit floating point multiply-accumulate (~4.6 pJ)
    JOULES_PER_SYNOP = 0.9e-12  # Event-driven neuromorphic synaptic operation (~0.9 pJ)

    def __init__(self, in_features: int = 32, num_neurons: int = 128, seed: int = 42):
        self.in_features = in_features
        self.num_neurons = num_neurons
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

    def profile_architecture(
        self,
        arch_type: str,
        steps: int = 100,
    ) -> Dict[str, Any]:
        """Profiles a specific neural architecture over `steps` simulation cycles."""
        x_inputs = [
            torch.tensor(self.rng.normal(0.0, 1.0, size=(self.in_features,)), dtype=torch.float32)
            for _ in range(steps)
        ]

        if arch_type == 'dense_rate':
            layer = ContinuousRateLayer(self.in_features, self.num_neurons)
        elif arch_type == 'sparse_rate':
            layer = ContinuousRateLayer(self.in_features, self.num_neurons)
            # Sparsify 80% of weights to simulate structural pruning
            with torch.no_grad():
                mask = torch.rand_like(layer.w_rec) > 0.8
                layer.w_rec.mul_(mask.float())
        elif arch_type == 'lif_spiking':
            layer = LIFSpikingLayer(self.in_features, self.num_neurons)
        elif arch_type == 'adex_spiking':
            layer = AdExSpikingLayer(self.in_features, self.num_neurons)
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")

        # Warmup
        for x in x_inputs[:5]:
            layer(x)
        layer.reset_state()

        # Profiling pass
        latencies_ms = []
        t0 = time.perf_counter()

        for x in x_inputs:
            t_step0 = time.perf_counter()
            with torch.no_grad():
                out = layer(x)
            dt_step = (time.perf_counter() - t_step0) * 1000.0
            latencies_ms.append(dt_step)

        total_wall_ms = (time.perf_counter() - t0) * 1000.0

        # Operation counts & energy proxy
        if hasattr(layer, 'total_macs'):
            total_ops = layer.total_macs
            op_type = 'MACs'
            energy_nj = (total_ops * self.JOULES_PER_MAC) * 1e9
        elif hasattr(layer, 'total_synops'):
            total_ops = layer.total_synops
            op_type = 'SynOps'
            energy_nj = (total_ops * self.JOULES_PER_SYNOP) * 1e9
        else:
            total_ops = self.num_neurons * self.in_features * steps
            op_type = 'Ops'
            energy_nj = (total_ops * self.JOULES_PER_MAC) * 1e9

        # Approximate parameter memory footprint
        param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
        param_kb = param_bytes / 1024.0

        return {
            'architecture': arch_type,
            'steps': steps,
            'num_neurons': self.num_neurons,
            'wall_clock_time_ms': round(total_wall_ms, 3),
            'mean_step_latency_ms': round(float(np.mean(latencies_ms)), 4),
            'p95_step_latency_ms': round(float(np.percentile(latencies_ms, 95)), 4),
            'total_operations': total_ops,
            'operation_type': op_type,
            'estimated_energy_nJ': round(energy_nj, 4),
            'parameter_memory_kb': round(param_kb, 2),
        }

    def compute_pareto_front(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifies non-dominated architectures on the (Latency, Energy) Pareto frontier."""
        pareto = []
        for cand in profiles:
            is_dominated = False
            for other in profiles:
                if other['architecture'] == cand['architecture']:
                    continue
                # Other dominates cand if it is strictly better in both latency and energy
                if (other['mean_step_latency_ms'] <= cand['mean_step_latency_ms'] and
                    other['estimated_energy_nJ'] <= cand['estimated_energy_nJ'] and
                    (other['mean_step_latency_ms'] < cand['mean_step_latency_ms'] or
                     other['estimated_energy_nJ'] < cand['estimated_energy_nJ'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(cand)
        return pareto

    def run_benchmark_suite(self, steps: int = 100) -> Dict[str, Any]:
        """Runs the complete neuromorphic efficiency comparison across all architectures."""
        architectures = ['dense_rate', 'sparse_rate', 'lif_spiking', 'adex_spiking']
        profiles = [self.profile_architecture(arch, steps=steps) for arch in architectures]
        pareto = self.compute_pareto_front(profiles)

        return {
            'num_neurons': self.num_neurons,
            'steps_profiled': steps,
            'profiles': {p['architecture']: p for p in profiles},
            'pareto_frontier': [p['architecture'] for p in pareto],
            'spiking_synops_efficiency_ratio': round(
                profiles[0]['total_operations'] / max(1, profiles[2]['total_operations']), 2
            ),
            'energy_savings_ratio': round(
                profiles[0]['estimated_energy_nJ'] / max(1e-6, profiles[2]['estimated_energy_nJ']), 2
            ),
        }
