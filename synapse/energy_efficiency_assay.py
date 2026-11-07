"""Energy & Compute Efficiency Assay.

Compares full-dense matrix recurrent updates against event-driven sparse updates
across activity levels and threshold values. Measures theoretical SynOps, FLOPs,
wall-clock execution latency, and numerical fidelity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from synapse.event_driven_backend import EventDrivenSparseBackend


@dataclass
class EfficiencyTrialResult:
    threshold: float
    sparsity_ratio: float
    synaptic_events: int
    dense_flops: int
    theoretical_speedup: float
    wallclock_dense_ms: float
    wallclock_sparse_ms: float
    max_absolute_error: float


@dataclass
class EnergyEfficiencyAssayReport:
    num_neurons: int
    trials: List[EfficiencyTrialResult]
    recommended_threshold: float
    summary: str


class EnergyEfficiencyAssay:
    """Rigorous evaluation of dense vs sparse event-driven simulation costs."""

    def __init__(self, num_neurons: int = 1000, num_steps: int = 50, seed: int = 42):
        self.num_neurons = num_neurons
        self.num_steps = num_steps
        self.seed = seed

    def run_assay(self, thresholds: Optional[List[float]] = None) -> EnergyEfficiencyAssayReport:
        if thresholds is None:
            thresholds = [0.01, 0.05, 0.1, 0.2]

        torch.manual_seed(self.seed)
        N = self.num_neurons
        # Weight matrix with ~10% sparsity
        W = torch.randn(N, N) * 0.05
        mask = torch.rand(N, N) > 0.90
        W = W * mask.float()

        # Input drive
        inputs = [torch.randn(N) * 0.1 for _ in range(self.num_steps)]

        # Baseline: Dense run
        t0 = time.perf_counter()
        r_dense = torch.zeros(N)
        dense_history = []
        for inp in inputs:
            r_dense = torch.tanh(torch.mv(W, r_dense) + inp)
            dense_history.append(r_dense.clone())
        t_dense = (time.perf_counter() - t0) * 1000.0

        trials = []
        best_thresh = 0.05
        best_balance = -1.0

        for th in thresholds:
            backend = EventDrivenSparseBackend(delta_threshold=th)
            t_s0 = time.perf_counter()
            r_sparse = torch.zeros(N)
            sparse_history = []
            synops_accum = 0
            dense_flops_accum = 0
            sparsity_accum = 0.0

            for inp in inputs:
                # Sparse event detection on previous activation
                sparse_deltas, active_idx, metrics = backend.detect_events(r_sparse)
                synops_accum += metrics.synaptic_events
                dense_flops_accum += metrics.dense_equivalent_flops
                sparsity_accum += metrics.sparsity_ratio

                # Update state: only propagate active deltas
                if active_idx.numel() > 0:
                    delta_W = W[:, active_idx]
                    inc = torch.mv(delta_W, sparse_deltas[active_idx])
                    r_sparse = torch.tanh(r_sparse + inc + inp)
                else:
                    r_sparse = torch.tanh(r_sparse + inp)
                sparse_history.append(r_sparse.clone())

            t_sparse = (time.perf_counter() - t_s0) * 1000.0

            # Error relative to dense
            max_err = 0.0
            for d, s in zip(dense_history, sparse_history):
                err = float(torch.max(torch.abs(d - s)).item())
                if err > max_err:
                    max_err = err

            avg_sparsity = sparsity_accum / self.num_steps
            speedup = dense_flops_accum / max(synops_accum * 2, 1)

            # Heuristic balance: high speedup with acceptable error (< 0.15)
            balance = speedup / (1.0 + max_err * 10.0)
            if balance > best_balance and max_err < 0.2:
                best_balance = balance
                best_thresh = th

            trials.append(
                EfficiencyTrialResult(
                    threshold=th,
                    sparsity_ratio=round(avg_sparsity, 4),
                    synaptic_events=int(synops_accum),
                    dense_flops=int(dense_flops_accum),
                    theoretical_speedup=round(speedup, 2),
                    wallclock_dense_ms=round(t_dense, 2),
                    wallclock_sparse_ms=round(t_sparse, 2),
                    max_absolute_error=round(max_err, 4),
                )
            )

        sample_trial = trials[1] if len(trials) > 1 else (trials[0] if trials else None)
        if sample_trial:
            summary = (
                f"Event-driven sparse execution achieves theoretical speedup of "
                f"{sample_trial.theoretical_speedup}x at threshold {sample_trial.threshold} with "
                f"max error {sample_trial.max_absolute_error} over {N} neurons."
            )
        else:
            summary = f"No efficiency trials run over {N} neurons."

        return EnergyEfficiencyAssayReport(
            num_neurons=N,
            trials=trials,
            recommended_threshold=best_thresh,
            summary=summary,
        )
