"""Memory Interference & Capacity Scaling Assay.

Empirically benchmarks capacity scaling (superposition of 1..16 pairs), SNR degradation,
distractor resistance, and query latency for Holographic Reduced Representations (HRR).

Scientific note: Results demonstrate mathematical associative capacity bounds of HRRs
under circular convolution and vector cleanup; this is an algebraic memory model,
not a claim of biological episodic memory.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from synapse.compositional_retrieval import CleanupMatch, CodebookCleanupMemory, CompositionalRetriever
from synapse.relational_memory import HRRCore, RelationalMemoryBank


@dataclass
class CapacityDataPoint:
    num_pairs: int
    mean_snr_db: float
    retrieval_accuracy: float
    mean_cosine_similarity: float
    mean_latency_ms: float


@dataclass
class MemoryInterferenceAssayResult:
    dim: int
    capacity_curve: List[CapacityDataPoint]
    max_clean_capacity: int  # Max pairs with 100% top-1 accuracy
    distractor_resistance_score: float  # Top-1 accuracy under 100 distractor symbols
    notes: str


class MemoryInterferenceAssay:
    """Evaluates interference, capacity scaling, and distractor resistance."""

    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def run_assay(self, max_pairs: int = 16, num_trials: int = 10, num_distractors: int = 50) -> MemoryInterferenceAssayResult:
        torch.manual_seed(self.seed)

        # Setup cleanup codebook with distractors
        cleanup = CodebookCleanupMemory(dim=self.dim, similarity_threshold=0.2)
        distractor_symbols = {}
        for d_idx in range(num_distractors):
            v = torch.randn(self.dim)
            v = v / torch.norm(v, p=2)
            distractor_symbols[f"distractor_{d_idx}"] = v
        cleanup.register_symbols(distractor_symbols)

        capacity_points: List[CapacityDataPoint] = []
        max_clean_capacity = 0

        for n_pairs in range(1, max_pairs + 1):
            snr_vals = []
            acc_vals = []
            cos_vals = []
            latencies = []

            for _ in range(num_trials):
                bank = RelationalMemoryBank(dim=self.dim, max_slots=n_pairs * 2 + 10)
                trial_cleanup = CodebookCleanupMemory(dim=self.dim, similarity_threshold=0.2)
                trial_cleanup.register_symbols(distractor_symbols)

                roles = []
                fillers = []
                for p in range(n_pairs):
                    r_name = f"role_{p}"
                    f_name = f"filler_{p}"
                    r_vec = bank.hrr.random_unitary()
                    f_vec = bank.hrr.random_unitary()
                    bank.symbols[r_name] = r_vec
                    bank.symbols[f_name] = f_vec
                    trial_cleanup.register_symbol(r_name, r_vec)
                    trial_cleanup.register_symbol(f_name, f_vec)
                    roles.append(r_name)
                    fillers.append(f_name)

                # Bind pairs and superpose
                bound_pairs = []
                for r, f in zip(roles, fillers):
                    bound_pairs.append(bank.hrr.bind(bank.symbols[r], bank.symbols[f]))

                trace = bank.hrr.superpose(bound_pairs)

                # Query each pair
                retriever = CompositionalRetriever(bank, trial_cleanup)
                for r, expected_f in zip(roles, fillers):
                    t0 = time.perf_counter()
                    unbound = bank.hrr.unbind(trace, bank.symbols[r])
                    matches = trial_cleanup.lookup(unbound, top_k=3)
                    t1 = time.perf_counter()

                    latencies.append((t1 - t0) * 1000.0)

                    # Check top-1 match
                    if matches and matches[0].symbol == expected_f:
                        acc_vals.append(1.0)
                    else:
                        acc_vals.append(0.0)

                    target_vec = np.asarray(bank.symbols[expected_f], dtype=np.float32)
                    unb_arr = np.asarray(unbound, dtype=np.float32)
                    unb_norm = float(np.linalg.norm(unb_arr)) + 1e-8
                    unb_unit = unb_arr / unb_norm
                    t_norm = float(np.linalg.norm(target_vec)) + 1e-8
                    t_unit = target_vec / t_norm

                    target_cos = float(np.dot(unb_unit, t_unit))
                    cos_vals.append(target_cos)

                    # Compute SNR: signal energy vs noise energy
                    proj = target_cos * t_unit
                    noise = unb_unit - proj
                    sig_pow = target_cos ** 2
                    noise_pow = float(np.dot(noise, noise)) + 1e-8
                    snr_db = 10.0 * math.log10(max(sig_pow / noise_pow, 1e-6))
                    snr_vals.append(snr_db)

            mean_acc = sum(acc_vals) / max(len(acc_vals), 1)
            mean_snr = sum(snr_vals) / max(len(snr_vals), 1)
            mean_cos = sum(cos_vals) / max(len(cos_vals), 1)
            mean_lat = sum(latencies) / max(len(latencies), 1)

            if mean_acc >= 0.999:
                max_clean_capacity = n_pairs

            capacity_points.append(
                CapacityDataPoint(
                    num_pairs=n_pairs,
                    mean_snr_db=round(mean_snr, 2),
                    retrieval_accuracy=round(mean_acc, 3),
                    mean_cosine_similarity=round(mean_cos, 4),
                    mean_latency_ms=round(mean_lat, 3),
                )
            )

        # Distractor resistance test at 4 pairs with 100 distractors
        distractor_accs = []
        for _ in range(5):
            bank = RelationalMemoryBank(dim=self.dim)
            heavy_cleanup = CodebookCleanupMemory(dim=self.dim, similarity_threshold=0.15)
            # 100 distractors
            for d_idx in range(100):
                v = torch.randn(self.dim)
                v = v / torch.norm(v, p=2)
                heavy_cleanup.register_symbol(f"heavy_distractor_{d_idx}", v)

            roles = [f"r_{i}" for i in range(4)]
            fillers = [f"f_{i}" for i in range(4)]
            pairs = []
            for r, f in zip(roles, fillers):
                rv = bank.hrr.random_unitary()
                fv = bank.hrr.random_unitary()
                bank.symbols[r] = rv
                bank.symbols[f] = fv
                heavy_cleanup.register_symbol(r, rv)
                heavy_cleanup.register_symbol(f, fv)
                pairs.append(bank.hrr.bind(rv, fv))
            trace = bank.hrr.superpose(pairs)
            retriever = CompositionalRetriever(bank, heavy_cleanup)
            for r, f in zip(roles, fillers):
                ok, _ = retriever.verify_binding(r, f, memory_vector=trace)
                distractor_accs.append(1.0 if ok else 0.0)

        distractor_score = sum(distractor_accs) / max(len(distractor_accs), 1)

        return MemoryInterferenceAssayResult(
            dim=self.dim,
            capacity_curve=capacity_points,
            max_clean_capacity=max_clean_capacity,
            distractor_resistance_score=round(distractor_score, 3),
            notes="Evaluated circular-convolution HRR capacity with cosine-cleanup codebook. SNR drops gracefully with sqrt(N) crosstalk.",
        )
