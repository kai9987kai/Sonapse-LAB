"""Compute Profiler for Sonapse NeuroWave-X2.

Measures parameter counts, buffer allocations, analytical FLOPs, MACs,
and empirical CPU execution latency percentiles (p50, p95, p99).
Strictly separates analytical estimates from empirical measurements.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class ComputeProfiler:
    """Profiles memory footprint, computational complexity, and execution latency."""

    def __init__(self, history_size: int = 100) -> None:
        self.history_size = int(history_size)
        self.latency_history: List[float] = []

    def record_latency(self, elapsed_seconds: float) -> None:
        self.latency_history.append(float(elapsed_seconds))
        if len(self.latency_history) > self.history_size:
            self.latency_history.pop(0)

    def measure_model_memory(self, module: nn.Module) -> Dict[str, Any]:
        """Calculates exact parameter counts, buffer memory, and estimated optimizer state memory."""
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())

        # Adam/AdamW maintains 2 state tensors (exp_avg, exp_avg_sq) per trainable parameter
        estimated_adam_bytes = trainable_params * 4 * 2

        return {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "parameter_memory_kb": round(param_bytes / 1024.0, 3),
            "buffer_memory_kb": round(buffer_bytes / 1024.0, 3),
            "estimated_optimizer_memory_kb": round(estimated_adam_bytes / 1024.0, 3),
            "total_memory_footprint_kb": round((param_bytes + buffer_bytes + estimated_adam_bytes) / 1024.0, 3),
        }

    def estimate_step_flops(
        self,
        active_routes: int,
        message_dim: int,
        target_size: int,
        spectral_modes: int = 0,
        uses_attention: bool = False,
    ) -> Dict[str, float]:
        """Calculates analytical floating-point operations (FLOPs) per communication step.
        
        Analytical model:
            - Linear projection E(p): 2 * pool_dim * hidden + 2 * hidden * msg_dim
            - Spectral synthesis: 2 * spectral_modes * msg_dim
            - Decoder projection D(m): 2 * msg_dim * target_size
            - Phase CTC gating: 3 FLOPs (cosine + add + mult)
        """
        encoder_flops = 2 * 32 * 64 + 2 * 64 * message_dim
        spectral_flops = 2 * spectral_modes * message_dim if spectral_modes > 0 else 0
        decoder_flops = 2 * message_dim * target_size
        route_flops = encoder_flops + spectral_flops + decoder_flops + 5
        total_flops = active_routes * route_flops
        total_macs = total_flops / 2.0

        return {
            "analytical_flops_per_step": float(total_flops),
            "analytical_macs_per_step": float(total_macs),
            "flops_per_route": float(route_flops),
        }

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Returns empirical latency distribution over recorded steps (p50, p95, p99 in milliseconds)."""
        if len(self.latency_history) == 0:
            return {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0}

        arr = np.array(self.latency_history, dtype=np.float64) * 1000.0  # to ms
        return {
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_p99_ms": float(np.percentile(arr, 99)),
            "latency_mean_ms": float(np.mean(arr)),
        }
