"""Resource Budget & Cost-Aware Communication Objective for Sonapse NeuroWave-X2.

Enforces bandwidth and compute limits, computes multi-objective penalties,
and calculates cost-aware transmission eligibility.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ResourceBudgetController:
    """Manages computational and bandwidth quotas, and computes multi-objective regularization."""

    def __init__(
        self,
        max_wire_bits_per_step: int = 2048,
        max_flops_per_step: float = 1e6,
        lambda_bits: float = 1e-4,
        lambda_compute: float = 1e-6,
        lambda_latency: float = 1e-3,
        lambda_stability: float = 1e-2,
    ) -> None:
        self.max_wire_bits_per_step = int(max_wire_bits_per_step)
        self.max_flops_per_step = float(max_flops_per_step)
        self.lambda_bits = float(lambda_bits)
        self.lambda_compute = float(lambda_compute)
        self.lambda_latency = float(lambda_latency)
        self.lambda_stability = float(lambda_stability)

        # Step accounting
        self.accumulated_wire_bits = 0
        self.accumulated_flops = 0.0
        self.budget_exceeded_count = 0

    def check_within_budget(self, prospective_wire_bits: int, prospective_flops: float) -> bool:
        """Determines if prospective communication step respects allocated budgets."""
        if prospective_wire_bits > self.max_wire_bits_per_step:
            self.budget_exceeded_count += 1
            return False
        if prospective_flops > self.max_flops_per_step:
            self.budget_exceeded_count += 1
            return False
        return True

    def compute_communication_loss_penalties(
        self,
        total_wire_bits: float,
        total_flops: float,
        mean_latency: float = 0.0,
        residual_norm: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Calculates differentiable resource penalty terms to encourage efficient communication.
        
        Terminology note:
            - L_bits: wire-cost proxy
            - L_compute: compute-cost proxy
        """
        loss_bits = self.lambda_bits * float(total_wire_bits)
        loss_compute = self.lambda_compute * float(total_flops)
        loss_latency = self.lambda_latency * float(mean_latency)
        # Stability penalty: discourages unbounded residual perturbations
        loss_stability = self.lambda_stability * float(residual_norm ** 2)

        total_penalty = loss_bits + loss_compute + loss_latency + loss_stability

        return {
            "loss_comm_total_penalty": torch.tensor(total_penalty, dtype=torch.float32),
            "loss_comm_bits": torch.tensor(loss_bits, dtype=torch.float32),
            "loss_comm_compute": torch.tensor(loss_compute, dtype=torch.float32),
            "loss_comm_stability": torch.tensor(loss_stability, dtype=torch.float32),
        }
