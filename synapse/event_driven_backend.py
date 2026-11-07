"""Event-Driven Sparse Update Engine.

Prioritizes active neurons and transmits updates only when activation deltas exceed
a configurable threshold. Tracks theoretical Synaptic Operations (SynOps) vs dense FLOPs.

Scientific note: This is an algorithmic event-driven execution backend for efficiency
in digital neural simulation; it is not a neuromorphic hardware emulation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


@dataclass
class EventDrivenMetrics:
    total_neurons: int
    active_neurons: int
    sparsity_ratio: float  # (1 - active/total)
    synaptic_events: int  # SynOps
    dense_equivalent_flops: int
    theoretical_energy_saving_factor: float  # Dense FLOPs / SynOps


class EventDrivenSparseBackend:
    """Delta-thresholded event-driven propagation engine."""

    def __init__(self, delta_threshold: float = 0.05, max_queue_events: int = 4096):
        self.delta_threshold = delta_threshold
        self.max_queue_events = max_queue_events
        self.last_emitted_activations: Optional[torch.Tensor] = None

    def detect_events(self, current_activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, EventDrivenMetrics]:
        """Detect neurons whose activation changed by more than delta_threshold.

        Returns:
            sparse_deltas: Float tensor with deltas for active neurons, 0 elsewhere
            active_indices: 1D Int tensor of triggered neuron indices
            metrics: EventDrivenMetrics
        """
        N = current_activations.numel()
        if self.last_emitted_activations is None or self.last_emitted_activations.shape != current_activations.shape:
            self.last_emitted_activations = torch.zeros_like(current_activations)

        deltas = current_activations - self.last_emitted_activations
        abs_deltas = torch.abs(deltas)
        mask = abs_deltas >= self.delta_threshold

        active_indices = torch.nonzero(mask).squeeze(-1)
        if active_indices.ndim == 0 and active_indices.numel() > 0:
            active_indices = active_indices.unsqueeze(0)

        # Update emitted activation cache for active units
        self.last_emitted_activations = torch.where(mask, current_activations, self.last_emitted_activations)

        active_count = int(mask.sum().item())
        sparsity = 1.0 - (active_count / max(N, 1))

        # Theoretical SynOps: active neurons * average fan-out (assumed ~10% connectivity in connectome)
        assumed_fanout = max(1, int(N * 0.08))
        synops = active_count * assumed_fanout
        dense_flops = N * assumed_fanout * 2  # MAC = 2 FLOPs

        savings = dense_flops / max(synops * 2, 1)

        metrics = EventDrivenMetrics(
            total_neurons=N,
            active_neurons=active_count,
            sparsity_ratio=round(sparsity, 4),
            synaptic_events=synops,
            dense_equivalent_flops=dense_flops,
            theoretical_energy_saving_factor=round(savings, 2),
        )

        sparse_deltas = torch.where(mask, deltas, torch.zeros_like(deltas))
        return sparse_deltas, active_indices, metrics

    def reset(self) -> None:
        self.last_emitted_activations = None

    def export_state(self) -> Dict[str, Any]:
        return {
            "last_emitted_activations": None if self.last_emitted_activations is None else self.last_emitted_activations.detach().cpu().clone(),
            "delta_threshold": float(self.delta_threshold),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        lea = state.get("last_emitted_activations")
        self.last_emitted_activations = None if lea is None else lea.clone()
        self.delta_threshold = float(state.get("delta_threshold", self.delta_threshold))
