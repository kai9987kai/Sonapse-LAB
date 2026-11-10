"""Bounded three-factor learning and reversible topology operators."""
from __future__ import annotations

import math
from typing import Any

import torch


def three_factor_update(
    weight: torch.Tensor,
    eligibility: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    modulator: float,
    *,
    learning_rate: float,
    decay: float,
    eligibility_decay: float = 0.9,
    bound: float = 0.25,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a projected weight and eligibility update without mutating inputs."""
    if not math.isfinite(float(modulator)):
        raise ValueError("modulator must be finite")
    if learning_rate < 0.0 or decay < 0.0 or not 0.0 <= eligibility_decay <= 1.0 or bound <= 0.0:
        raise ValueError("learning_rate/decay must be non-negative and bound positive")
    correlation = torch.einsum("bk,bj->kj", post, pre) / max(1, pre.shape[0])
    next_eligibility = eligibility_decay * eligibility + correlation
    delta = learning_rate * float(modulator) * next_eligibility - learning_rate * decay * weight
    if mask is not None:
        delta = delta * mask
    return (weight + delta).clamp(-bound, bound), next_eligibility


class DSAGTopologyController:
    """Namespaced reversible branch growth, pruning, and reassignment."""

    def __init__(self, model: Any):
        self.model = model

    def add_axon_input_branch(self, segment_index: int) -> dict[str, Any]:
        return self.model.apply_topology("add_axon_input_branch", segment_index=segment_index)

    def prune_axon_input_branch(self, branch_index: int) -> dict[str, Any]:
        return self.model.apply_topology("prune_axon_input_branch", branch_index=branch_index)

    def move_axon_input_branch_to_segment(self, branch_index: int, segment_index: int) -> dict[str, Any]:
        return self.model.apply_topology(
            "move_axon_input_branch_to_segment", branch_index=branch_index, segment_index=segment_index,
        )

    def rollback(self, receipt: dict[str, Any]) -> None:
        self.model.rollback_topology(receipt)
