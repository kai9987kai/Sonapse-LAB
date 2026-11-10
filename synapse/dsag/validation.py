"""Cheap deterministic validation and resource estimates for DSAG candidates."""
from __future__ import annotations

from typing import Any

from .config import DSAGSpecV1


def validate_spec(spec: DSAGSpecV1, *, input_dim: int | None = None) -> dict[str, Any]:
    if not isinstance(spec, DSAGSpecV1):
        raise TypeError("spec must be a DSAGSpecV1")
    if input_dim is not None and (type(input_dim) is not int or not 1 <= input_dim <= 100_000):
        raise ValueError("input_dim must be an integer in [1,100000]")
    if input_dim is not None and spec.target_indices is not None and any(index >= input_dim for index in spec.target_indices):
        raise ValueError("target_indices must be below input_dim")
    d, k, j = spec.soma_dim, spec.segment_count, spec.max_branches
    n = input_dim or 0
    # Counts parameters from the actual v1 tensor layout, including dormant
    # branch capacity, so structural growth cannot exceed the declared budget.
    parameter_count = (
        6 * d * d + 2 * d * n + 3 * d * max(1, spec.context_dim) + d * spec.message_dim
        + j * d * n + 2 * j * d * d + 5 * k * d * d + k * j
        + len(spec.target_indices or tuple(range(n))) * d + d
    )
    return {
        "valid": True,
        "schema_version": 1,
        "spec_digest": spec.spec_digest,
        "topology": {
            "somas": 2 if spec.dual_soma else 1,
            "segments": k,
            "active_branches": len(spec.branch_segments),
            "branch_budget": j,
            "targets": len(spec.target_indices) if spec.target_indices is not None else input_dim,
        },
        "parameter_count_upper_bound": int(parameter_count),
        "parameter_bytes_float32": int(parameter_count * 4),
        "gain_default": float(spec.gain),
        "evidence_tier": "engineering_validation",
    }
