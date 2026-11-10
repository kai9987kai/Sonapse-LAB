"""Versioned, default-off configuration for the synthetic DSAG graft."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DSAGSpecV1:
    """Complete immutable v1 rate-kernel configuration.

    All topology and numeric choices that affect the kernel live here so the
    digest changes whenever the experiment changes.  ``target_indices=None``
    declares every host rate as a possible terminal target; a tuple restricts
    release to those existing host indices.
    """

    version: int = 1
    enabled: bool = False
    observe_only: bool = False
    gain: float = 0.0
    max_gain: float = 0.1
    ramp_steps: int = 16
    seed: int = 42
    soma_dim: int = 8
    segment_count: int = 4
    branch_segments: tuple[int, ...] = (1, 3)
    max_branches: int = 8
    branch_delays: tuple[int, ...] = (0, 0)
    segment_delay: int = 0
    context_dim: int = 64
    message_dim: int = 32
    eta_soma_a: float = 0.5
    eta_soma_b: float = 0.25
    eta_branch: float = 0.5
    eta_axon: float = 0.5
    eligibility_decay: float = 0.9
    plasticity_rate: float = 0.001
    plasticity_decay: float = 0.0001
    plasticity_enabled: bool = False
    phase_gate_enabled: bool = False
    richer_messages: bool = False
    dual_soma: bool = True
    axo_branches: bool = True
    structural_growth_enabled: bool = False
    packet_ttl: int = 64
    max_queue: int = 256
    packet_loss_policy: str = "no_event"
    target_indices: tuple[int, ...] | None = None
    parent_commit: str | None = None

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("DSAGSpecV1 only accepts version=1")
        for name in (
            "enabled", "observe_only", "plasticity_enabled", "phase_gate_enabled",
            "richer_messages", "dual_soma", "axo_branches", "structural_growth_enabled",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        integer_ranges = {
            "ramp_steps": (0, 100_000),
            "seed": (0, 2**31 - 1),
            "soma_dim": (1, 128),
            "segment_count": (1, 8),
            "max_branches": (0, 8),
            "segment_delay": (0, 128),
            "context_dim": (0, 256),
            "message_dim": (1, 128),
            "packet_ttl": (0, 100_000),
            "max_queue": (1, 4_096),
        }
        for name, (low, high) in integer_ranges.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")
        for name in (
            "gain", "max_gain", "eta_soma_a", "eta_soma_b", "eta_branch", "eta_axon",
            "eligibility_decay", "plasticity_rate", "plasticity_decay",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.gain <= self.max_gain <= 0.5:
            raise ValueError("gain and max_gain must satisfy 0 <= gain <= max_gain <= 0.5")
        for name in ("eta_soma_a", "eta_soma_b", "eta_branch", "eta_axon"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0,1]")
        if not 0.0 <= self.eligibility_decay <= 1.0:
            raise ValueError("eligibility_decay must be in [0,1]")
        if not 0.0 <= self.plasticity_rate <= 0.1 or not 0.0 <= self.plasticity_decay <= 0.1:
            raise ValueError("plasticity_rate and plasticity_decay must be in [0,0.1]")
        if self.packet_loss_policy not in {"no_event", "last_value", "drop"}:
            raise ValueError("packet_loss_policy must be no_event, last_value, or drop")

        segments = tuple(self.branch_segments)
        delays = tuple(self.branch_delays)
        if len(segments) > self.max_branches:
            raise ValueError("branch_segments cannot exceed max_branches")
        if not self.axo_branches and segments:
            raise ValueError("branch_segments must be empty when axo_branches is false")
        if any(type(i) is not int or not 0 <= i < self.segment_count for i in segments):
            raise ValueError("every branch segment must reference an existing axon segment")
        if len(delays) != len(segments):
            raise ValueError("branch_delays must have one entry per active branch")
        if any(type(i) is not int or not 0 <= i <= 128 for i in delays):
            raise ValueError("branch delays must be integers in [0,128]")
        object.__setattr__(self, "branch_segments", segments)
        object.__setattr__(self, "branch_delays", delays)

        if self.target_indices is not None:
            targets = tuple(self.target_indices)
            if any(type(i) is not int or i < 0 for i in targets):
                raise ValueError("target_indices must contain non-negative integers")
            if len(set(targets)) != len(targets):
                raise ValueError("target_indices cannot contain duplicates")
            object.__setattr__(self, "target_indices", targets)
        if self.parent_commit is not None and (not isinstance(self.parent_commit, str) or not self.parent_commit):
            raise ValueError("parent_commit must be a non-empty string or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def spec_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DSAGSpecV1":
        if not isinstance(data, dict):
            raise TypeError("DSAG spec must be a mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown DSAG spec field(s): {', '.join(sorted(unknown))}")
        values = dict(data)
        for name in ("branch_segments", "branch_delays", "target_indices"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        return cls(**values)

    @classmethod
    def from_file(cls, path: str | Path) -> "DSAGSpecV1":
        raw = Path(path).read_text(encoding="utf-8")
        try:
            import json
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml  # optional; JSON configs need no YAML dependency
            except ImportError as exc:
                raise RuntimeError("reading non-JSON YAML requires PyYAML") from exc
            data = yaml.safe_load(raw)
        return cls.from_dict(data)
