"""Sonapse-LAB adapter for the existing capped NeuralGraftProtocol bus."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from ..graft_bus import NeuralGraftProtocol
from .config import DSAGSpecV1
from .model import DualSomaAxodendriticGraft
from .packet_adapter import DSAGLocalEnvelope
from .state import DSAGCheckpointError, seal_checkpoint, verify_checkpoint


def _code_digest() -> str:
    root = Path(__file__).resolve().parent
    names = ("__init__.py", "config.py", "model.py", "packet_adapter.py", "plasticity.py", "state.py", "telemetry.py", "transaction.py", "validation.py", "bridge.py")
    hasher = hashlib.sha256()
    for name in names:
        data = (root / name).read_bytes()
        hasher.update(name.encode("utf-8") + b"\0" + data)
    return hasher.hexdigest()


class DSAGBusChannel(NeuralGraftProtocol):
    """Thin stateful residual adapter; final current clipping stays in the bus."""

    def __init__(self, spec: DSAGSpecV1, input_dim: int, name: str = "dsag"):
        if not isinstance(spec, DSAGSpecV1):
            raise TypeError("spec must be a DSAGSpecV1")
        self.spec = spec
        self.model = DualSomaAxodendriticGraft(input_dim=input_dim, spec=spec)
        self._name = name
        self._enabled = spec.enabled
        self._last_latency_ms = 0.0
        self._composed_aux_current_inf_norm = 0.0
        self._effective_current_inf_norm = 0.0
        self._lineage_id = hashlib.sha256((spec.spec_digest + self.model.topology_digest).encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.spec.enabled)

    def set_enabled(self, value: bool) -> None:
        if type(value) is not bool:
            raise ValueError("enabled must be a boolean")
        self._enabled = value

    def reset(self, batch_size: int = 1) -> None:
        self.model.reset(batch_size=batch_size)
        self._last_latency_ms = 0.0
        self._composed_aux_current_inf_norm = 0.0
        self._effective_current_inf_norm = 0.0

    def compute_current(
        self,
        r: torch.Tensor,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = True,
        phase_angles: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(r)
        started = perf_counter_ns()
        last_tick = self.model._runtime.get("tick", -1)
        if advance:
            tick = max(int(microstep), last_tick + 1)
        else:
            tick = max(int(microstep), last_tick, 0)
        current = self.model(
            r,
            context=context,
            phase_angles=phase_angles,
            advance=advance,
            adapt=adapt,
            microstep=tick,
        )
        if advance:
            self._last_latency_ms = (perf_counter_ns() - started) / 1_000_000.0
        return current

    def enqueue_envelope(self, envelope: DSAGLocalEnvelope) -> None:
        self.model.enqueue_envelope(envelope)

    def enqueue_packet(self, packet: Any, **route: Any) -> DSAGLocalEnvelope:
        return self.model.enqueue_packet(packet, **route)

    def get_telemetry_dict(self, include_timing: bool = True) -> dict[str, Any]:
        if type(include_timing) is not bool:
            raise ValueError("include_timing must be a boolean")
        values = self.model.telemetry()
        if not values:
            values = {
                "schema_version": 1,
                "graft_id": self.model.graft_id,
                "spec_digest": self.spec.spec_digest,
                "topology_digest": self.model.topology_digest,
                "topology_version": self.model.topology_version,
                "segments": [0.0] * self.spec.segment_count,
                "branches": [0.0] * len(self.spec.branch_segments),
                "branch_events": 0,
                "event_drop_fraction": 0.0,
                "proposed_current_inf_norm": 0.0,
                "plasticity_updates": 0,
                "queued_packets": 0,
                "accepted_packets": 0,
                "dropped_packets": 0,
                "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
                "model_bytes": sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters()),
                "evidence_tier": "synthetic_fixture",
            }
        values.update({
            "enabled": self.enabled,
            "observe_only": self.spec.observe_only,
            "alpha": values.get("alpha", self.model.alpha),
            "gain_configured": self.spec.gain,
            "composed_aux_current_inf_norm": self._composed_aux_current_inf_norm,
            "effective_current_inf_norm": self._effective_current_inf_norm,
            "ram_bytes": values.get("model_bytes"),
            "parameters": values.get("parameter_count", 0),
            "lineage_id": self._lineage_id,
            "evidence_tier": "synthetic_fixture",
            "telemetry_namespace": f"dsag/{self.model.graft_id}",
        })
        if include_timing:
            values["latency_ms"] = self._last_latency_ms
        return values

    def record_composition(self, composed: torch.Tensor, effective: torch.Tensor) -> None:
        self._composed_aux_current_inf_norm = float(composed.detach().abs().max().cpu()) if composed.numel() else 0.0
        self._effective_current_inf_norm = float(effective.detach().abs().max().cpu()) if effective.numel() else 0.0

    def export_state(self) -> dict[str, Any]:
        payload = {
            "schema": "DSAGCheckpointV1",
            "schema_version": 1,
            "spec": self.spec.to_dict(),
            "spec_digest": self.spec.spec_digest,
            "parent_commit": self.spec.parent_commit,
            "code_digest": _code_digest(),
            "input_dim": self.model.input_dim,
            "target_indices": self.model.target_indices,
            "topology_digest": self.model.topology_digest,
            "model_state": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
            "runtime_state": self.model.export_runtime_state(),
            "enabled_override": self._enabled,
            "optimizer": None,
            "source_provenance": {
                "kind": "synthetic_computational_graft",
                "measured_anatomy": False,
            },
        }
        return seal_checkpoint(payload)

    def import_state(self, state: dict[str, Any]) -> None:
        verify_checkpoint(state)
        if state.get("schema") != "DSAGCheckpointV1" or state.get("schema_version") != 1:
            raise DSAGCheckpointError("unsupported DSAG checkpoint schema")
        if state.get("spec_digest") != self.spec.spec_digest:
            raise DSAGCheckpointError("DSAG spec digest mismatch")
        if state.get("parent_commit") != self.spec.parent_commit:
            raise DSAGCheckpointError("stale DSAG parent commit")
        if state.get("code_digest") != _code_digest():
            raise DSAGCheckpointError("DSAG checkpoint was produced by different source code")
        if state.get("input_dim") != self.model.input_dim or tuple(state.get("target_indices", ())) != self.model.target_indices:
            raise DSAGCheckpointError("DSAG checkpoint host shape or target ports mismatch")
        if set(state.get("model_state", {})) != set(self.model.state_dict()):
            raise DSAGCheckpointError("DSAG checkpoint parameter inventory mismatch")

        old_weights = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        old_runtime = self.model.export_runtime_state()
        old_enabled = self._enabled
        old_lineage = self._lineage_id
        try:
            self.model.load_state_dict(state["model_state"], strict=True)
            self.model.import_runtime_state(state["runtime_state"])
            if self.model.topology_digest != state.get("topology_digest"):
                raise DSAGCheckpointError("DSAG topology digest mismatch")
            self._enabled = bool(state.get("enabled_override", self.spec.enabled))
            self._lineage_id = hashlib.sha256((self.spec.spec_digest + self.model.topology_digest).encode()).hexdigest()[:16]
        except Exception:
            self.model.load_state_dict(old_weights, strict=True)
            self.model.import_runtime_state(old_runtime)
            self._enabled = old_enabled
            self._lineage_id = old_lineage
            raise
