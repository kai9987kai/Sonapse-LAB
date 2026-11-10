"""CPU-first, continuous-rate Dual-Soma Axodendritic Graft kernel.

This is a deterministic synthetic rate model.  It does not simulate membrane
voltage or action-potential propagation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import DSAGSpecV1
from .packet_adapter import DSAGLocalEnvelope
from .plasticity import three_factor_update


class DualSomaAxodendriticGraft(nn.Module):
    """Two synchronously coupled rate somas, routed branches, and axon segments."""

    graft_id = "dsag-001"

    def __init__(self, input_dim: int, spec: DSAGSpecV1 | None = None):
        super().__init__()
        if type(input_dim) is not int or not 1 <= input_dim <= 100_000:
            raise ValueError("input_dim must be an integer in [1,100000]")
        self.spec = spec or DSAGSpecV1()
        self.input_dim = input_dim
        self.target_indices = self._resolve_targets(input_dim)
        self._active_branch_segments = tuple(self.spec.branch_segments)
        self._topology_version = 0
        self._generator = torch.Generator(device="cpu").manual_seed(self.spec.seed)
        self._generator_state = self._generator.get_state().clone()

        d = self.spec.soma_dim
        k = self.spec.segment_count
        j = self.spec.max_branches
        nctx = max(1, self.spec.context_dim)
        msg = self.spec.message_dim

        def parameter(shape: tuple[int, ...], scale: float = 0.04) -> nn.Parameter:
            value = torch.randn(shape, generator=self._generator, dtype=torch.float32) * scale
            return nn.Parameter(value)

        self.soma_a_recurrent = parameter((d, d))
        self.soma_b_recurrent = parameter((d, d))
        self.soma_a_from_b = parameter((d, d), 0.025)
        self.soma_b_from_a = parameter((d, d), 0.025)
        self.soma_a_input = parameter((d, input_dim), 0.02)
        self.soma_b_input = parameter((d, input_dim), 0.02)
        self.soma_a_context = parameter((d, nctx), 0.015)
        self.soma_b_context = parameter((d, nctx), 0.015)
        self.packet_encoder = parameter((d, msg), 0.025)
        self.branch_input = parameter((j, d, input_dim), 0.012)
        self.branch_recurrent = parameter((j, d, d), 0.025)
        self.branch_from_axon = parameter((j, d, d), 0.02)
        self.segment_recurrent = parameter((k, d, d), 0.025)
        self.segment_forward = parameter((k, d, d), 0.02)
        self.segment_from_soma_a = parameter((k, d, d), 0.02)
        self.segment_from_soma_b = parameter((k, d, d), 0.02)
        self.branch_to_segment = parameter((k, j), 0.025)
        self.segment_from_packet = parameter((k, d, d), 0.015)
        self.terminal_from_axon = parameter((d, d), 0.025)
        self.terminal_from_soma_b = parameter((d, d), 0.02)
        self.terminal_from_context = parameter((d, nctx), 0.015)
        self.release_gate = parameter((1, d), 0.015)
        self.terminal_projection = parameter((len(self.target_indices), d), 0.01)
        self.register_buffer("target_index_tensor", torch.tensor(self.target_indices, dtype=torch.long))

        self._runtime: dict[str, Any] = {}
        self._queue: list[DSAGLocalEnvelope] = []
        self._last_values: dict[tuple[str, int, str, int], DSAGLocalEnvelope] = {}
        self._seen_sequences: dict[tuple[str, int, str, int], tuple[int, int]] = {}
        self._seen_checksums: set[str] = set()
        self._metrics: dict[str, Any] = {}
        self.reset()

    def _resolve_targets(self, input_dim: int) -> tuple[int, ...]:
        targets = self.spec.target_indices
        if targets is None:
            return tuple(range(input_dim))
        if not targets:
            raise ValueError("target_indices cannot be empty; use None to declare all host rates")
        if max(targets) >= input_dim:
            raise ValueError("target_indices must reference existing host rates")
        return tuple(targets)

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    @property
    def alpha(self) -> float:
        if not self.spec.enabled or self.spec.gain == 0.0 or self.spec.observe_only:
            return 0.0
        if self.spec.ramp_steps <= 0:
            return self.spec.gain if self._runtime.get("ramp_progress", 0) > 0 else 0.0
        fraction = min(1.0, self._runtime.get("ramp_progress", 0) / self.spec.ramp_steps)
        return self.spec.gain * fraction

    @property
    def topology_version(self) -> int:
        return self._topology_version

    @property
    def topology_digest(self) -> str:
        encoded = json.dumps({
            "spec_digest": self.spec.spec_digest,
            "input_dim": self.input_dim,
            "target_indices": self.target_indices,
            "active_branch_segments": self._active_branch_segments,
            "topology_version": self._topology_version,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def reset(self, batch_size: int = 1) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        device = self.terminal_projection.device
        dtype = self.terminal_projection.dtype
        d, k, j = self.spec.soma_dim, self.spec.segment_count, self.spec.max_branches
        self._runtime = {
            "soma_a": torch.zeros((batch_size, d), device=device, dtype=dtype),
            "soma_b": torch.zeros((batch_size, d), device=device, dtype=dtype),
            "branches": torch.zeros((batch_size, j, d), device=device, dtype=dtype),
            "segments": torch.zeros((batch_size, k, d), device=device, dtype=dtype),
            "eligibility": torch.zeros((k, j), device=device, dtype=dtype),
            "rate_history": [],
            "segment_history": [],
            "tick": -1,
            "ramp_progress": 0,
            "plasticity_updates": 0,
            "accepted_packets": 0,
            "dropped_packets": 0,
            "expired_packets": 0,
            "last_phase_coherence": None,
            "last_modulator": None,
        }
        self._queue = []
        self._last_values = {}
        self._seen_sequences = {}
        self._seen_checksums = set()
        self._metrics = {}

    def step_ramp(self, steps: int = 1) -> float:
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if self.spec.enabled:
            cap = self.spec.ramp_steps if self.spec.ramp_steps > 0 else 1
            self._runtime["ramp_progress"] = min(cap, self._runtime["ramp_progress"] + steps)
        return self.alpha

    def enqueue_envelope(self, envelope: DSAGLocalEnvelope) -> None:
        if not isinstance(envelope, DSAGLocalEnvelope):
            raise TypeError("envelope must be a DSAGLocalEnvelope")
        if envelope.graft_id != self.graft_id:
            raise ValueError("packet is addressed to another DSAG graft")
        if envelope.topology_version != self._topology_version:
            raise ValueError("stale DSAG topology version")
        if envelope.destination_kind == "axon_segment" and envelope.destination_index >= self.spec.segment_count:
            raise ValueError("packet segment destination is outside this topology")
        if envelope.destination_kind == "axon_branch" and envelope.destination_index >= len(self._active_branch_segments):
            raise ValueError("packet branch destination is outside the active topology")
        if envelope.destination_kind.startswith("soma_") and envelope.destination_index != 0:
            raise ValueError("soma packet destinations use index zero")
        if envelope.checksum in self._seen_checksums:
            raise ValueError("duplicate DSAG envelope")
        key = (envelope.source_kind, envelope.source_region, envelope.destination_kind, envelope.destination_index)
        previous = self._seen_sequences.get(key)
        if previous is None and len(self._seen_sequences) >= self.spec.max_queue:
            raise ValueError("DSAG source-route tracking capacity exceeded")
        if previous is not None:
            previous_sequence, previous_tick = previous
            if envelope.source_sequence_id <= previous_sequence or envelope.event_tick < previous_tick:
                raise ValueError("out-of-order DSAG envelope")
        if len(self._queue) >= self.spec.max_queue:
            raise ValueError("DSAG packet queue capacity exceeded")
        self._queue.append(envelope)
        self._queue.sort(key=lambda item: (item.event_tick, item.source_sequence_id, item.destination_index))
        self._seen_sequences[key] = (envelope.source_sequence_id, envelope.event_tick)
        self._seen_checksums.add(envelope.checksum)
        self._runtime["accepted_packets"] += 1

    def enqueue_packet(self, packet: Any, **route: Any) -> DSAGLocalEnvelope:
        envelope = DSAGLocalEnvelope.from_packet(packet, graft_id=self.graft_id, **route)
        self.enqueue_envelope(envelope)
        return envelope

    def _initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        d, k, j = self.spec.soma_dim, self.spec.segment_count, self.spec.max_branches
        return {
            "soma_a": torch.zeros((batch, d), device=device, dtype=dtype),
            "soma_b": torch.zeros((batch, d), device=device, dtype=dtype),
            "branches": torch.zeros((batch, j, d), device=device, dtype=dtype),
            "segments": torch.zeros((batch, k, d), device=device, dtype=dtype),
            "eligibility": torch.zeros((k, j), device=device, dtype=dtype),
            "rate_history": [],
            "segment_history": [],
            "tick": self._runtime.get("tick", -1),
            "ramp_progress": self._runtime.get("ramp_progress", 0),
            "plasticity_updates": self._runtime.get("plasticity_updates", 0),
            "accepted_packets": self._runtime.get("accepted_packets", 0),
            "dropped_packets": self._runtime.get("dropped_packets", 0),
            "expired_packets": self._runtime.get("expired_packets", 0),
            "last_phase_coherence": self._runtime.get("last_phase_coherence"),
            "last_modulator": self._runtime.get("last_modulator"),
        }

    def _normalize_context(self, context: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        width = self.spec.context_dim
        if width == 0:
            return torch.zeros((batch, 1), device=device, dtype=dtype)
        if context is None:
            return torch.zeros((batch, width), device=device, dtype=dtype)
        value = torch.as_tensor(context, device=device, dtype=dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[0] not in (1, batch):
            raise ValueError("context must have shape [C] or [B,C]")
        if value.shape[0] == 1 and batch > 1:
            value = value.expand(batch, -1)
        if value.shape[-1] < width:
            value = F.pad(value, (0, width - value.shape[-1]))
        elif value.shape[-1] > width:
            value = value[..., :width]
        if not torch.isfinite(value).all():
            raise ValueError("context contains non-finite values")
        return torch.tanh(value)

    def _phase_groups(self, phase_angles: torch.Tensor | None, batch: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.spec.phase_gate_enabled or phase_angles is None:
            return None, None
        phase = torch.as_tensor(phase_angles, device=device, dtype=dtype)
        if phase.ndim == 1:
            phase = phase.unsqueeze(0)
        if phase.ndim != 2 or phase.shape[0] not in (1, batch):
            raise ValueError("phase_angles must have shape [N] or [B,N]")
        if phase.shape[0] == 1 and batch > 1:
            phase = phase.expand(batch, -1)
        if not torch.isfinite(phase).all():
            raise ValueError("phase_angles contains non-finite values")
        needed = self.spec.segment_count + max(1, len(self._active_branch_segments))
        if phase.shape[-1] == needed:
            groups = phase
        elif phase.shape[-1] >= needed:
            chunks = torch.tensor_split(phase, needed, dim=-1)
            groups = torch.stack([chunk.mean(dim=-1) for chunk in chunks], dim=-1)
        else:
            raise ValueError("phase input must cover all DSAG segments/branches or match the host width")
        return groups[:, :self.spec.segment_count], groups[:, self.spec.segment_count:self.spec.segment_count + len(self._active_branch_segments)]

    def _events_for_tick(self, tick: int, advance: bool) -> list[tuple[DSAGLocalEnvelope, bool]]:
        due: list[tuple[DSAGLocalEnvelope, bool]] = []
        remaining: list[DSAGLocalEnvelope] = []
        expired: list[DSAGLocalEnvelope] = []
        for envelope in self._queue:
            if envelope.event_tick > tick:
                remaining.append(envelope)
                continue
            age = tick - envelope.event_tick + envelope.age_ticks
            ttl = min(envelope.ttl, self.spec.packet_ttl)
            if age > ttl:
                expired.append(envelope)
            else:
                due.append((envelope, False))
        if advance:
            self._queue = remaining
            self._seen_checksums = {item.checksum for item in remaining}
            self._runtime["expired_packets"] += len(expired)
            self._runtime["dropped_packets"] += len(expired)
            if self.spec.packet_loss_policy == "last_value":
                current_keys = {
                    (env.source_kind, env.source_region, env.destination_kind, env.destination_index)
                    for env, _ in due
                }
                for envelope, _ in due:
                    key = (envelope.source_kind, envelope.source_region, envelope.destination_kind, envelope.destination_index)
                    self._last_values[key] = envelope
                for key, envelope in self._last_values.items():
                    if key in current_keys:
                        continue
                    age = tick - envelope.event_tick + envelope.age_ticks
                    if age <= min(envelope.ttl, self.spec.packet_ttl):
                        due.append((envelope, True))
        return due

    def _message_features(self, envelope: DSAGLocalEnvelope, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vector = torch.tensor(envelope.payload_vector, device=device, dtype=dtype)
        mask = torch.tensor(envelope.validity_mask, device=device, dtype=torch.bool)
        vector = vector[mask]
        if vector.numel() == 0:
            vector = torch.zeros((1,), device=device, dtype=dtype)
        if self.spec.richer_messages:
            if vector.numel() < self.spec.message_dim:
                vector = F.pad(vector, (0, self.spec.message_dim - vector.numel()))
            else:
                vector = vector[:self.spec.message_dim]
        else:
            scalar = vector.mean().reshape(1)
            vector = F.pad(scalar, (0, self.spec.message_dim - 1))
        return vector

    def _route_messages(self, events: list[tuple[DSAGLocalEnvelope, bool]], batch: int, device: torch.device, dtype: torch.dtype, tick: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None, dict[int, float]]:
        d, k, j = self.spec.soma_dim, self.spec.segment_count, self.spec.max_branches
        soma_a = torch.zeros((batch, d), device=device, dtype=dtype)
        soma_b = torch.zeros_like(soma_a)
        branch = torch.zeros((batch, j, d), device=device, dtype=dtype)
        segment = torch.zeros((batch, k, d), device=device, dtype=dtype)
        modulators: list[float] = []
        branch_phases: dict[int, float] = {}
        for envelope, stale in events:
            if envelope.topology_version != self._topology_version:
                continue
            feature = self._message_features(envelope, device, dtype)
            embedded = F.linear(feature, self.packet_encoder)
            age = max(0, tick - envelope.event_tick + envelope.age_ticks)
            if stale:
                embedded = embedded * math.exp(-age / max(1, envelope.ttl))
            if envelope.destination_kind == "soma_a":
                soma_a.add_(embedded.unsqueeze(0).expand(batch, -1))
            elif envelope.destination_kind == "soma_b":
                soma_b.add_(embedded.unsqueeze(0).expand(batch, -1))
            elif envelope.destination_kind == "axon_branch":
                branch[:, envelope.destination_index, :] += embedded
            elif envelope.destination_kind == "axon_segment":
                segment[:, envelope.destination_index, :] += embedded
            if envelope.modulation_vector:
                modulators.extend(envelope.modulation_vector)
            if envelope.destination_kind == "axon_branch" and envelope.phase_vector:
                branch_phases[envelope.destination_index] = float(envelope.phase_vector[0])
        modulator = sum(modulators) / len(modulators) if modulators else None
        return soma_a, soma_b, branch, segment, modulator, branch_phases

    def forward(
        self,
        full_rates: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        phase_angles: torch.Tensor | None = None,
        advance: bool = True,
        adapt: bool = True,
        microstep: int | None = None,
    ) -> torch.Tensor:
        if not self.spec.enabled:
            return torch.zeros_like(full_rates)
        rates = torch.as_tensor(full_rates, device=self.terminal_projection.device, dtype=self.terminal_projection.dtype)
        unbatched = rates.ndim == 1
        if unbatched:
            rates = rates.unsqueeze(0)
        if rates.ndim != 2 or rates.shape[-1] != self.input_dim:
            raise ValueError(f"full_rates must have shape [N] or [B,{self.input_dim}]")
        if not torch.isfinite(rates).all():
            raise ValueError("full_rates contains non-finite values")
        batch = rates.shape[0]
        current_tick = self._runtime["tick"]
        tick = current_tick + 1 if microstep is None else int(microstep)
        if tick < 0 or (advance and tick < current_tick):
            raise ValueError("microstep must be non-negative and monotonic when advancing")
        if advance and self._runtime["soma_a"].shape[0] != batch:
            self.reset(batch_size=batch)
        state = self._runtime if self._runtime["soma_a"].shape[0] == batch else self._initial_state(batch, rates.device, rates.dtype)
        rates = torch.tanh(rates)
        ctx = self._normalize_context(context, batch, rates.device, rates.dtype)
        messages = self._events_for_tick(tick, advance)
        pkt_a, pkt_b, pkt_branch, pkt_segment, modulator, packet_phases = self._route_messages(messages, batch, rates.device, rates.dtype, tick)

        branch_sources = []
        for branch_index in range(self.spec.max_branches):
            delay = self.spec.branch_delays[branch_index] if branch_index < len(self.spec.branch_delays) else 0
            if delay == 0:
                branch_sources.append(rates)
            elif len(state["rate_history"]) >= delay:
                branch_sources.append(state["rate_history"][-delay])
            else:
                branch_sources.append(torch.zeros_like(rates))
        routed_rates = torch.stack(branch_sources, dim=1) if branch_sources else rates.new_zeros((batch, 0, self.input_dim))
        branch_drive = torch.einsum("bjn,jdn->bjd", routed_rates, self.branch_input)

        old_a, old_b = state["soma_a"], state["soma_b"]
        if old_a.device != rates.device or old_a.dtype != rates.dtype:
            old_a, old_b = old_a.to(rates), old_b.to(rates)
        old_branch = state["branches"].to(rates)
        old_segments = state["segments"].to(rates)
        ctx_a = F.linear(ctx, self.soma_a_context)
        ctx_b = F.linear(ctx, self.soma_b_context)
        drive_a = F.linear(rates, self.soma_a_input) + ctx_a + pkt_a
        drive_b = F.linear(rates, self.soma_b_input) + ctx_b + pkt_b
        next_a = (1.0 - self.spec.eta_soma_a) * old_a + self.spec.eta_soma_a * torch.tanh(
            F.linear(old_a, self.soma_a_recurrent)
            + F.linear(old_b, self.soma_a_from_b)
            + drive_a
        )
        if self.spec.dual_soma:
            next_b = (1.0 - self.spec.eta_soma_b) * old_b + self.spec.eta_soma_b * torch.tanh(
                F.linear(old_b, self.soma_b_recurrent)
                + F.linear(old_a, self.soma_b_from_a)
                + drive_b
            )
        else:
            next_b = torch.zeros_like(old_b)

        route_indices = torch.tensor(self._active_branch_segments, device=rates.device, dtype=torch.long)
        active_count = len(self._active_branch_segments)
        if self._active_branch_segments:
            local_axon = old_segments.index_select(1, route_indices)
            branch_drive[:, :active_count, :] += torch.einsum("bjd,jde->bje", local_axon, self.branch_from_axon[:active_count])
        else:
            local_axon = rates.new_zeros((batch, 0, self.spec.soma_dim))
        branch_drive[:, :active_count, :] += pkt_branch[:, :active_count, :]
        if self.spec.axo_branches:
            next_branch = (1.0 - self.spec.eta_branch) * old_branch + self.spec.eta_branch * torch.tanh(
                branch_drive + torch.einsum("bjd,jde->bje", old_branch, self.branch_recurrent)
                if active_count else old_branch
            )
            if active_count < self.spec.max_branches:
                next_branch[:, active_count:, :] = 0.0
        else:
            next_branch = torch.zeros_like(old_branch)

        segment_delay = self.spec.segment_delay
        if segment_delay == 0:
            source_segments = old_segments
        elif len(state["segment_history"]) >= segment_delay:
            source_segments = state["segment_history"][-segment_delay]
        else:
            source_segments = torch.zeros_like(old_segments)
        left = torch.zeros_like(old_segments)
        if self.spec.segment_count > 1:
            left[:, 1:, :] = source_segments[:, :-1, :]
        branch_mask = torch.zeros((self.spec.segment_count, self.spec.max_branches), device=rates.device, dtype=rates.dtype)
        for branch_index, segment_index in enumerate(self._active_branch_segments):
            branch_mask[segment_index, branch_index] = 1.0
        route_gain = self.branch_to_segment * branch_mask
        branch_current = torch.einsum("kj,bjd->bkd", route_gain, next_branch)
        axon_drive = (
            torch.einsum("bkd,kde->bke", old_segments, self.segment_recurrent)
            + torch.einsum("bkd,kde->bke", left, self.segment_forward)
            + branch_current
            + torch.einsum("bd,kde->bke", next_a, self.segment_from_soma_a)
            + torch.einsum("bd,kde->bke", next_b, self.segment_from_soma_b)
            + torch.einsum("bkd,kde->bke", pkt_segment.reshape(batch, self.spec.segment_count, self.spec.soma_dim), self.segment_from_packet)
        )
        segment_phase, branch_phase = self._phase_groups(phase_angles, batch, rates.device, rates.dtype)
        if branch_phase is not None and packet_phases:
            branch_phase = branch_phase.clone()
            for branch_index, phase_value in packet_phases.items():
                if branch_index < branch_phase.shape[-1]:
                    branch_phase[:, branch_index] = phase_value
        phase_coherence = None
        if self.spec.phase_gate_enabled and segment_phase is not None and branch_phase is not None and active_count:
            gates = 0.5 * (1.0 + torch.cos(branch_phase[..., :active_count] - segment_phase.index_select(1, route_indices)))
            if not self.spec.richer_messages:
                gates = gates.detach()
            branch_current = torch.einsum("kj,bjd->bkd", route_gain[:, :active_count], next_branch[:, :active_count, :] * gates.unsqueeze(-1))
            axon_drive = axon_drive - torch.einsum("kj,bjd->bkd", route_gain[:, :active_count], next_branch[:, :active_count, :]) + branch_current
            phase_coherence = float(gates.detach().mean().cpu())
        next_segments = (1.0 - self.spec.eta_axon) * old_segments + self.spec.eta_axon * torch.tanh(axon_drive)
        final_hidden = F.linear(next_segments[:, -1, :], self.terminal_from_axon)
        final_hidden = final_hidden + F.linear(next_b, self.terminal_from_soma_b) + F.linear(ctx, self.terminal_from_context)
        released = torch.sigmoid(F.linear(final_hidden, self.release_gate)) * torch.tanh(final_hidden)
        target_current = F.linear(released, self.terminal_projection)
        used_alpha = self.alpha
        proposed = target_current * used_alpha
        full_current = rates.new_zeros((batch, self.input_dim)).index_copy(1, self.target_index_tensor, proposed)
        if self.spec.observe_only:
            full_current = torch.zeros_like(full_current)

        next_eligibility = state["eligibility"].to(rates)
        plasticity_update = False
        if self.spec.plasticity_enabled and advance and adapt and modulator is not None and active_count:
            pre = next_branch.new_zeros((batch, self.spec.max_branches))
            pre[:, :active_count] = next_branch[:, :active_count, :].mean(dim=-1)
            post = next_segments.mean(dim=-1)
            with torch.no_grad():
                mod = max(-1.0, min(1.0, float(modulator)))
                mask = torch.zeros_like(self.branch_to_segment)
                for branch_index, segment_index in enumerate(self._active_branch_segments):
                    mask[segment_index, branch_index] = 1.0
                updated_weight, next_eligibility = three_factor_update(
                    self.branch_to_segment,
                    next_eligibility,
                    pre,
                    post,
                    mod,
                    learning_rate=self.spec.plasticity_rate,
                    decay=self.spec.plasticity_decay,
                    eligibility_decay=self.spec.eligibility_decay,
                    bound=0.25,
                    mask=mask,
                )
                self.branch_to_segment.copy_(updated_weight)
            plasticity_update = True

        if advance:
            self._runtime["soma_a"] = next_a.detach()
            self._runtime["soma_b"] = next_b.detach()
            self._runtime["branches"] = next_branch.detach()
            self._runtime["segments"] = next_segments.detach()
            self._runtime["eligibility"] = next_eligibility.detach()
            self._runtime["rate_history"].append(rates.detach())
            self._runtime["segment_history"].append(old_segments.detach())
            max_history = max([self.spec.segment_delay, *self.spec.branch_delays, 0])
            if max_history:
                self._runtime["rate_history"] = self._runtime["rate_history"][-max_history:]
                self._runtime["segment_history"] = self._runtime["segment_history"][-max_history:]
            else:
                self._runtime["rate_history"] = []
                self._runtime["segment_history"] = []
            self._runtime["tick"] = tick
            if self.spec.ramp_steps > 0:
                self._runtime["ramp_progress"] = min(self.spec.ramp_steps, self._runtime["ramp_progress"] + 1)
            elif self.spec.enabled:
                self._runtime["ramp_progress"] = 1
            if plasticity_update:
                self._runtime["plasticity_updates"] += 1
            if phase_coherence is not None:
                self._runtime["last_phase_coherence"] = phase_coherence
            if modulator is not None:
                self._runtime["last_modulator"] = float(modulator)

        if advance:
            self._metrics = {
                "schema_version": 1,
                "graft_id": self.graft_id,
                "spec_digest": self.spec.spec_digest,
                "topology_digest": self.topology_digest,
                "topology_version": self._topology_version,
                "alpha": used_alpha,
                "soma_a_norm": float(next_a.detach().norm(dim=-1).mean().cpu()),
                "soma_b_norm": float(next_b.detach().norm(dim=-1).mean().cpu()),
                "coupling_flux": float((F.linear(old_b, self.soma_a_from_b).norm(dim=-1).mean() + F.linear(old_a, self.soma_b_from_a).norm(dim=-1).mean()).detach().cpu()),
                "segments": next_segments.detach().norm(dim=-1).mean(dim=0).cpu().tolist(),
                "branches": next_branch.detach().norm(dim=-1).mean(dim=0).cpu().tolist()[:active_count],
                "active_branch_segments": list(self._active_branch_segments),
                "branch_events": len(messages),
                "packet_flow": [
                    {
                        "source_kind": envelope.source_kind,
                        "source_region": envelope.source_region,
                        "destination_kind": envelope.destination_kind,
                        "destination_index": envelope.destination_index,
                        "source_sequence_id": envelope.source_sequence_id,
                        "age_ticks": max(0, tick - envelope.event_tick + envelope.age_ticks),
                        "stale": stale,
                        "parent_packet_checksum": envelope.parent_packet_checksum,
                        "envelope_checksum": envelope.checksum,
                        "provenance": envelope.provenance,
                    }
                    for envelope, stale in messages
                ],
                "event_drop_fraction": self._runtime["dropped_packets"] / max(1, self._runtime["accepted_packets"]),
                "phase_coherence": phase_coherence if phase_coherence is not None else self._runtime.get("last_phase_coherence"),
                "proposed_current_inf_norm": float(full_current.detach().abs().max().cpu()) if full_current.numel() else 0.0,
                "plasticity_updates": int(self._runtime["plasticity_updates"]),
                "tick": tick,
                "queued_packets": len(self._queue),
                "accepted_packets": int(self._runtime["accepted_packets"]),
                "dropped_packets": int(self._runtime["dropped_packets"]),
                "expired_packets": int(self._runtime["expired_packets"]),
                "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
                "model_bytes": sum(parameter.numel() * parameter.element_size() for parameter in self.parameters()),
                "evidence_tier": "synthetic_fixture",
            }
        return full_current.squeeze(0) if unbatched else full_current

    def telemetry(self) -> dict[str, Any]:
        return deepcopy(self._metrics)

    def apply_topology(self, operation: str, *, branch_index: int | None = None, segment_index: int | None = None) -> dict[str, Any]:
        """Apply one bounded branch topology edit and return an exact undo receipt."""
        if not self.spec.structural_growth_enabled:
            raise ValueError("structural growth is disabled in this DSAGSpecV1")
        if self._queue:
            raise ValueError("drain or explicitly route pending packets before changing topology")
        before = {
            "segments": tuple(self._active_branch_segments),
            "version": self._topology_version,
            "branch_to_segment": self.branch_to_segment.detach().clone(),
        }
        candidate = list(self._active_branch_segments)
        if operation == "add_axon_input_branch":
            if len(candidate) >= self.spec.max_branches:
                raise ValueError("DSAG branch budget exhausted")
            if segment_index is None or not 0 <= segment_index < self.spec.segment_count:
                raise ValueError("segment_index must reference an existing axon segment")
            candidate.append(segment_index)
        elif operation == "prune_axon_input_branch":
            if branch_index is None or not 0 <= branch_index < len(candidate):
                raise ValueError("branch_index must reference an active branch")
            candidate.pop(branch_index)
        elif operation == "move_axon_input_branch_to_segment":
            if branch_index is None or not 0 <= branch_index < len(candidate):
                raise ValueError("branch_index must reference an active branch")
            if segment_index is None or not 0 <= segment_index < self.spec.segment_count:
                raise ValueError("segment_index must reference an existing axon segment")
            candidate[branch_index] = segment_index
        else:
            raise ValueError("unsupported DSAG topology operation")
        if len(candidate) > self.spec.max_branches or any(not 0 <= item < self.spec.segment_count for item in candidate):
            raise ValueError("proposed DSAG topology violates resource or port limits")
        old_segments = list(before["segments"])
        self._active_branch_segments = tuple(candidate)
        self._topology_version += 1
        with torch.no_grad():
            self.branch_to_segment.copy_(before["branch_to_segment"].to(self.branch_to_segment))
            if operation == "add_axon_input_branch":
                self.branch_to_segment[:, len(old_segments)].zero_()
            elif operation == "prune_axon_input_branch":
                pruned_index = int(branch_index)
                old_weights = before["branch_to_segment"].to(self.branch_to_segment)
                self.branch_to_segment.zero_()
                if pruned_index:
                    self.branch_to_segment[:, :pruned_index].copy_(old_weights[:, :pruned_index])
                if pruned_index < len(candidate):
                    self.branch_to_segment[:, pruned_index:len(candidate)].copy_(old_weights[:, pruned_index + 1:len(old_segments)])
            elif operation == "move_axon_input_branch_to_segment":
                old_segment = old_segments[int(branch_index)]
                self.branch_to_segment[old_segment, int(branch_index)] = 0.0
                self.branch_to_segment[int(segment_index), int(branch_index)] = 0.0
        return {
            "schema_version": 1,
            "operation": operation,
            "before_topology_digest": hashlib.sha256(json.dumps({"segments": before["segments"], "version": before["version"]}, sort_keys=True).encode()).hexdigest(),
            "after_topology_digest": self.topology_digest,
            "before": before,
            "after_version": self._topology_version,
        }

    def rollback_topology(self, receipt: dict[str, Any]) -> None:
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
            raise ValueError("invalid DSAG topology receipt")
        if receipt.get("after_topology_digest") != self.topology_digest:
            raise ValueError("stale DSAG topology receipt")
        before = receipt.get("before")
        if not isinstance(before, dict) or "segments" not in before or "branch_to_segment" not in before:
            raise ValueError("incomplete DSAG topology receipt")
        self._active_branch_segments = tuple(before["segments"])
        self._topology_version = int(before["version"])
        with torch.no_grad():
            self.branch_to_segment.copy_(before["branch_to_segment"].to(self.branch_to_segment))

    def export_runtime_state(self) -> dict[str, Any]:
        def clone(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, list):
                return [clone(item) for item in value]
            if isinstance(value, dict):
                return {key: clone(item) for key, item in value.items()}
            return deepcopy(value)
        return {
            "runtime": clone(self._runtime),
            "active_branch_segments": tuple(self._active_branch_segments),
            "topology_version": self._topology_version,
            "queue": [envelope.to_dict() for envelope in self._queue],
            "last_values": [envelope.to_dict() for envelope in self._last_values.values()],
            "seen_sequences": [(key, value) for key, value in self._seen_sequences.items()],
            "seen_checksums": sorted(self._seen_checksums),
            "generator_state": self._generator_state.clone(),
            "metrics": deepcopy(self._metrics),
        }

    def import_runtime_state(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("DSAG runtime state must be a mapping")
        from .packet_adapter import DSAGLocalEnvelope
        active_segments = tuple(data["active_branch_segments"])
        topology_version = int(data["topology_version"])
        if topology_version < 0 or len(active_segments) > self.spec.max_branches:
            raise ValueError("invalid DSAG topology version or branch count")
        if any(type(index) is not int or not 0 <= index < self.spec.segment_count for index in active_segments):
            raise ValueError("invalid DSAG branch target in checkpoint")
        queue = [DSAGLocalEnvelope.from_dict(item) for item in data.get("queue", [])]
        if len(queue) > self.spec.max_queue:
            raise ValueError("DSAG checkpoint queue exceeds configured capacity")
        if any(envelope.topology_version != topology_version for envelope in queue):
            raise ValueError("DSAG checkpoint contains a stale queued topology version")
        if queue != sorted(queue, key=lambda item: (item.event_tick, item.source_sequence_id, item.destination_index)):
            raise ValueError("DSAG checkpoint queue is not in canonical delivery order")
        if set(data.get("seen_checksums", [])) != {envelope.checksum for envelope in queue}:
            raise ValueError("DSAG checkpoint checksum tracker disagrees with packet queue")
        runtime = deepcopy(data["runtime"])
        if not isinstance(runtime, dict):
            raise ValueError("DSAG runtime checkpoint is malformed")
        batch = runtime["soma_a"].shape[0]
        expected = {
            "soma_a": (batch, self.spec.soma_dim),
            "soma_b": (batch, self.spec.soma_dim),
            "branches": (batch, self.spec.max_branches, self.spec.soma_dim),
            "segments": (batch, self.spec.segment_count, self.spec.soma_dim),
            "eligibility": (self.spec.segment_count, self.spec.max_branches),
        }
        if any(not isinstance(runtime.get(key), torch.Tensor) or tuple(runtime[key].shape) != shape for key, shape in expected.items()):
            raise ValueError("DSAG checkpoint runtime tensor shape mismatch")
        if type(runtime.get("tick")) is not int or type(runtime.get("ramp_progress")) is not int or runtime["ramp_progress"] < 0:
            raise ValueError("DSAG checkpoint clock or ramp is invalid")
        self._runtime = runtime
        self._active_branch_segments = active_segments
        self._topology_version = topology_version
        self._queue = queue
        self._last_values = {
            (env.source_kind, env.source_region, env.destination_kind, env.destination_index): env
            for env in (DSAGLocalEnvelope.from_dict(item) for item in data.get("last_values", []))
        }
        self._seen_sequences = {tuple(key): tuple(value) for key, value in data.get("seen_sequences", [])}
        self._seen_checksums = set(data.get("seen_checksums", []))
        self._generator_state = data["generator_state"].clone()
        self._generator.set_state(self._generator_state)
        self._metrics = deepcopy(data.get("metrics", {}))

    def step_ramp_compat(self, rate: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Compatibility spelling used by evolutionary graft interfaces."""
        return self.forward(rate, context=context)
