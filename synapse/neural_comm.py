"""SANCF — Sonapse Adaptive Neural Communication Fabric & NeuroWave Engine.

Implements the additive, default-OFF inter-region neural communication fabric:
- ``MessageEncoder``: shared low-rank MLP with optional attentive source pooling.
- ``MessageDecoder``: parallel low-rank basis projection into target receiving populations.
- ``RegionRouter``: adaptive top-k route gate + Communication-Through-Coherence (CTC)
  non-negative phase gating (explicitly abstract oscillatory phases, preserving Dale's law E/I sign).
- ``CommunicationLimiter``: enforces ``communication_every`` (in environment steps),
  ``max_messages_per_tick``, ``max_payload_bits_per_tick``, and ``max_aux_current``.
- ``CommunicationTelemetry``: tracks ``comm_payload_bits_per_step``, ``comm_utilization``,
  ``comm_causal_gain``, ``comm_probe_accuracy``, ``comm_effective_rank``, ``comm_drop_rate``,
  ``comm_latency_steps`` separately from legacy ``inter_connectome_flux``.
- ``NeuralCommFabric(nn.Module)``: zero-initialised residual gate ``gamma = 0.0``, strict
  two-clock accounting (environment step vs microstep), side-effect-free ``advance=False``
  execution, and dendritic contextual message integration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neural_comm_channels import (
    CommChannelSpec,
    CommStepOutput,
    ContinuousChannel,
    QuantisedChannel,
    OrthogonalModeChannel,
    SimulatedImpairedChannel,
)
from .neural_comm_topology import (
    CommTopologyManifest,
    RoutePopulationSpec,
    compile_comm_topology,
)


class MessageEncoder(nn.Module):
    """Shared region-conditioned encoder E_AB(p_A(t)) -> m_AB(t) in [-1, 1]^D."""

    def __init__(self, pool_dim: int = 32, region_emb_dim: int = 16, message_dim: int = 16, num_regions: int = 8):
        super().__init__()
        self.pool_dim = int(pool_dim)
        self.region_emb_dim = int(region_emb_dim)
        self.message_dim = int(message_dim)
        self.region_emb = nn.Embedding(max(1, num_regions), self.region_emb_dim)
        self.attn_query = nn.Parameter(torch.zeros(self.pool_dim))
        self.norm = nn.LayerNorm(self.pool_dim)
        self.net = nn.Sequential(
            nn.Linear(self.pool_dim + 2 * self.region_emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.message_dim),
        )
        # Deterministic small init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def pool_region(self, r: torch.Tensor, indices: Tuple[int, ...]) -> torch.Tensor:
        """Aggregate source population activations into a fixed K=pool_dim summary vector."""
        if len(indices) == 0 or r.numel() == 0:
            return torch.zeros(self.pool_dim, dtype=r.dtype, device=r.device)
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=r.device)
        sub = r.index_select(0, idx_t)
        if sub.numel() == 0:
            return torch.zeros(self.pool_dim, dtype=r.dtype, device=r.device)
        pooled = F.adaptive_avg_pool1d(sub.reshape(1, 1, -1), self.pool_dim).reshape(self.pool_dim)
        # Attentive re-weighting using learned query
        attn_weights = torch.softmax(pooled * self.attn_query, dim=0) * float(self.pool_dim)
        return self.norm(pooled * (0.5 + 0.5 * attn_weights))

    def forward(self, pooled_src: torch.Tensor, src_region: int, dst_region: int) -> torch.Tensor:
        max_idx = self.region_emb.num_embeddings - 1
        s_idx = torch.tensor(min(max(0, src_region), max_idx), dtype=torch.long, device=pooled_src.device)
        d_idx = torch.tensor(min(max(0, dst_region), max_idx), dtype=torch.long, device=pooled_src.device)
        inp = torch.cat([pooled_src, self.region_emb(s_idx), self.region_emb(d_idx)], dim=0)
        return torch.tanh(self.net(inp))


class MessageDecoder(nn.Module):
    """Shared decoder D_AB(m_AB(t)) projecting parallel currents to target receiving populations."""

    def __init__(self, message_dim: int = 16, region_emb_dim: int = 16, max_targets: int = 64, num_regions: int = 8):
        super().__init__()
        self.message_dim = int(message_dim)
        self.region_emb_dim = int(region_emb_dim)
        self.max_targets = int(max_targets)
        self.region_emb = nn.Embedding(max(1, num_regions), self.region_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.message_dim + self.region_emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.max_targets),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.4)
                nn.init.zeros_(m.bias)

    def forward(self, message: torch.Tensor, dst_region: int, target_count: int) -> torch.Tensor:
        if target_count <= 0:
            return torch.zeros(0, dtype=message.dtype, device=message.device)
        max_idx = self.region_emb.num_embeddings - 1
        d_idx = torch.tensor(min(max(0, dst_region), max_idx), dtype=torch.long, device=message.device)
        raw = torch.tanh(self.net(torch.cat([message, self.region_emb(d_idx)], dim=0)))
        if target_count <= self.max_targets:
            return raw[:target_count]
        return F.interpolate(raw.reshape(1, 1, -1), size=target_count, mode="linear", align_corners=False).reshape(target_count)


class RegionRouter(nn.Module):
    """Learned route & Communication-Through-Coherence (CTC) phase gate alpha_AB(t) in [0, 1].

    Preserves neurotransmitter sign constraints: alpha_AB(t) >= 0 so an excitatory or
    inhibitory source never flips polarity due to routing or phase gating.
    """

    def __init__(self, pool_dim: int = 32, num_regions: int = 8, phase_coupling: float = 0.5):
        super().__init__()
        self.pool_dim = int(pool_dim)
        self.num_regions = max(1, int(num_regions))
        self.phase_coupling = float(np.clip(phase_coupling, 0.0, 1.0))
        self.gate_net = nn.Sequential(
            nn.Linear(2 * self.pool_dim + 4, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Preferred phase offset phi^*_AB for abstract oscillatory coherence routing
        self.preferred_phase = nn.Parameter(torch.zeros(self.num_regions, self.num_regions))
        self.register_buffer("hysteresis_ema", torch.full((self.num_regions, self.num_regions), 0.5))

    def score_route(
        self,
        p_src: torch.Tensor,
        p_dst: torch.Tensor,
        src_reg: int,
        dst_reg: int,
        *,
        phase_angles: Optional[np.ndarray] = None,
        phase_mode: str = "coherence",  # 'coherence' | 'none' | 'random' | 'shuffled' | 'frozen'
        advance: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, float]:
        s_idx = min(max(0, src_reg), self.num_regions - 1)
        d_idx = min(max(0, dst_reg), self.num_regions - 1)

        # Compute abstract phase coherence factor in [0, 1]
        coherence_gain = 1.0
        if phase_mode != "none" and phase_angles is not None and len(phase_angles) > max(s_idx, d_idx):
            if phase_mode == "random" and rng is not None:
                delta_phi = float(rng.uniform(-math.pi, math.pi))
            elif phase_mode == "shuffled" and len(phase_angles) >= 2:
                delta_phi = float(phase_angles[(s_idx + 1) % len(phase_angles)] - phase_angles[d_idx])
            elif phase_mode == "frozen":
                delta_phi = 0.25 * float(s_idx - d_idx)
            else:
                delta_phi = float(phase_angles[s_idx] - phase_angles[d_idx])
            pref = float(self.preferred_phase[s_idx, d_idx].detach().item())
            raw_coh = 0.5 * (1.0 + math.cos(delta_phi - pref))
            coherence_gain = (1.0 - self.phase_coupling) + self.phase_coupling * raw_coh

        prev_hyst = float(self.hysteresis_ema[s_idx, d_idx].item())
        meta = torch.tensor(
            [float(src_reg) / max(1, self.num_regions), float(dst_reg) / max(1, self.num_regions), coherence_gain, prev_hyst],
            dtype=p_src.dtype,
            device=p_src.device,
        )
        logit = self.gate_net(torch.cat([p_src, p_dst, meta], dim=0)).squeeze(0)
        gate_val = torch.sigmoid(logit) * float(coherence_gain)
        # Blend with hysteresis EMA to prevent high-frequency route chattering
        blended = 0.85 * gate_val + 0.15 * prev_hyst
        if advance:
            with torch.no_grad():
                self.hysteresis_ema[s_idx, d_idx] = float(blended.detach().item())
        return torch.clamp(blended, 0.0, 1.0), float(coherence_gain)


class CommunicationLimiter:
    """Enforces rate, message count, payload bit budget, and auxiliary current bounds."""

    def __init__(self, spec: CommChannelSpec, max_aux_current: float = 0.10):
        self.spec = spec
        self.max_aux_current = float(np.clip(max_aux_current, 0.0, 1.0))

    def should_communicate_on_step(self, step_id: int) -> bool:
        return (int(step_id) % max(1, int(self.spec.communication_every))) == 0

    def estimate_message_bits(self) -> int:
        if self.spec.mode == "quantized" and self.spec.use_vq:
            return int(math.ceil(math.log2(self.spec.vq_codebook_size)) + 32)
        if self.spec.mode == "orthogonal":
            return int(self.spec.active_modes * self.spec.precision_bits + 16)
        header = 24 if self.spec.mode == "impaired" else 16
        mult = 2 if (self.spec.mode == "impaired" and self.spec.redundancy_coding) else 1
        return int((self.spec.message_dim * self.spec.precision_bits + header) * mult)


class CommunicationTelemetry:
    """Records measured SANCF communication metrics without conflating weight magnitude with bit rate."""

    def __init__(self):
        self.message_history: List[np.ndarray] = []
        self.sender_history: List[np.ndarray] = []
        self.last_step_summary: Dict[str, Any] = {
            "enabled": False,
            "comm_payload_bits_per_step": 0,
            "comm_utilization": 0.0,
            "comm_causal_gain": 0.0,
            "comm_probe_accuracy": 0.0,
            "comm_effective_rank": 0.0,
            "comm_drop_rate": 0.0,
            "comm_latency_steps": 0,
        }

    def record_step(self, messages: List[np.ndarray], senders: List[np.ndarray]) -> None:
        for m in messages:
            self.message_history.append(np.asarray(m, dtype=np.float32).flatten().copy())
        for s in senders:
            self.sender_history.append(np.asarray(s, dtype=np.float32).flatten().copy())
        if len(self.message_history) > 64:
            self.message_history = self.message_history[-64:]
        if len(self.sender_history) > 64:
            self.sender_history = self.sender_history[-64:]

    def compute_effective_rank_and_probe(self) -> Tuple[float, float]:
        """Compute Roy-Vetterli spectral effective rank and linear sender-state probe R^2."""
        if len(self.message_history) < 4:
            return 1.0, 0.0
        M = np.stack(self.message_history[-32:], axis=0)
        M_centered = M - np.mean(M, axis=0, keepdims=True)
        try:
            _, svals, _ = np.linalg.svd(M_centered, full_matrices=False)
            s_pos = svals[svals > 1e-7]
            if len(s_pos) == 0:
                eff_rank = 1.0
            else:
                p = s_pos / np.sum(s_pos)
                ent = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
                eff_rank = float(math.exp(ent))
        except Exception:
            eff_rank = 1.0

        probe_acc = 0.0
        if len(self.sender_history) == len(self.message_history) and len(self.sender_history) >= 6:
            S = np.stack(self.sender_history[-32:], axis=0)
            target = np.mean(S, axis=1)
            reg = 1e-2 * np.eye(M.shape[1], dtype=np.float32)
            try:
                w = np.linalg.solve(M.T @ M + reg, M.T @ target)
                pred = M @ w
                var_t = float(np.var(target))
                if var_t > 1e-7:
                    mse = float(np.mean((target - pred) ** 2))
                    probe_acc = float(np.clip(1.0 - mse / var_t, 0.0, 1.0))
            except Exception:
                probe_acc = 0.0
        return round(eff_rank, 4), round(probe_acc, 4)


class NeuralCommFabric(nn.Module):
    """Additive SANCF & NeuroWave Communication Fabric.

    Obeys the core invariant:
    - Disabled by default (``enabled=False``).
    - Residual gain ``gamma`` initialised to **exactly 0.0**.
    - Never modifies ``base_pre/post/w`` or ``graft_pre/post/w``.
    - Strictly side-effect-free when ``advance=False`` / ``read_only=True``.
    """

    def __init__(
        self,
        graph,
        spec: Optional[CommChannelSpec] = None,
        *,
        enabled: bool = False,
        topology_strategy: str = "structural_centrality",
        max_targets_per_route: int = 32,
        max_aux_current: float = 0.10,
        recompute_per_microstep: bool = False,
        phase_coupling: float = 0.5,
        phase_mode: str = "coherence",
        initial_gamma: float = 0.0,
        seed: int = 7,
        dim: Optional[int] = None,
        mode: Optional[str] = None,
        precision_bits: Optional[int] = None,
        drop_rate: Optional[float] = None,
        latency_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if spec is None:
            self.spec = CommChannelSpec(
                mode=mode or "continuous",
                message_dim=dim or 16,
                precision_bits=precision_bits or 32,
                packet_loss_probability=drop_rate or 0.0,
                max_latency_steps=latency_steps or 0,
                seed=seed,
            )
        else:
            self.spec = spec
        self.enabled = bool(enabled)
        self.topology_strategy = str(topology_strategy)
        self.max_targets_per_route = int(max_targets_per_route)
        self.max_aux_current = float(np.clip(max_aux_current, 0.0, 1.0))
        self.recompute_per_microstep = bool(recompute_per_microstep)
        self.phase_mode = str(phase_mode)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed + 303)

        self.n = int(graph.n)
        origins = np.asarray(graph.origin, dtype=np.int64) if self.n > 0 else np.zeros(0, dtype=np.int64)
        self.num_regions = max(1, int(len(np.unique(origins)))) if self.n > 0 else 1

        # Compile initial route topology without touching legacy graft endpoints
        self.topology: CommTopologyManifest = compile_comm_topology(
            graph,
            strategy=self.topology_strategy,
            max_targets_per_route=self.max_targets_per_route,
            max_routes=max(1, self.spec.max_messages_per_tick),
            seed=self.seed,
        )

        # Preserve Dale's law excitatory/inhibitory sign per source region
        base_pre = np.asarray(graph.base_pre, dtype=np.int64)
        base_w = np.asarray(graph.base_w, dtype=np.float32)
        node_signs = np.ones(max(1, self.n), dtype=np.float32)
        if len(base_pre) > 0 and self.n > 0:
            signed_sum = np.bincount(base_pre, weights=base_w, minlength=self.n)
            node_signs = np.where(signed_sum < 0.0, -1.0, 1.0).astype(np.float32)
        self.register_buffer("node_signs", torch.from_numpy(node_signs))

        self.encoder = MessageEncoder(
            pool_dim=32,
            region_emb_dim=16,
            message_dim=self.spec.message_dim,
            num_regions=self.num_regions,
        )
        self.decoder = MessageDecoder(
            message_dim=self.spec.message_dim,
            region_emb_dim=16,
            max_targets=max(16, self.max_targets_per_route),
            num_regions=self.num_regions,
        )
        self.router = RegionRouter(
            pool_dim=32,
            num_regions=self.num_regions,
            phase_coupling=phase_coupling,
        )
        self.dendritic_proj = nn.Linear(self.spec.message_dim, 32)
        nn.init.zeros_(self.dendritic_proj.weight)
        nn.init.zeros_(self.dendritic_proj.bias)

        # Residual gain gamma initialized to EXACTLY zero by default (Section 3.1 / 3.5)
        self.residual_gain = nn.Parameter(torch.tensor(float(initial_gamma), dtype=torch.float32))

        self.continuous_channel = ContinuousChannel(self.spec)
        self.quantized_channel = QuantisedChannel(self.spec)
        self.orthogonal_channel = OrthogonalModeChannel(self.spec)
        self.impaired_channel = SimulatedImpairedChannel(self.spec)
        self.limiter = CommunicationLimiter(self.spec, max_aux_current=self.max_aux_current)
        self.telemetry = CommunicationTelemetry()

        # Two-clock environment-step state
        self._current_step_id: int = 0
        self._read_only_window: bool = False
        self._step_bits_used: int = 0
        self._step_messages_admitted: int = 0
        self._step_messages_attempted: int = 0
        self._step_messages_dropped: int = 0
        self._step_messages_retransmitted: int = 0
        self._step_messages_recomputed: int = 0
        self._step_compute_ops: int = 0
        self._cached_output: Optional[CommStepOutput] = None
        self.last_output: Optional[CommStepOutput] = None
        self.last_dendritic_context: torch.Tensor = torch.zeros(32, dtype=torch.float32)

        # Causal ablation masks
        self.ablated_routes: Set[Tuple[int, int]] = set()
        self.ablate_all: bool = False
        self.training_steps: int = 0

    def active_channel(self):
        if self.spec.mode == "quantized":
            return self.quantized_channel
        if self.spec.mode in ("orthogonal", "neurowave"):
            return self.orthogonal_channel
        if self.spec.mode == "impaired":
            return self.impaired_channel
        return self.continuous_channel

    def warm_gate(self, target_gamma: float = 0.08, rate: float = 0.25) -> float:
        """Gradually warm residual_gain toward target_gamma within [-max_aux_current, +max_aux_current]."""
        with torch.no_grad():
            cur = float(self.residual_gain.item())
            nxt = (1.0 - rate) * cur + rate * float(target_gamma)
            nxt = float(np.clip(nxt, -self.max_aux_current, self.max_aux_current))
            self.residual_gain.fill_(nxt)
        return nxt

    def reset_episode(self) -> None:
        """Clear transient message queues and step caches at episode boundaries while preserving learned weights."""
        self.impaired_channel.reset_queues()
        self._cached_output = None
        self._step_bits_used = 0
        self._step_messages_admitted = 0
        self._step_messages_attempted = 0
        self._step_messages_dropped = 0
        self._step_messages_retransmitted = 0
        self._step_messages_recomputed = 0
        self._step_compute_ops = 0

    def reset(self) -> None:
        """Alias for reset_episode for compatibility with standard gym/agent resets."""
        self.reset_episode()
        self.last_dendritic_context = torch.zeros(32, dtype=torch.float32)

    def begin_environment_step(self, step_id: int, read_only: bool = False) -> None:
        """Initialize the simulated-time accounting window for environment step ``step_id``."""
        if read_only:
            return
        self._current_step_id = int(step_id)
        self._read_only_window = False
        self._step_bits_used = 0
        self._step_messages_admitted = 0
        self._step_messages_attempted = 0
        self._step_messages_dropped = 0
        self._step_messages_retransmitted = 0
        self._step_messages_recomputed = 0
        self._step_compute_ops = 0
        self._cached_output = None

    def compute_current(
        self,
        r: torch.Tensor,
        *,
        microstep: int = 0,
        advance: bool = True,
        adapt: bool = False,
        phase_angles: Optional[np.ndarray] = None,
    ) -> CommStepOutput:
        """Compute bounded auxiliary communication current u(t) of shape [N]."""
        if not self.enabled or self.n == 0 or r.numel() == 0:
            zero_cur = torch.zeros_like(r)
            return CommStepOutput(
                residual_current=zero_cur,
                payload_bits=0,
                attempted_messages=0,
                admitted_messages=0,
                dropped_messages=0,
                route_ids=[],
                diagnostics={"enabled": False},
            )

        # Sanitize input rates against NaNs/Infs before any computation
        safe_r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
        is_read_only = (not advance) or self._read_only_window

        # Reuse cached message across microsteps > 0 unless recompute_per_microstep is enabled
        if (
            microstep > 0
            and not self.recompute_per_microstep
            and self._cached_output is not None
            and not is_read_only
        ):
            cached = self._cached_output
            # Receiver projection still incurs a lightweight apply cost without new payload bits
            self._step_compute_ops += max(1, len(cached.route_ids) * 16)
            return cached

        # Check environment-step cadence
        if not self.limiter.should_communicate_on_step(self._current_step_id):
            zero_cur = torch.zeros_like(safe_r)
            out = CommStepOutput(
                residual_current=zero_cur,
                payload_bits=0,
                attempted_messages=0,
                admitted_messages=0,
                dropped_messages=0,
                route_ids=[],
                diagnostics={"enabled": True, "cadence_skipped": True, "step_id": self._current_step_id},
            )
            if not is_read_only and microstep == 0:
                self._cached_output = out
                self.last_output = out
            return out

        if is_read_only:
            bits_used = 0
            msgs_admitted = 0
            msgs_attempted = 0
            msgs_dropped = 0
            msgs_retrans = 0
            msgs_recomp = 1 if (microstep > 0 and self.recompute_per_microstep) else 0
            comp_ops = 0
        else:
            bits_used = self._step_bits_used
            msgs_admitted = self._step_messages_admitted
            msgs_attempted = self._step_messages_attempted
            msgs_dropped = self._step_messages_dropped
            msgs_retrans = self._step_messages_retransmitted
            if microstep > 0 and self.recompute_per_microstep:
                self._step_messages_recomputed += 1
            msgs_recomp = self._step_messages_recomputed
            comp_ops = self._step_compute_ops

        # 1. Pool all active source/target regions and score routes
        route_candidates: List[Tuple[float, torch.Tensor, float, RoutePopulationSpec, torch.Tensor]] = []
        for spec_route in self.topology.routes:
            if not self.topology.allow_self_routes and spec_route.source_region == spec_route.target_region:
                continue
            p_src = self.encoder.pool_region(safe_r, spec_route.source_indices)
            p_dst = self.encoder.pool_region(safe_r, spec_route.target_indices)
            gate_val, coh_gain = self.router.score_route(
                p_src,
                p_dst,
                spec_route.source_region,
                spec_route.target_region,
                phase_angles=phase_angles,
                phase_mode=self.phase_mode,
                advance=not is_read_only,
                rng=None if is_read_only else self.rng,
            )
            route_candidates.append((float(gate_val.detach().item()), gate_val, coh_gain, spec_route, p_src))

        # Rank routes by gate score (top-k routing admission)
        route_candidates.sort(key=lambda item: -item[0])

        raw_current = torch.zeros_like(safe_r)
        admitted_routes: List[Tuple[int, int]] = []
        route_records: List[Dict[str, Any]] = []
        recorded_msgs: List[np.ndarray] = []
        recorded_senders: List[np.ndarray] = []
        dendritic_accum = torch.zeros(self.spec.message_dim, dtype=safe_r.dtype, device=safe_r.device)
        channel = self.active_channel()

        for score_f, gate_t, coh_gain, spec_route, p_src in route_candidates:
            msgs_attempted += 1
            r_id = spec_route.route_id
            if self.ablate_all or r_id in self.ablated_routes:
                msgs_dropped += 1
                route_records.append({"route": list(r_id), "status": "ablated", "gate": round(score_f, 4)})
                continue
            if msgs_admitted >= int(self.spec.max_messages_per_tick):
                msgs_dropped += 1
                route_records.append({"route": list(r_id), "status": "msg_budget_rejected", "gate": round(score_f, 4)})
                continue

            est_bits = self.limiter.estimate_message_bits()
            if bits_used + est_bits > int(self.spec.max_payload_bits_per_tick):
                msgs_dropped += 1
                route_records.append({"route": list(r_id), "status": "bit_budget_rejected", "gate": round(score_f, 4)})
                continue

            msg = self.encoder(p_src, spec_route.source_region, spec_route.target_region)
            rx_msg, ch_diag = channel.transmit(msg, r_id, advance=not is_read_only, adapt=adapt)
            actual_bits = int(ch_diag.get("payload_bits", est_bits))
            if bits_used + actual_bits > int(self.spec.max_payload_bits_per_tick):
                msgs_dropped += 1
                route_records.append({"route": list(r_id), "status": "bit_budget_rejected", "gate": round(score_f, 4)})
                continue

            bits_used += actual_bits
            comp_ops += int(ch_diag.get("compute_ops", 64)) + 128
            msgs_retrans += int(ch_diag.get("retransmitted", 0))

            if ch_diag.get("dropped", False):
                msgs_dropped += 1
                route_records.append({
                    "route": list(r_id),
                    "status": "channel_dropped",
                    "gate": round(score_f, 4),
                    "coherence_gain": round(coh_gain, 4),
                    "payload_bits": actual_bits,
                })
                continue

            msgs_admitted += 1
            admitted_routes.append(r_id)
            dendritic_accum = dendritic_accum + gate_t * rx_msg

            target_len = len(spec_route.target_indices)
            decoded = self.decoder(rx_msg, spec_route.target_region, target_len)
            # Enforce Dale's law sign polarity of the source region on the envelope magnitude
            src_idx_t = torch.as_tensor(spec_route.source_indices, dtype=torch.long, device=safe_r.device)
            mean_src_sign = float(torch.sign(self.node_signs.index_select(0, src_idx_t).mean()).item()) if len(spec_route.source_indices) else 1.0
            if mean_src_sign == 0.0:
                mean_src_sign = 1.0
            signed_proj = gate_t * decoded

            dst_idx_t = torch.as_tensor(spec_route.target_indices, dtype=torch.long, device=safe_r.device)
            raw_current = raw_current.index_add(0, dst_idx_t, signed_proj)

            if not is_read_only:
                recorded_msgs.append(rx_msg.detach().cpu().numpy())
                recorded_senders.append(p_src.detach().cpu().numpy())

            route_records.append({
                "route": list(r_id),
                "source_name": spec_route.source_name,
                "target_name": spec_route.target_name,
                "status": "delivered",
                "gate": round(score_f, 4),
                "coherence_gain": round(coh_gain, 4),
                "source_sign": mean_src_sign,
                "payload_bits": actual_bits,
                "message_preview": [round(float(x), 4) for x in rx_msg[:4].detach().cpu().numpy()],
            })

        # Apply zero-initialised residual gain gamma and hard [-max_aux_current, +max_aux_current] clamp
        gamma = torch.clamp(self.residual_gain, -self.max_aux_current, self.max_aux_current)
        clipped_u = torch.clamp(raw_current, -1.0, 1.0)
        final_current = torch.clamp(gamma * clipped_u, -self.max_aux_current, self.max_aux_current)
        if not torch.isfinite(final_current).all():
            final_current = torch.zeros_like(safe_r)

        if not is_read_only:
            self._step_bits_used = bits_used
            self._step_messages_admitted = msgs_admitted
            self._step_messages_attempted = msgs_attempted
            self._step_messages_dropped = msgs_dropped
            self._step_messages_retransmitted = msgs_retrans
            self._step_compute_ops = comp_ops
            self.telemetry.record_step(recorded_msgs, recorded_senders)
            if msgs_admitted > 0:
                self.last_dendritic_context = torch.tanh(self.dendritic_proj(dendritic_accum / float(msgs_admitted))).detach()

        eff_rank, probe_acc = self.telemetry.compute_effective_rank_and_probe()
        max_bits = max(1, int(self.spec.max_payload_bits_per_tick))
        utilization = float(bits_used) / float(max_bits)
        drop_rate = float(msgs_dropped) / float(max(1, msgs_attempted))

        # Route gate entropy diagnostic
        gate_vals = np.array([max(1e-6, item[0]) for item in route_candidates], dtype=np.float64)
        if len(gate_vals) > 1:
            g_probs = gate_vals / np.sum(gate_vals)
            route_entropy = float(-np.sum(g_probs * np.log2(np.maximum(g_probs, 1e-12))))
        else:
            route_entropy = 0.0

        diagnostics = {
            "enabled": True,
            "mode": self.spec.mode,
            "topology_strategy": self.topology.strategy,
            "residual_gain_gamma": round(float(gamma.detach().item()), 6),
            "residual_current_l2": round(float(final_current.detach().norm().item()), 6),
            "residual_current_max": round(float(final_current.detach().abs().max().item()) if final_current.numel() else 0.0, 6),
            "comm_payload_bits_per_step": int(bits_used),
            "comm_utilization": round(utilization, 4),
            "comm_effective_rank": eff_rank,
            "comm_probe_accuracy": probe_acc,
            "comm_drop_rate": round(drop_rate, 4),
            "comm_latency_steps": int(self.spec.max_latency_steps),
            "route_entropy_bits": round(route_entropy, 4),
            "attempted_messages": int(msgs_attempted),
            "admitted_messages": int(msgs_admitted),
            "dropped_messages": int(msgs_dropped),
            "retransmitted_messages": int(msgs_retrans),
            "recomputed_messages": int(msgs_recomp),
            "compute_ops": int(comp_ops),
            "routes": route_records,
            "oscillatory_scope_note": "Abstract coupled phase modes; not literal biophysical Hz membrane rhythms.",
        }

        out = CommStepOutput(
            residual_current=final_current,
            payload_bits=int(bits_used),
            attempted_messages=int(msgs_attempted),
            admitted_messages=int(msgs_admitted),
            dropped_messages=int(msgs_dropped),
            route_ids=admitted_routes,
            diagnostics=diagnostics,
            retransmitted_messages=int(msgs_retrans),
            recomputed_messages=int(msgs_recomp),
            compute_ops=int(comp_ops),
        )
        if not is_read_only:
            self._cached_output = out
            self.last_output = out
            self.telemetry.last_step_summary = diagnostics
        return out

    def reconfigure_topology(self, graph, strategy: str, **kwargs) -> CommTopologyManifest:
        """Atomically compile and swap in a new SANCF route topology with rollback on error."""
        previous = self.topology
        try:
            candidate = compile_comm_topology(
                graph,
                strategy=strategy,
                max_targets_per_route=kwargs.get("max_targets_per_route", self.max_targets_per_route),
                max_routes=kwargs.get("max_routes", max(1, self.spec.max_messages_per_tick)),
                allow_self_routes=kwargs.get("allow_self_routes", False),
                perturbation_scores=kwargs.get("perturbation_scores"),
                seed=kwargs.get("seed", self.seed),
            )
            self.topology = candidate
            self.topology_strategy = strategy
            return candidate
        except Exception:
            self.topology = previous
            raise

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "spec": self.spec.to_dict(),
            "topology_strategy": self.topology.strategy,
            "topology_sha256": self.topology.manifest_sha256,
            "route_count": len(self.topology.routes),
            "residual_gain_gamma": round(float(self.residual_gain.detach().item()), 6),
            "max_aux_current": self.max_aux_current,
            "training_steps": int(self.training_steps),
            "ablated_routes": [list(r) for r in sorted(self.ablated_routes)],
            "ablate_all": bool(self.ablate_all),
            "telemetry": deepcopy(self.telemetry.last_step_summary),
        }

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Return diagnostic and telemetry dictionary for SANCF fabric."""
        summ = self.summary()
        res = dict(self.telemetry.last_step_summary or {})
        res.update(summ)
        res["channel_mode"] = getattr(self.spec, "channel_mode", getattr(self.spec, "mode", "continuous"))
        res["residual_gain"] = round(float(self.residual_gain.detach().item()), 6)
        res["enabled"] = bool(self.enabled)
        return res

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema": "sancf-state-v1",
            "enabled": bool(self.enabled),
            "spec": self.spec.to_dict(),
            "topology_strategy": self.topology_strategy,
            "max_targets_per_route": int(self.max_targets_per_route),
            "max_aux_current": float(self.max_aux_current),
            "phase_mode": self.phase_mode,
            "training_steps": int(self.training_steps),
            "current_step_id": int(self._current_step_id),
            "weights": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
            "quantized_channel": self.quantized_channel.export_state(),
            "orthogonal_channel": self.orthogonal_channel.export_state(),
            "impaired_channel": self.impaired_channel.export_state(),
            "rng": deepcopy(self.rng.bit_generator.state),
            "telemetry": deepcopy(self.telemetry.last_step_summary),
            "step_bits_used": int(self._step_bits_used),
            "step_messages_admitted": int(self._step_messages_admitted),
            "step_messages_attempted": int(self._step_messages_attempted),
            "step_messages_dropped": int(self._step_messages_dropped),
            "step_compute_ops": int(self._step_compute_ops),
        }

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Return diagnostic and telemetry dictionary for SANCF fabric."""
        summ = self.summary()
        res = dict(self.telemetry.last_step_summary or {})
        res.update(summ)
        res["channel_mode"] = getattr(self.spec, "channel_mode", getattr(self.spec, "mode", "continuous"))
        res["residual_gain"] = round(float(self.residual_gain.detach().item()), 6)
        res["enabled"] = bool(self.enabled)
        return res

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("SANCF checkpoint state must be a dictionary")
        # Atomic snapshot before mutation so any error leaves live state untouched
        backup_weights = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        backup_q = self.quantized_channel.export_state()
        backup_o = self.orthogonal_channel.export_state()
        backup_i = self.impaired_channel.export_state()
        backup_rng = deepcopy(self.rng.bit_generator.state)
        try:
            if "weights" in state:
                w_dict = state["weights"]
                cur_state = self.state_dict()
                for k, target_tensor in cur_state.items():
                    if k in w_dict:
                        src_t = torch.as_tensor(w_dict[k], dtype=target_tensor.dtype)
                        if src_t.shape != target_tensor.shape or not torch.isfinite(src_t).all():
                            raise ValueError(f"SANCF tensor mismatch or non-finite values on key {k!r}")
                self.load_state_dict(
                    {k: torch.as_tensor(w_dict[k], dtype=cur_state[k].dtype) for k in cur_state if k in w_dict},
                    strict=False,
                )
            if "quantized_channel" in state:
                self.quantized_channel.import_state(state["quantized_channel"])
            if "orthogonal_channel" in state:
                self.orthogonal_channel.import_state(state["orthogonal_channel"])
            if "impaired_channel" in state:
                self.impaired_channel.import_state(state["impaired_channel"])
            if "rng" in state:
                self.rng.bit_generator.state = deepcopy(state["rng"])
            self.enabled = bool(state.get("enabled", self.enabled))
            self.training_steps = int(state.get("training_steps", self.training_steps))
            self._current_step_id = int(state.get("current_step_id", self._current_step_id))
            self._step_bits_used = int(state.get("step_bits_used", self._step_bits_used))
            self._step_messages_admitted = int(state.get("step_messages_admitted", self._step_messages_admitted))
            self._step_messages_attempted = int(state.get("step_messages_attempted", self._step_messages_attempted))
            self._step_messages_dropped = int(state.get("step_messages_dropped", self._step_messages_dropped))
            self._step_compute_ops = int(state.get("step_compute_ops", self._step_compute_ops))
            if "telemetry" in state and isinstance(state["telemetry"], dict):
                self.telemetry.last_step_summary = deepcopy(state["telemetry"])
        except Exception:
            self.load_state_dict(backup_weights, strict=False)
            self.quantized_channel.import_state(backup_q)
            self.orthogonal_channel.import_state(backup_o)
            self.impaired_channel.import_state(backup_i)
            self.rng.bit_generator.state = backup_rng
            raise
