"""SANCF Virtual Communication Channels & NeuroWave Mode Multiplexing.

Implements four typed virtual communication channels behind a unified protocol:
1. ``ContinuousChannel``: fp32 (or simulated fp16/bfloat16) continuous latent vectors.
2. ``QuantisedChannel``: deterministic uniform integer quantisation (2/4/8/16-bit) and optional
   Vector Quantisation (VQ codebook with 16/64/256 entries, straight-through estimator, EMA
   codebook updates, dead-code revival, and full header/scale/index payload bit accounting).
3. ``OrthogonalModeChannel``: digital orthogonal-mode representation analogy
   ``M_ij(t) = sum_{k=1}^K a_{ij,k}(t) psi_k`` using orthonormal Walsh-Hadamard, DCT, or
   Cayley-parameterised Stiefel basis vectors, optional crosstalk matrix, and norm conservation.
   Explicitly documented as a digital basis transform, NOT physical OAM wave multiplexing.
4. ``SimulatedImpairedChannel``: bounded step-latency queue, packet drop, burst loss, additive
   receiver noise, parity/diversity redundancy coding, and deterministic ARQ retransmission
   accounting with strict ``advance=False`` read-only invariance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ChannelMode = Literal["continuous", "quantized", "orthogonal", "impaired", "neurowave"]
ALLOWED_PRECISION_BITS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class CommChannelSpec:
    mode: ChannelMode = "continuous"
    message_dim: int = 16
    precision_bits: int = 32
    communication_every: int = 1
    max_messages_per_tick: int = 8
    max_payload_bits_per_tick: int = 4096
    max_latency_steps: int = 0
    packet_loss_probability: float = 0.0
    burst_loss_probability: float = 0.0
    noise_std: float = 0.0
    use_vq: bool = False
    vq_codebook_size: int = 64
    orthogonal_basis: str = "dct"  # 'hadamard' | 'dct' | 'learned'
    active_modes: int = 8
    crosstalk_coeff: float = 0.0
    redundancy_coding: bool = False
    retransmit_on_drop: bool = False
    seed: int = 7

    def __post_init__(self) -> None:
        if self.mode not in ("continuous", "quantized", "orthogonal", "impaired", "neurowave"):
            raise ValueError(f"Unsupported channel mode: {self.mode!r}")
        if not (1 <= int(self.message_dim) <= 128):
            raise ValueError("message_dim must be in [1, 128]")
        if int(self.precision_bits) not in ALLOWED_PRECISION_BITS:
            raise ValueError(f"precision_bits must be one of {ALLOWED_PRECISION_BITS}")
        if not (1 <= int(self.communication_every) <= 128):
            raise ValueError("communication_every must be in [1, 128]")
        if not (0 <= int(self.max_messages_per_tick) <= 64):
            raise ValueError("max_messages_per_tick must be in [0, 64]")
        if not (0 <= int(self.max_payload_bits_per_tick) <= 65536):
            raise ValueError("max_payload_bits_per_tick must be in [0, 65536]")
        if not (0 <= int(self.max_latency_steps) <= 16):
            raise ValueError("max_latency_steps must be in [0, 16]")
        if not (0.0 <= float(self.packet_loss_probability) <= 1.0):
            raise ValueError("packet_loss_probability must be in [0, 1]")
        if not (0.0 <= float(self.burst_loss_probability) <= 1.0):
            raise ValueError("burst_loss_probability must be in [0, 1]")
        if not (0.0 <= float(self.noise_std) <= 5.0):
            raise ValueError("noise_std must be in [0, 5]")
        if int(self.vq_codebook_size) not in (16, 64, 256):
            raise ValueError("vq_codebook_size must be 16, 64, or 256")
        if not (1 <= int(self.active_modes) <= int(self.message_dim)):
            object.__setattr__(self, "active_modes", min(max(1, int(self.active_modes)), int(self.message_dim)))

    @property
    def channel_mode(self) -> str:
        return str(self.mode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommStepOutput:
    residual_current: torch.Tensor  # shape [N], same dtype/device as native activity
    payload_bits: int
    attempted_messages: int
    admitted_messages: int
    dropped_messages: int
    route_ids: List[Tuple[int, int]]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    retransmitted_messages: int = 0
    recomputed_messages: int = 0
    compute_ops: int = 0


def build_orthonormal_basis(dim: int, kind: str = "dct", seed: int = 7) -> np.ndarray:
    """Construct an exact D x D orthonormal basis matrix Psi (Psi^T Psi = I)."""
    if kind == "hadamard" and (dim & (dim - 1) == 0):
        H = np.array([[1.0]], dtype=np.float64)
        while H.shape[0] < dim:
            H = np.block([[H, H], [H, -H]])
        H = H / math.sqrt(dim)
        return H.astype(np.float32)
    if kind == "dct":
        basis = np.zeros((dim, dim), dtype=np.float64)
        for k in range(dim):
            alpha = math.sqrt(1.0 / dim) if k == 0 else math.sqrt(2.0 / dim)
            for n in range(dim):
                basis[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2.0 * dim))
        return basis.astype(np.float32)
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 1.0, size=(dim, dim))
    q, r = np.linalg.qr(raw)
    d = np.diag(r)
    q = q * np.where(d >= 0, 1.0, -1.0)
    return q.astype(np.float32)


class ContinuousChannel(nn.Module):
    """Full-precision or reduced-float continuous latent vector channel."""

    HEADER_BITS = 16  # 8-bit src/dst route tag + 8-bit sequence metadata

    def __init__(self, spec: CommChannelSpec):
        super().__init__()
        self.spec = spec

    def transmit(
        self,
        message: torch.Tensor,
        route: Tuple[int, int],
        *,
        advance: bool = True,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.isfinite(message).all():
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        bits_per_elem = 16 if self.spec.precision_bits == 16 else 32
        if bits_per_elem == 16:
            out = message.to(torch.float16).to(message.dtype)
        else:
            out = message
        payload_bits = int(message.numel() * bits_per_elem + self.HEADER_BITS)
        return out, {
            "channel_mode": "continuous",
            "payload_bits": payload_bits,
            "header_bits": self.HEADER_BITS,
            "dropped": False,
            "retransmitted": 0,
            "latency_steps": 0,
            "compute_ops": int(message.numel() * 2),
        }

    def export_state(self) -> Dict[str, Any]:
        return {"mode": "continuous"}

    def import_state(self, state: Dict[str, Any]) -> None:
        return


class QuantisedChannel(nn.Module):
    """Uniform b-bit quantiser (STE) and optional Vector Quantisation (VQ) codebook."""

    HEADER_BITS = 16
    SCALE_METADATA_BITS = 16

    def __init__(self, spec: CommChannelSpec):
        super().__init__()
        self.spec = spec
        self.dim = int(spec.message_dim)
        self.k_codes = int(spec.vq_codebook_size)
        rng = np.random.default_rng(spec.seed)
        init_cb = rng.normal(0.0, 0.35, size=(self.k_codes, self.dim)).astype(np.float32)
        self.codebook = nn.Parameter(torch.from_numpy(init_cb))
        self.register_buffer("ema_counts", torch.ones(self.k_codes, dtype=torch.float32))
        self.register_buffer("usage_hist", torch.zeros(self.k_codes, dtype=torch.float32))

    def _uniform_quantize(self, x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if bits >= 32:
            payload_bits = int(x.numel() * 32 + self.HEADER_BITS)
            return x, {"quant_type": "fp32", "payload_bits": payload_bits, "entropy_bits": 0.0}
        levels = (1 << bits) - 1
        x_clamped = torch.clamp(x, -1.0, 1.0)
        scaled = (x_clamped + 1.0) * 0.5 * levels
        rounded = torch.round(scaled)
        x_q = (rounded / max(1, levels)) * 2.0 - 1.0
        # Straight-through estimator preserves exact gradient dx_ste/dx = 1
        x_ste = x_clamped + (x_q - x_clamped).detach()
        # Empirical bin entropy diagnostic (distinct from semantic mutual information)
        with torch.no_grad():
            bins = rounded.detach().cpu().numpy().astype(np.int64).flatten()
            _, counts = np.unique(bins, return_counts=True)
            probs = counts / max(1, len(bins))
            ent = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
        payload_bits = int(x.numel() * bits + self.HEADER_BITS + self.SCALE_METADATA_BITS)
        return x_ste, {
            "quant_type": f"uniform_{bits}bit",
            "levels": int(levels + 1),
            "payload_bits": payload_bits,
            "header_bits": self.HEADER_BITS + self.SCALE_METADATA_BITS,
            "empirical_bin_entropy_bits": round(ent, 4),
            "quantization_l2_error": round(float((x_clamped - x_q).norm().item()), 5),
        }

    def _vq_quantize(self, x: torch.Tensor, *, advance: bool, adapt: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        vec = x.reshape(-1, self.dim)
        # Squared Euclidean distance to codebook entries
        dists = (
            torch.sum(vec ** 2, dim=1, keepdim=True)
            - 2.0 * (vec @ self.codebook.t())
            + torch.sum(self.codebook ** 2, dim=1).unsqueeze(0)
        )
        idx = torch.argmin(dists, dim=1)
        quant = self.codebook[idx].reshape_as(x)
        commitment_loss = F.mse_loss(vec, quant.detach().reshape(-1, self.dim))
        codebook_loss = F.mse_loss(quant.reshape(-1, self.dim), vec.detach())
        # Straight-through estimator
        x_ste = x + (quant - x).detach()

        if advance:
            with torch.no_grad():
                one_hot = F.one_hot(idx, num_classes=self.k_codes).to(torch.float32).sum(dim=0)
                self.usage_hist.mul_(0.98).add_(0.02 * one_hot)
                if adapt:
                    self.ema_counts.mul_(0.95).add_(0.05 * one_hot)
                    # Dead-code revival: if an entry has near-zero EMA count, re-anchor toward active input
                    dead_mask = self.ema_counts < 0.15
                    if bool(dead_mask.any()):
                        first_dead = int(torch.nonzero(dead_mask, as_tuple=False)[0].item())
                        self.codebook.data[first_dead] = (0.8 * self.codebook.data[first_dead] + 0.2 * vec[0].detach())
                        self.ema_counts[first_dead] = 0.5

        with torch.no_grad():
            p = self.usage_hist / torch.clamp(self.usage_hist.sum(), min=1e-8)
            ent = float(-(p * torch.log2(torch.clamp(p, min=1e-12))).sum().item())
            perplexity = float(2.0 ** ent)
            active_codes = int((self.usage_hist > 1e-3).sum().item())

        index_bits = int(math.ceil(math.log2(self.k_codes))) * int(idx.numel())
        payload_bits = int(index_bits + self.HEADER_BITS + self.SCALE_METADATA_BITS)
        return x_ste, {
            "quant_type": f"vq_{self.k_codes}",
            "code_index": int(idx[0].item()),
            "payload_bits": payload_bits,
            "index_bits": index_bits,
            "header_bits": self.HEADER_BITS + self.SCALE_METADATA_BITS,
            "codebook_perplexity": round(perplexity, 4),
            "active_codebook_entries": active_codes,
            "commitment_loss": round(float(commitment_loss.item()), 6),
            "vq_loss_tensor": commitment_loss * 0.25 + codebook_loss,
        }

    def transmit(
        self,
        message: torch.Tensor,
        route: Tuple[int, int],
        *,
        advance: bool = True,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.isfinite(message).all():
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.spec.use_vq:
            out, diag = self._vq_quantize(message, advance=advance, adapt=adapt)
        else:
            out, diag = self._uniform_quantize(message, self.spec.precision_bits)
        diag.update({
            "channel_mode": "quantized",
            "dropped": False,
            "retransmitted": 0,
            "latency_steps": 0,
            "compute_ops": int(message.numel() * (self.k_codes if self.spec.use_vq else 6)),
        })
        return out, diag

    def export_state(self) -> Dict[str, Any]:
        return {
            "codebook": self.codebook.detach().cpu().clone(),
            "ema_counts": self.ema_counts.detach().cpu().clone(),
            "usage_hist": self.usage_hist.detach().cpu().clone(),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        if "codebook" in state:
            t = torch.as_tensor(state["codebook"], dtype=torch.float32)
            if t.shape != self.codebook.shape or not torch.isfinite(t).all():
                raise ValueError("Invalid VQ codebook shape or non-finite values in checkpoint")
            with torch.no_grad():
                self.codebook.copy_(t)
        for buf_name in ("ema_counts", "usage_hist"):
            if buf_name in state:
                b = torch.as_tensor(state[buf_name], dtype=torch.float32)
                if b.shape != getattr(self, buf_name).shape or not torch.isfinite(b).all():
                    raise ValueError(f"Invalid {buf_name} buffer in checkpoint")
                getattr(self, buf_name).copy_(b)


class OrthogonalModeChannel(nn.Module):
    """Digital orthogonal-mode representation analogy: M_ij(t) = sum_{k=1}^K a_{ij,k}(t) psi_k.

    Scientific Scope Note:
    This is a digital orthonormal basis decomposition (Walsh-Hadamard / DCT / Stiefel),
    NOT physical orbital-angular-momentum (OAM) electromagnetic waves or quantum superdense
    coding. Under equal dimensionality, bit budget, and power norm, orthonormal transforms
    preserve Euclidean energy (Parseval's theorem) and do not create free Shannon capacity.
    """

    HEADER_BITS = 16

    def __init__(self, spec: CommChannelSpec):
        super().__init__()
        self.spec = spec
        self.dim = int(spec.message_dim)
        self.active_modes = min(int(spec.active_modes), self.dim)
        basis_np = build_orthonormal_basis(self.dim, kind=spec.orthogonal_basis, seed=spec.seed)
        self.register_buffer("basis", torch.from_numpy(basis_np))  # [D, D], rows are psi_k
        self.skew_param = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.mode_gates = nn.Parameter(torch.ones(self.dim))
        self.rng = np.random.default_rng(spec.seed + 101)

    def current_basis(self) -> torch.Tensor:
        """Return exact orthonormal basis rows psi_k (via Cayley transform if learned)."""
        if self.spec.orthogonal_basis == "learned":
            skew = 0.5 * (self.skew_param - self.skew_param.t())
            eye = torch.eye(self.dim, device=skew.device, dtype=skew.dtype)
            orth = torch.linalg.solve(eye + skew, eye - skew)
            return orth @ self.basis
        return self.basis

    def transmit(
        self,
        message: torch.Tensor,
        route: Tuple[int, int],
        *,
        advance: bool = True,
        adapt: bool = False,
        context_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.isfinite(message).all():
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)
        psi = self.current_basis()  # [D, D]
        # Project message onto orthonormal mode basis: a = Psi @ m
        coeffs = psi @ message  # [D]
        input_energy = float((message ** 2).sum().item())
        full_mode_energy = float((coeffs ** 2).sum().item())

        # Mode allocation mask: retain top K active virtual modes
        mask = torch.zeros_like(coeffs)
        mask[: self.active_modes] = torch.sigmoid(self.mode_gates[: self.active_modes]) * 2.0 / (1.0 + math.exp(-1.0))
        if self.spec.orthogonal_basis != "learned":
            # Exact unit gain on active modes by default so K=D is exact identity
            mask[: self.active_modes] = 1.0
        if context_gate is not None and context_gate.numel() >= self.active_modes:
            mask[: self.active_modes] = mask[: self.active_modes] * torch.clamp(context_gate[: self.active_modes], 0.0, 1.0)

        active_coeffs = coeffs * mask

        # Optional known mode cross-talk matrix C = (1 - eps) I + (eps / max(1, D-1)) (11^T - I)
        eps = float(self.spec.crosstalk_coeff)
        crosstalk_leakage = 0.0
        if eps > 0.0 and self.dim > 1:
            off_diag = eps / float(self.dim - 1)
            C = (1.0 - eps - off_diag) * torch.eye(self.dim, dtype=coeffs.dtype, device=coeffs.device) + off_diag
            mixed_coeffs = C @ active_coeffs
            crosstalk_leakage = float((mixed_coeffs - active_coeffs).norm().item() / (active_coeffs.norm().item() + 1e-8))
        else:
            mixed_coeffs = active_coeffs

        # Optional receiver noise (never consumes RNG when advance=False)
        if self.spec.noise_std > 0.0 and advance:
            noise_np = self.rng.normal(0.0, float(self.spec.noise_std), size=mixed_coeffs.shape).astype(np.float32)
            mixed_coeffs = mixed_coeffs + torch.from_numpy(noise_np).to(mixed_coeffs.device)

        # Reconstruct multiplexed signal M_ij(t) = sum_{k=1}^K a_{ij,k}(t) psi_k = Psi^T @ mixed_coeffs
        reconstructed = psi.t() @ mixed_coeffs
        output_energy = float((reconstructed ** 2).sum().item())

        payload_bits = int(self.active_modes * self.spec.precision_bits + self.HEADER_BITS)
        return reconstructed, {
            "channel_mode": "orthogonal",
            "representation_scope": "digital_orthogonal_mode_analogy_not_physical_oam",
            "basis_type": self.spec.orthogonal_basis,
            "active_modes": self.active_modes,
            "total_modes": self.dim,
            "input_energy": round(input_energy, 6),
            "parseval_mode_energy": round(full_mode_energy, 6),
            "output_energy": round(output_energy, 6),
            "crosstalk_leakage_ratio": round(crosstalk_leakage, 6),
            "payload_bits": payload_bits,
            "header_bits": self.HEADER_BITS,
            "dropped": False,
            "retransmitted": 0,
            "latency_steps": 0,
            "compute_ops": int(2 * self.dim * self.dim),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "basis": self.basis.detach().cpu().clone(),
            "skew_param": self.skew_param.detach().cpu().clone(),
            "mode_gates": self.mode_gates.detach().cpu().clone(),
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        for key in ("basis", "skew_param", "mode_gates"):
            if key in state:
                t = torch.as_tensor(state[key], dtype=torch.float32)
                target = getattr(self, key)
                if t.shape != target.shape or not torch.isfinite(t).all():
                    raise ValueError(f"Invalid {key} tensor in OrthogonalModeChannel state")
                with torch.no_grad():
                    target.copy_(t)
        if "rng" in state:
            self.rng.bit_generator.state = deepcopy(state["rng"])


class SimulatedImpairedChannel(nn.Module):
    """Step-delayed, lossy, noisy channel with redundancy coding & ARQ accounting."""

    HEADER_BITS = 24  # route tag + sequence number + parity/CRC tag

    def __init__(self, spec: CommChannelSpec):
        super().__init__()
        self.spec = spec
        self.dim = int(spec.message_dim)
        self.rng = np.random.default_rng(spec.seed + 202)
        # Per-route delay queues: route -> list of (remaining_steps, tensor)
        self.queues: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = {}
        self.in_burst_loss: Dict[Tuple[int, int], bool] = {}

    def reset_queues(self) -> None:
        self.queues.clear()
        self.in_burst_loss.clear()

    def transmit(
        self,
        message: torch.Tensor,
        route: Tuple[int, int],
        *,
        advance: bool = True,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not torch.isfinite(message).all():
            message = torch.nan_to_num(message, nan=0.0, posinf=1.0, neginf=-1.0)

        # Strictly non-mutating read-only path when advance=False
        if not advance:
            q = self.queues.get(route, [])
            ready = [tensor for rem, tensor in q if rem <= 0]
            out = ready[0] if ready else (message if self.spec.max_latency_steps == 0 else torch.zeros_like(message))
            base_bits = int(self.dim * self.spec.precision_bits + self.HEADER_BITS)
            return out, {
                "channel_mode": "impaired",
                "payload_bits": base_bits * (2 if self.spec.redundancy_coding else 1),
                "dropped": False,
                "retransmitted": 0,
                "latency_steps": int(self.spec.max_latency_steps),
                "queue_depth": len(q),
                "compute_ops": int(self.dim * 4),
            }

        # Advance=True: deterministic local RNG draws for loss, burst state, and noise
        in_burst = self.in_burst_loss.get(route, False)
        if in_burst:
            # Stay in burst with probability 0.5, else recover
            in_burst = bool(self.rng.random() < 0.5)
        elif self.spec.burst_loss_probability > 0.0:
            in_burst = bool(self.rng.random() < float(self.spec.burst_loss_probability))
        self.in_burst_loss[route] = in_burst

        drop_prob = float(self.spec.packet_loss_probability)
        raw_dropped = in_burst or (drop_prob > 0.0 and bool(self.rng.random() < drop_prob))
        retransmissions = 0
        effective_dropped = raw_dropped

        if raw_dropped and self.spec.redundancy_coding:
            # Paired diversity / parity code recovers single-subchannel drop unless burst-corrupted
            if not in_burst:
                effective_dropped = False
        if effective_dropped and self.spec.retransmit_on_drop:
            retransmissions = 1
            retry_drop = bool(self.rng.random() < (drop_prob * 0.5))
            effective_dropped = retry_drop

        tx_vec = message
        if self.spec.noise_std > 0.0 and not effective_dropped:
            noise_scale = float(self.spec.noise_std) * (0.5 if self.spec.redundancy_coding else 1.0)
            noise_np = self.rng.normal(0.0, noise_scale, size=message.shape).astype(np.float32)
            tx_vec = tx_vec + torch.from_numpy(noise_np).to(message.device)

        q = self.queues.setdefault(route, [])
        # Decrement remaining latency on queued messages
        q = [(rem - 1, tensor) for rem, tensor in q]
        if not effective_dropped:
            q.append((int(self.spec.max_latency_steps), tx_vec))
        # Bound queue length to prevent unbounded memory growth
        if len(q) > 32:
            q = q[-32:]

        delivered = torch.zeros_like(message)
        remaining_q: List[Tuple[int, torch.Tensor]] = []
        delivered_count = 0
        for rem, tensor in q:
            if rem <= 0 and delivered_count == 0:
                delivered = tensor
                delivered_count += 1
            elif rem > 0:
                remaining_q.append((rem, tensor))
        self.queues[route] = remaining_q

        base_bits = int(self.dim * self.spec.precision_bits + self.HEADER_BITS)
        coding_mult = 2 if self.spec.redundancy_coding else 1
        payload_bits = base_bits * coding_mult * (1 + retransmissions)

        return delivered, {
            "channel_mode": "impaired",
            "payload_bits": payload_bits,
            "header_bits": self.HEADER_BITS,
            "dropped": bool(effective_dropped),
            "raw_drop_event": bool(raw_dropped),
            "in_burst_loss": bool(in_burst),
            "retransmitted": int(retransmissions),
            " delivered": bool(delivered_count > 0),
            "latency_steps": int(self.spec.max_latency_steps),
            "queue_depth": len(remaining_q),
            "compute_ops": int(self.dim * 6),
        }

    def export_state(self) -> Dict[str, Any]:
        serialized_queues = [
            ( [int(r[0]), int(r[1])], [(int(rem), t.detach().cpu().clone()) for rem, t in items] )
            for r, items in sorted(self.queues.items())
        ]
        burst_list = [([int(r[0]), int(r[1])], bool(val)) for r, val in sorted(self.in_burst_loss.items())]
        return {
            "queues": serialized_queues,
            "in_burst_loss": burst_list,
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        new_queues: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor]]] = {}
        for route_pair, items in state.get("queues", []):
            r_key = (int(route_pair[0]), int(route_pair[1]))
            new_queues[r_key] = [(int(rem), torch.as_tensor(t, dtype=torch.float32).clone()) for rem, t in items]
        self.queues = new_queues
        self.in_burst_loss = {
            (int(r[0]), int(r[1])): bool(v) for r, v in state.get("in_burst_loss", [])
        }
        if "rng" in state:
            self.rng.bit_generator.state = deepcopy(state["rng"])
