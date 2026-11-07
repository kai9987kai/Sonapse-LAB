"""Differentiable PyTorch transport engine for Sonapse NeuroWave-X2.

Implements the differentiable communication pathway, spectral decoding, phase-alignment
modulation, and bounded residual injection, verified against ``ReferenceTransport``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .packets import NeuralPacket


def quantize_torch(x: torch.Tensor, bits: int = 8, v_min: float = -1.0, v_max: float = 1.0) -> torch.Tensor:
    """Straight-through estimator uniform quantization in PyTorch."""
    if bits >= 32:
        return x
    levels = 2 ** bits
    step = (v_max - v_min) / float(levels - 1)
    clipped = torch.clamp(x, v_min, v_max)
    # Detached round with straight-through gradient
    quantized_idx = torch.round((clipped - v_min) / step)
    quantized = quantized_idx * step + v_min
    return clipped + (quantized - clipped).detach()


def spectral_synthesize_torch(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Synthesizes regional signal from mode amplitudes and orthogonal basis matrix.
    
    Args:
        coefficients: (K,) or (B, K)
        basis: (K, D) orthogonal basis matrix
    Returns:
        (D,) or (B, D) vector
    """
    K, D = basis.shape
    if coefficients.dim() == 1:
        c = coefficients[:K]
        if c.numel() < K:
            c = F.pad(c, (0, K - c.numel()))
        return torch.matmul(c, basis)
    else:
        c = coefficients[:, :K]
        if c.shape[1] < K:
            c = F.pad(c, (0, K - c.shape[1]))
        return torch.matmul(c, basis)


def phase_coherence_gain_torch(phase_src: torch.Tensor, phase_dst: torch.Tensor) -> torch.Tensor:
    """Calculates non-negative phase alignment gain in [0.0, 1.0] preserving Dale's law."""
    d_theta = phase_src - phase_dst
    return 0.5 * (1.0 + torch.cos(d_theta))


class PyTorchTransport(nn.Module):
    """Differentiable transport module with deterministic queue and bounded residual injection."""

    def __init__(
        self,
        dimension: int = 16,
        precision_bits: int = 32,
        max_latency: int = 0,
        packet_loss: float = 0.0,
        noise_std: float = 0.0,
        max_aux_current: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.precision_bits = int(precision_bits)
        self.max_latency = int(max_latency)
        self.packet_loss = float(packet_loss)
        self.noise_std = float(noise_std)
        self.max_aux_current = float(max_aux_current)
        
        # Dedicated RNG for communication noise and latency
        self._rng = np.random.RandomState(seed)
        self.queue: List[Tuple[int, NeuralPacket]] = []
        self.delivered_packets: List[NeuralPacket] = []
        self.dropped_count = 0
        self.transmitted_count = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.queue.clear()
        self.delivered_packets.clear()
        self.dropped_count = 0
        self.transmitted_count = 0

    def send(self, packet: NeuralPacket, current_step: int) -> bool:
        """Enqueue packet subject to loss and simulated latency."""
        self.transmitted_count += 1
        if self.packet_loss > 0.0 and self._rng.uniform(0.0, 1.0) < self.packet_loss:
            self.dropped_count += 1
            return False

        latency = 0
        if self.max_latency > 0:
            latency = int(self._rng.randint(0, self.max_latency + 1))
        delivery_step = current_step + latency
        self.queue.append((delivery_step, packet))
        return True

    def receive(self, current_step: int, receiver_id: int) -> List[NeuralPacket]:
        """Collect all packets ready for delivery to receiver_id at current_step."""
        ready: List[NeuralPacket] = []
        remaining: List[Tuple[int, NeuralPacket]] = []
        for step, pkt in self.queue:
            if step <= current_step and pkt.receiver_region == receiver_id:
                ready.append(pkt)
                self.delivered_packets.append(pkt)
            else:
                remaining.append((step, pkt))
        self.queue = remaining
        return ready

    def decode_and_inject(
        self,
        packets: Sequence[NeuralPacket],
        base_activation: torch.Tensor,
        decoder_weights: torch.Tensor,  # shape (D, target_size)
        basis: Optional[torch.Tensor] = None,  # shape (K, D)
        receiver_phase: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes received packets into bounded residual current added to base_activation.
        
        Formula:
            h_new = h_base + gamma * clip(delta_comm, -max_aux, max_aux)
            
        Returns:
            (h_new, accumulated_delta)
        """
        target_size = base_activation.shape[0]
        device = base_activation.device
        dtype = base_activation.dtype

        if len(packets) == 0 or gamma == 0.0:
            return base_activation.clone(), torch.zeros(target_size, dtype=dtype, device=device)

        accumulated_delta = torch.zeros(target_size, dtype=dtype, device=device)

        for pkt in packets:
            payload = torch.tensor(pkt.payload, dtype=dtype, device=device)

            # 1. Quantization
            if pkt.precision_bits < 32:
                payload = quantize_torch(payload, bits=pkt.precision_bits)

            # 2. Add noise if active in training
            if self.noise_std > 0.0 and self.training:
                noise = torch.randn_like(payload) * self.noise_std
                payload = payload + noise

            # 3. Spectral synthesis if basis provided
            if "spectral" in pkt.protocol_id and basis is not None:
                decoded_vector = spectral_synthesize_torch(payload, basis.to(device=device, dtype=dtype))
            else:
                decoded_vector = payload

            # Match dimension with decoder input
            d_in = decoder_weights.shape[0]
            if decoded_vector.numel() != d_in:
                if decoded_vector.numel() < d_in:
                    decoded_vector = F.pad(decoded_vector, (0, d_in - decoded_vector.numel()))
                else:
                    decoded_vector = decoded_vector[:d_in]

            # 4. Project through decoder weights
            pop_current = torch.matmul(decoded_vector, decoder_weights.to(device=device, dtype=dtype))

            # 5. Phase gating
            if "phase" in pkt.protocol_id and pkt.phase_metadata and receiver_phase is not None:
                src_phase = torch.tensor(pkt.phase_metadata[0], dtype=dtype, device=device)
                gain = phase_coherence_gain_torch(src_phase, receiver_phase.to(device=device, dtype=dtype))
                pop_current = pop_current * gain

            accumulated_delta = accumulated_delta + pop_current

        # 6. Bounded clipping and residual injection
        clipped_delta = torch.clamp(accumulated_delta, -self.max_aux_current, self.max_aux_current)
        h_new = base_activation + gamma * clipped_delta
        return h_new, accumulated_delta
