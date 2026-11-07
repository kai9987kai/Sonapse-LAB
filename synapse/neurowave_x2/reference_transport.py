"""Independent NumPy reference transport implementation for Sonapse NeuroWave-X2.

Used exclusively for scientific conformance testing, numerical verification,
and causal oracle comparisons (independent of PyTorch autograd engine).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .packets import NeuralPacket, PacketValidationError


def quantize_numpy(x: np.ndarray, bits: int = 8, v_min: float = -1.0, v_max: float = 1.0) -> np.ndarray:
    """Uniform mid-tread quantization in NumPy."""
    if bits >= 32:
        return x.copy()
    levels = 2 ** bits
    step = (v_max - v_min) / (levels - 1)
    clipped = np.clip(x, v_min, v_max)
    quantized_idx = np.round((clipped - v_min) / step)
    return quantized_idx * step + v_min


def spectral_synthesize_numpy(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Synthesize regional signal from spectral coefficients and eigenvector carriers.
    
    Args:
        coefficients: (K,) mode amplitudes
        basis: (K, D) orthogonal basis matrix (each row is a mode carrier)
    Returns:
        (D,) synthesized signal vector = sum_k c_k * basis[k]
    """
    K, D = basis.shape
    c = coefficients[:K]
    if len(c) < K:
        padded = np.zeros(K, dtype=basis.dtype)
        padded[:len(c)] = c
        c = padded
    return np.dot(c, basis)


def spectral_analyze_numpy(signal: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project signal vector onto orthogonal spectral modes.
    
    Args:
        signal: (D,) signal vector
        basis: (K, D) orthogonal basis matrix
    Returns:
        (K,) spectral coefficients
    """
    return np.dot(basis, signal)


def phase_coherence_gain_numpy(phase_src: float, phase_dst: float) -> float:
    """Calculates non-negative phase alignment gain in [0.0, 1.0].
    Preserves Dale's law and avoids arbitrary sign-inversions.
    """
    d_theta = float(phase_src - phase_dst)
    return float(0.5 * (1.0 + math.cos(d_theta)))


class ReferenceTransport:
    """Independent NumPy reference transport engine.
    Simulates transmission, impairment, latency queue, quantisation,
    spectral projection, and bounded residual injection.
    """

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
        self.dimension = int(dimension)
        self.precision_bits = int(precision_bits)
        self.max_latency = int(max_latency)
        self.packet_loss = float(packet_loss)
        self.noise_std = float(noise_std)
        self.max_aux_current = float(max_aux_current)
        self.rng = np.random.RandomState(seed)
        # Latency queue: list of (delivery_step, NeuralPacket)
        self.queue: List[Tuple[int, NeuralPacket]] = []
        self.delivered_packets: List[NeuralPacket] = []
        self.dropped_count = 0
        self.transmitted_count = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.queue.clear()
        self.delivered_packets.clear()
        self.dropped_count = 0
        self.transmitted_count = 0

    def send(self, packet: NeuralPacket, current_step: int) -> bool:
        """Enqueue packet subject to loss and simulated latency."""
        self.transmitted_count += 1
        if self.packet_loss > 0.0 and self.rng.uniform(0.0, 1.0) < self.packet_loss:
            self.dropped_count += 1
            return False  # Dropped

        latency = 0
        if self.max_latency > 0:
            latency = int(self.rng.randint(0, self.max_latency + 1))
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
        base_activation: np.ndarray,
        decoder_weights: np.ndarray,  # shape (D, target_size)
        basis: Optional[np.ndarray] = None,  # shape (K, D) for spectral modes
        receiver_phase: Optional[float] = None,
        gamma: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decodes received packets and applies bounded zero-gain residual injection.
        
        Formula:
            h_new = h_base + gamma * clip(delta_comm, -max_aux, max_aux)
        
        Returns:
            (h_new, raw_delta_comm)
        """
        target_size = len(base_activation)
        if len(packets) == 0 or gamma == 0.0:
            return base_activation.copy(), np.zeros(target_size, dtype=np.float32)

        accumulated_delta = np.zeros(target_size, dtype=np.float32)

        for pkt in packets:
            payload = np.array(pkt.payload, dtype=np.float32)

            # 1. Quantization if applicable
            if pkt.precision_bits < 32:
                payload = quantize_numpy(payload, bits=pkt.precision_bits)

            # 2. Add channel noise if configured
            if self.noise_std > 0.0:
                payload = payload + self.rng.normal(0.0, self.noise_std, size=payload.shape)

            # 3. Spectral synthesis if basis provided and protocol is spectral/hybrid
            if "spectral" in pkt.protocol_id and basis is not None:
                decoded_vector = spectral_synthesize_numpy(payload, basis)
            else:
                decoded_vector = payload

            # Ensure dimensional match with decoder input
            if len(decoded_vector) != decoder_weights.shape[0]:
                if len(decoded_vector) < decoder_weights.shape[0]:
                    padded = np.zeros(decoder_weights.shape[0], dtype=np.float32)
                    padded[:len(decoded_vector)] = decoded_vector
                    decoded_vector = padded
                else:
                    decoded_vector = decoded_vector[:decoder_weights.shape[0]]

            # 4. Project through decoder weights to target population
            pop_current = np.dot(decoded_vector, decoder_weights)

            # 5. Phase gating if phase metadata is present
            if "phase" in pkt.protocol_id and pkt.phase_metadata and receiver_phase is not None:
                src_phase = pkt.phase_metadata[0]
                gain = phase_coherence_gain_numpy(src_phase, receiver_phase)
                pop_current = pop_current * gain

            accumulated_delta += pop_current

        # 6. Bounded clipping and zero-gain residual injection
        clipped_delta = np.clip(accumulated_delta, -self.max_aux_current, self.max_aux_current)
        h_new = base_activation + gamma * clipped_delta
        return h_new, accumulated_delta
