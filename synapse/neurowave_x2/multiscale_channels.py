"""Multi-Timescale & Event-Driven Communication Channels for Sonapse NeuroWave-X2.

Maintains hierarchical communication timescales with separate queue accounting:
- FAST: Sub-step microstep signalling
- MEDIUM: Step-level inter-region coordination (cadence: communication_every)
- SLOW: Episodic / context updates triggered by environmental phase shifts
- EVENT: Surprise / prediction-error thresholded event transmissions
Integrates cleanly with Sonapse's EventDrivenBackend without duplicate queues.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .packets import NeuralPacket


class MultiTimescaleChannelManager:
    """Manages independent queues and bandwidth metrics across communication timescales."""

    def __init__(self, medium_cadence: int = 1, surprise_threshold: float = 0.35) -> None:
        self.medium_cadence = max(1, int(medium_cadence))
        self.surprise_threshold = float(surprise_threshold)

        # Separate queue buffers for each timescale
        self.fast_queue: List[NeuralPacket] = []
        self.medium_queue: List[NeuralPacket] = []
        self.slow_queue: List[NeuralPacket] = []
        self.event_queue: List[NeuralPacket] = []

        # Wire cost statistics per timescale
        self.timescale_wire_bits: Dict[str, int] = {
            "fast": 0, "medium": 0, "slow": 0, "event": 0
        }
        self.timescale_packet_counts: Dict[str, int] = {
            "fast": 0, "medium": 0, "slow": 0, "event": 0
        }

    def reset(self) -> None:
        self.fast_queue.clear()
        self.medium_queue.clear()
        self.slow_queue.clear()
        self.event_queue.clear()
        for k in self.timescale_wire_bits:
            self.timescale_wire_bits[k] = 0
            self.timescale_packet_counts[k] = 0

    def should_transmit_medium(self, step: int) -> bool:
        """Determines if medium-timescale step channel should fire."""
        return (step % self.medium_cadence) == 0

    def should_transmit_event(self, surprise_or_error: float) -> bool:
        """Determines if event-driven channel should fire based on sensory surprise/prediction error."""
        return float(surprise_or_error) >= self.surprise_threshold

    def enqueue_packet(self, packet: NeuralPacket, timescale: str = "medium") -> None:
        """Enqueues packet into the appropriate timescale buffer with separate accounting."""
        if timescale == "fast":
            self.fast_queue.append(packet)
        elif timescale == "slow":
            self.slow_queue.append(packet)
        elif timescale == "event":
            self.event_queue.append(packet)
        else:
            self.medium_queue.append(packet)

        key = timescale if timescale in self.timescale_wire_bits else "medium"
        self.timescale_wire_bits[key] += packet.total_wire_bits
        self.timescale_packet_counts[key] += 1

    def drain_active_packets(self, timescale: str = "medium") -> List[NeuralPacket]:
        """Drains and returns pending packets for the requested timescale."""
        if timescale == "fast":
            pkts = list(self.fast_queue)
            self.fast_queue.clear()
            return pkts
        elif timescale == "slow":
            pkts = list(self.slow_queue)
            self.slow_queue.clear()
            return pkts
        elif timescale == "event":
            pkts = list(self.event_queue)
            self.event_queue.clear()
            return pkts
        else:
            pkts = list(self.medium_queue)
            self.medium_queue.clear()
            return pkts

    def get_timescale_telemetry(self) -> Dict[str, Any]:
        return {
            "bits_fast": self.timescale_wire_bits["fast"],
            "bits_medium": self.timescale_wire_bits["medium"],
            "bits_slow": self.timescale_wire_bits["slow"],
            "bits_event": self.timescale_wire_bits["event"],
            "packets_fast": self.timescale_packet_counts["fast"],
            "packets_medium": self.timescale_packet_counts["medium"],
            "packets_slow": self.timescale_packet_counts["slow"],
            "packets_event": self.timescale_packet_counts["event"],
        }
