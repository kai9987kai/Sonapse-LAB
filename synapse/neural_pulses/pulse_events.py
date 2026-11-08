"""Propagating Neural Pulse Events and Trigger Field.

Defines schemas and triggers for global/regional neuromodulatory pulse events:
- Acute threat / emergency alarm
- Surprisal / massive prediction error
- Salient reward arrival
- Sleep / arousal state transitions
- Social orienting pulses
"""
from __future__ import annotations

import dataclasses
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


class PulseEventType(str, Enum):
    ACUTE_THREAT = "acute_threat"
    SURPRISAL = "surprisal"
    REWARD_SALIENT = "reward_salient"
    SLEEP_TRANSITION = "sleep_transition"
    SOCIAL_ORIENTING = "social_orienting"
    RESET_ATTENTION = "reset_attention"


@dataclasses.dataclass
class PulseEvent:
    event_id: str
    event_type: PulseEventType
    source_region: int
    origin_time: float = dataclasses.field(default_factory=time.time)
    amplitude: float = 1.0
    frequency_hz: float = 12.0  # e.g., alpha / beta oscillation frequency
    velocity: float = 2.5       # propagation speed across graph distance
    damping: float = 0.15       # spatial/temporal attenuation
    target_regions: Optional[List[int]] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    receipt: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": str(self.event_type.value if isinstance(self.event_type, PulseEventType) else self.event_type),
            "source_region": self.source_region,
            "origin_time": self.origin_time,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "velocity": self.velocity,
            "damping": self.damping,
            "target_regions": self.target_regions,
            "metadata": self.metadata,
            "receipt": self.receipt,
        }


class NeuromodulatoryPulseField:
    """Manages active propagating wave pulses and triggers."""

    def __init__(self, num_regions: int = 8, max_active_pulses: int = 16):
        self.num_regions = num_regions
        self.max_active_pulses = max_active_pulses
        self.active_pulses: List[PulseEvent] = []
        self._counter = 0

    def trigger_pulse(
        self,
        event_type: PulseEventType,
        source_region: int = 0,
        amplitude: float = 1.0,
        frequency_hz: float = 12.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PulseEvent:
        """Instantiates and registers a new wave pulse."""
        self._counter += 1
        source_region = int(np.clip(source_region, 0, self.num_regions - 1))
        amp = float(np.clip(amplitude, 0.05, 5.0))

        receipt = EvidenceReceipt(
            item_id=f"pulse_event_{self._counter}",
            kind=EvidenceKind.MODELLED_BIOLOGICAL,
            source=f"NeuromodulatoryPulseField:{event_type.value}",
            metrics={
                "source_region": source_region,
                "amplitude": amp,
                "frequency_hz": frequency_hz,
            },
        )

        pulse = PulseEvent(
            event_id=f"pulse_{int(time.time()*1000)}_{self._counter}",
            event_type=event_type,
            source_region=source_region,
            amplitude=amp,
            frequency_hz=frequency_hz,
            metadata=metadata or {},
            receipt=receipt.to_dict(),
        )

        self.active_pulses.append(pulse)
        if len(self.active_pulses) > self.max_active_pulses:
            self.active_pulses.pop(0)  # Drop oldest

        return pulse

    def check_triggers(
        self,
        prediction_error: float = 0.0,
        acute_threat_level: float = 0.0,
        reward_delta: float = 0.0,
        is_sleep_transition: bool = False,
    ) -> List[PulseEvent]:
        """Automatically checks biological trigger conditions and fires pulses if thresholds are met."""
        fired = []
        # 1. Threat trigger: sudden danger
        if acute_threat_level > 0.65:
            pulse = self.trigger_pulse(
                PulseEventType.ACUTE_THREAT,
                source_region=1,  # Threat region
                amplitude=acute_threat_level * 1.5,
                frequency_hz=25.0,  # High-frequency gamma alert
                metadata={"threat": acute_threat_level},
            )
            fired.append(pulse)

        # 2. Surprisal trigger: large prediction error
        if prediction_error > 0.70:
            pulse = self.trigger_pulse(
                PulseEventType.SURPRISAL,
                source_region=0,  # Sensory salience
                amplitude=min(prediction_error * 1.2, 3.0),
                frequency_hz=16.0,  # Beta orienting
                metadata={"error": prediction_error},
            )
            fired.append(pulse)

        # 3. Reward trigger: strong positive RPE
        if reward_delta > 0.50:
            pulse = self.trigger_pulse(
                PulseEventType.REWARD_SALIENT,
                source_region=2,  # Reward valuation
                amplitude=reward_delta * 1.3,
                frequency_hz=10.0,  # Alpha/theta burst
                metadata={"reward": reward_delta},
            )
            fired.append(pulse)

        # 4. Sleep state transition
        if is_sleep_transition:
            pulse = self.trigger_pulse(
                PulseEventType.SLEEP_TRANSITION,
                source_region=7,  # Rest / consolidation
                amplitude=1.0,
                frequency_hz=1.5,  # Slow wave delta oscillation
                metadata={"mode": "slow_wave_init"},
            )
            fired.append(pulse)

        return fired

    def get_active_pulses(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.active_pulses]

    def clear(self) -> None:
        self.active_pulses.clear()
