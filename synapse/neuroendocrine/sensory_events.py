"""Sensory Event Fabric for Multimodal Biological and Environmental Ingress.

Provides standardized event representations for sensory modalities:
- vision: visual field luminance, contrast, motion vectors
- sound: acoustic pressure, frequency spectra, auditory events
- touch: mechanical contact, shear force, surface texture
- smell: chemical odorant concentrations, volatile plumes
- taste: chemical nutrient/toxin indicators (sugar, salt, bitter, umami, acid)
- body_interoception: energy reserves, thermal state, damage/nociception, fatigue
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from .evidence_types import EvidenceKind, EvidenceReceipt


@dataclasses.dataclass
class SensoryEvent:
    """Multimodal sensory event with strict provenance and timing."""
    modality: str  # 'vision', 'sound', 'touch', 'smell', 'taste', 'body_interoception'
    timestamp: float
    data: np.ndarray
    channel_names: List[str]
    evidence_kind: EvidenceKind = EvidenceKind.SYNTHETIC_SIMULATION
    provenance_id: str = "environment_sensor"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "timestamp": self.timestamp,
            "data_shape": list(self.data.shape),
            "data_mean": float(np.mean(self.data)),
            "data_l2_norm": float(np.linalg.norm(self.data)),
            "channel_names": self.channel_names,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
            "provenance_id": self.provenance_id,
            "confidence": self.confidence,
        }


class SensoryEventFabric:
    """Buffers, validates, and routes incoming multimodal sensory events."""

    def __init__(self, max_buffer_len: int = 128, buffer_capacity: Optional[int] = None):
        self.max_buffer_len = buffer_capacity if buffer_capacity is not None else max_buffer_len
        self.buffers: Dict[str, List[SensoryEvent]] = {
            "vision": [],
            "sound": [],
            "touch": [],
            "smell": [],
            "taste": [],
            "body_interoception": [],
        }
        self.event_counts: Dict[str, int] = {m: 0 for m in self.buffers}

    def emit_event(self, event: SensoryEvent) -> None:
        if event.modality not in self.buffers:
            self.buffers[event.modality] = []
            self.event_counts[event.modality] = 0

        buf = self.buffers[event.modality]
        buf.append(event)
        if len(buf) > self.max_buffer_len:
            buf.pop(0)
        self.event_counts[event.modality] += 1

    def ingest_event(
        self,
        modality: str,
        feature_vector: Any,
        intensity: float = 1.0,
        evidence_kind: EvidenceKind = EvidenceKind.SYNTHETIC_SIMULATION,
        channel_names: Optional[List[str]] = None,
    ) -> SensoryEvent:
        arr = np.asarray(feature_vector, dtype=np.float32)
        evt = SensoryEvent(
            modality=modality,
            timestamp=time.time(),
            data=arr * float(intensity),
            channel_names=channel_names or [f"{modality}_{i}" for i in range(len(arr))],
            evidence_kind=evidence_kind,
        )
        self.emit_event(evt)
        return evt

    def buffer_size(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def get_latest(self, modality: str) -> Optional[SensoryEvent]:
        buf = self.buffers.get(modality, [])
        return buf[-1] if buf else None

    def export_feature_vector(self, target_dim: int = 64) -> np.ndarray:
        """
        Assemble a normalized multimodal representation across all active modalities.
        Zero-padded or projected to target_dim.
        """
        modalities = ["vision", "sound", "touch", "smell", "taste", "body_interoception"]
        pieces = []
        for m in modalities:
            evt = self.get_latest(m)
            if evt is not None:
                flat = evt.data.flatten()
                norm = np.linalg.norm(flat)
                if norm > 1e-6:
                    pieces.append(flat / norm)
                else:
                    pieces.append(flat)
            else:
                pieces.append(np.zeros(8, dtype=np.float32))

        combined = np.concatenate(pieces).astype(np.float32)
        if len(combined) < target_dim:
            combined = np.pad(combined, (0, target_dim - len(combined)))
        elif len(combined) > target_dim:
            combined = combined[:target_dim]
        return combined

    def assemble_multimodal_vector(self, target_dim: int = 64) -> torch.Tensor:
        """Helper returning torch.Tensor."""
        arr = self.export_feature_vector(target_dim)
        return torch.from_numpy(arr)

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "buffered_modalities": list(self.buffers.keys()),
            "event_counts": dict(self.event_counts),
            "buffer_depths": {m: len(b) for m, b in self.buffers.items()},
        }
