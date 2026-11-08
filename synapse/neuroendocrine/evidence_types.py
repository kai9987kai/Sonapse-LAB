"""Scientific Evidence Categorisation and Provenance Types.

Strictly enforces scientific taxonomy for all observations, representations, and assays:
- measured_biological: Authentic neurophysiological recordings (e.g., CRCNS, DANDI, Neuropixels)
- modelled_biological: Biologically calibrated computational models (e.g., MaleCNS, FlyWire)
- synthetic_simulation: Synthetic simulation signals (e.g., virtual environment sensory streams)
- trained_computational: Learned representations and weights from training updates
- heuristic: Engineered algorithmic priors and heuristic baselines
- exploratory_result: Non-confirmatory hypothesis-generating metrics
- confirmatory_result: Controlled hypothesis-testing results with paired statistics
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np


class EvidenceKind(str, Enum):
    MEASURED_BIOLOGICAL = "measured_biological"
    MODELLED_BIOLOGICAL = "modelled_biological"
    SYNTHETIC_SIMULATION = "synthetic_simulation"
    TRAINED_COMPUTATIONAL = "trained_computational"
    HEURISTIC = "heuristic"
    EXPLORATORY_RESULT = "exploratory_result"
    CONFIRMATORY_RESULT = "confirmatory_result"


def compute_tensor_digest(data: Any) -> str:
    """Computes a deterministic SHA-256 hex digest for numpy arrays, torch tensors, or nested dicts."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        raw = data.tobytes()
        return hashlib.sha256(raw).hexdigest()
    elif isinstance(data, (dict, list)):
        raw = json.dumps(data, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()
    else:
        raw = str(data).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()


@dataclasses.dataclass
class EvidenceReceipt:
    """Cryptographically verifiable evidence receipt for claims, states, and metrics."""
    kind: EvidenceKind
    source_identifier: str = ""
    description: str = ""
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = dataclasses.field(default_factory=time.time)
    checksum: str = ""

    # Flexible compatibility aliases
    item_id: Optional[str] = None
    source: Optional[str] = None
    sha256_digest: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.item_id and not self.description:
            self.description = self.item_id
        if self.source and not self.source_identifier:
            self.source_identifier = self.source
        if self.metadata and not self.metrics:
            self.metrics = dict(self.metadata)
        if self.sha256_digest and not self.checksum:
            self.checksum = self.sha256_digest
        if not self.checksum:
            self.checksum = self.compute_digest()

    def compute_digest(self) -> str:
        payload = {
            "kind": str(self.kind.value if isinstance(self.kind, EvidenceKind) else self.kind),
            "source_identifier": self.source_identifier or str(self.source or ""),
            "description": self.description or str(self.item_id or ""),
            "metrics": self.metrics or self.metadata or {},
            "confidence": round(float(self.confidence), 4),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": str(self.kind.value if isinstance(self.kind, EvidenceKind) else self.kind),
            "source_identifier": self.source_identifier,
            "description": self.description,
            "metrics": self.metrics,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
        }
