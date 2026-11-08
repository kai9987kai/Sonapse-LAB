"""Structured Knowledge Record Schema.

Typed, cryptographically authenticated knowledge units for cross-model exchange,
memory consolidation, and vector-symbolic reasoning.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from synapse.neuroendocrine.evidence_types import (
    EvidenceKind,
    EvidenceReceipt,
    compute_tensor_digest,
)


class KnowledgeContentType(str, Enum):
    TACTICAL_RULE = "tactical_rule"
    SPATIAL_LANDMARK = "spatial_landmark"
    CAUSAL_RELATION = "causal_relation"
    AFFECTIVE_ASSOCIATION = "affective_association"
    DONOR_PRIMITIVE = "donor_primitive"
    SEMANTIC_CONCEPT = "semantic_concept"


@dataclasses.dataclass
class KnowledgeRecord:
    record_id: str
    content_type: KnowledgeContentType
    source_id: str
    description: str
    key_vector: List[float]
    value_vector: List[float]
    confidence: float = 1.0
    creation_time: float = dataclasses.field(default_factory=time.time)
    last_accessed_time: float = dataclasses.field(default_factory=time.time)
    access_count: int = 0
    evidence_kind: EvidenceKind = EvidenceKind.TRAINED_COMPUTATIONAL
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    checksum_sha256: str = ""

    def __post_init__(self):
        if not self.checksum_sha256:
            self.checksum_sha256 = self.compute_checksum()

    def compute_checksum(self) -> str:
        payload = {
            "record_id": self.record_id,
            "content_type": str(self.content_type.value if isinstance(self.content_type, KnowledgeContentType) else self.content_type),
            "source_id": self.source_id,
            "description": self.description,
            "key_vector": [round(float(x), 5) for x in self.key_vector],
            "value_vector": [round(float(x), 5) for x in self.value_vector],
            "confidence": round(float(self.confidence), 4),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "content_type": str(self.content_type.value if isinstance(self.content_type, KnowledgeContentType) else self.content_type),
            "source_id": self.source_id,
            "description": self.description,
            "key_vector": self.key_vector,
            "value_vector": self.value_vector,
            "confidence": self.confidence,
            "creation_time": self.creation_time,
            "last_accessed_time": self.last_accessed_time,
            "access_count": self.access_count,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
            "metadata": self.metadata,
            "checksum_sha256": self.checksum_sha256,
        }
