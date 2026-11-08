"""Shared Structured Knowledge Pool and Vector Symbolic Subsystem.

Provides:
- Cryptographically verified structured knowledge records (KnowledgeRecord, KnowledgeContentType).
- Holographic vector-symbolic binding and unbinding (VectorSymbolicBinder).
- Continuous soft-attention neural retrieval pool (NeuralKnowledgePool).
- Zero-gain bounded knowledge graft adapter (KnowledgeGraftAdapter).
"""

from synapse.shared_knowledge.records import (
    KnowledgeRecord,
    KnowledgeContentType,
)
from synapse.shared_knowledge.vector_binding import (
    VectorSymbolicBinder,
)
from synapse.shared_knowledge.neural_retrieval import (
    NeuralKnowledgePool,
)
from synapse.shared_knowledge.knowledge_graft import (
    KnowledgeGraftAdapter,
)

__all__ = [
    "KnowledgeRecord",
    "KnowledgeContentType",
    "VectorSymbolicBinder",
    "NeuralKnowledgePool",
    "KnowledgeGraftAdapter",
]
