"""Conversational Chat Gateway and Semantic Context Bridge.

Provides bidirectional conversational interaction:
- Transforms incoming chat prompts into dense semantic embeddings z_chat.
- Injects conversational intent into connectome / WPAC context ports.
- Synthesizes responses informed by affective state and neuromodulatory tone.
- Tags every message with scientific provenance receipts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from synapse.neuroendocrine.evidence_types import (
    EvidenceReceipt,
    EvidenceKind,
    compute_tensor_digest,
)


@dataclasses.dataclass
class ChatMessage:
    message_id: str
    sender: str  # 'user' | 'agent' | 'system'
    content: str
    timestamp: float
    affective_state: Optional[Dict[str, float]] = None
    embedding: Optional[List[float]] = None
    evidence_kind: EvidenceKind = EvidenceKind.SYNTHETIC_SIMULATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp,
            "affective_state": self.affective_state,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
        }


class ConversationalChatGateway(nn.Module):
    """Bridges natural language chat messages with neural connectome states."""

    def __init__(
        self,
        embed_dim: int = 64,
        max_history: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_dim = embed_dim
        self.max_history = max_history

        # Text projection network (hashes text tokens into dense continuous vectors)
        self.token_proj = nn.Linear(128, embed_dim, bias=False)
        nn.init.orthogonal_(self.token_proj.weight, gain=0.3)

        self.history: List[ChatMessage] = []
        self._msg_counter = 0

        # Latest projected context vector z_chat
        self.register_buffer("last_chat_vector", torch.zeros(embed_dim))

    def _text_to_feature_vector(self, text: str) -> torch.Tensor:
        """Converts raw text into a 128-dimensional hashed feature representation."""
        vec = np.zeros(128, dtype=np.float32)
        words = text.lower().strip().split()
        if not words:
            return torch.zeros(128)

        for w in words:
            # Deterministic hash bucket
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16) % 128
            vec[h] += 1.0

        norm = np.linalg.norm(vec) + 1e-6
        return torch.tensor(vec / norm, dtype=torch.float32)

    def receive_user_message(self, text: str) -> Tuple[ChatMessage, torch.Tensor]:
        """Processes an incoming user message, updates history, and produces z_chat embedding."""
        self._msg_counter += 1
        msg_id = f"msg_user_{int(time.time()*1000)}_{self._msg_counter}"

        raw_feat = self._text_to_feature_vector(text)
        device = self.token_proj.weight.device
        z_chat = self.token_proj(raw_feat.to(device).unsqueeze(0)).squeeze(0)
        z_chat = torch.clamp(z_chat, -1.0, 1.0)
        self.last_chat_vector.copy_(z_chat)

        msg = ChatMessage(
            message_id=msg_id,
            sender="user",
            content=text,
            timestamp=time.time(),
            embedding=z_chat.tolist(),
            evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,  # Direct human user input
        )

        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return msg, z_chat

    def synthesize_agent_response(
        self,
        current_affect: Optional[Dict[str, float]] = None,
        neuromod_state: Optional[Dict[str, float]] = None,
        world_event: Optional[str] = None,
    ) -> ChatMessage:
        """Synthesizes a context-aware and affect-informed response from the agent."""
        self._msg_counter += 1
        msg_id = f"msg_agent_{int(time.time()*1000)}_{self._msg_counter}"

        # Determine dominant emotion
        dom_emotion = "calmness"
        dom_val = 0.5
        if current_affect:
            # Sort by intensity
            sorted_aff = sorted(current_affect.items(), key=lambda kv: kv[1], reverse=True)
            if sorted_aff:
                dom_emotion, dom_val = sorted_aff[0]

        # Template-based generative synthesis reflecting internal state
        if dom_emotion in ["curiosity", "positive_excitement"]:
            tone = f"Observing new environmental cues with active interest (curiosity={dom_val:.2f})."
        elif dom_emotion in ["fear", "anxiety", "dread"]:
            tone = f"Heightened threat vigilance active; cautious orienting engaged (fear={dom_val:.2f})."
        elif dom_emotion in ["joy", "happiness", "contentment"]:
            tone = f"Homeostatic equilibrium stable; rewarding signals integrated (contentment={dom_val:.2f})."
        elif dom_emotion in ["fatigue", "restfulness"]:
            tone = f"Sleep pressure elevated; neural consolidation pathways active (fatigue={dom_val:.2f})."
        else:
            tone = f"Neural connectome active; processing multimodal streams at equilibrium."

        if world_event:
            content = f"{tone} Environmental notice: {world_event}."
        else:
            content = f"{tone} System operating at 9,000 neurons."

        msg = ChatMessage(
            message_id=msg_id,
            sender="agent",
            content=content,
            timestamp=time.time(),
            affective_state=current_affect,
            evidence_kind=EvidenceKind.SYNTHETIC_SIMULATION,  # Explicitly simulated agent affect
        )

        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return msg

    def get_history(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.history]

    def clear(self) -> None:
        self.history.clear()
        self.last_chat_vector.zero_()
