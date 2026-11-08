"""Multimodal Live Inputs and Interactive Gateways Subsystem.

Provides:
- Permission-gated real-time microphone stream and feature extractor (LiveAudioGateway, AudioFeatures).
- Cochlear filterbank auditory nerve spike encoder (AuditorySpikeEncoder).
- Interactive conversational chat gateway with affect-informed synthesis (ConversationalChatGateway, ChatMessage).
"""

from synapse.multimodal_live.audio_gateway import (
    LiveAudioGateway,
    AudioFeatures,
)
from synapse.multimodal_live.auditory_encoder import (
    AuditorySpikeEncoder,
)
from synapse.multimodal_live.chat_gateway import (
    ConversationalChatGateway,
    ChatMessage,
)

__all__ = [
    "LiveAudioGateway",
    "AudioFeatures",
    "AuditorySpikeEncoder",
    "ConversationalChatGateway",
    "ChatMessage",
]
