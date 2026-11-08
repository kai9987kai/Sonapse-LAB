"""Comprehensive Test Suite for Multimodal Live Inputs and Virtual Ecology V2.

Verifies:
1. LiveAudioGateway permission gating and audio feature extraction (RMS, mel filterbanks).
2. AuditorySpikeEncoder cochlear frequency mapping and rate-coded spike trains.
3. ConversationalChatGateway semantic text embedding and affect-informed synthesis.
4. VirtualEcologyWorld 2D/3D scent plumes diffusion, rest shelters, conspecifics, and metabolic energy.
"""
import pytest
import numpy as np
import torch

from synapse.multimodal_live import (
    LiveAudioGateway,
    AudioFeatures,
    AuditorySpikeEncoder,
    ConversationalChatGateway,
    ChatMessage,
)
from synapse.world_v2 import (
    VirtualEcologyWorld,
)


def test_live_audio_gateway_and_privacy():
    # Privacy invariant: disabled by default
    gw = LiveAudioGateway(enabled=False)
    assert gw.enabled is False

    # When disabled, ingesting samples does not store raw audio
    gw.ingest_samples(np.random.randn(1024).astype(np.float32))
    assert len(gw.buffer) == 0

    # When explicitly enabled, features are extracted
    gw.set_enabled(True)
    assert gw.enabled is True
    gw.ingest_samples(np.random.randn(2048).astype(np.float32) * 0.2)
    assert len(gw.buffer) > 0

    features = gw.process_latest_frame()
    assert features is not None
    assert features.rms_energy > 0.0
    assert len(features.mel_energies) == 16


def test_auditory_spike_encoder():
    encoder = AuditorySpikeEncoder(num_channels=16, dt=0.01)

    # Resting baseline spontaneous spikes
    spikes_base, rates_base = encoder.encode(None)
    assert len(spikes_base) == 16
    assert len(rates_base) == 16

    # Acoustic stimulation
    features = AudioFeatures(
        timestamp=0.0,
        rms_energy=0.15,
        spectral_centroid_hz=1500.0,
        zero_crossing_rate=0.05,
        mel_energies=[-1.0, -0.5, 0.2, 0.8] + [-2.0] * 12,
        is_speech_active=True,
    )
    spikes_stim, rates_stim = encoder.encode(features)
    assert np.mean(rates_stim) > np.mean(rates_base)

    feat_vec = encoder.to_feature_vector()
    assert feat_vec.shape == (32,)


def test_conversational_chat_gateway():
    gateway = ConversationalChatGateway(embed_dim=64)

    # Ingest user message
    msg, z_chat = gateway.receive_user_message("Explore the eastern ridge for resources.")
    assert msg.sender == "user"
    assert z_chat.shape == (64,)
    assert len(gateway.history) == 1

    # Synthesize affect-informed response
    current_affect = {"curiosity": 0.85, "calmness": 0.40, "fear": 0.10}
    resp = gateway.synthesize_agent_response(current_affect=current_affect)
    assert resp.sender == "agent"
    assert "curiosity" in resp.content.lower()
    assert len(gateway.history) == 2


def test_virtual_ecology_v2_scents_and_shelters():
    world = VirtualEcologyWorld(seed=42)
    assert len(world.shelters) >= 2
    assert len(world.conspecifics) >= 2

    # Scent intensity sampling
    scents = world.sample_scent_intensity(0.0, 0.0)
    assert "food_plume" in scents
    assert "predator_musk" in scents
    assert "social_pheromone" in scents
    for s_name, s_val in scents.items():
        assert 0.0 <= s_val <= 1.0

    # Shelter check
    sh_x = world.shelters[0]["x"]
    sh_y = world.shelters[0]["y"]
    in_sh, rec_mult = world.is_agent_in_shelter(sh_x, sh_y)
    assert in_sh is True
    assert rec_mult > 1.0

    # Simulation step
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, info = world.step(action)

    assert "ecology" in info
    assert "scents" in info["ecology"]
    assert "metabolic_energy" in info["ecology"]
    assert info["ecology"]["metabolic_energy"] <= 100.0
