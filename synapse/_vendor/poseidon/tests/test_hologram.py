"""Tests for Holographic Reduced Representation Associative Memory."""
import numpy as np
from poseidon.hologram import HolographicAssociativeMemory


def test_hologram_binding_and_unbinding():
    mem = HolographicAssociativeMemory(dimension=256)
    r = mem.vocabulary["ROLE_FOOD"]
    s = mem.vocabulary["STATE_CRITICAL"]
    bound = mem.bind(r, s)
    recovered = mem.unbind(r, bound)
    sim = float(np.dot(recovered, s))
    # Unbinding a single clean bound pair should yield high cosine similarity (> 0.5)
    assert sim > 0.5


def test_hologram_multiple_associations_superposition():
    mem = HolographicAssociativeMemory(dimension=256, decay_rate=1.0)
    mem.remember_association("ROLE_FOOD", "STATE_CRITICAL")
    mem.remember_association("ROLE_PREDATOR", "STATE_THREATENING")

    food_match, food_sim = mem.query("ROLE_FOOD")
    threat_match, threat_sim = mem.query("ROLE_PREDATOR")

    assert food_match == "STATE_CRITICAL"
    assert threat_match == "STATE_THREATENING"
    assert food_sim > 0.25
    assert threat_sim > 0.25


def test_hologram_observation_encoding_and_telemetry():
    mem = HolographicAssociativeMemory(dimension=256)
    # Low food (0.1), low water (0.1), high threat (0.8)
    obs = [0.1, 0.1, 0.9, 0.9, 0.8, 0.0] + [0.0] * 10
    res = mem.encode_observation(obs)
    assert res["associations_added"] >= 3
    telemetry = mem.telemetry()
    assert telemetry["dimension"] == 256
    assert telemetry["items_stored"] >= 3
    assert "top_associations" in telemetry
    assert "ROLE_FOOD" in telemetry["top_associations"]
    assert "ROLE_WATER" in telemetry["top_associations"]
