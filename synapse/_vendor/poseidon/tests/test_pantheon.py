"""Tests for Project PANTHEON (Stage 16): 24-Pillar Unified Cognitive Synthesis Engine."""
from __future__ import annotations

import copy
import pytest

from poseidon.pantheon import (
    PANTHEON_24_PILLARS,
    SCHEMA,
    PantheonEngine,
    ReaActorOrchestrator,
    build_ntag215_ndef_frame,
    iso14443a_crc16,
    verify_pantheon_receipt,
)


def test_pantheon_24_pillar_registry_completeness() -> None:
    assert len(PANTHEON_24_PILLARS) == 24
    ids = [p["id"] for p in PANTHEON_24_PILLARS]
    assert ids == list(range(1, 25))
    slugs = [p["slug"] for p in PANTHEON_24_PILLARS]
    assert len(set(slugs)) == 24
    # Every parent must appear earlier in topological order
    seen = set()
    for p in PANTHEON_24_PILLARS:
        for parent in p["parents"]:
            assert parent in seen, f"Parent {parent} of {p['slug']} not in topological order"
        seen.add(p["slug"])


def test_iso14443a_crc16_and_ntag215_zero_residue() -> None:
    frame = build_ntag215_ndef_frame("n_queens:92", "a" * 64, step=1)
    assert frame["chip_type"] == "NTAG215_540B_135_PAGES"
    assert frame["crc_residue_zero_verified"] is True
    assert frame["used_pages"] <= frame["total_pages"]
    # Direct test on known bytes
    sample = b"POSEIDON_NTAG215_TEST"
    crc = iso14443a_crc16(sample)
    crc_le = bytes([crc & 0xFF, (crc >> 8) & 0xFF])
    assert iso14443a_crc16(sample + crc_le) == 0


def test_rea_actor_orchestrator_lamport_ordering() -> None:
    bus = ReaActorOrchestrator(max_mailbox=8)
    m1 = bus.send("a", "b", "T1", {"x": 1}, priority=1)
    m2 = bus.send("b", "c", "T2", {"x": 2}, priority=3)
    assert m1.lamport_clock == 1
    assert m2.lamport_clock == 2
    s = bus.summary()
    assert s["total_messages_dispatched"] == 2
    assert s["backpressure_healthy"] is True
    assert s["causal_ordering_verified"] is True


def test_pantheon_synthesize_and_quibble_merkle_verification() -> None:
    engine = PantheonEngine(seed=42)
    receipt = engine.synthesize("Solve the 5-queens problem", seed=42)
    assert receipt["schema"] == SCHEMA
    assert receipt["verified"] is True
    assert receipt["pillars_count"] == 24
    assert len(receipt["pillars_executed"]) == 24
    assert receipt["primary_solution"]["domain"] == "n_queens"
    assert receipt["counterfactual_dream"]["dream_verified"] is True

    verification = verify_pantheon_receipt(receipt)
    assert verification["verified"] is True
    assert verification["pillars_verified"] == 24
    assert verification["quibble_merkle_root"] == receipt["quibble_merkle_root"]


def test_pantheon_tamper_detection() -> None:
    engine = PantheonEngine(seed=7)
    receipt = engine.synthesize("Tower of Hanoi with 3 disks", seed=7)
    tampered = copy.deepcopy(receipt)
    tampered["pillars_executed"][7]["telemetry"]["chsh_bell_parameter"] = 1.5
    with pytest.raises(ValueError, match="Merkle node hash mismatch"):
        verify_pantheon_receipt(tampered)

    tampered_root = copy.deepcopy(receipt)
    tampered_root["quibble_merkle_root"] = "0" * 64
    with pytest.raises(ValueError, match="Quibble Merkle root digest mismatch"):
        verify_pantheon_receipt(tampered_root)


def test_pantheon_math_and_open_ended_synthesis() -> None:
    engine = PantheonEngine(seed=99)
    r_math = engine.synthesize("x^2 - 5*x + 6 = 0")
    assert r_math["verified"] is True
    assert r_math["primary_solution"]["domain"] == "symbolic_algebra_calculus"
    assert verify_pantheon_receipt(r_math)["verified"] is True

    r_open = engine.synthesize("Synthesize a resilient bio-inspired underwater autonomous navigation architecture")
    assert r_open["verified"] is True
    assert r_open["primary_solution"]["domain"] == "cross_domain_scientific_synthesis"
    assert verify_pantheon_receipt(r_open)["verified"] is True


def test_pantheon_world_controller_interface() -> None:
    engine = PantheonEngine(seed=42)
    obs = [0.6, 0.7, 0.8, 0.9] + [0.5] * 12
    plan = engine.plan(obs)
    assert 0 <= plan["action"] < 6
    assert plan["pillars_active"] == 24
    engine.observe_transition(obs, plan["action"], 1.0, obs)
    st = engine.status()
    assert st["ready"] is True
    assert st["pillars_count"] == 24
    assert len(st["repositories"]) == 24


def test_pantheon_runtime_and_omni_reasoner_integration() -> None:
    from poseidon.omni_reasoner import OmniReasoner
    from poseidon.runtime import Poseidon

    reasoner = OmniReasoner(seed=42)
    assert reasoner.classify_intent("Run a 24-pillar Pantheon synthesis for quantum entanglement") == "pantheon_24_pillar_synthesis"
    res = reasoner.reason("Run a 24-pillar Pantheon synthesis for quantum entanglement amplitudes [1, 0, 0, 1]", seed=42)
    assert res.empirical_simulation["pantheon_receipt"]["verified"] is True
    assert res.empirical_simulation["pillars_verified"] == 24
    assert len(res.empirical_simulation["quibble_merkle_root"]) == 64

    rt = Poseidon(".")
    st = rt.pantheon_status()
    assert st["ready"] is True
    assert st["total_pillars"] == 24
    resp = rt.respond("Calculate Shannon-Hartley channel capacity for B = 1 MHz and SNR = 20 dB", mode="pantheon", seed=42)
    assert resp["verified"] is True
    assert resp["domain"] == "signal_channel_modulation"
    assert verify_pantheon_receipt(resp)["verified"] is True

