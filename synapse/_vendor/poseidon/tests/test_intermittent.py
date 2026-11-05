"""Tests for Intermittent Passive Energy Harvesting & NTAG Resuscitation."""
import pytest
from poseidon.intermittent import NtagHarvestEngine


def test_intermittent_initialization_and_reset():
    engine = NtagHarvestEngine(capacitor_max=1.0)
    assert engine.virtual_capacitor == 0.15
    assert engine.total_checkpoints == 0
    assert engine.total_resuscitations == 0

    engine.reset()
    assert engine.virtual_capacitor == 0.15


def test_intermittent_ambient_harvesting():
    engine = NtagHarvestEngine(capacitor_max=1.0, harvest_efficiency=0.05)
    obs = [0.8, 0.8, 1.0, 1.0] + [0.0] * 12

    # Resting in rich resource area should harvest energy
    init_cap = engine.virtual_capacitor
    harvested = engine.update_harvest(obs, action=3)
    assert harvested > 0.0
    assert engine.virtual_capacitor > init_cap
    assert engine.total_harvested_energy > 0.0


def test_intermittent_ntag_packing_and_resuscitation():
    engine = NtagHarvestEngine(capacitor_max=1.0, resuscitation_threshold=0.10, resuscitation_boost=0.20)
    engine.virtual_capacitor = 0.50  # Charged capacitor

    phases = [0.1, 0.5, 1.2, 2.0, 3.1, 4.5]
    lattice_pos = [1.2, -3.4, 5.6]

    # Test binary NTAG packing
    frame = engine.pack_ntag_frame(step=42, phases=phases, lattice_pos=lattice_pos, health=0.9, stamina=0.8)
    assert len(frame) == 38  # 30-byte payload + 8-byte checksum = 38 bytes
    assert frame.startswith(b"NTAG")

    # Critical energy brownout: stamina = 0.05
    resuscitated, boost = engine.evaluate_resuscitation(stamina=0.05, health=0.8)
    assert resuscitated
    assert boost > 0.0
    assert engine.total_resuscitations == 1
    assert engine.virtual_capacitor < 0.50
