import pytest
import numpy as np

from poseidon.molt import InstarStage, MoltEngine, PROFILES, ExuviaRecord


def test_molt_profiles_exist_and_bounded():
    assert InstarStage.LARVAL in PROFILES
    assert InstarStage.PUPA in PROFILES
    assert InstarStage.IMAGO in PROFILES

    pupa = PROFILES[InstarStage.PUPA]
    assert pupa.exposure_vulnerability < 0.5  # Hardened against exposure
    assert pupa.basal_metabolic_cost < 0.5   # Diapause reduction


def test_molt_developmental_transitions():
    engine = MoltEngine(larval_intake_threshold=0.5, pupa_duration_steps=3)
    assert engine.stage == InstarStage.LARVAL

    # Observation with high energy: [health, energy, hydration, stamina, exposure, ...]
    obs = [0.9, 0.95, 0.8, 0.8, 0.1, 0.2, 0.2, 0.0]
    
    # Step until larval threshold met
    exuvia = None
    for _ in range(30):
        exuvia = engine.update_development(obs)
        if exuvia is not None:
            break

    assert exuvia is not None
    assert exuvia.from_stage == InstarStage.LARVAL
    assert exuvia.to_stage == InstarStage.PUPA
    assert engine.stage == InstarStage.PUPA
    assert len(engine.exuviae) == 1

    # Now step 3 times through pupa duration to emerge as IMAGO
    e2 = engine.update_development(obs)
    assert e2 is None
    e3 = engine.update_development(obs)
    assert e3 is None
    e4 = engine.update_development(obs)
    assert e4 is not None
    assert e4.from_stage == InstarStage.PUPA
    assert e4.to_stage == InstarStage.IMAGO
    assert engine.stage == InstarStage.IMAGO
    assert len(engine.exuviae) == 2


def test_molt_observation_filter():
    engine = MoltEngine()
    obs = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.8, 0.8, 0.0], dtype=np.float32)
    filtered = engine.filter_observation(obs)
    assert len(filtered) == len(obs)
    assert filtered[4] > 0.0  # Modulated exposure
