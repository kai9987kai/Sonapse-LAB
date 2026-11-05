"""Tests for Autopoietic Self-Maintenance, PAD Affect, and Curiosity Engine."""
import pytest
from poseidon.autopoiesis import (
    AutopoieticEngine,
    PADAffectiveState,
)


def test_autopoietic_affective_update_positive_valence():
    engine = AutopoieticEngine()
    # Initial nominal state
    obs_t0 = [0.8, 0.5, 0.5, 0.5]
    engine.update(obs_t0)

    # State improves significantly (feeding/drinking)
    obs_t1 = [0.9, 0.8, 0.8, 0.7]
    audit = engine.update(obs_t1, separatrix_distance=0.9)

    assert audit.affect.valence > 0.0
    assert audit.affect.dominance > 0.0
    assert audit.negentropy_flux > 0.0
    assert len(audit.hypotheses_generated) > 0


def test_autopoietic_affective_terror_on_collapse():
    engine = AutopoieticEngine()
    engine.update([1.0, 0.9, 0.9, 0.9])

    # Sudden crisis: low vitals, low separatrix distance
    obs_crisis = [0.3, 0.1, 0.08, 0.1]
    audit = engine.update(obs_crisis, separatrix_distance=0.05)

    assert audit.affect.valence < 0.0
    assert audit.affect.dominance < 0.0
    assert audit.affect.mood_label in ("acute_terror", "morose_frustration", "catatonic_despair", "defensive_rage")


def test_autopoietic_micro_sleep_trigger():
    engine = AutopoieticEngine(entropy_threshold_sleep=0.5)
    # Expose to consecutive high-arousal steps to accumulate entropy
    sleep_triggered = False
    for step in range(15):
        obs = [0.5, 0.1, 0.1, 0.2]
        pred = [1.0, 0.9, 0.9, 0.9]  # Massive prediction error -> high arousal
        audit = engine.update(obs, predicted_obs=pred, separatrix_distance=0.2)
        if audit.micro_sleep_active:
            sleep_triggered = True
            break

    assert sleep_triggered is True
    assert engine.total_sleep_resets >= 1
