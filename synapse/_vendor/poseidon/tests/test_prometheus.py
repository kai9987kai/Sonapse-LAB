"""Tests for Prometheus Meta-Plasticity & Homeostatic Drive Self-Regulation."""
import pytest
from poseidon.prometheus import PrometheusPlasticityEngine


def test_prometheus_initialization_and_reset():
    engine = PrometheusPlasticityEngine(base_causeway_gain=0.5, window_size=8)
    assert engine.step_count == 0
    assert len(engine.observation_window) == 0

    obs = [0.4, 0.6, 0.9, 0.8] + [0.0] * 12
    info = engine.evaluate(obs)
    assert engine.step_count == 1
    assert "adapted_gain" in info
    assert "scarcity_entropy" in info
    assert "dominant_drive" in info

    engine.reset()
    assert engine.step_count == 0
    assert len(engine.observation_window) == 0


def test_prometheus_entropy_and_gain_adaptation():
    engine = PrometheusPlasticityEngine(base_causeway_gain=0.50, window_size=8, entropy_gain_sensitivity=0.40)

    # Low resource volatile observation stream
    for i in range(10):
        # Fluctuating resource values
        food = 0.1 if i % 2 == 0 else 0.8
        water = 0.2 if i % 3 == 0 else 0.7
        obs = [food, water, 0.5, 0.5] + [0.0] * 12
        info = engine.evaluate(obs)

    # Scarcity entropy should be positive and gain adapted above baseline
    assert info["scarcity_entropy"] >= 0.0
    assert info["adapted_gain"] >= engine.base_gain
    assert 0.70 <= info["adapted_decay"] <= 0.99


def test_prometheus_homeostatic_drives():
    engine = PrometheusPlasticityEngine()

    # Severe hunger
    obs_hunger = [0.05, 0.90, 0.90, 0.90] + [0.0] * 12
    info_h = engine.evaluate(obs_hunger)
    assert info_h["dominant_drive"] == "hunger"
    assert info_h["homeostatic_urgency"] >= 0.90

    # Severe fatigue
    obs_fatigue = [0.90, 0.90, 0.90, 0.05] + [0.0] * 12
    info_f = engine.evaluate(obs_fatigue)
    assert info_f["dominant_drive"] == "fatigue"
    assert info_f["homeostatic_urgency"] >= 0.90
