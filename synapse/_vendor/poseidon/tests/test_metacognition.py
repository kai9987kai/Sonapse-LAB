import pytest

from poseidon.metacognition import (
    MetacognitionConfig,
    MetacognitiveMonitor,
    MetacognitiveState,
)
from poseidon.world import OBS_SIZE


def test_metacognitive_volatility_and_alert():
    monitor = MetacognitiveMonitor(MetacognitionConfig(window_size=5))

    # Feed consistent low errors -> low volatility
    for _ in range(4):
        monitor.register_transition(
            predicted_obs=[0.5] * OBS_SIZE,
            realized_obs=[0.51] * OBS_SIZE,
            predicted_uncertainty=0.04,
        )

    obs = [0.8, 0.7, 0.7, 0.8] + [0.0] * (OBS_SIZE - 4)
    state = monitor.evaluate_state(obs)
    assert state.alert_level == "nominal"
    assert not state.epistemic_caution_active

    # Sudden volatile jump in errors
    monitor.register_transition(
        predicted_obs=[0.1] * OBS_SIZE,
        realized_obs=[0.9] * OBS_SIZE,
        predicted_uncertainty=0.05,
    )
    state_volatile = monitor.evaluate_state(obs)
    assert state_volatile.epistemic_volatility > state.epistemic_volatility


def test_metacognitive_thermodynamic_horizon_allocation():
    monitor = MetacognitiveMonitor()

    # High energy & stamina -> deep horizon
    high_energy_obs = [1.0, 0.9, 0.9, 0.8] + [0.0] * (OBS_SIZE - 4)
    state_high = monitor.evaluate_state(high_energy_obs)
    assert state_high.recommended_horizon == 16

    # Starving/exhausted -> shallow emergency horizon to conserve deliberation energy
    exhausted_obs = [0.4, 0.1, 0.1, 0.05] + [0.0] * (OBS_SIZE - 4)
    state_low = monitor.evaluate_state(exhausted_obs)
    assert state_low.recommended_horizon == 4
