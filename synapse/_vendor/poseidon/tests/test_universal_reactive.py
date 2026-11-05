"""Tests for UniversalModder and Reactive Actor Pipeline."""
import pytest
from poseidon.universal_reactive import (
    ModderTransaction,
    ReactiveActorSystem,
    UniversalModder,
)


class DummyTarget:
    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier

    def compute(self, val: float) -> float:
        return val * self.multiplier


def test_universal_modder_intercept_and_revert():
    target = DummyTarget(3.0)
    assert target.compute(10.0) == 30.0

    modder = UniversalModder()

    # Intercept method
    def custom_interceptor(original_fn, self_or_cls, args, kwargs):
        val = args[0]
        # Pre-multiply input by 2 before invoking original
        return original_fn(val * 2.0, **kwargs)

    modder.intercept(target, "compute", custom_interceptor)
    assert target.compute(10.0) == 60.0

    # Revert method
    assert modder.revert(target, "compute") is True
    assert target.compute(10.0) == 30.0


def test_universal_modder_transaction_rollback():
    target = DummyTarget(5.0)
    modder = UniversalModder()

    try:
        with ModderTransaction(modder) as tx:
            tx.hotpatch_value(target, "multiplier", 100.0)
            assert target.multiplier == 100.0
            # Trigger intentional exception inside transaction
            raise RuntimeError("Simulation abort")
    except RuntimeError:
        pass

    # Should have safely rolled back to 5.0
    assert target.multiplier == 5.0


def test_reactive_actor_pipeline_dispatch():
    system = ReactiveActorSystem()
    system.start_all()
    try:
        # Standard observation where energy is fine
        obs_nominal = [1.0, 0.8, 0.8, 0.9, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
        action_out = system.step_cognitive_pipeline(obs_nominal, candidate_action=4)
        assert action_out == 4

        # Critically low energy (< 0.12) -> should shield to forage (1)
        obs_starving = [0.8, 0.05, 0.8, 0.5, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
        shielded_out = system.step_cognitive_pipeline(obs_starving, candidate_action=4)
        assert shielded_out == 1
        assert system.motor_shield.intervention_count >= 1
    finally:
        system.stop_all()
