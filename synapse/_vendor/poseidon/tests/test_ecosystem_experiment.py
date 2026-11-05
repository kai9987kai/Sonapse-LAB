"""Tests for Ecosystem Co-Evolution Benchmark Experiment."""
import pytest
from poseidon.ecosystem_experiment import (
    run_ecosystem_experiment,
    verify_ecosystem_receipt,
)


def test_ecosystem_experiment_run_and_verify():
    receipt = run_ecosystem_experiment(seeds=[101, 102], steps_per_episode=15)

    assert receipt["schema"] == "poseidon-ecosystem-experiment-v1"
    assert receipt["total_seeds"] == 2
    assert receipt["total_steps_executed"] == 30
    assert len(receipt["episodes"]) == 2
    assert "total_pulses_exchanged" in receipt["aggregate_metrics"]
    assert len(receipt["receipt_sha256"]) == 64

    # Verification passes
    assert verify_ecosystem_receipt(receipt) is True

    # Tampered receipt fails
    tampered = dict(receipt)
    tampered["aggregate_metrics"] = dict(receipt["aggregate_metrics"])
    tampered["aggregate_metrics"]["total_pulses_exchanged"] += 999
    assert verify_ecosystem_receipt(tampered) is False
