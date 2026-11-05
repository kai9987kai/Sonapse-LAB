"""Tests for Poseidon-X Benchmark Experiment and Verification."""
import pytest
from poseidon.poseidon_x_experiment import (
    run_poseidon_x_experiment,
    verify_poseidon_x_receipt,
)


def test_poseidon_x_experiment_run_and_verify():
    receipt = run_poseidon_x_experiment(seeds=[201, 202], max_steps=12)

    assert receipt["schema"] == "poseidon-x-experiment-v1"
    assert receipt["total_seeds"] == 2
    assert receipt["total_steps_executed"] == 24
    assert len(receipt["episodes"]) == 2
    assert "mean_survival_rate" in receipt["aggregate_metrics"]
    assert len(receipt["receipt_sha256"]) == 64

    # Verification passes
    assert verify_poseidon_x_receipt(receipt) is True

    # Tampered receipt fails
    tampered = dict(receipt)
    tampered["aggregate_metrics"] = dict(receipt["aggregate_metrics"])
    tampered["aggregate_metrics"]["mean_survival_rate"] = 0.123
    assert verify_poseidon_x_receipt(tampered) is False
