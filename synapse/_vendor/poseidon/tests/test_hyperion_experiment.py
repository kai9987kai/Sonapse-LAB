"""Tests for HYPERION Paired Evaluation Benchmark and Receipt Verifier."""
from poseidon.core import CoreRuntime, active_core_path
from poseidon.hyperion_experiment import run_hyperion_experiment, verify_hyperion_receipt


def test_hyperion_experiment_execution_and_verification():
    core = CoreRuntime(active_core_path())
    receipt = run_hyperion_experiment(
        core=core,
        seeds=[303000001, 303000002],
        max_steps=16,
        scarcity=2.5,
    )

    assert "receipt_sha256" in receipt
    assert "arm_stats" in receipt
    assert "hyperion" in receipt["arm_stats"]
    assert receipt["arm_stats"]["hyperion"]["episodes"] == 2
    assert "hyperion_telemetry" in receipt

    ver = verify_hyperion_receipt(receipt)
    assert ver["verified"] is True
    assert ver["receipt_sha256"] == receipt["receipt_sha256"]
