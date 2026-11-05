"""Tests for CHIMERA experiment execution and verification."""
import pytest
from poseidon.chimera_experiment import run_chimera_experiment, verify_chimera_receipt
from poseidon.core import CoreRuntime, active_core_path


def test_chimera_experiment_execution_and_verification():
    core = CoreRuntime(active_core_path())
    receipt = run_chimera_experiment(
        core=core,
        seeds=[202000001, 202000002],
        max_steps=16,
        scarcity=2.5,
    )

    assert receipt["schema"] == "poseidon-chimera-experiment-v1"
    assert "experiment_id" in receipt
    assert "receipt_sha256" in receipt
    assert len(receipt["episodes"]) == 8  # 2 seeds * 4 arms
    assert "chimera" in receipt["arm_stats"]
    assert "chimera_telemetry" in receipt

    verified = verify_chimera_receipt(receipt)
    assert verified["verified"] is True
    assert verified["receipt_sha256"] == receipt["receipt_sha256"]

    # Check tampering is caught
    tampered = dict(receipt)
    tampered["arm_stats"] = dict(receipt["arm_stats"])
    tampered["arm_stats"]["chimera"] = dict(receipt["arm_stats"]["chimera"])
    tampered["arm_stats"]["chimera"]["mean_reward"] = 999.0
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_chimera_receipt(tampered)
