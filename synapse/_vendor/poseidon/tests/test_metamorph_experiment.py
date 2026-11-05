import pytest
from poseidon.metamorph_experiment import run_metamorph_experiment, verify_metamorph_receipt


class DummyCore:
    def act(self, obs):
        return 0


def test_metamorph_experiment_execution_and_verification():
    core = DummyCore()
    receipt = run_metamorph_experiment(core=core, seeds=[101, 102], max_steps=8, scarcity=1.5)

    assert "schema" in receipt
    assert "receipt_sha256" in receipt
    assert "arm_stats" in receipt
    assert "core" in receipt["arm_stats"]
    assert "aura" in receipt["arm_stats"]
    assert "metamorph" in receipt["arm_stats"]

    # Verify receipt integrity
    verification = verify_metamorph_receipt(receipt)
    assert verification["verified"] is True
    assert verification["receipt_sha256"] == receipt["receipt_sha256"]
