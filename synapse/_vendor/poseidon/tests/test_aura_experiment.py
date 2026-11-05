import pytest

from poseidon.aura_experiment import run_aura_experiment, verify_aura_receipt


class DummyCore:
    def act(self, obs):
        return 1


def test_aura_experiment_micro_run_and_replay_verification():
    core = DummyCore()
    # Micro run: 2 seeds, 16 steps
    receipt = run_aura_experiment(core=core, seeds=[144000001, 144000002], max_steps=16, scarcity=2.0)
    
    assert receipt["schema"] == "poseidon-aura-experiment-v1"
    assert len(receipt["episodes"]) == 10  # 2 seeds * 5 arms
    assert "summary" in receipt
    assert "aura" in receipt["summary"]
    assert "policy" in receipt["summary"]
    
    # Verify receipt digest
    verification = verify_aura_receipt(receipt)
    assert verification["verified"]
    assert verification["episodes_verified"] == 10
    assert verification["arms_verified"] == 5
