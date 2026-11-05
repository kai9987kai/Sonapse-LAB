"""Tests for TITAN Paired Benchmark & Receipt Verifier."""
from poseidon.titan_experiment import run_titan_benchmark, verify_titan_receipt


def test_titan_experiment_execution_and_verification():
    # Fast 6-arm micro benchmark: 2 episodes x 8 steps
    receipt = run_titan_benchmark(episodes=2, max_steps=8, scarcity=2.5, seed_base=99200000)

    assert receipt["schema"] == "poseidon-titan-experiment-v2"
    assert len(receipt["seeds"]) == 2
    assert "titan" in receipt["arms"]
    assert "core" in receipt["arms"]
    assert "hyperion" in receipt["arms"]

    # Verify receipt offline
    verification = verify_titan_receipt(receipt)
    assert verification["verified"] is True
    assert verification["verification_scope"] == "receipt integrity, controller replay, and TidePool transition replay"
    assert receipt["titan_telemetry"]["decision_count"] == sum(
        len(episode["trajectory"]) for episode in receipt["arms"]["titan"]["episodes_data"]
    )
