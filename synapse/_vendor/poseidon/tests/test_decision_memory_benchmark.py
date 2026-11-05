import copy
import hashlib
import json
import pytest

from poseidon.research.contracts import canonical_bytes
from poseidon.research.decision_memory_benchmark import (
    BENCHMARK_SCHEMA,
    bootstrap_ci_mean,
    run_decision_memory_benchmark,
    verify_decision_memory_receipt,
)


def test_bootstrap_ci_mean():
    diffs = [1.0, 1.2, 0.9, 1.1, 1.05]
    mean, ci_low, ci_high = bootstrap_ci_mean(diffs, reps=500, seed=123)
    assert 0.9 <= ci_low <= mean <= ci_high <= 1.2


def test_decision_memory_benchmark_micro_run():
    receipt = run_decision_memory_benchmark(
        base_seed=710000000,
        episodes=2,
        max_steps=6,
        conditions=("id_alias",),
    )

    assert receipt["schema"] == BENCHMARK_SCHEMA
    assert "sha256" in receipt
    assert "results" in receipt
    assert "id_alias" in receipt["results"]

    id_results = receipt["results"]["id_alias"]
    assert "arms" in id_results
    assert "paired_comparisons" in id_results

    # Verify receipt passes verification
    assert verify_decision_memory_receipt(receipt, replay_seeds=True) is True


def test_tampered_benchmark_receipt_fails():
    receipt = run_decision_memory_benchmark(
        base_seed=710000010,
        episodes=1,
        max_steps=4,
        conditions=("id_alias",),
    )

    tampered = copy.deepcopy(receipt)
    tampered["results"]["id_alias"]["arms"]["intact"]["mean_reward"] += 5.0

    # Tampered sha must fail
    with pytest.raises(ValueError, match="Receipt SHA-256 hash mismatch"):
        verify_decision_memory_receipt(tampered, replay_seeds=False)

    # Resealed with false data must fail replay
    tampered.pop("sha256")
    tampered["sha256"] = hashlib.sha256(canonical_bytes(tampered)).hexdigest()
    with pytest.raises(ValueError, match="Replay"):
        # Alter a trajectory reward
        tampered["results"]["id_alias"]["trajectories"]["baseline"][0]["total_reward"] += 99.0
        tampered.pop("sha256")
        tampered["sha256"] = hashlib.sha256(canonical_bytes(tampered)).hexdigest()
        verify_decision_memory_receipt(tampered, replay_seeds=True)
