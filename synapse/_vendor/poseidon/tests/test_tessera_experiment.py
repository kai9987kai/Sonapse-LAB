import pytest

from poseidon.tessera_experiment import run_tessera_experiment, verify_tessera_receipt


def test_tessera_experiment_and_receipt_verification():
    receipt = run_tessera_experiment(seeds=[155000001, 155000002, 155000003, 155000004])
    
    assert receipt["schema"] == "poseidon-tessera-experiment-v1"
    # Fluke [2, 2, 4] was only proposed by seed 1, so quarantine count must be 1
    assert receipt["quarantined_candidates_count"] == 1
    # [0, 5] and [1, 4, 5] were independently ratified by multiple seeds
    assert receipt["ratified_opcodes_count"] == 2
    
    # Replay verify receipt digest
    verification = verify_tessera_receipt(receipt)
    assert verification["verified"]
    assert verification["ratified_count"] == 2
    assert verification["quarantined_count"] == 1
