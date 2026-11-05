"""Unit tests for Odysseus paired experiment and weightless replay."""
import copy
import pytest
import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.odysseus import fit_odysseus
from poseidon.odysseus_experiment import (
    ARMS,
    run_odysseus_experiment,
    verify_odysseus_receipt,
)


def test_micro_odysseus_experiment_and_replay():
    torch.set_num_threads(2)
    core = CoreRuntime(active_core_path())

    atlas, _ = fit_odysseus(
        core,
        train_seeds=[131000001, 131000002],
        calibration_seeds=[132000001, 132000002],
        max_steps=16,
    )

    # Run micro experiment on 1 seed with max_steps=16
    receipt = run_odysseus_experiment(
        core,
        atlas,
        seeds=[133000001],
        max_steps=16,
        scarcity=2.5,
    )

    assert receipt["schema"] == "poseidon-odysseus-experiment-v1"
    assert receipt["status"] == "completed"
    assert len(receipt["episodes"]) == len(ARMS) * 1
    assert "summary" in receipt
    assert "paired" in receipt

    # Weightless replay verification
    ver = verify_odysseus_receipt(receipt)
    assert ver["verified"] is True
    assert ver["episodes_replayed"] == len(ARMS) * 1

    # Tampered receipt is rejected
    tampered = copy.deepcopy(receipt)
    tampered["episodes"][0]["total_reward"] += 1.0
    with pytest.raises(ValueError, match="Receipt checksum mismatch"):
        verify_odysseus_receipt(tampered)
