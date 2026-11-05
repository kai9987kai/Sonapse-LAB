import hashlib
import json
import pytest

from tools import run_decision_memory_pilot as pilot


def test_pilot_runs_micro_and_fails_on_existing_dir(tmp_path):
    output = tmp_path / "pilot_out"
    result = pilot.run(
        output,
        seed=720000000,
        train_seeds_count=2,
        eval_episodes_per_condition=2,
        max_steps=6,
        updates=10,
    )

    assert result["complete"] is True
    assert result["verified"] is True
    assert result["promoted"] is False
    assert (output / "protocol.json").exists()
    assert (output / "summary.json").exists()
    assert (output / "benchmark_receipt.json").exists()
    assert (output / "source.zip").exists()

    # Re-running on existing directory must fail
    with pytest.raises(FileExistsError):
        pilot.run(output, seed=720000000)
