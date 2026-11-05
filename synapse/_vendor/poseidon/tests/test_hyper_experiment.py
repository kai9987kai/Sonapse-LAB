import pytest
from poseidon.core import CoreRuntime, active_core_path
from poseidon.hyper_experiment import run_hyper_experiment, HYPER_EXPERIMENT_SCHEMA


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


def test_hyper_experiment_execution(core):
    # Quick 1-episode test across arms
    result = run_hyper_experiment(seed=999, episodes=1, max_steps=8, scarcity=1.5, core=core)
    assert result["schema"] == HYPER_EXPERIMENT_SCHEMA
    assert "receipt_hash" in result
    assert len(result["receipt_hash"]) == 64
    assert "summary_pomdp" in result
    assert "hyper_v3" in result["summary_pomdp"]
    assert "summary_3d" in result
    assert "hyper_v3" in result["summary_3d"]
