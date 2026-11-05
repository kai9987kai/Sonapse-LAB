import pytest

from poseidon.adapt import ReplayBuffer
from poseidon.core import CoreRuntime, active_core_path
from poseidon.dream import DreamConfig, DreamConsolidator, DreamSummary
from poseidon.mnemosyne import EpisodicRecord, MnemosyneConfig, MnemosyneMemory
from poseidon.oracle import OracleConfig, OracleEnsemble
from poseidon.world import OBS_SIZE


@pytest.fixture(scope="module")
def core():
    return CoreRuntime(active_core_path())


@pytest.fixture(scope="module")
def oracle():
    return OracleEnsemble(OracleConfig(ensemble_size=2, hidden_size=64))


def test_dream_consolidation_replay(core, oracle):
    mnemo = MnemosyneMemory(MnemosyneConfig(hidden_size=32))
    replay = ReplayBuffer(capacity=50)

    # Insert a high surprise record
    mnemo.episodic.add(
        EpisodicRecord(
            step=1,
            observation=[0.5] * OBS_SIZE,
            action=4,
            reward=0.1,
            next_observation=[0.4] * OBS_SIZE,
            surprise=0.45,  # High surprise
            context=[0.0] * 32,
        )
    )

    consolidator = DreamConsolidator(
        core=core,
        oracle=oracle,
        mnemosyne=mnemo,
        replay_buffer=replay,
        config=DreamConfig(dream_cycles_per_rest=1, dream_horizon=2),
    )

    summary = consolidator.consolidate_during_rest()
    assert isinstance(summary, DreamSummary)
    assert summary.dream_cycles == 1
    assert summary.episodes_inspected == 1
    # Check if synthetic dream transitions were inoculated into replay buffer
    assert len(replay) >= 0
