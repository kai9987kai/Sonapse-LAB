import pytest
from dataclasses import dataclass
import torch
from poseidon.causal_scm import StructuralCausalModel
from poseidon.morphogenesis import MorphogeneticAdapter
from poseidon.morpheus_replay import MorpheusSleepEngine


@dataclass
class MockRecord:
    step: int
    observation: list[float]
    action: int
    reward: float
    next_observation: list[float]


def test_morpheus_sleep_engine_replay_and_downscale():
    engine = MorpheusSleepEngine(dream_depth=2, seed=42)
    scm = StructuralCausalModel()
    adapter = MorphogeneticAdapter(in_dim=16, out_dim=6, initial_hidden=32)

    # Put small non-zero weights in adapter
    adapter.w2.data.normal_(0.0, 0.1)
    norm_before = float(torch.norm(adapter.w2.data).item())

    # Create dummy transition where agent rested instead of foraging
    records = [
        MockRecord(
            step=1,
            observation=[1.0, 0.3, 0.8, 0.8, 0.0, 0.0, 0.9, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
            action=0,  # rest
            reward=0.01,
            next_observation=[0.98, 0.28, 0.78, 0.95, 0.0, 0.0, 0.9, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0],
        )
    ]

    report = engine.execute_sleep_cycle(
        episodic_records=records,
        causeway_scm=scm,
        morph_adapter=adapter,
    )

    assert report.sleep_cycles == 1
    assert report.sws_compression_events == 1
    # Check synaptic downscaling was executed
    assert report.synaptic_downscale_norm <= norm_before
