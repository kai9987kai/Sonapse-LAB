import pytest
import torch
import numpy as np

from poseidon.morphogenesis import MorphogeneticAdapter


def test_morphogenesis_identity_initialization():
    adapter = MorphogeneticAdapter(in_dim=16, out_dim=6, initial_hidden=64)
    x = torch.randn(1, 16)
    delta = adapter(x)
    assert delta.shape == (1, 6)
    # Output must be precisely zero at initialization
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)


def test_morphogenesis_neurogenesis_preserves_output():
    adapter = MorphogeneticAdapter(in_dim=16, out_dim=6, initial_hidden=32)
    # Give some non-zero values to original w2 to test preservation
    adapter.w2.data.normal_(0.0, 0.1)
    x = torch.randn(2, 16)
    out_before = adapter(x).clone()

    initial_h = adapter.current_hidden
    success = adapter.trigger_neurogenesis(growth_factor=16)
    assert success is True
    assert adapter.current_hidden == initial_h + 16

    out_after = adapter(x)
    # Outputs before and after neurogenesis must match exactly due to zero-init for new units
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_morphogenesis_pruning_and_triggers():
    adapter = MorphogeneticAdapter(in_dim=16, out_dim=6, initial_hidden=32, min_hidden=16)
    # All w2 are 0 except first 20 columns
    adapter.w2.data[:, :20] = 0.5
    adapter.w2.data[:, 20:] = 0.0

    success = adapter.trigger_pruning(prune_threshold=0.01)
    assert success is True
    assert adapter.current_hidden == 20

    # Metacognitive trigger
    res_neuro = adapter.step_metacognitive_trigger(epistemic_volatility=0.45, surprise=0.7)
    assert res_neuro == "neurogenesis"
    assert adapter.current_hidden == 36
