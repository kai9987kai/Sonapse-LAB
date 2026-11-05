import pytest
import torch

from poseidon.neuro import NeuroConfig, NeuromodulatedRouter, SparseConnectomeMask


def test_sparse_connectome_mask_generation():
    dim = 32
    sparsity = 0.75
    mask = SparseConnectomeMask.generate_mask(dim=dim, sparsity=sparsity)
    assert mask.shape == (dim, dim)
    zero_ratio = float((mask == 0.0).sum() / (dim * dim))
    # Sparsity should be approximately 75% (+-15% after rewiring)
    assert 0.60 <= zero_ratio <= 0.90


def test_neuromodulated_router_forward():
    cfg = NeuroConfig(dim=32, sparsity=0.7)
    router = NeuromodulatedRouter(cfg)

    x = torch.randn(4, 32)
    # vitals: health, energy, hydration, threat
    vitals = torch.tensor([
        [1.0, 0.9, 0.8, 0.1],  # Calm, healthy
        [0.4, 0.1, 0.1, 0.9],  # Stressed, depleted
        [0.8, 0.5, 0.5, 0.3],
        [0.2, 0.2, 0.2, 0.8],
    ], dtype=torch.float32)

    out = router(x, vitals)
    assert out["output"].shape == (4, 32)
    assert 0.5 <= out["gain"] <= 2.0
    assert 0.0 <= out["active_sparsity"] <= 1.0
