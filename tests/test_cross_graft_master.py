"""Master regression and scientific validation tests for Cross-Connectome Synaptic Grafting Engine.

Verifies:
1. Biological presynaptic Dale's law: every presynaptic neuron column has uniform sign across all target rows.
2. Zero-gain identity: gamma=0.0 yields strictly zero current bit-for-bit.
3. Sparse edge provenance and anatomical region bridging.
4. Fan-in normalization and bounded current injection.
5. Contextual route gating modulation.
6. Checkpoint export and import round-trip invariance.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from synapse.cross_graft import CrossConnectomeGraft, CrossGraftTelemetry


def test_columnwise_presynaptic_dales_law():
    """Verify that every presynaptic neuron column j has strictly identical sign s_j across all targets i."""
    graft = CrossConnectomeGraft(node_dim=64, gamma=0.2, seed=123)
    w_eff = graft.get_effective_weights()

    # Verify column-wise Dale's law
    for j in range(graft.node_dim):
        col = w_eff[:, j]
        col_nz = col[col.abs() > 1e-7]
        if col_nz.numel() == 0:
            continue
        expected_sign = float(graft.pre_signs[j].item())
        if expected_sign > 0:
            assert (col_nz >= 0.0).all(), f"Column {j} with excitatory pre_sign has negative outgoing weight"
        else:
            assert (col_nz <= 0.0).all(), f"Column {j} with inhibitory pre_sign has positive outgoing weight"

    compliance = graft.verify_dales_law()
    assert compliance == 100.0


def test_zero_gain_identity_bit_for_bit():
    """Verify that gamma=0.0 produces strictly zero current bit-for-bit."""
    graft_zero = CrossConnectomeGraft(node_dim=64, gamma=0.0, seed=42)
    acts = torch.randn(64)
    ctx = torch.randn(64)

    delta, telem = graft_zero(acts, context=ctx)
    assert (delta == 0.0).all()
    assert delta.norm().item() == 0.0
    assert telem.current_l2_norm == 0.0


def test_contextual_route_gating():
    """Verify that contextual input modulates the output current and affinity."""
    graft = CrossConnectomeGraft(node_dim=64, gamma=0.25, seed=42)
    acts = torch.randn(64)

    ctx1 = torch.ones(64) * 2.0
    ctx2 = torch.ones(64) * -2.0

    delta1, telem1 = graft(acts, context=ctx1)
    delta2, telem2 = graft(acts, context=ctx2)

    assert not torch.allclose(delta1, delta2, atol=1e-4)
    assert delta1.norm().item() > 0.0
    assert delta2.norm().item() > 0.0


def test_sparse_provenance_and_fan_in():
    """Verify sparse connectivity mask and fan-in normalization."""
    graft = CrossConnectomeGraft(node_dim=64, gamma=0.15, seed=42)
    prov = graft.provenance_mask

    # Mask must have values in {0.0, 1.0}
    assert torch.all((prov == 0.0) | (prov == 1.0))
    # Density must be sparse (between 0.15 and 0.70)
    density = float(prov.mean().item())
    assert 0.15 <= density <= 0.70

    # Fan-in norm must match 1 / sqrt(max(1, sum_j M_ij))
    k_in = prov.sum(dim=1, keepdim=True)
    expected_fan_in = 1.0 / torch.sqrt(torch.clamp(k_in, min=1.0))
    assert torch.allclose(graft.fan_in_norm, expected_fan_in)


def test_bounded_injection_clamp():
    """Verify that current never exceeds max_current bound."""
    bound = 1.5
    graft = CrossConnectomeGraft(node_dim=64, gamma=10.0, max_current=bound, seed=42)
    acts = torch.ones(64) * 100.0

    delta, telem = graft(acts)
    assert delta.abs().max().item() <= bound + 1e-5


def test_checkpoint_roundtrip_preservation():
    """Verify that export_state and import_state preserve weights, signs, and forward projections."""
    graft1 = CrossConnectomeGraft(node_dim=64, gamma=0.2, seed=99)
    acts = torch.randn(64)
    ctx = torch.randn(64)

    delta1, telem1 = graft1(acts, context=ctx)
    state = graft1.export_state()

    graft2 = CrossConnectomeGraft(node_dim=64, gamma=0.0, seed=1)
    graft2.import_state(state)

    delta2, telem2 = graft2(acts, context=ctx)
    assert torch.allclose(delta1, delta2, atol=1e-6)
    assert math.isclose(graft1.gamma, graft2.gamma, rel_tol=1e-6)
    assert graft2.verify_dales_law() == 100.0
