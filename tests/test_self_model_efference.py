"""Unit tests for Action-Grounded Efference-Copy Self-Model."""
import pytest
import numpy as np
import torch
from synapse.self_model import (
    EfferenceCopyComparator,
    EfferenceForwardModel,
    ControlMode,
    AuthorshipReceipt
)


def test_forward_model_gradient_flow():
    model = EfferenceForwardModel(sensory_dim=10, action_dim=2, hidden_dim=16)
    s = torch.randn(4, 10)
    a = torch.randn(4, 2)
    delta_pred = model(s, a)
    assert delta_pred.shape == (4, 10)

    target = torch.randn(4, 10)
    loss = torch.nn.functional.mse_loss(delta_pred, target)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None


def test_comparator_active_mode_and_learning():
    comparator = EfferenceCopyComparator(sensory_dim=8, action_dim=2, hidden_dim=16, seed=42)
    s_t = np.array([0.1, 0.2, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0], dtype=np.float32)
    a_t = np.array([0.5, -0.2], dtype=np.float32)
    s_next = s_t + np.array([0.05, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    receipt = comparator.process_transition(s_t, a_t, s_next, learn=True)
    assert isinstance(receipt, AuthorshipReceipt)
    assert 0.0 <= receipt.self_authorship <= 1.0
    assert receipt.external_disturbance >= 0.0
    assert receipt.actuator_fault >= 0.0
    assert receipt.control_mode == 'active'

    summary = comparator.summary()
    assert summary['step_counter'] == 1
    assert summary['recent_samples'] == 1


def test_comparator_external_disturbance_spike():
    comparator = EfferenceCopyComparator(sensory_dim=8, action_dim=2, hidden_dim=16, seed=42)
    s_t = np.zeros(8, dtype=np.float32)
    a_t = np.array([0.1, 0.0], dtype=np.float32)

    # Train model on normal smooth motion
    for _ in range(30):
        s_next = s_t + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        comparator.process_transition(s_t, a_t, s_next, learn=True)

    # Now simulate a sudden violent external collision / shock
    shock_s_next = s_t + np.array([0.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    shock_receipt = comparator.process_transition(s_t, a_t, shock_s_next, learn=False)

    assert shock_receipt.external_disturbance > 1.0
    assert shock_receipt.self_authorship < 0.5
    assert not shock_receipt.is_self_caused


def test_matched_controls_destroy_authorship():
    # Comparing active vs shuffled vs ablated
    comp_active = EfferenceCopyComparator(sensory_dim=8, action_dim=4, hidden_dim=16, seed=42)
    comp_shuffled = EfferenceCopyComparator(sensory_dim=8, action_dim=4, hidden_dim=16, seed=42)
    comp_shuffled.set_control_mode(ControlMode.SHUFFLED_ACTION)
    comp_ablated = EfferenceCopyComparator(sensory_dim=8, action_dim=4, hidden_dim=16, seed=42)
    comp_ablated.set_control_mode(ControlMode.ABLATED)

    s_t = np.ones(8, dtype=np.float32) * 0.2
    a_t = np.array([0.8, -0.4, 0.1, 0.3], dtype=np.float32)
    # Target delta strongly correlated with action
    s_next = s_t + np.pad(a_t * 0.2, (0, 4))

    # Pretrain active
    for _ in range(50):
        comp_active.process_transition(s_t, a_t, s_next, learn=True)

    rec_active = comp_active.process_transition(s_t, a_t, s_next, learn=False)
    rec_shuffled = comp_shuffled.process_transition(s_t, a_t, s_next, learn=False)
    rec_ablated = comp_ablated.process_transition(s_t, a_t, s_next, learn=False)

    # Active should show high self-authorship, while controls show lower authorship
    assert rec_active.self_authorship > rec_ablated.self_authorship
    assert rec_active.discrepancy_norm < rec_ablated.discrepancy_norm
