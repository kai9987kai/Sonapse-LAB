"""Unit tests for Tripartite Uncertainty Decomposition and Calibration."""
import pytest
import numpy as np
import torch
from synapse.uncertainty_calibration import (
    AleatoricUncertaintyHead,
    LatentDynamicsEnsemble,
    OutDistributionDetector,
    TripartiteUncertaintyDecomposer,
    TripartiteUncertaintyEstimate
)


def test_aleatoric_head_and_gaussian_nll():
    head = AleatoricUncertaintyHead(in_dim=16, out_dim=1, hidden_dim=32)
    x = torch.randn(8, 16)
    mu, log_var = head(x)
    assert mu.shape == (8, 1)
    assert log_var.shape == (8, 1)

    target = torch.randn(8, 1)
    loss = AleatoricUncertaintyHead.gaussian_nll_loss(mu, log_var, target)
    assert torch.isfinite(loss)
    assert loss.ndim == 0

    # Ensure gradients flow back
    loss.backward()
    for p in head.parameters():
        assert p.grad is not None


def test_latent_dynamics_ensemble_disagreement():
    ensemble = LatentDynamicsEnsemble(state_dim=16, action_dim=4, ensemble_size=3, seed=42)
    z = torch.randn(5, 16)
    a = torch.randn(5, 4)

    all_z_next, all_r_mu, all_r_log_var = ensemble.forward_all(z, a)
    assert all_z_next.shape == (3, 5, 16)
    assert all_r_mu.shape == (3, 5, 1)
    assert all_r_log_var.shape == (3, 5, 1)

    mean_r, ep_var, al_var = ensemble.estimate_uncertainty(z, a)
    assert mean_r.shape == (5,)
    assert ep_var.shape == (5,)
    assert al_var.shape == (5,)
    assert (ep_var >= 0.0).all()
    assert (al_var >= 0.0).all()


def test_out_of_distribution_detector():
    detector = OutDistributionDetector(latent_dim=8, seed=42)
    rng = np.random.default_rng(42)

    # In-distribution: Gaussian centered at 0 with small std
    in_dist = rng.normal(loc=0.0, scale=0.5, size=(100, 8)).astype(np.float32)
    detector.update_support(in_dist)

    test_in = rng.normal(loc=0.0, scale=0.5, size=(8,)).astype(np.float32)
    score_in = detector.compute_ood_score(test_in)

    # OOD: shifted significantly
    test_out = rng.normal(loc=10.0, scale=0.5, size=(8,)).astype(np.float32)
    score_out = detector.compute_ood_score(test_out)

    assert score_out > score_in
    assert score_out > 5.0


def test_tripartite_uncertainty_decomposer_flow():
    decomposer = TripartiteUncertaintyDecomposer(
        state_dim=16,
        action_dim=4,
        ensemble_size=3,
        seed=42
    )

    rng = np.random.default_rng(42)
    z_batch = rng.normal(0.0, 1.0, size=(20, 16)).astype(np.float32)
    act_batch = rng.uniform(-1.0, 1.0, size=(20, 4)).astype(np.float32)
    rew_batch = rng.normal(0.0, 0.5, size=(20,)).astype(np.float32)
    next_z_batch = z_batch + 0.1 * rng.normal(0.0, 1.0, size=(20, 16)).astype(np.float32)

    # Train batch
    metrics = decomposer.train_batch(z_batch, act_batch, rew_batch, next_z_batch)
    assert 'loss_total' in metrics
    assert 'loss_nll_mean' in metrics
    assert 'loss_trans_mean' in metrics

    # Single evaluation
    z_eval = rng.normal(0.0, 1.0, size=(16,)).astype(np.float32)
    a_eval = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)

    est = decomposer.evaluate_uncertainty(z_eval, a_eval)
    assert isinstance(est, TripartiteUncertaintyEstimate)
    assert est.epistemic >= 0.0
    assert est.aleatoric >= 0.0
    assert est.ood_score >= 0.0
    assert est.composite_penalty >= 0.0
    assert 0.0 <= est.confidence <= 1.0

    d = est.to_dict()
    assert 'epistemic' in d
    assert 'confidence' in d


def test_calibration_and_risk_coverage():
    decomposer = TripartiteUncertaintyDecomposer(state_dim=8, action_dim=2, ensemble_size=3, seed=42)
    rng = np.random.default_rng(42)

    N = 30
    z_val = rng.normal(0.0, 1.0, size=(N, 8)).astype(np.float32)
    act_val = rng.uniform(-1.0, 1.0, size=(N, 2)).astype(np.float32)
    rew_val = rng.normal(0.0, 1.0, size=(N,)).astype(np.float32)

    res = decomposer.calibrate_on_held_out(z_val, act_val, rew_val)
    assert 'ece' in res
    assert 'risk_coverage_curve' in res
    assert len(res['risk_coverage_curve']) == 5
    assert 'calibration_scale' in res
    assert res['ece'] >= 0.0
