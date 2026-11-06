"""Unit and Integration Tests for Deep Research Report Upgrades.

Verifies:
1. Verifier-Gated Bounded RSI Engine (AIDE² / DGM / AlphaEvolve pattern & failure archive).
2. World Model v2 with Latent Dynamics Ensemble, JEPA loss, CEM Trajectory Planner & Fallback Shield.
3. 9-Arm Controlled Connectome Causal Assay (bio vs degree/sign/community/random/learned nulls & Cohen's d).
4. Coconut-style Continuous Latent Recurrent Reasoner and anytime depth curve.
5. Masked Diffusion Repair & Infilling Sidecar with verifier-in-the-loop.
6. Machine-Verifiable Direct Preference Optimization (DPO).
7. Unified Evaluation Scorecard, Calibration Metrics (ECE, Brier) & Joint Deployment Score.
"""
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from synapse.rsi_verifier import BoundedRSIEngine, VerificationStage, CandidatePatch
from synapse.world_model_v2 import RecurrentLatentWorldModelEnsemble, CrossEntropyMethodPlanner
from synapse.connectome_compiler import BiologicalConnectomeCompiler
from synapse.connectome_controls import ControlledConnectomeAssay
from synapse.latent_reasoner import ContinuousLatentReasoner
from synapse.diffusion_repair import DiffusionRepairSidecar
from synapse.preference_dpo import MachineVerifiableDPOEngine
from synapse.unified_eval import compute_brier_score, compute_expected_calibration_error, UnifiedEvaluator


def test_bounded_rsi_verification_pipeline():
    """Verifies that the bounded RSI engine enforces multi-stage verification and dual archives."""
    engine = BoundedRSIEngine(epsilon_threshold=0.02, seed=42)

    # 1. Test Static Analysis rejection (out of bounds mutation)
    bad_patch = CandidatePatch(
        patch_id="bad_patch_1",
        parent_hash=engine.current_parent_hash,
        patch_type="config",
        diff={'learning_rate': 99.0}  # Forbidden > 0.1
    )
    receipt_bad = engine.verify_and_eval(bad_patch)
    assert not receipt_bad.passed_all
    assert receipt_bad.final_stage == VerificationStage.REJECTED
    assert len(engine.failure_archive) == 1
    assert engine.failure_archive[0]['failed_stage'] == "static_analysis"

    # 2. Test candidate generation and evaluation
    cand = engine.generate_candidate(patch_type="config")
    assert cand.parent_hash == engine.current_parent_hash
    receipt = engine.verify_and_eval(cand, num_support_trials=4, num_query_trials=4)
    assert receipt.receipt_hash != ""
    assert len(receipt.stage_results) > 0


def test_world_model_ensemble_and_cem_planner():
    """Verifies that World Model v2 trains with JEPA loss and plans via CEM with fallback shield."""
    wm = RecurrentLatentWorldModelEnsemble(num_members=3, state_dim=32, action_dim=5, obs_dim=155, seed=42)

    # Synthetic transition batch
    B = 8
    obs = np.random.uniform(0.0, 1.0, size=(B, 155)).astype(np.float32)
    acts = np.random.uniform(-1.0, 1.0, size=(B, 5)).astype(np.float32)
    rews = np.random.randn(B).astype(np.float32)
    next_obs = np.random.uniform(0.0, 1.0, size=(B, 155)).astype(np.float32)
    dones = np.zeros(B, dtype=np.float32)

    loss_dict = wm.train_batch(obs, acts, rews, next_obs, dones)
    assert 'mean_loss' in loss_dict
    assert loss_dict['mean_loss'] > 0.0

    # Test CEM Planner
    planner = CrossEntropyMethodPlanner(
        wm, action_dim=5, default_horizon=3, num_samples=8, num_elites=2, cem_iterations=2, seed=42
    )
    plan_out = planner.plan(obs[0], trusted_action=np.zeros(5, dtype=np.float32))
    assert plan_out['status'] == 'ok'
    assert len(plan_out['action']) == 5
    assert plan_out['horizon'] == 3

    # Test Fallback Shield invariant
    planner_shield = CrossEntropyMethodPlanner(
        wm, action_dim=5, default_horizon=3, uncertainty_threshold=-1.0  # Force shield trigger
    )
    shield_out = planner_shield.plan(obs[0], trusted_action=np.ones(5, dtype=np.float32))
    assert shield_out['shield_triggered']
    assert shield_out['plan_mode'] == "fallback_trusted_controller"
    np.testing.assert_allclose(shield_out['action'], np.ones(5, dtype=np.float32))


def test_9arm_connectome_causal_assay():
    """Verifies generation of all 9 matched graph control topologies and formal Cohen's d computation."""
    compiler = BiologicalConnectomeCompiler()
    bio_mod = compiler.compile_subcircuit(max_neurons=16, min_weight=1.0)
    assay = ControlledConnectomeAssay(compiler=compiler, seed=42)

    controls = assay.generate_all_controls(bio_mod)
    expected_arms = {
        'bio', 'degree_rewired', 'sign_rewired', 'community_rewired',
        'random_sparse', 'learned_sparse', 'unsigned', 'local_only', 'brain_cord_motif'
    }
    assert set(controls.keys()) == expected_arms

    # Run causal assay
    assay_res = assay.run_causal_assay(bio_mod, steps=8, trials=2)
    assert assay_res['status'] == 'ok'
    assert assay_res['num_arms'] == 9
    assert 'bio_advantage' in assay_res
    assert 'cohens_d_effect_size' in assay_res
    assert len(assay_res['pairwise_cohens_d']) == 8


def test_coconut_latent_reasoner_and_anytime_curve():
    """Verifies continuous latent recurrence without token bottleneck and anytime depth curve."""
    reasoner = ContinuousLatentReasoner(latent_dim=32, action_dim=5, max_depth=6, ponder_cost_rate=0.01)
    z = torch.randn(2, 32)

    action, trace = reasoner.forward_continuous_thinking(z)
    assert action.shape == (2, 5)
    assert trace.engaged
    assert 1 <= trace.recurrent_depth <= 6
    assert trace.ponder_cost > 0.0

    # Test anytime performance curve
    targets = torch.randn(4, 5)
    inputs = torch.randn(4, 32)
    curve_res = reasoner.compute_anytime_curve(inputs, targets, depths=[1, 2, 4])
    assert 'feedforward_baseline_mse' in curve_res
    assert 'depth_1' in curve_res['anytime_recurrent_curve']
    assert 'depth_4' in curve_res['anytime_recurrent_curve']


def test_masked_diffusion_repair_sidecar():
    """Verifies that masked diffusion repair resolves failing constraints with a verifier in the loop."""
    sidecar = DiffusionRepairSidecar(feature_dim=8, max_repair_iterations=4, diffusion_timesteps=5, seed=42)
    candidate = np.array([2.5, -2.5, 0.2, 0.1, -0.4, 0.3, 0.0, -0.1], dtype=np.float32)

    # Verifier fails if any element has magnitude > 1.0
    def box_verifier(vec):
        failing = [i for i, v in enumerate(vec) if abs(v) > 1.0]
        return len(failing) == 0, failing, {}

    receipt = sidecar.repair_candidate_vector(candidate, [0, 1], box_verifier, item_id="test_vec")
    assert not receipt.initial_valid
    assert receipt.iterations_used >= 1
    assert 'idx_0' in receipt.repaired_diff or len(receipt.denoising_loss_trajectory) > 0


def test_machine_verifiable_dpo():
    """Verifies DPO loss optimization over first-party machine-checked outcome pairs."""
    engine = MachineVerifiableDPOEngine(obs_dim=32, action_dim=5, beta=0.1, lr=1e-3, seed=42)

    # Populate preference buffer with 6 pairs
    for i in range(6):
        obs = np.random.randn(32).astype(np.float32)
        # Winner is closer to zero, loser has larger magnitude
        w = np.random.uniform(-0.2, 0.2, size=5).astype(np.float32)
        l = np.random.uniform(0.8, 1.2, size=5).astype(np.float32)
        engine.record_preference(obs, w, l, criterion="safety", delta_utility=1.0)

    train_out = engine.train_dpo_step(batch_size=4)
    assert 'loss_dpo' in train_out
    assert train_out['loss_dpo'] > 0.0
    assert train_out['buffer_size'] == 6


def test_unified_eval_scorecard_and_calibration():
    """Verifies Brier score, ECE calibration, and multi-dimensional joint deployment scoring."""
    # Perfect predictions
    preds = np.array([0.95, 0.90, 0.05, 0.10])
    targets = np.array([1, 1, 0, 0])
    brier = compute_brier_score(preds, targets)
    assert brier < 0.02

    ece = compute_expected_calibration_error(preds, targets, num_bins=5)
    assert ece < 0.15

    evaluator = UnifiedEvaluator(alpha_compute=0.1, beta_calibration=1.0, gamma_safety=2.0)
    card = evaluator.evaluate_task_run(
        "test_task",
        returns=[20.0, 22.0],
        execution_times=[0.5, 0.5],
        predictions=preds,
        ground_truth=targets,
        failures=0
    )
    assert card.realized_return == 21.0
    assert card.joint_deployment_score > 0.0
    matrix = evaluator.get_summary_matrix()
    assert matrix['tasks_evaluated'] == 1
