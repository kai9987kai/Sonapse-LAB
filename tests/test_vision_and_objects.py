"""Tests for Genuine Visual Perception, Object-Centric Understanding, Hierarchical Reasoning, Active Inference, and Uncertainty Guard (Items 21-25).
"""
import numpy as np
import torch
from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.vision_encoder import TrainableVisualEncoder, evaluate_viewpoint_generalization
from synapse.object_memory import ObjectCentricMemory, SlotAttention
from synapse.skill_hierarchy import HierarchicalSkillController, BehaviorSkill
from synapse.active_inference import ActiveInferenceAgent
from synapse.uncertainty_guard import UncertaintyGuard, ModificationProposal, apply_safe_modification


def test_trainable_visual_encoder_and_camera_consumption():
    encoder = TrainableVisualEncoder(latent_dim=32, seed=42)
    # Synthetic RGB frame (3, 32, 32)
    frame = np.random.uniform(0.0, 1.0, size=(3, 32, 32)).astype(np.float32)
    res = encoder.encode_frame(frame)
    
    assert res['latent'].shape == (32,)
    assert res['spatial_map'].shape == (64, 4, 4)
    assert res['invariant_code'].shape == (16,)
    assert 0.0 <= res['mean_brightness'] <= 1.0
    
    # Invariance check
    inv_eval = evaluate_viewpoint_generalization(encoder, frame)
    assert 'shift_invariance' in inv_eval
    assert 'lighting_invariance' in inv_eval
    assert -1.0 <= inv_eval['feature_stability'] <= 1.0


def test_agent_consumes_camera():
    cfg = SynapseConfig(nodes_per_source=16, cognition_interval=1, seed=42)
    agent = IntegratedAgent(cfg)
    obs = np.linspace(0.1, 0.9, 155, dtype=np.float32)
    camera_frame = np.random.uniform(0.0, 1.0, size=(3, 32, 32)).astype(np.float32)
    
    # Initial status
    stat_before = agent.status()
    assert stat_before['camera'] == 'inactive; image not consumed'
    
    # Act with camera
    out = agent.act(obs, camera=camera_frame)
    assert out['telemetry']['camera_consumed'] is True
    assert len(out['telemetry']['visual_latents']) == 32
    assert np.isfinite(out['action']).all()
    
    # Status should now indicate active
    stat_after = agent.status()
    assert stat_after['camera'] == 'active; visual encoder fused'


def test_object_centric_memory_persistence_and_push():
    mem = ObjectCentricMemory(num_slots=4, slot_dim=32)
    spatial_features = torch.randn(1, 32, 4, 4)
    
    # Step 1: Detect objects
    slots_t1 = mem.process_spatial_features(spatial_features)
    assert len(slots_t1) == 4
    id_t1 = slots_t1[0].track_id
    
    # Step 2: Slightly shifted features (simulating moving object)
    shifted_features = spatial_features + 0.02 * torch.randn_like(spatial_features)
    slots_t2 = mem.process_spatial_features(shifted_features)
    tracked_ids = [s.track_id for s in slots_t2]
    
    # ID persistence: original object ID is preserved
    assert id_t1 in tracked_ids
    
    # Relational push prediction
    push_report = mem.predict_push_effect(target_track_id=id_t1, push_force=np.array([1.0, 0.5]))
    assert push_report['target_track_id'] == id_t1
    assert id_t1 in push_report['predictions']
    assert 'predicted_position' in push_report['predictions'][id_t1]


def test_hierarchical_skill_controller():
    ctrl = HierarchicalSkillController()
    goal = {'type': 'deliver'}
    plan = ctrl.plan_sequence(goal, {})
    assert plan == ['navigate_to', 'push_object']
    
    obs = np.zeros(155, dtype=np.float32)
    ctx = {
        'target_pos': [1.0, 1.0],
        'agent_pos': [0.0, 0.0],
        'box_pos': [1.0, 1.0],
        'target_dest': [3.0, 3.0]
    }
    
    act, tele = ctrl.step(obs, ctx, plan=plan)
    assert tele['active_skill'] == 'navigate_to'
    assert len(act) == 5
    
    # Reach target to trigger skill transition
    ctx['agent_pos'] = [1.0, 1.0] # At target
    act2, tele2 = ctrl.step(obs, ctx, plan=plan)
    assert tele2['finished'] is True or tele2['transitioned'] is True


def test_active_inference_epistemic_selection():
    agent = ActiveInferenceAgent(state_dim=4, action_dim=5)
    obs = np.zeros(155, dtype=np.float32)
    obs[:4] = [0.2, -0.1, 0.5, 0.0]
    
    action, tele = agent.select_action(obs)
    assert len(action) == 5
    assert 'efe_scores' in tele
    assert len(tele['efe_scores']) == 5
    assert tele['selected_epistemic_gain'] >= 0.0
    assert tele['belief_entropy'] is not None


def test_uncertainty_guard_safety_gating():
    guard = UncertaintyGuard(ensemble_size=3, disagreement_threshold=0.3, ood_threshold=2.5)
    guard.register_safe_experience(np.zeros(10))
    guard.register_safe_experience(np.ones(10) * 0.1)
    
    # 1. Safe, consistent proposals
    w_safe = torch.randn(5, 64) * 0.01
    proposals_safe = [
        ModificationProposal("motor", w_safe, np.zeros(10), 0.2),
        ModificationProposal("motor", w_safe + 0.001 * torch.randn(5, 64), np.zeros(10), 0.2),
        ModificationProposal("motor", w_safe + 0.001 * torch.randn(5, 64), np.zeros(10), 0.2),
    ]
    target_layer = torch.nn.Linear(64, 5)
    assess_safe = guard.assess_ensemble_proposals(proposals_safe, target_layer)
    assert assess_safe.approved is True
    
    approved_mean = torch.stack([proposal.delta_weights for proposal in proposals_safe]).mean(dim=0)
    applied = apply_safe_modification(target_layer.weight, assess_safe, approved_mean)
    assert applied is True
    
    # 2. Conflicting/divergent proposals (high disagreement)
    proposals_conflicting = [
        ModificationProposal("motor", torch.randn(5, 64) * 2.0, np.zeros(10), 0.1),
        ModificationProposal("motor", -torch.randn(5, 64) * 2.0, np.zeros(10), 0.1),
        ModificationProposal("motor", torch.randn(5, 64) * 0.1, np.zeros(10), 0.1),
    ]
    assess_reject = guard.assess_ensemble_proposals(proposals_conflicting, target_layer)
    assert assess_reject.approved is False
    assert "exceeds safety threshold" in assess_reject.rejection_reason
    
    applied_reject = apply_safe_modification(target_layer.weight, assess_reject, torch.randn(5, 64))
    assert applied_reject is False
