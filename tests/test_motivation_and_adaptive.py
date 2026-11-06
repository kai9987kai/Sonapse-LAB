"""Tests for Language, Compositional Discovery, Motivation, Autonomy, and Adaptive Compute (Items 36-50).
"""
import numpy as np
import torch
from synapse.neuro_symbolic import (
    NeuralProgramCompiler,
    GroundedInstructionParser,
    FewShotImitationLearner,
    AutonomousCausalExperimenter,
    InvariantRepresentationLearner
)
from synapse.motivation import (
    LearningProgressCuriosity,
    CompetenceGoalSelector,
    ActiveSensoryAttention,
    EmergentCommunicationChannel
)
from synapse.adaptive_compute import (
    EventDrivenSparseEngine,
    LearnableTimeConstants,
    DifficultyAwareComputeRouter,
    AdversarialEnvironmentGenerator,
    VerifiedActionSafetyLayer
)
from synapse.evidence_engine import ResearchEvidenceEngine, compute_cohens_d, bootstrap_confidence_interval


def test_neural_program_compiler():
    comp = NeuralProgramCompiler()
    trajectory = [np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([1.2, 1.2])]
    actions = [np.array([0.5, 0.0, 0.0, 0.0, 0.0]), np.array([0.8, 0.0, 0.0, 0.0, 1.0])]
    
    prog = comp.compile_trajectory("push_block", trajectory, actions)
    assert len(prog) >= 3
    ops = [instr.op for instr in prog]
    assert "CHECK_PRECONDITION" in ops
    assert "MOVE_TO" in ops
    assert "PUSH" in ops
    
    res = comp.execute_program(prog, {'energy': 0.8})
    assert res['success'] is True


def test_grounded_instruction_parser():
    parser = GroundedInstructionParser()
    objects = [
        {'category': 'box', 'track_id': 1},
        {'category': 'food', 'track_id': 2}
    ]
    
    # 1. Push box north
    parsed = parser.parse_and_ground("push the yellow box north", objects)
    assert parsed['verb'] == 'push'
    assert parsed['target_object']['category'] == 'box'
    assert parsed['target_direction'] == [0.0, 1.0]
    assert parsed['grounded'] is True


def test_few_shot_imitation_learner():
    learner = FewShotImitationLearner(obs_dim=155, act_dim=5)
    demo_obs = np.random.uniform(-1, 1, size=(8, 155)).astype(np.float32)
    demo_act = np.random.uniform(-0.5, 0.5, size=(8, 5)).astype(np.float32)
    demos = [{'obs': demo_obs, 'act': demo_act}]
    
    loss = learner.fit_demonstrations(demos, epochs=10)
    assert np.isfinite(loss)
    assert loss < 1.0


def test_causal_experimenter_and_invariant_representation():
    exp = AutonomousCausalExperimenter()
    interv = exp.design_intervention("heavy_surface", [{'track_id': 1}])
    assert interv['treatment_action'][4] > interv['control_action'][4]
    
    eval_res = exp.evaluate_causal_effect(treatment_outcome=1.5, control_outcome=0.2)
    assert eval_res['statistically_significant'] is True
    assert eval_res['average_treatment_effect'] > 1.0
    
    # Invariant representation
    rep = InvariantRepresentationLearner(in_dim=32, latent_dim=16)
    z = rep(torch.randn(4, 32))
    assert z.shape == (4, 16)
    assert torch.allclose(torch.norm(z, dim=-1), torch.ones(4), atol=1e-4)


def test_motivation_curiosity_and_competence_selection():
    # 1. Learning progress curiosity
    cur = LearningProgressCuriosity(window_size=4)
    # Decreasing error (learning progress)
    errors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    r_prog = 0.0
    for e in errors:
        r_prog = cur.compute_intrinsic_reward(e)
    assert r_prog > 0.0 # Positive learning velocity
    
    # Flat error (random unlearnable noise or stalled)
    for _ in range(10):
        r_flat = cur.compute_intrinsic_reward(0.5)
    assert r_flat == 0.0
    
    # 2. Competence goal selector
    sel = CompetenceGoalSelector()
    sel.register_goal("easy_reach", "Reach", np.zeros(2))
    sel.register_goal("medium_push", "Push", np.ones(2))
    sel.record_attempt("easy_reach", 1.0)
    sel.record_attempt("easy_reach", 1.0)
    sel.record_attempt("medium_push", 0.4)
    sel.record_attempt("medium_push", 0.7) # High learning rate!
    
    next_goal = sel.select_next_goal()
    assert next_goal.goal_id == "medium_push"


def test_active_sensory_attention_and_emergent_comm():
    # 1. Active sensory attention
    attn = ActiveSensoryAttention(visual_grid_size=8)
    frame = np.zeros((32, 32), dtype=np.float32)
    frame[0:8, 0:8] = 1.0 # High energy patch in top-left
    attn.update_with_frame(frame)
    gx, gy = attn.saccade_to_highest_information()
    assert 0 <= gx < 8 and 0 <= gy < 8
    
    # 2. Emergent communication channel & ablation
    chan = EmergentCommunicationChannel()
    raw_msg = np.array([1.0, 0.5, -0.5])
    normal_msg = chan.transmit("agentA", "agentB", raw_msg, ablate=False)
    ablated_msg = chan.transmit("agentA", "agentB", raw_msg, ablate=True)
    assert np.array_equal(normal_msg, raw_msg)
    assert np.array_equal(ablated_msg, np.zeros_like(raw_msg))
    
    util = chan.evaluate_communication_utility(0.9, 0.4)
    assert util['has_emergent_utility'] is True


def test_adaptive_compute_and_verified_safety():
    # 1. Event-driven sparse computation
    sparse_eng = EventDrivenSparseEngine(num_units=16, event_threshold=0.1)
    inp1 = np.zeros(16)
    inp2 = np.zeros(16)
    inp2[0] = 0.5 # only unit 0 exceeds event threshold
    state, mask, sparsity = sparse_eng.step(inp2)
    assert mask[0] == 1.0
    assert np.sum(mask[1:]) == 0.0
    assert sparsity > 0.8
    
    # 2. Learnable neural time constants
    lt = LearnableTimeConstants(num_units=8, tau_min=5.0, tau_max=50.0)
    state = lt(torch.ones(8) * 2.0, dt=1.0)
    assert state.shape == (8,)
    assert torch.all(lt.tau >= 5.0) and torch.all(lt.tau <= 50.0)
    
    # 3. Dynamic compute router
    router = DifficultyAwareComputeRouter(complexity_threshold=0.3)
    sys_reflex, iters1 = router.route(np.ones(155), uncertainty=0.0)
    sys_delib, iters2 = router.route(np.ones(155), uncertainty=0.9)
    assert sys_reflex == 'SYSTEM_1'
    assert sys_delib == 'SYSTEM_2'
    assert iters2 > iters1
    
    # 4. Adversarial scenario generator
    adv_gen = AdversarialEnvironmentGenerator(seed=42)
    scen = adv_gen.generate_adversarial_scenario("sudden_obstacle")
    assert scen['type'] == 'sudden_obstacle'
    
    # 5. Verified action safety layer
    safety = VerifiedActionSafetyLayer(max_thrust=0.8, min_safe_dist=0.3)
    # Dangerous action: forward thrust when obstacle is at 0.1m
    unsafe_act = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    safe_act, intervened = safety.shield_action(unsafe_act, min_obstacle_distance=0.1, current_speed=1.0)
    assert intervened is True
    assert safe_act[0] <= 0.0 # Forced braking


def test_reproducible_research_evidence_engine():
    engine = ResearchEvidenceEngine()
    baseline = [1.2, 1.15, 1.25, 1.18, 1.22]
    ablations = {
        'no_vision': [0.7, 0.72, 0.68, 0.71, 0.69],
        'no_hierarchy': [0.85, 0.88, 0.82, 0.84, 0.86]
    }
    
    suite = engine.run_ablation_suite(baseline, ablations)
    assert 'baseline' in suite
    assert 'no_vision' in suite['ablations']
    assert suite['ablations']['no_vision']['cohens_d'] > 2.0
    assert suite['ablations']['no_vision']['statistically_significant'] is True
    
    # Markdown report generation
    report = engine.generate_markdown_report()
    assert "# SYNAPSE Scientific Evidence Report" in report
    assert "no_vision" in report
    assert "Component Ablation Matrix" in report
