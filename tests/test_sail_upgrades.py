"""Comprehensive Test Suite for Sonapse Adaptive Intelligence Laboratory (SAIL) Upgrades.

Covers:
1. Continual Backpropagation & Utility-Guided Neuron Replacement
2. Trainable Student-Teacher Sleep Consolidation
3. Universal Fast-Weight Memory (uTTT & Titans)
4. Recurrent Multistep Predictive Planner & MPC
5. Verifiable Curriculum Engine (ReLIFT Process Rewards)
6. Causal Connectome Evolution & Interventional Screening
7. Flagship Continual Learning 4-Arm Milestone Benchmark
"""
import numpy as np
import pytest
import torch
from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.continual_backprop import ContinualBackpropEngine, benchmark_continual_backprop
from synapse.sleep import SleepConsolidationEngine
from synapse.fast_weight_memory import UniversalFastWeightMemory
from synapse.multistep_planner import MultistepRecurrentPlanner, compare_planning_horizons
from synapse.verified_curriculum import VerifiableCurriculumEngine
from synapse.causal_growth import CausalConnectomeEvolutionEngine, CausalMutationProposal
from synapse.flagship_benchmark import FlagshipContinualBenchmark

def test_continual_backprop_utility_and_replacement():
    cfg = SynapseConfig(nodes_per_source=16, seed=42)
    agent = IntegratedAgent(cfg)
    optimizer = torch.optim.Adam(agent._optimization_parameters(), lr=1e-3)

    engine = ContinualBackpropEngine(
        num_neurons=agent.brain.n,
        beta=0.9,
        utility_threshold=0.1,
        maturity_threshold=5,
        max_replacements_per_step=2,
        seed=42
    )

    # 1. Simulate steps to age neurons and lower utility for dead units
    for t in range(25):
        # Give first 10 neurons zero activity
        rates = np.ones(agent.brain.n, dtype=np.float32) * 0.5
        rates[:10] = 0.001
        engine.update_utility(rates)

    assert engine.step_count == 25
    candidates = engine.identify_replacement_candidates()
    assert len(candidates) > 0

    # 2. Execute replacement with Adam moment zeroing
    rec = engine.replace_neurons(agent, candidates, optimizer=optimizer)
    assert rec['replaced_count'] > 0
    assert engine.total_replacements > 0

    # 3. Benchmark runner
    bench = benchmark_continual_backprop(agent, steps=10, seed=42)
    assert bench['status'] == 'ok'
    assert 'final_mean_utility' in bench

def test_trainable_sleep_consolidation():
    cfg = SynapseConfig(nodes_per_source=16, seed=42)
    agent = IntegratedAgent(cfg)

    engine = SleepConsolidationEngine(buffer_size=32, lr=1e-3)

    # Log experiences
    for i in range(10):
        obs = np.random.uniform(0.0, 1.0, size=155).astype(np.float32)
        act = np.random.uniform(-0.5, 0.5, size=5).astype(np.float32)
        engine.record_experience(obs, act, reward=1.0 if i % 2 == 0 else -0.5, surprisal=0.3)

    assert len(engine.wake_buffer) == 10

    # Mock evaluation function
    eval_calls = []
    def mock_eval(a):
        eval_calls.append(True)
        return 1.5

    result = engine.run_sleep_cycle(agent, sleep_steps=5, eval_fn=mock_eval)
    assert result['cycle'] == 1
    assert result['accepted'] is True
    assert len(engine.wake_buffer) == 0  # Cleared after successful consolidation
    assert len(engine.long_term_replay) > 0

    # Test serialization
    state = engine.export_state()
    engine2 = SleepConsolidationEngine()
    engine2.import_state(state)
    assert engine2.sleep_cycles_completed == 1
    assert engine2.accepted_consolidations == 1

def test_universal_fast_weight_memory():
    mem = UniversalFastWeightMemory(in_dim=32, out_dim=32, learning_rate=0.05, seed=42)

    # Test read
    x = torch.randn(32)
    readout, gate = mem.read(x)
    assert readout.shape == (32,)
    assert gate.shape == (32,)

    # Test step and test-time inner update
    res1 = mem.step_and_update(x, surprisal=0.8, adapt=True)
    assert res1['step'] == 1
    assert res1['inner_loss'] > 0.0

    # Verify memory norm increased from zero
    assert float(mem.M.norm().item()) > 0.0

    # Second query on same input should have reduced inner loss (learning effect)
    res2 = mem.step_and_update(x, surprisal=0.8, adapt=False)
    assert res2['inner_loss'] < res1['inner_loss']

    # State serialization
    state = mem.export_state()
    mem2 = UniversalFastWeightMemory(in_dim=32, out_dim=32)
    mem2.import_state(state)
    assert float(mem2.M.norm().item()) == pytest.approx(float(mem.M.norm().item()), rel=1e-5)

def test_multistep_recurrent_planner():
    planner = MultistepRecurrentPlanner(state_dim=16, action_dim=5, obs_dim=155, seed=42)

    # Train step
    batch_size = 4
    obs = np.random.uniform(0.0, 1.0, size=(batch_size, 155)).astype(np.float32)
    act = np.random.uniform(-1.0, 1.0, size=(batch_size, 5)).astype(np.float32)
    rew = np.random.randn(batch_size).astype(np.float32)
    next_obs = np.random.uniform(0.0, 1.0, size=(batch_size, 155)).astype(np.float32)
    done = np.zeros(batch_size, dtype=np.float32)

    train_res = planner.train_step(obs, act, rew, next_obs, done)
    assert 'loss_total' in train_res
    assert train_res['loss_total'] > 0.0

    # Multi-step plan
    single_obs = obs[0]
    base_act = np.array([0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    plan_res = planner.plan(single_obs, base_act, horizon=3)
    assert 'action' in plan_res
    assert plan_res['horizon'] == 3
    assert plan_res['action'].shape == (5,)

    # Compare horizons
    comp = compare_planning_horizons(planner, single_obs, base_act, horizons=[1, 3, 5])
    assert 'horizon_1' in comp['results']
    assert 'horizon_3' in comp['results']
    assert 'horizon_5' in comp['results']

def test_verifiable_curriculum_relift():
    curric = VerifiableCurriculumEngine(base_seed=42, mastery_threshold=0.75, window_size=4)

    # Generate procedural challenge
    ch = curric.generate_procedural_challenge()
    assert ch['stage'] == 1
    assert len(ch['subgoals']) >= 2

    # Check subgoals
    target_pos = ch['subgoals'][0]['pos']
    sub_res = curric.check_subgoals(target_pos)
    assert len(sub_res['newly_achieved']) > 0
    assert sub_res['process_reward'] > 0.0

    # Complete episode and get verifiable receipt
    receipt = curric.complete_episode(success=True)
    assert receipt.verified is True
    assert len(receipt.proof_hash) == 16

    # Test curriculum progression after 4 consecutive successes
    curric.complete_episode(success=True)
    curric.complete_episode(success=True)
    curric.complete_episode(success=True)
    assert curric.current_stage == 2  # Promoted from stage 1 to stage 2!

def test_causal_connectome_evolution():
    cfg = SynapseConfig(nodes_per_source=16, seed=42)
    agent = IntegratedAgent(cfg)

    engine = CausalConnectomeEvolutionEngine(lambda_w=0.01, lambda_c=0.01, lambda_f=0.5, seed=42)

    # Generate candidate mutation
    mut = engine.generate_candidate_mutation(agent)
    assert isinstance(mut, CausalMutationProposal)
    assert mut.pre_idx != mut.post_idx

    # Evaluate mutation via counterfactual paired rollouts
    receipt = engine.evaluate_interventional_mutation(agent, mut, test_episodes=2, horizon=10)
    assert isinstance(receipt.causal_delta, float)
    assert isinstance(receipt.net_fitness, float)
    assert receipt.wiring_cost > 0.0
    assert engine.summary()['evaluated_mutations_count'] == 1

def test_flagship_continual_milestone_benchmark():
    # Run with small step budget for fast unit testing
    benchmark = FlagshipContinualBenchmark(
        task_sequence=['forage', 'delayed_match'],
        steps_per_stage=10,
        eval_episodes_per_task=1,
        seed=42
    )
    results = benchmark.run_benchmark()

    assert results['status'] == 'ok'
    assert 'arm_results' in results
    assert 'fixed_gradient' in results['arm_results']
    assert 'dynamic_plasticity' in results['arm_results']
    assert 'continual_sleep' in results['arm_results']
    assert 'full_developmental_sail' in results['arm_results']

    comp = results['comparative_summary']
    assert 'full_developmental_sail' in comp
    assert 'retention_rate' in comp['full_developmental_sail']
