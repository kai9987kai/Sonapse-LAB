"""Tests for Adaptive Self-Generated Curriculum and Sealed Evaluator."""
import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.curriculum import AdaptiveCurriculumGenerator, SealedEvaluator


def test_sealed_evaluator_immutability():
    evaluator = SealedEvaluator(num_episodes=2)
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))

    res1 = evaluator.evaluate(agent)
    res2 = evaluator.evaluate(agent)
    assert res1['sealed_generalization_score'] == res2['sealed_generalization_score']
    assert res1['immutable_evaluation_hash'] == hex(0x5EA1ED)


def test_adaptive_curriculum_generator():
    agent = IntegratedAgent(SynapseConfig(seed=42, nodes_per_source=16))
    gen = AdaptiveCurriculumGenerator(seed=42)

    weaknesses = gen.diagnose_weaknesses(agent, episodes=1)
    assert len(weaknesses) == 4
    for dim in ['obstacle_density', 'threat_urgency', 'reward_sparsity', 'memory_delay']:
        assert dim in weaknesses

    curriculum = gen.synthesize_curriculum(weaknesses)
    assert len(curriculum) == 3
    assert all(t.tier in [1, 2, 3] for t in curriculum)

    res = gen.execute_curriculum_training(agent, curriculum, steps_per_tier=5)
    assert 'pre_curriculum_sealed_score' in res
    assert 'post_curriculum_sealed_score' in res
    assert 'generalization_gain' in res
