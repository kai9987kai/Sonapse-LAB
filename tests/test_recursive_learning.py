"""Tests for Recursive Learning Rule Evolution and Phylogenetic Lineages."""
import pytest
from synapse.recursive_learning import LearningRulePolicy, RecursiveEvolutionEngine


def test_learning_rule_policy_mutation():
    parent = LearningRulePolicy(lr_base=3e-4, eligibility_decay=0.85)
    import numpy as np
    rng = np.random.default_rng(42)
    child = parent.mutate(rng)

    assert child.policy_id != parent.policy_id
    assert child.parent_id == parent.policy_id
    assert 1e-5 <= child.lr_base <= 1e-2
    assert 0.5 <= child.eligibility_decay <= 0.99


def test_recursive_evolution_engine():
    engine = RecursiveEvolutionEngine(population_size=4, seed=42)
    res = engine.run_recursive_evolution(generations=3, task_name='forage_push', compute_budget_per_indiv=50)

    assert res['total_generations'] == 3
    assert len(res['generation_history']) == 3
    assert res['evolutionary_efficiency_verified'] is True
    assert res['total_lineage_nodes'] == 12
