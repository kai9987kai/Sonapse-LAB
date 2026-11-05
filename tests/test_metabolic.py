import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.metabolic import compute_metabolic_cost, compute_metabolic_pareto_front

def test_compute_metabolic_cost():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    cost = compute_metabolic_cost(agent)
    assert cost['total_metabolic_cost'] > 0
    assert cost['wiring_cost'] > 0
    assert cost['active_graft_synapses'] > 0
    assert 'spike_energy' in cost
    assert 'homeostatic_penalty' in cost

def test_compute_metabolic_pareto_front():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    res = compute_metabolic_pareto_front(
        agent,
        task='forage',
        num_eval_episodes=1,
        thresholds=[0.0, 0.1, 0.3],
        seed=42,
    )
    assert res['status'] == 'ok'
    assert len(res['frontier_points']) == 3
    assert res['pareto_optimal_count'] >= 1
    assert any(p['pareto_efficient'] for p in res['frontier_points'])
