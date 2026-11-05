import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.auto_programmer import (
    run_diagnostic_episodes,
    diagnose_failures,
    synthesize_weight_patch,
    execute_closed_loop_repair,
)
from synapse.patches import preview_patch

def test_run_diagnostic_episodes():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    agent = IntegratedAgent(config)
    diag = run_diagnostic_episodes(agent, task='forage', num_episodes=2, max_steps_per_episode=10, seed=42)
    assert diag['task'] == 'forage'
    assert diag['num_episodes'] == 2
    assert len(diag['returns']) == 2
    assert len(diag['lengths']) == 2
    assert 'mean_graph_activity' in diag
    assert 'mean_approach_valence' in diag
    assert 'mean_avoid_valence' in diag

def test_diagnose_failures():
    dummy_diag = {
        'mean_return': 0.1,
        'mean_approach_valence': -0.2,
        'mean_avoid_valence': 0.5,
        'mean_graph_activity': 0.02,
        'action_mean': [0.01, 0.0, 0.0, 0.0, 0.0],
    }
    hyp = diagnose_failures(dummy_diag)
    assert 'primary_surface' in hyp
    assert 'suggested_fix' in hyp
    assert len(hyp['hypotheses']) > 0

def test_synthesize_weight_patch():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    agent = IntegratedAgent(config)
    diagnosis = {
        'primary_surface': 'mb.w_mbon',
        'suggested_fix': 'reinforce_approach',
        'mean_return': 0.0,
    }
    packet = synthesize_weight_patch(agent, diagnosis)
    assert packet['schema'] == 'synapse-weight-program-v1'
    assert packet['writes'] > 0
    preview = preview_patch(agent, packet)
    assert preview['ok'] is True

def test_execute_closed_loop_repair():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    agent = IntegratedAgent(config)
    report = execute_closed_loop_repair(
        agent,
        task='forage',
        num_diagnostic_episodes=1,
        num_eval_episodes=1,
        min_gain=-1.0,  # permissive to test successful execution and commit
        seed=42,
    )
    assert report['status'] == 'ok'
    assert 'accepted' in report
    assert 'baseline_return' in report
    assert 'diagnosis' in report
    assert 'provenance' in report
    assert report['provenance']['writer_disconnected_for_confirmation'] is True
