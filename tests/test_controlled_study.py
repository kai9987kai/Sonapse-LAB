"""Unit tests for the Controlled 6-Arm Comparative Study."""
import pytest
from synapse.controlled_study import ControlledSixArmStudy, deterministic_seed


def test_deterministic_seed():
    s1 = deterministic_seed(42, "test_task")
    s2 = deterministic_seed(42, "test_task")
    s3 = deterministic_seed(42, "other_task")
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**31


def test_controlled_study_execution():
    study = ControlledSixArmStudy(
        task_sequence=['forage', 'maze'],
        steps_per_stage=6,
        eval_episodes_per_task=1,
        seed=42,
        nodes_per_source=16
    )
    results = study.run_study()

    assert 'task_sequence' in results
    assert len(results['task_sequence']) == 2
    assert 'arms' in results
    assert len(results['arms']) == 6

    # Verify all 6 arms are present
    expected_arms = [
        'frozen_actor',
        'online_rl',
        'actor_replay',
        'continual_backprop',
        'connectome_evolution',
        'composed_system'
    ]
    for arm in expected_arms:
        assert arm in results['arms']
        arm_data = results['arms'][arm]
        assert 'stage_returns' in arm_data
        assert len(arm_data['stage_returns']) == 2
        assert 'final_retention_rate' in arm_data
        assert 'catastrophic_forgetting' in arm_data
        assert 'mean_step_latency_ms' in arm_data
        assert arm_data['mean_step_latency_ms'] > 0.0

    assert 'comparative_table' in results
    assert len(results['comparative_table']) == 6
    assert 'delta_composed_vs_frozen' in results
