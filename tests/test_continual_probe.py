"""Contracts for a controlled hidden-regime outcome-learning probe."""
import json

import numpy as np
import pytest


def test_reward_has_visible_action_effects_and_exact_hidden_reversal():
    from synapse.continual_probe import outcome_reward
    # The quadratic and sinusoidal terms do not reverse; the state/action term does.
    assert outcome_reward([0.5, 0.8], [0.0], 1) == 0.0
    positive = outcome_reward([0.5, 0.8], [0.4], 1)
    reversed_value = outcome_reward([0.5, 0.8], [0.4], -1)
    assert positive - reversed_value == pytest.approx(0.4)
    assert positive == pytest.approx(0.2 - 0.032 + 0.15 * np.sin(1.2))
    assert outcome_reward([0.5, -0.9], [0.4], 1) == positive


def test_probe_is_deterministic_json_compatible_and_preserves_frozen_state():
    from synapse.continual_probe import run_continual_probe
    result = run_continual_probe(seeds=(7,), support_steps=8, query_steps=8)
    assert result == run_continual_probe(seeds=(7,), support_steps=8, query_steps=8)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert [phase['name'] for phase in result['config']['phases']] == ['stable', 'reversal', 'recurrence']
    assert [phase['sign'] for phase in result['config']['phases']] == [1, -1, 1]
    assert len(result['raw']['support']) == 24
    assert len(result['raw']['query']) == 3 * 2 * 8 * 4
    assert len(result['checkpoints']) == 3
    assert [row['model_samples'] for row in result['checkpoints']] == [8, 16, 24]
    assert [row['memory_updates'] for row in result['checkpoints']] == [8, 16, 24]
    for checkpoint in result['checkpoints']:
        assert checkpoint['freeze_ok'] is True
        assert checkpoint['restore_exact'] is True
        assert checkpoint['learned_before'] == checkpoint['learned_after']
    assert all(len(value) == 64 for value in result['hashes'].values())


def test_shared_support_and_query_outcomes_are_recomputable_from_ground_truth():
    from synapse.continual_probe import outcome_reward, run_continual_probe
    result = run_continual_probe(seeds=(19,), support_steps=12, query_steps=8)
    for row in result['raw']['support']:
        assert row['reward'] == outcome_reward(row['state'], row['action'], row['sign'])
        assert set(row['predictions']) == {'off', 'model', 'memory', 'shuffled'}
    for phase in ['stable', 'reversal', 'recurrence']:
        support = [row for row in result['raw']['support'] if row['phase'] == phase]
        assert sorted(row['reward'] for row in support) == sorted(row['shuffled_reward'] for row in support)
    grouped = {}
    for row in result['raw']['query']:
        assert row['chosen_reward'] == outcome_reward(row['state'], row['chosen_action'], row['sign'])
        assert row['probe_reward'] == outcome_reward(row['state'], row['probe_action'], row['sign'])
        assert abs(row['chosen_action'][0]) <= 0.350001
        assert row['local_oracle_reward'] + 1e-7 >= row['chosen_reward']
        key = (row['phase'], row['evaluation'], row['index'])
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        assert len(rows) == 4
        assert len({tuple(row['state']) for row in rows}) == 1
        assert len({tuple(row['probe_action']) for row in rows}) == 1


def test_negative_paired_outcomes_are_retained_in_summaries():
    from synapse.continual_probe import summarize_records
    rows = []
    for arm, reward in [('off', 0.0), ('model', 0.2), ('memory', -0.3), ('shuffled', 0.1)]:
        rows.append({'seed': 7, 'phase': 'reversal', 'evaluation': 'current',
                     'arm': arm, 'chosen_reward': reward, 'probe_reward': 0.5,
                     'predicted_probe_reward': 0.0, 'local_oracle_reward': 0.6,
                     'planner_used': arm != 'off'})
    summary = summarize_records([], rows)
    comparison = summary['paired'][0]
    assert comparison['memory_minus_model'] == pytest.approx(-0.5)
    assert comparison['memory_minus_off'] == pytest.approx(-0.3)
    assert comparison['memory_minus_shuffled'] == pytest.approx(-0.4)


@pytest.mark.parametrize('kwargs', [
    {'seeds': ()}, {'seeds': (7, 7)}, {'seeds': (True,)}, {'seeds': (float('nan'),)},
    {'support_steps': 0}, {'query_steps': np.inf}, {'support_steps': 1.5},
])
def test_invalid_probe_configuration_is_rejected(kwargs):
    from synapse.continual_probe import run_continual_probe
    with pytest.raises(ValueError):
        run_continual_probe(**kwargs)


@pytest.mark.parametrize('state, action, sign', [
    ([0.1], [0.0], 1), ([np.nan, 0.0], [0.0], 1),
    ([0.1, 0.2], [np.inf], 1), ([0.1, 0.2], [0.0], 0),
])
def test_invalid_reward_inputs_are_rejected(state, action, sign):
    from synapse.continual_probe import outcome_reward
    with pytest.raises(ValueError):
        outcome_reward(state, action, sign)
