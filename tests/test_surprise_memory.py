"""Behavioral checks for bounded, outcome-grounded associative memory."""
from copy import deepcopy
import json

import numpy as np
import pytest


def make_memory(**kwargs):
    from synapse.surprise_memory import SurpriseEpisodicMemory
    return SurpriseEpisodicMemory(**kwargs)


def test_actual_outcomes_are_recalled_and_surprising_reversal_replaces_them():
    mem = make_memory(state_dim=2, action_dim=1, initial_coverage=2)
    state = np.array([0.2, -0.1])
    actions = np.array([[-0.5], [0.5]])
    mem.update(state, actions[0], 0.004, 0.0)
    mem.update(state, actions[1], -0.004, 0.0)
    recalled = mem.retrieve(state, actions)
    assert recalled['reward'][0] > recalled['reward'][1]
    assert np.all(recalled['confidence'] > 0.5)
    # The prediction error admits an observed reward, never becomes its value.
    mem.update(state, actions[0], -0.004, -0.03)
    mem.update(state, actions[1], 0.004, 0.03)
    reversed_recall = mem.retrieve(state, actions)
    assert reversed_recall['reward'][1] > reversed_recall['reward'][0]
    assert np.max(np.abs(reversed_recall['reward'])) <= 0.004
    assert mem.summary()['size'] == 2
    assert mem.summary()['replacements'] == 2


def test_error_gate_rejects_predictable_repeats_but_admits_new_coverage():
    mem = make_memory(state_dim=1, action_dim=1, initial_coverage=1)
    mem.update([0.0], [0.0], 0.25, 0.0)
    mem.update([0.0], [0.0], 0.75, 0.0)
    assert mem.summary()['rejected'] == 1
    assert mem.retrieve([0.0], [[0.0]])['reward'][0] == pytest.approx(0.25)
    mem.update([2.0], [0.0], -0.25, 0.0)
    assert mem.summary()['size'] == 2


def test_capacity_evicts_oldest_outcome_deterministically():
    mem = make_memory(state_dim=1, action_dim=1, capacity=2, initial_coverage=3)
    for state, reward in [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]:
        mem.update([state], [0.0], reward, 0.0)
    state = mem.export_state()
    assert mem.summary()['size'] == 2
    assert mem.summary()['evictions'] == 1
    assert sorted(row[0] for row in state['states']) == [0.0, 1.0]
    assert mem.retrieve([1.0], [[0.0]])['reward'][0] > 0.5


def test_exact_recall_recency_weighting_and_distance_confidence():
    mem = make_memory(state_dim=1, action_dim=1, initial_coverage=2,
                      recency_half_life=1.0, match_distance=0.0)
    mem.update([-0.2], [0.0], -1.0, 0.0)
    mem.update([0.2], [0.0], 1.0, 0.0)
    between = mem.retrieve([0.0], [[0.0]])
    assert between['reward'][0] > 0.0
    close = mem.retrieve([0.2], [[0.0]])
    far = mem.retrieve([10.0], [[1.0]])
    assert far['confidence'][0] < close['confidence'][0]
    assert far['distance'][0] > close['distance'][0]
    for _ in range(5):
        mem.update([0.2], [0.0], 1.0, 0.0)
    assert mem.retrieve([0.2], [[0.0]])['confidence'][0] < close['confidence'][0]


def test_empty_and_repeated_reads_are_finite_and_never_mutate_memory():
    mem = make_memory(state_dim=2, action_dim=1)
    before = mem.export_state()
    for _ in range(3):
        out = mem.retrieve([0.0, 0.0], [[0.0], [1.0]])
        assert set(out) == {'reward', 'confidence', 'distance'}
        assert all(value.shape == (2,) and np.isfinite(value).all() for value in out.values())
        np.testing.assert_array_equal(out['reward'], [0.0, 0.0])
        np.testing.assert_array_equal(out['confidence'], [0.0, 0.0])
    assert mem.export_state() == before
    assert mem.retrieve([0.0, 0.0], np.empty((0, 1)))['reward'].shape == (0,)
    mem.update([0.0, 0.0], [0.0], 1.0, 0.0)
    before = mem.export_state()
    result = mem.retrieve([0.0, 0.0], [[0.0]])
    result['reward'][0] = -100.0
    assert mem.export_state() == before


@pytest.mark.parametrize('state, action, reward, error', [
    ([0.0], [0.0], 1.0, 0.0),
    ([0.0, np.nan], [0.0], 1.0, 0.0),
    ([0.0, 0.0], [0.0, 0.0], 1.0, 0.0),
    ([0.0, 0.0], [np.inf], 1.0, 0.0),
    ([0.0, 0.0], [0.0], np.nan, 0.0),
    ([0.0, 0.0], [0.0], 1.0, np.inf),
    ([0.0, 0.0], [0.0], [1.0, 2.0], 0.0),
])
def test_invalid_outcome_is_rejected_before_any_mutation(state, action, reward, error):
    mem = make_memory(state_dim=2, action_dim=1)
    mem.update([0.0, 0.0], [0.0], 0.2, 0.0)
    before = mem.export_state()
    with pytest.raises(ValueError):
        mem.update(state, action, reward, error)
    assert mem.export_state() == before


@pytest.mark.parametrize('state, actions', [
    ([0.0], [[0.0]]), ([0.0, 0.0], [0.0]),
    ([0.0, 0.0], [[np.nan]]), ([0.0, 0.0], [[0.0, 1.0]]),
])
def test_invalid_read_is_rejected_without_mutation(state, actions):
    mem = make_memory(state_dim=2, action_dim=1)
    before = mem.export_state()
    with pytest.raises(ValueError):
        mem.retrieve(state, actions)
    assert mem.export_state() == before


def test_json_restore_replays_future_writes_and_reads_exactly():
    mem = make_memory(state_dim=2, action_dim=1, capacity=3, initial_coverage=2)
    for s, a, r, e in [([0.0, 0.1], [0.0], 0.2, 0.0),
                        ([0.5, 0.0], [1.0], -0.3, -0.2),
                        ([0.0, 0.1], [0.0], 0.8, 0.9)]:
        mem.update(s, a, r, e)
    snapshot = json.loads(json.dumps(mem.export_state()))
    restored = make_memory(state_dim=2, action_dim=1, capacity=3, initial_coverage=2)
    restored.import_state(snapshot)
    assert restored.summary() == mem.summary()
    for s, a, r, e in [([-0.8, 0.2], [0.2], 0.6, 0.0),
                        ([0.5, 0.0], [1.0], -0.9, -2.0),
                        ([0.0, 0.1], [0.0], 0.2, 0.0)]:
        assert restored.update(s, a, r, e) == mem.update(s, a, r, e)
        for name, value in mem.retrieve(s, [[0.0], [1.0]]).items():
            np.testing.assert_array_equal(value, restored.retrieve(s, [[0.0], [1.0]])[name])
    assert restored.export_state() == mem.export_state()
    snapshot['states'][0][0] = 999.0
    assert restored.export_state() == mem.export_state()


@pytest.mark.parametrize('corruption', ['shape', 'nan', 'counters', 'config', 'time'])
def test_invalid_restore_is_atomic(corruption):
    mem = make_memory(state_dim=2, action_dim=1)
    mem.update([0.0, 0.1], [0.0], 0.2, 0.0)
    before = mem.export_state()
    bad = deepcopy(before)
    if corruption == 'shape':
        bad['states'][0] = [1.0]
    elif corruption == 'nan':
        bad['rewards'][0] = float('nan')
    elif corruption == 'counters':
        bad['counters']['writes'] = -1
    elif corruption == 'config':
        bad['config']['state_dim'] = 3
    else:
        bad['timestamps'][0] = 100000
    with pytest.raises(ValueError):
        mem.import_state(bad)
    assert mem.export_state() == before


@pytest.mark.parametrize('kwargs', [
    {'capacity': 0}, {'state_dim': True}, {'initial_coverage': -1},
    {'error_decay': 1.0}, {'bandwidth': 0.0}, {'recency_half_life': np.inf},
    {'neighbors': 0}, {'error_floor': -0.1}, {'novelty_distance': -1.0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_memory(**kwargs)


def test_extreme_finite_outcomes_keep_retrieval_finite():
    mem = make_memory(state_dim=1, action_dim=1, initial_coverage=2)
    mem.update([1e300], [1e300], 1e300, -1e300)
    mem.update([-1e300], [-1e300], -1e300, 1e300)
    out = mem.retrieve([1e300], [[1e300], [-1e300]])
    assert all(np.isfinite(value).all() for value in out.values())
    assert np.isfinite(mem.summary()['error_threshold'])
