"""Behavioral contracts for bounded, pure planning with learned transitions."""
from copy import deepcopy

import numpy as np
import pytest

from synapse.experience import ExperienceWorldModel
from synapse.stateio import digest
from synapse.surprise_memory import SurpriseEpisodicMemory


def planner_type():
    # A missing feature is a test failure while the initial RED stays collectable.
    import synapse.experience as experience
    from importlib.util import find_spec
    assert find_spec('synapse.experience_planner') is not None, 'bounded experience planner is missing'
    from synapse.experience_planner import ExperienceMPC
    assert hasattr(experience.ExperienceWorldModel, 'predict_members'), 'pure member prediction is missing'
    return ExperienceMPC


@pytest.fixture(scope='module')
def delayed_model():
    model = ExperienceWorldModel(state_dim=3, action_dim=1, context_dim=1,
                                 forgetting=1.0, seed=17)
    rng = np.random.default_rng(143)
    for _ in range(600):
        state = rng.uniform(-1.5, 1.5, 3)
        action = rng.uniform(-1, 1, 1)
        # Position and velocity are visible. Action affects position only after
        # another transition; the final coordinate is conditioning, held fixed.
        next_state = [state[0] + state[1], action[0], state[2] + 0.7]
        model.update(state, action, next_state,
                     state[0] + 0.2 * state[2] - 0.1 * action[0] ** 2)
    return model


def test_three_steps_find_delayed_reward_that_one_step_rejects(delayed_model):
    ExperienceMPC = planner_type()
    state = [0.0, 0.0, 0.0]
    short, one = ExperienceMPC(delayed_model, horizon=1).plan(state, [0.0])
    long, three = ExperienceMPC(delayed_model, horizon=3).plan(state, [0.0])
    np.testing.assert_array_equal(short, [0.0])
    assert one['used'] is False
    assert 0.3 < long[0] <= 0.350001
    assert three['used'] is True
    assert three['predicted_gain'] > 0.15
    assert len(three['predicted_trajectory']) == 4
    assert three['baseline_return'] < three['predicted_return']


def test_member_predictions_are_pure_and_keep_distinct_conditioning_states(delayed_model):
    planner_type()
    before = digest(delayed_model.export_state())
    states = np.array([[0.2, 0.1, 0.4], [-0.4, 0.8, -0.6], [0.5, -0.2, 0.3]])
    result = delayed_model.predict_members(states, [[0.3], [-0.2]])
    assert result['reward'].shape == (3, 2)
    assert result['delta'].shape == (3, 2, 3)
    assert result['coverage'].shape == (3, 2)
    np.testing.assert_allclose(result['delta'][:, 0, :2], [[0.1, 0.2], [0.8, -0.5], [-0.2, 0.5]], atol=0.006)
    np.testing.assert_allclose(result['reward'][:, 0], [0.271, -0.529, 0.551], atol=0.006)
    np.testing.assert_array_equal(result['delta'][:, :, -1], 0)
    assert digest(delayed_model.export_state()) == before


def test_search_freezes_inputs_model_rng_memory_and_context(delayed_model):
    ExperienceMPC = planner_type()
    memory = SurpriseEpisodicMemory(state_dim=3, action_dim=1)
    memory.update([0, 0, 0.7], [0], 0.14, 0.1)
    state = np.array([0., 0., 0.7]); baseline = np.array([0.])
    before = digest({'model': delayed_model.export_state(), 'memory': memory.export_state()})
    action, diagnostics = ExperienceMPC(delayed_model, horizon=3).plan(state, baseline, memory)
    again, repeated = ExperienceMPC(delayed_model, horizon=3).plan(state, baseline, memory)
    np.testing.assert_array_equal(action, again)
    assert diagnostics == repeated
    np.testing.assert_array_equal(state, [0, 0, 0.7])
    np.testing.assert_array_equal(baseline, [0])
    np.testing.assert_array_equal(np.asarray(diagnostics['predicted_trajectory'])[:, -1], 0.7)
    assert digest({'model': delayed_model.export_state(), 'memory': memory.export_state()}) == before


def test_each_member_propagates_its_own_state(delayed_model):
    ExperienceMPC = planner_type()
    model = deepcopy(delayed_model)
    model.weights.fill(0)
    # Three exact local dynamics: x'=.98x, x'=x, x'=1.02x.
    model.weights[:, 1, 0] = [-0.02, 0, 0.02]
    model.weights[:, 1, -1] = 1
    _, diagnostics = ExperienceMPC(model, horizon=3).plan([0.5, 0, 0.4], [0])
    trajectories = np.asarray(diagnostics['member_trajectories'])
    np.testing.assert_allclose(trajectories[:, :, 0],
                               [[.5, .49, .4802, .470596], [.5, .5, .5, .5],
                                [.5, .51, .5202, .530604]], atol=1e-8)
    np.testing.assert_array_equal(trajectories[:, :, 2], .4)


def test_cold_start_and_bad_dynamics_abstain(delayed_model):
    ExperienceMPC = planner_type()
    cold = ExperienceWorldModel(state_dim=3, action_dim=1, context_dim=1)
    action, diagnostics = ExperienceMPC(cold).plan([0, 0, 0], [.2])
    np.testing.assert_allclose(action, [.2])
    assert diagnostics['used'] is False and diagnostics['evaluations'] == 0
    model = deepcopy(delayed_model)
    model.delta_error_ema = model.delta_baseline_error_ema * 2
    action, diagnostics = ExperienceMPC(model).plan([0, 0, 0], [.2])
    np.testing.assert_allclose(action, [.2])
    assert diagnostics['used'] is False
    assert 'dynamics' in diagnostics['reason']


def test_large_disagreement_abstains(delayed_model):
    ExperienceMPC = planner_type()
    model = deepcopy(delayed_model)
    model.weights[0, 0, -1] += 100
    model.weights[1, 0, -1] -= 100
    action, diagnostics = ExperienceMPC(model).plan([0, 0, 0], [0])
    np.testing.assert_array_equal(action, [0])
    assert diagnostics['used'] is False
    assert 'uncertainty' in diagnostics['reason']


def test_baseline_leaving_encoded_domain_abstains_before_further_prediction(delayed_model):
    ExperienceMPC = planner_type()
    action, diagnostics = ExperienceMPC(delayed_model).plan([9.9, .5, 0], [0])
    np.testing.assert_array_equal(action, [0])
    assert diagnostics['used'] is False
    assert diagnostics['evaluations'] == 1
    assert diagnostics['reason'] == 'unsupported predicted state'
    assert np.isfinite(np.asarray(diagnostics['predicted_trajectory'])).all()


def test_memory_correction_cannot_hide_model_disagreement(delayed_model):
    ExperienceMPC = planner_type()
    model = deepcopy(delayed_model)
    model.weights[0, 0, -1] += .4
    model.weights[1, 0, -1] -= .4
    memory = SurpriseEpisodicMemory(state_dim=3, action_dim=1, bandwidth=100)
    memory.update([0, 0, 0], [0], 0, 1)
    action, diagnostics = ExperienceMPC(model).plan([0, 0, 0], [0], memory)
    np.testing.assert_array_equal(action, [0])
    assert 'uncertainty' in diagnostics['reason']


def test_width_one_can_choose_an_immediately_better_first_action(delayed_model):
    ExperienceMPC = planner_type()
    model = deepcopy(delayed_model)
    # A learned immediate action incentive with the same supported dynamics.
    model.weights[:, 1 + model.state_dim, -1] += .5
    action, diagnostics = ExperienceMPC(model, horizon=3, beam_width=1).plan([0, 0, 0], [0])
    assert action[0] > .3
    assert diagnostics['used'] is True
    assert any(candidate['baseline'] for candidate in diagnostics['candidates'])


@pytest.mark.parametrize('budget', [1, 3, 7, 11, 32])
def test_work_budget_preserves_full_baseline_comparator(delayed_model, budget):
    ExperienceMPC = planner_type()
    _, diagnostics = ExperienceMPC(delayed_model, horizon=3, max_evaluations=budget).plan([0, 0, 0], [0])
    assert diagnostics['evaluations'] <= budget
    if budget < 3:
        assert diagnostics['used'] is False and diagnostics['reason'] == 'insufficient planning budget'
    else:
        baseline = next(row for row in diagnostics['candidates'] if row['baseline'])
        np.testing.assert_array_equal(baseline['actions'], [[0], [0], [0]])


@pytest.mark.parametrize('kwargs', [dict(horizon=0), dict(horizon=6), dict(horizon=True),
                                   dict(beam_width=0), dict(beam_width=257),
                                   dict(discount=0), dict(discount=1.01), dict(discount=np.nan),
                                   dict(discount='bad'), dict(discount=1j),
                                   dict(max_evaluations=0), dict(max_evaluations=1.5)])
def test_invalid_search_arguments_are_rejected(delayed_model, kwargs):
    ExperienceMPC = planner_type()
    with pytest.raises(ValueError):
        ExperienceMPC(delayed_model, **kwargs)


@pytest.mark.parametrize('state,baseline', [([0, 0], [0]), ([0, np.nan, 0], [0]),
                                          ([0, 0, 0], [1.1]), ([0, 0, 0], [0, 0]),
                                          ([11, 0, 0], [0])])
def test_invalid_plan_inputs_fail_before_reads_or_mutation(delayed_model, state, baseline):
    ExperienceMPC = planner_type()
    before = digest(delayed_model.export_state())
    with pytest.raises(ValueError):
        ExperienceMPC(delayed_model).plan(state, baseline)
    assert digest(delayed_model.export_state()) == before


def test_horizon_one_preserves_existing_action_and_reason(delayed_model):
    ExperienceMPC = planner_type()
    expected, old = delayed_model.plan([0.8, .2, .1], [.1])
    actual, new = ExperienceMPC(delayed_model, horizon=1).plan([0.8, .2, .1], [.1])
    np.testing.assert_array_equal(actual, expected)
    for key, value in old.items():
        assert new[key] == value


def test_dynamics_gate_and_member_predictions_restore_with_legacy_defaults(delayed_model):
    planner_type()
    restored = ExperienceWorldModel(state_dim=3, action_dim=1, context_dim=1)
    restored.import_state(delayed_model.export_state())
    assert restored.summary()['dynamics_gate_passed'] is True
    assert digest(restored.export_state()) == digest(delayed_model.export_state())
    legacy = delayed_model.export_state()
    legacy.pop('delta_baseline_error_ema')
    legacy.pop('delta_baseline_squared_error')
    restored.import_state(legacy)
    assert restored.summary()['dynamics_gate_passed'] is False
    assert np.isfinite(restored.predict_members([0, 0, 0], [[0]])['reward']).all()


def test_no_change_error_is_measured_before_transition_fit():
    model = ExperienceWorldModel(state_dim=2, action_dim=1)
    model.update([0, 0], [1], [.4, -.2], .5)
    # Mean squared delta versus a predictor that simply holds the current state.
    assert getattr(model, 'delta_baseline_error_ema', None) == pytest.approx(.1)
    assert model.delta_error_ema == pytest.approx(.1)
    assert model.summary().get('dynamics_gate_passed') is False
