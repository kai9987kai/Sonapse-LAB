"""Behavioral checks for prediction from executed transitions and frozen planning."""
import numpy as np
import pytest

from synapse.experience import ExperienceWorldModel, encode_experience_state
from synapse.stateio import digest


def trained_model(seed=4):
    model = ExperienceWorldModel(state_dim=2, action_dim=1, seed=seed)
    rng = np.random.default_rng(19)
    for _ in range(240):
        state = rng.uniform(-1, 1, 2)
        action = rng.uniform(-1, 1, 1)
        # Independent, action-dependent targets, with a known state/action interaction.
        delta = np.array([0.2 * action[0], -0.1 * state[0] * action[0]])
        reward = 0.5 * state[0] * action[0] - 0.2 * action[0] ** 2
        model.update(state, action, state + delta, reward)
    return model


def test_learns_unseen_action_conditioned_rewards_and_transitions():
    model = trained_model()
    prediction = model.predict(np.array([0.8, -0.3]), np.array([[0.6], [-0.6]]))
    np.testing.assert_allclose(prediction['reward'], [0.168, -0.312], atol=0.025)
    np.testing.assert_allclose(prediction['delta'], [[0.12, -0.048], [-0.12, 0.048]], atol=0.015)
    assert np.all(prediction['coverage'] >= 0)
    assert np.all(prediction['disagreement'] >= 0)


def test_cold_start_abstains_and_trained_planning_is_bounded_and_pure():
    model = ExperienceWorldModel(state_dim=2, action_dim=1)
    baseline = np.array([0.0])
    action, diagnostics = model.plan(np.array([0.8, -0.3]), baseline)
    np.testing.assert_array_equal(action, baseline)
    assert diagnostics['used'] is False
    model = trained_model()
    before = digest(model.export_state())
    action, diagnostics = model.plan(np.array([0.8, -0.3]), baseline)
    assert 0.1 < action[0] <= 0.35
    assert diagnostics['used'] is True
    assert digest(model.export_state()) == before


def test_restore_replays_bootstrap_updates_and_predictions_exactly():
    original = trained_model()
    restored = ExperienceWorldModel(state_dim=2, action_dim=1, seed=999)
    restored.import_state(original.export_state())
    state, action = np.array([0.2, 0.3]), np.array([-0.2])
    for model in (original, restored):
        model.update(state, action, state + 0.01, -0.04)
    assert digest(original.export_state()) == digest(restored.export_state())


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_invalid_transition_does_not_mutate(value):
    model = trained_model()
    before = digest(model.export_state())
    with pytest.raises(ValueError):
        model.update(np.array([0.2, 0.3]), np.array([0.1]), np.array([0.3, 0.4]), value)
    assert digest(model.export_state()) == before


def test_sensory_encoder_uses_graph_and_donor_context_and_validates_input():
    obs = np.zeros(155, dtype=np.float32)
    first = encode_experience_state(obs, np.zeros(64))
    features = np.zeros(64); features[:32] = 0.5; features[32:48] = -0.25
    second = encode_experience_state(obs, features)
    assert second.shape == (16,)
    assert np.linalg.norm(second - first) > 0.1
    with pytest.raises(ValueError):
        encode_experience_state(np.zeros(154))


def test_shuffled_rewards_cannot_match_action_effect_predictions():
    rng = np.random.default_rng(8)
    correct = ExperienceWorldModel(state_dim=2, action_dim=1, seed=4)
    sham = ExperienceWorldModel(state_dim=2, action_dim=1, seed=4)
    states = rng.uniform(-1, 1, (180, 2)); actions = rng.uniform(-1, 1, (180, 1))
    rewards = actions[:, 0] * states[:, 0]
    shuffled = rng.permutation(rewards)
    for state, action, reward, fake in zip(states, actions, rewards, shuffled):
        correct.update(state, action, state, reward)
        sham.update(state, action, state, fake)
    truth = np.array([0.8, -0.8])
    probe = np.array([[1.0], [-1.0]])
    real_error = np.mean((correct.predict([0.8, 0.1], probe)['reward'] - truth) ** 2)
    sham_error = np.mean((sham.predict([0.8, 0.1], probe)['reward'] - truth) ** 2)
    assert real_error < 0.01
    assert sham_error > real_error + 0.1


def test_neural_context_conditions_reward_without_claiming_to_predict_its_dynamics():
    model = ExperienceWorldModel(state_dim=3, action_dim=1, context_dim=1)
    for _ in range(30):
        model.update([0.1, 0.2, 0.3], [0.2], [0.2, 0.2, 0.9], 0.1)
    prediction = model.predict([0.1, 0.2, 0.3], [[0.2]])
    assert prediction['delta'][0, 0] > 0.08
    assert prediction['delta'][0, 2] == 0.0
    assert model.summary()['predicted_state_dimensions'] == 2


def test_repeated_unexcited_learning_cannot_overflow_covariance():
    model = ExperienceWorldModel(state_dim=2, action_dim=1, forgetting=0.9)
    for _ in range(480):
        model.update([0.0, 0.0], [0.0], [0.0, 0.0], 0.0)
    state = model.export_state()
    assert np.isfinite(state['covariance']).all()
    assert np.max(np.trace(state['covariance'], axis1=1, axis2=2)) <= 10000.000001


def test_unpredictable_rewards_cannot_steer_without_beating_running_mean():
    rng = np.random.default_rng(6)
    model = ExperienceWorldModel(state_dim=2, action_dim=1)
    for _ in range(120):
        state = rng.uniform(-1, 1, 2); action = rng.uniform(-1, 1, 1)
        model.update(state, action, state, rng.normal(0, 0.1))
    action, diagnostics = model.plan([0.8, -0.3], [0.0])
    np.testing.assert_array_equal(action, [0.0])
    assert diagnostics['used'] is False
    assert model.summary()['prediction_gate_passed'] is False


@pytest.mark.parametrize('key,value', [('member_updates', [0.5, 0, 0]), ('member_updates', [1, 0, 0]), ('terminals', 1)])
def test_restore_rejects_impossible_counts_atomically(key, value):
    model = ExperienceWorldModel(state_dim=2, action_dim=1)
    before = digest(model.export_state())
    invalid = model.export_state(); invalid[key] = value
    with pytest.raises(ValueError):
        model.import_state(invalid)
    assert digest(model.export_state()) == before
