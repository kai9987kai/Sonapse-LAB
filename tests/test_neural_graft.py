"""Learning, intervention and restoration contracts for the residual graft."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from synapse.fast_weight_memory import UniversalFastWeightMemory
from synapse.stateio import digest


def graft(**kwargs):
    from synapse.neural_graft import NeuralResidualGraft
    return NeuralResidualGraft(hidden_dim=64, seed=11, **kwargs)


def inputs():
    return np.linspace(0, 1, 155, dtype=np.float32), np.zeros(64, np.float32), np.zeros(5, np.float32)


def record(model, count=48, manual=False):
    obs, features, baseline = inputs()
    for index in range(count):
        action, receipt = model.propose(obs, features, baseline, learn=True)
        # A controlled action-contingent reward tests learning, not environment competence.
        reward = float(action[0]) * 4
        model.observe(obs + .01 * action[0], reward, True, action, receipt=receipt, manual=manual)


def test_initial_residual_preserves_actor_and_query_is_pure():
    model = graft()
    obs, features, baseline = inputs()
    baseline[:] = [.2, -.3, .4, .9, -1]
    before = digest(model.export_state())
    action, receipt = model.propose(obs, features, baseline)
    np.testing.assert_array_equal(action, baseline)
    assert receipt['learning'] is False
    assert digest(model.export_state()) == before


def test_exploration_is_bounded_and_evaluation_does_not_consume_rng():
    model = graft()
    obs, features, baseline = inputs()
    action, receipt = model.propose(obs, features, baseline, learn=True)
    assert 0 < np.linalg.norm(action - baseline)
    assert np.max(np.abs(action - baseline)) <= .25
    saved = model.export_state()
    model.propose(obs, features, baseline, learn=False)
    assert digest(model.export_state()) == digest(saved)
    assert receipt['learning'] is True


def test_active_gradient_uses_real_reward_and_changes_residual_policy():
    model = graft()
    record(model, 96)
    before = model.propose(*inputs())[0]
    result = model.train_replay(updates=12, batch_size=96)
    after = model.propose(*inputs())[0]
    assert result['actor_updates'] > 0
    assert result['optimizer_updates'] == 12
    assert result['samples'] == 96
    assert result['cpu_seconds'] >= 0 and result['wall_seconds'] > 0
    assert after[0] > before[0] + .001


def test_manual_interventions_train_prediction_but_cannot_train_policy():
    model = graft()
    record(model, manual=True)
    before = model.propose(*inputs())[0]
    result = model.train_replay(updates=4, batch_size=48)
    np.testing.assert_array_equal(model.propose(*inputs())[0], before)
    assert result['actor_updates'] == 0
    assert result['optimizer_updates'] == 4


def test_auxiliary_pretraining_keeps_actor_exactly_unchanged_and_reduces_loss():
    model = graft()
    record(model)
    before = model.propose(*inputs())[0]
    result = model.train_replay(updates=24, batch_size=48, actor=False)
    np.testing.assert_array_equal(model.propose(*inputs())[0], before)
    assert result['actor_updates'] == 0
    assert result['final_prediction_loss'] < result['initial_prediction_loss']


def test_checkpoint_resumes_exploration_and_training_exactly():
    model = graft()
    record(model)
    model.train_replay(updates=2, batch_size=24)
    restored = graft()
    restored.import_state(model.export_state())
    for item in (model, restored):
        action, receipt = item.propose(*inputs(), learn=True)
        item.observe(inputs()[0] + .01, .2, False, action, receipt=receipt)
        item.train_replay(updates=3, batch_size=24)
    assert digest(model.export_state()) == digest(restored.export_state())


def test_invalid_outcome_is_atomic_and_receipts_cannot_be_reused():
    model = graft()
    action, receipt = model.propose(*inputs(), learn=True)
    before = digest(model.export_state())
    with pytest.raises(ValueError):
        model.observe(inputs()[0], float('nan'), False, action, receipt=receipt)
    assert digest(model.export_state()) == before
    model.observe(inputs()[0], .1, False, action, receipt=receipt)
    with pytest.raises(ValueError):
        model.observe(inputs()[0], .1, False, action, receipt=receipt)


def test_invalid_checkpoint_does_not_replace_resident_model():
    model = graft()
    state = model.export_state()
    state['network']['actor_head.weight'][0, 0] = float('nan')
    before = digest(model.export_state())
    with pytest.raises(ValueError):
        model.import_state(state)
    assert digest(model.export_state()) == before


def test_restore_keeps_registered_network_and_optimizer_tensor_bindings():
    model = graft()
    registered_network = model.network
    parameters = list(model.network.parameters())
    record(model)
    model.train_replay(updates=1, batch_size=24)
    state = model.export_state()
    model.import_state(state)
    assert model.network is registered_network
    assert all(a is b for a, b in zip(parameters, model.network.parameters()))
    assert all(any(p is live for group in model.optimizer.param_groups for live in group['params']) for p in parameters)


def test_fixed_connectome_sidecar_survives_checkpoint_and_has_causal_conditioning():
    model = graft()
    prior = dict(pre=[0, 1, 2], post=[1, 2, 3], weights=[.7, -.5, .4], n=4,
                 provenance={'source': 'fixture', 'measured': False})
    model.bind_connectome(**prior)
    # Train policy to expose the causal input path beyond its initial zero head.
    record(model)
    model.train_replay(updates=3, batch_size=48)
    state = model.export_state()
    restored = graft()
    restored.import_state(state)
    assert restored.summary()['connectome']['edges'] == 3
    np.testing.assert_array_equal(model.propose(*inputs())[0], restored.propose(*inputs())[0])
    assert digest(model.export_state()) == digest(restored.export_state())
    restored.bind_connectome(**{**prior, 'weights': [-.7, .5, -.4]})
    assert np.linalg.norm(model.propose(*inputs())[0] - restored.propose(*inputs())[0]) > 1e-7


def test_fast_weight_analytic_updates_remain_detached_and_cloneable():
    memory = UniversalFastWeightMemory()
    for _ in range(3):
        memory.inner_update(torch.ones(1, 32))
    assert not memory.M.requires_grad
    assert memory.M.grad_fn is None
    assert torch.equal(deepcopy(memory).M, memory.M)
