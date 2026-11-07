"""Pretraining consumes recorded transitions and preserves the live world."""
import numpy as np
from synapse.stateio import digest


def test_pretraining_from_executed_records_trains_auxiliary_without_policy_change():
    from synapse.neural_graft import NeuralResidualGraft
    from synapse.active_training import pretrain_from_transitions
    model = NeuralResidualGraft(hidden_dim=64)
    obs = np.zeros(155, np.float32)
    features = np.zeros(64, np.float32)
    transitions = []
    for index in range(24):
        transitions.append({'obs': obs.copy(), 'features': features.copy(),
                            'baseline': np.zeros(5, np.float32),
                            'action': np.full(5, .1, np.float32),
                            'next_obs': obs + .01, 'reward': .2, 'done': False})
    before = model.propose(obs, features, np.zeros(5, np.float32))[0]
    result = pretrain_from_transitions(model, transitions, updates=4, batch_size=24)
    assert result['recorded_transitions'] == 24
    assert result['optimizer_updates'] == 4
    assert result['actor_updates'] == 0
    np.testing.assert_array_equal(model.propose(obs, features, np.zeros(5, np.float32))[0], before)


def test_invalid_pretraining_batch_cannot_partially_add_transitions():
    from synapse.neural_graft import NeuralResidualGraft
    from synapse.active_training import pretrain_from_transitions
    model = NeuralResidualGraft(hidden_dim=64)
    before = digest(model.export_state())
    import pytest
    with pytest.raises(ValueError):
        pretrain_from_transitions(model, [{'obs': np.zeros(155)}])
    assert digest(model.export_state()) == before


def test_cloned_runtime_pretraining_keeps_live_world_and_actor_weights(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.runtime import LabRuntime
    from synapse.active_training import pretrain_runtime
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16, neural_graft=True), tmp_path)
    before_world = digest({'world': runtime._world_state(), 'obs': runtime.obs,
                           'steps': runtime.total_steps})
    before_policy = {key: value.clone() for key, value in runtime.agent.neural_graft.network.state_dict().items()
                     if key.startswith('actor_')}
    result = pretrain_runtime(runtime, steps=8, updates=2, batch_size=8)
    assert result['recorded_transitions'] == 8
    assert result['applied'] is True
    assert result['actor_updates'] == 0
    assert runtime.agent.neural_graft.summary()['replay_size'] == 8
    assert digest({'world': runtime._world_state(), 'obs': runtime.obs,
                   'steps': runtime.total_steps}) == before_world
    import torch
    for key, value in before_policy.items():
        assert torch.equal(value, runtime.agent.neural_graft.network.state_dict()[key])


def test_cancelled_pretraining_cannot_apply_a_partial_candidate(tmp_path):
    from synapse.config import SynapseConfig
    from synapse.runtime import LabRuntime
    from synapse.active_training import pretrain_runtime
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16, neural_graft=True), tmp_path)
    before = digest(runtime.agent.neural_graft.export_state())
    result = pretrain_runtime(runtime, steps=8, updates=2, batch_size=8, cancelled=lambda: True)
    assert result['status'] == 'cancelled'
    assert result['applied'] is False
    assert digest(runtime.agent.neural_graft.export_state()) == before
