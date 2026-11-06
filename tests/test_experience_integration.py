"""Behavioral regressions for persistent transition learning and live actor wiring."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.stateio import digest


def small_agent(**kwargs):
    return IntegratedAgent(SynapseConfig(nodes_per_source=16, cognition_interval=1, **kwargs))


def test_live_graft_training_continues_after_state_restore():
    agent = small_agent()
    saved = agent.export_state()
    original_parameter = agent.brain.graft_w
    agent.import_state(saved)
    assert agent.brain.graft_w is original_parameter
    before = agent.brain.graft_w.detach().clone()
    agent.brain.r.fill_(0.2)
    observations = np.linspace(0.05, 0.95, 155, dtype=np.float32)[None]
    report = agent.prepare_observations(observations, np.full((1, 5), 0.3, np.float32))
    assert report['gradient_norms']['brain.graft_w'] > 0
    assert not torch.equal(agent.brain.graft_w, before)


def test_camera_changes_actions_and_visual_weights_roundtrip():
    agent = small_agent()
    observations = np.linspace(0.05, 0.95, 155, dtype=np.float32)
    dark = np.zeros((3, 32, 32), np.float32)
    color = np.zeros_like(dark)
    color[0, :, 16:] = 1.0
    saved = agent.export_state()
    dark_action = agent.act(observations, camera=dark)['action']
    agent.import_state(saved)
    color_action = agent.act(observations, camera=color)['action']
    assert np.linalg.norm(dark_action - color_action) > 1e-6
    agent.import_state(saved)
    with torch.no_grad():
        agent.vis_to_sensory.weight.mul_(1.7)
        agent.vision_encoder.extractor.proj.bias.add_(0.1)
    visual_state = agent.export_state()
    expected = agent.act(observations, camera=color)['action']
    with torch.no_grad():
        agent.vis_to_sensory.weight.zero_()
        agent.vision_encoder.extractor.proj.bias.zero_()
    agent.import_state(visual_state)
    assert np.array_equal(agent.act(observations, camera=color)['action'], expected)


def test_camera_preparation_updates_the_visual_path():
    agent = small_agent()
    observations = np.linspace(0.05, 0.95, 155, dtype=np.float32)[None]
    frames = np.zeros((1, 3, 32, 32), np.float32)
    frames[:, 0, :, 16:] = 1
    before = agent.vis_to_sensory.weight.detach().clone()
    report = agent.prepare_observations(observations, np.full((1, 5), 0.3, np.float32), cameras=frames)
    assert report['gradient_norms']['vis_to_sensory.weight'] > 0
    assert not torch.equal(before, agent.vis_to_sensory.weight)
    assert any(value > 0 for name, value in report['gradient_norms'].items() if name.startswith('vision_encoder.'))


@pytest.mark.parametrize('field', ['experience_planning', 'surprise_memory'])
def test_experience_flags_reject_numeric_booleans(field):
    with pytest.raises(ValueError):
        SynapseConfig(**{field: 1})


def test_runtime_transition_learning_is_explicit_and_frozen_during_query(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    assert hasattr(runtime.agent, 'experience_model')
    assert hasattr(runtime.agent, 'surprise_memory')
    before_model = digest(runtime.agent.experience_model.export_state())
    before_memory = digest(runtime.agent.surprise_memory.export_state())
    runtime.step(3, learn=False)
    assert digest(runtime.agent.experience_model.export_state()) == before_model
    assert digest(runtime.agent.surprise_memory.export_state()) == before_memory
    runtime.step(3, learn=True)
    learned_model = digest(runtime.agent.experience_model.export_state())
    learned_memory = digest(runtime.agent.surprise_memory.export_state())
    assert learned_model != before_model
    assert learned_memory != before_memory
    runtime.step(3, learn=False)
    assert digest(runtime.agent.experience_model.export_state()) == learned_model
    assert digest(runtime.agent.surprise_memory.export_state()) == learned_memory


def test_pending_transition_and_learning_resume_exactly(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    assert hasattr(runtime.agent, 'experience_model')
    runtime.step(4, learn=True)
    runtime.agent.act(runtime.obs)
    runtime.save('learned')
    expected_update = runtime.agent.observe_transition(runtime.obs, 0.25)
    expected_snapshot = runtime.step(3, learn=True)
    expected_state = digest(runtime.agent.export_state())
    runtime.load('learned')
    assert runtime.agent.observe_transition(runtime.obs, 0.25) == expected_update
    assert runtime.step(3, learn=True) == expected_snapshot
    assert digest(runtime.agent.export_state()) == expected_state


def test_planner_override_bounds_and_untrained_abstention():
    agent = small_agent()
    assert hasattr(agent, 'experience_model')
    observations = np.linspace(0.05, 0.95, 155, dtype=np.float32)
    saved = agent.export_state()
    baseline = agent.act(observations, planner_mode='off')['action']
    agent.import_state(saved)
    planned = agent.act(observations, planner_mode='model')['action']
    assert np.array_equal(planned, baseline)
    with pytest.raises(ValueError):
        agent.act(observations, planner_mode='unknown')
    agent.observe_transition(observations, 0.5)
    trained = agent.export_state()
    for mode in ('model', 'memory'):
        agent.import_state(trained)
        model_before = digest(agent.experience_model.export_state())
        memory_before = digest(agent.surprise_memory.export_state())
        action = agent.act(observations, planner_mode=mode, learn=False)['action']
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1
        agent.import_state(trained)
        reference = agent.act(observations, planner_mode='off')['action']
        assert np.max(np.abs(action - reference)) <= 0.350001
        assert digest(agent.experience_model.export_state()) == model_before
        assert digest(agent.surprise_memory.export_state()) == memory_before


def test_explicit_transition_uses_executed_action_and_rejects_invalid_values():
    agent = small_agent()
    assert hasattr(agent, 'observe_transition')
    observations = np.linspace(0.05, 0.95, 155, dtype=np.float32)
    agent.act(observations)
    saved = agent.export_state()
    override = np.array([0.8, -0.7, 0.6, -0.5, 0.4], np.float32)
    agent.observe_transition(observations, 0.7, action=override)
    override_state = digest(agent.experience_model.export_state())
    agent.import_state(saved)
    agent.observe_transition(observations, 0.7)
    assert digest(agent.experience_model.export_state()) != override_state
    agent.import_state(saved)
    before = digest(agent.export_state())
    with pytest.raises(ValueError):
        agent.observe_transition(observations, float('nan'))
    assert digest(agent.export_state()) == before
    with pytest.raises(ValueError):
        agent.observe_transition(observations, 0.7, action=np.ones(4))


def test_legacy_agent_exports_load_without_new_optional_keys():
    agent = small_agent()
    assert hasattr(agent, 'experience_model')
    legacy = deepcopy(agent.export_state())
    for name in ('experience_planning', 'surprise_memory'):
        legacy['config'].pop(name)
    for name in ('experience_model', 'surprise_memory'):
        legacy.pop(name, None)
    legacy['transient'].pop('pending_transition', None)
    legacy['modules'].pop('vision_encoder', None)
    legacy['modules'].pop('vis_to_sensory', None)
    legacy['modules'].pop('predictive_network', None)
    legacy.pop('neuromod_context', None)
    # Historical optimizer had eight tensors, before visual paths were registered.
    old_parameters = legacy['optimizer']['param_groups'][0]['params'][:8]
    legacy['optimizer']['param_groups'][0]['params'] = old_parameters
    legacy['optimizer']['state'] = {key: value for key, value in legacy['optimizer']['state'].items() if key in old_parameters}
    agent.import_state(legacy)
    assert np.isfinite(agent.act(np.zeros(155, np.float32))['action']).all()
