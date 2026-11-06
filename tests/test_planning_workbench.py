import numpy as np
import pytest

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.stateio import digest


def test_preview_does_not_act_learn_or_advance_world(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    app = Workbench(runtime, output_root=tmp_path)
    before = digest({'agent': runtime.agent.export_state(), 'world': runtime._world_state(), 'obs': runtime.obs})
    preview = app.post('/api/experience/preview', {'horizon': 3, 'memory': True})
    assert preview['horizon'] == 3
    assert preview['source_step'] == 0
    assert preview['used'] is False
    assert preview['baseline'] == runtime.last_action.tolist()
    assert preview['action_order'] == ['drive', 'turn', 'reach', 'height', 'grip']
    assert digest({'agent': runtime.agent.export_state(), 'world': runtime._world_state(), 'obs': runtime.obs}) == before
    assert preview == app.post('/api/experience/preview', {'horizon': 3, 'memory': True})


@pytest.mark.parametrize('payload', [{'mode': 'model', 'horizon': 6}, {'mode': 'memory', 'horizon': True}, {'mode': 'bad', 'horizon': 3}])
def test_policy_validation_is_atomic(tmp_path, payload):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    app = Workbench(runtime, output_root=tmp_path)
    before = digest(runtime.agent.export_state())
    with pytest.raises(ValueError):
        app.post('/api/experience/mode', payload)
    assert digest(runtime.agent.export_state()) == before


def test_policy_horizon_persists_and_manual_learning_works_through_api(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    app = Workbench(runtime, output_root=tmp_path)
    state = app.post('/api/experience/mode', {'mode': 'model', 'horizon': 3})
    assert state['config']['experience_horizon'] == 3
    state = app.post('/api/step', {'steps': 1, 'learn': True, 'action': [0.5, -0.25, 0, 0, 0]})
    assert state['steps'] == 1
    assert state['experience']['model']['samples'] == 1
    np.testing.assert_allclose(state['action'][:2], [0.5, -0.25])
    runtime.save('planned')
    expected = runtime.step(1)
    runtime.load('planned')
    assert runtime.step(1) == expected


def test_json_config_roundtrip_preserves_selected_sources(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    app = Workbench(runtime, output_root=tmp_path)
    config = runtime.config.to_dict()
    state = app.post('/api/reset', {'config': config})
    assert state['config'] == config


def test_live_one_step_planner_reports_actual_model_work(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16, experience_planning=True), tmp_path)
    model = runtime.agent.experience_model
    rng = np.random.default_rng(17)
    for _ in range(160):
        state = rng.uniform(-.5, .5, 16)
        action = rng.uniform(-.8, .8, 5)
        model.update(state, action, state, float(action[0]))
    assert model.summary()['prediction_gate_passed']
    result = runtime.step(1)
    assert result['experience']['planner']['evaluations'] >= 11


def test_rollback_after_legacy_shape_mutation_restores_executing_agent(tmp_path):
    from synapse.evolution import TopologyEvolutionEngine
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    engine = TopologyEvolutionEngine()
    before = digest(runtime.agent.export_state())
    receipt = engine.add_neuron(runtime.agent)
    engine.rollback(runtime.agent, receipt.mutation_id)
    assert digest(runtime.agent.export_state()) == before
    assert runtime.agent.brain.graft_w.shape == runtime.agent.brain.graft_pre.shape
    assert len(runtime.agent.graph.graft_pre) == len(runtime.agent.brain.graft_pre)
    assert runtime.step(1)['steps'] == 1
