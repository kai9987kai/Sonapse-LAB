"""Workbench mode changes must keep accumulated learning available."""
import pytest

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.stateio import digest


def test_switching_planner_keeps_experience_and_syncs_checkpoint_configuration(tmp_path):
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), tmp_path)
    runtime.step(2, learn=True)
    app = Workbench(runtime, output_root=tmp_path)
    before = digest(runtime.agent.experience_model.export_state())
    snapshot = app.post('/api/experience/mode', {'mode': 'memory'})
    assert snapshot['experience']['mode'] == 'memory'
    assert runtime.config == runtime.agent.config
    assert digest(runtime.agent.experience_model.export_state()) == before
    runtime.save('enabled')
    assert runtime.load('enabled')['experience']['mode'] == 'memory'
    assert app.post('/api/experience/mode', {'mode': 'model'})['experience']['mode'] == 'model'
    assert app.post('/api/experience/mode', {'mode': 'off'})['experience']['mode'] == 'off'
    memory_updates = runtime.agent.surprise_memory.summary()['updates']
    runtime.step(1, learn=True)
    assert runtime.agent.surprise_memory.summary()['updates'] == memory_updates + 1
    with pytest.raises(ValueError):
        app.post('/api/experience/mode', {'mode': 'invented'})
