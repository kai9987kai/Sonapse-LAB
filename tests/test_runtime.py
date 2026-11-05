from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime

def test_runtime_replay(tmp_path):
    r=LabRuntime(SynapseConfig(nodes_per_source=16),tmp_path)
    r.step(2); r.save('test'); expected=r.step(2)
    r.load('test'); actual=r.step(2)
    assert actual==expected

def test_all_world_backends(tmp_path):
    from graftbrain.world import TASKS
    for task in TASKS:
        r=LabRuntime(SynapseConfig(nodes_per_source=16,task=task),tmp_path/task)
        r.step(1,learn=True); r.save('episode'); expected=r.step(1)
        r.load('episode'); assert r.step(1)==expected
