import time
import numpy as np

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.server import Workbench


def test_actual_background_pretraining_applies_only_graft(tmp_path):
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True),tmp_path)
    app=Workbench(runtime,output_root=tmp_path)
    before=runtime.agent.motor.weight.detach().numpy().copy()
    job=app.post('/api/training/pretrain',{'steps':8,'updates':2,'batch_size':8,'workers':1})
    deadline=time.monotonic()+30
    while app.jobs[job['id']]['status'] in ('queued','running') and time.monotonic()<deadline:
        time.sleep(.02)
    final=app.jobs[job['id']]
    assert final['status']=='completed',final
    assert final['result']['applied']
    assert runtime.total_steps==0
    assert runtime.agent.neural_graft.summary()['replay_size']==8
    np.testing.assert_array_equal(runtime.agent.motor.weight.detach().numpy(),before)


def test_training_budget_configuration_preserves_old_actor(tmp_path):
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16),tmp_path)
    app=Workbench(runtime,output_root=tmp_path)
    before=runtime.agent.motor.weight.detach().clone()
    result=app.post('/api/training/config',{'threads':2,'active':True})
    assert result['config']['neural_graft'] and result['config']['active_training']
    assert result['compute']['threads']==2
    assert (before==runtime.agent.motor.weight.detach()).all()
