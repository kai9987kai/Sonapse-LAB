import os
from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime


def test_parallel_pretraining_uses_distinct_workers_and_real_transitions(tmp_path):
    from synapse.active_training import pretrain_runtime
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True),tmp_path)
    result=pretrain_runtime(runtime,steps=4,updates=1,batch_size=4,workers=2)
    assert result['recorded_transitions']==4
    assert result['workers_used']==2
    assert len(set(result['worker_pids']))==2
    assert os.getpid() not in result['worker_pids']
    assert result['applied'] and result['policy_frozen']
    assert runtime.total_steps==0
