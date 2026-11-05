import numpy as np
import torch
from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent, AcquiringMushroomBody

def test_acquisition_from_neutral_and_reversal():
    torch.manual_seed(1)
    mb=AcquiringMushroomBody(4)
    cue=torch.tensor([1.,0.,0.,1.])
    mb(cue); mb.dopaminergic_update(1.)
    assert mb(cue)[0]>0
    for _ in range(8): mb(cue); mb.dopaminergic_update(-1.)
    assert mb(cue)[0]<0

def test_restore_and_inference_freeze():
    a=IntegratedAgent(SynapseConfig(nodes_per_source=16,cognition_interval=1))
    obs=np.linspace(0,1,155,dtype=np.float32)
    before={k:v.clone() for k,v in a.editable_tensors().items()}
    state=a.export_state(); first=a.act(obs); a.import_state(state); second=a.act(obs)
    assert np.array_equal(first['action'],second['action'])
    assert first['features'].shape==(64,)
    assert np.isfinite(first['action']).all()
    assert all(torch.equal(v,a.editable_tensors()[k]) for k,v in before.items())

def test_bidirectional_coupling_and_bridge_preparation():
    a=IntegratedAgent(SynapseConfig(nodes_per_source=16,cognition_interval=1))
    obs=np.linspace(0,1,155,dtype=np.float32)
    initial=a.export_state(); base=a.act(obs)
    a.import_state(initial)
    with torch.no_grad(): a.donor_to_graph.weight.zero_()
    lesion=a.act(obs)
    assert not np.array_equal(base['features'][:32],lesion['features'][:32])
    a.import_state(initial)
    a.brain.r.fill_(.8)
    changed=a.act(obs)
    assert not np.array_equal(base['features'][32:48],changed['features'][32:48])
    a.import_state(initial)
    report=a.prepare_observations(np.stack([obs,obs[::-1].copy()]),np.ones((2,5),np.float32)*.3)
    for prefix in ('encoder.','donor_to_graph.','graph_to_resource.','motor.'):
        assert any(v>0 for k,v in report['gradient_norms'].items() if k.startswith(prefix))
