from copy import deepcopy
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.stateio import digest


def test_configured_compute_replaces_global_single_thread_limit(tmp_path):
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,compute_threads=2),tmp_path)
    assert torch.get_num_threads()==2
    assert runtime.snapshot()['compute']['threads']==2


def test_live_graft_learns_actual_transitions_and_query_is_frozen(tmp_path):
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True,active_training=True),tmp_path)
    assert runtime.agent.neural_graft is not None
    runtime.step(32,learn=True)
    report=runtime.agent.neural_graft.summary()
    assert report['replay_size']==32
    assert report['optimizer_updates']>0
    state=digest(runtime.agent.neural_graft.export_state())
    runtime.step(3,learn=False)
    assert digest(runtime.agent.neural_graft.export_state())==state


def test_manual_transitions_are_recorded_without_actor_credit(tmp_path):
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True),tmp_path)
    runtime.step(3,learn=True,action=np.zeros(5))
    assert runtime.agent.neural_graft.summary()['replay_size']==3
    assert runtime.agent.neural_graft.summary()['actor_samples']==0


def test_neural_graft_checkpoint_restores_live_learned_state(tmp_path):
    cfg=SynapseConfig(nodes_per_source=16,neural_graft=True,active_training=True)
    runtime=LabRuntime(cfg,tmp_path)
    runtime.step(18,learn=True)
    runtime.save('graft')
    saved=digest(runtime.agent.neural_graft.export_state())
    runtime.step(4,learn=True)
    runtime.load('graft')
    assert digest(runtime.agent.neural_graft.export_state())==saved


def test_old_config_defaults_and_new_feature_validation():
    cfg=SynapseConfig(nodes_per_source=16)
    saved=cfg.to_dict()
    for key in ('neural_graft','active_training','programmed_connectome','compute_threads','graft_batch_size','graft_updates','connectome_mode'):
        saved.pop(key)
    assert cfg.matches_saved(saved)
    with pytest.raises(ValueError): SynapseConfig(active_training=True)
    with pytest.raises(ValueError): SynapseConfig(compute_threads=0)


def test_live_training_status_and_skill_library_are_runtime_owned(tmp_path):
    from synapse.server import Workbench
    app=Workbench(LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True),tmp_path),output_root=tmp_path)
    result=app.get('/api/training/status')
    assert result['graft']['enabled']
    assert result['compute']['threads']==1
    assert app.get('/api/reusable_skills')['consumer']=='current runtime or retained omnibus library'



def test_source_backed_connectomes_and_programmed_prior_bind_cleanly(tmp_path):
    cfg=SynapseConfig(nodes_per_source=16,neural_graft=True,programmed_connectome=True,connectome_mode='prefer_authentic')
    runtime=LabRuntime(cfg,tmp_path)
    snap=runtime.snapshot()
    assert snap['training']['programmed_prior']['enabled']
    if snap['training']['programmed_prior']['available']:
        assert snap['training']['graft']['connectome']['bound']
        assert snap['training']['graft']['connectome']['nodes']<=128
    assert len(snap['graph']['sources'])==3
    assert len(snap['graph']['graft_links'])>0


def test_server_connectome_inventory_and_live_patch_screen(tmp_path):
    from synapse.server import Workbench
    from synapse.patches import make_patch
    runtime=LabRuntime(SynapseConfig(nodes_per_source=16,neural_graft=True),tmp_path)
    app=Workbench(runtime,output_root=tmp_path)
    inv=app.get('/api/connectomes/inventory')
    assert len(inv['sources'])==8
    assert len(inv['active_sources'])==3
    pkt=make_patch(runtime.agent,[{'tensor':'motor.weight','indices':[[0,0]],'values':[0.0]}],description='zero screen')
    screen=app.post('/api/patch/evaluate',{'packet':pkt,'steps':8,'seeds':[19],'min_gain':0.0})
    assert screen['evidence_kind']=='measured_world_development_screen'
    assert 'paired_returns' in screen


def test_cli_connectome_inventory_pretrain_and_live_rsi(tmp_path,capsys):
    import json
    from synapse.cli import main
    assert main(['--output',str(tmp_path),'connectome-inventory'])==0
    inv=json.loads(capsys.readouterr().out)
    assert len(inv['sources'])==8
    assert main(['--output',str(tmp_path),'--nodes','16','pretrain','--steps','8','--updates','2','--batch-size','8','--workers','1','--save','cli_pre'])==0
    pre=json.loads(capsys.readouterr().out)
    assert pre['policy_frozen'] is True
    assert pre['checkpoint']['name']=='cli_pre'
    assert main(['--output',str(tmp_path),'--nodes','16','live-rsi','--steps','8','--candidates','1','--save','cli_rsi'])==0
    rsi=json.loads(capsys.readouterr().out)
    assert rsi['candidates_evaluated']==1
    assert rsi['checkpoint']['name']=='cli_rsi'
