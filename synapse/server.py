"""Loopback-only workbench and bounded background study jobs."""
from __future__ import annotations
import argparse
import dataclasses
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from .config import SynapseConfig, ARTIFACTS

MAX_BODY = 1024 * 1024

def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    return value

def safe_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value):
        raise ValueError('name must contain 1–64 letters, digits, underscores or hyphens')
    return value

class Workbench:
    def __init__(self,runtime=None,config=None,output_root=None):
        self.root=Path(output_root or ARTIFACTS).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        if runtime is None:
            from .runtime import LabRuntime
            runtime=LabRuntime(config or SynapseConfig(total_neurons=6500),output_root=self.root)
        self.runtime=runtime; self.lock=threading.RLock(); self.job_lock=threading.RLock()
        from .sim_worker import SimulationWorker
        self.sim_worker=SimulationWorker(self.runtime, self.lock)
        self.jobs={}; self.cancel={}; self.receipts={}; self.omnibus_runner=None; self.last_tandem_receipt=None
        jobsdir=self.root/'jobs'; jobsdir.mkdir(exist_ok=True)
        for path in jobsdir.glob('*.json'):
            try:
                job=json.loads(path.read_text()); self.jobs[job['id']]=job
                if job['status'] in ('queued','running'):
                    job.update(status='interrupted',error='Server restarted before this job completed'); self.persist(job)
            except (ValueError,KeyError,OSError): pass
    def persist(self,job):
        path=self.root/'jobs'/f"{job['id']}.json"; tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(job,allow_nan=False,default=str),encoding='utf-8'); tmp.replace(path)
    def start_study(self,config):
        if not isinstance(config,dict): raise ValueError('config must be an object')
        from .experiments import validate_config
        config=validate_config(config)
        for key,high in [('generations',20),('steps',10000),('episodes',100),('population',32),('seeds',16)]:
            if key in config:
                if key=='seeds' and isinstance(config[key],list):
                    if len(config[key])>high: raise ValueError('too many seeds')
                    for seed in config[key]: integer(seed,'seed',0,2**31-1)
                else: integer(config[key],key,1,high)
        with self.job_lock:
            if any(j['status'] in ('queued','running') for j in self.jobs.values()): raise ValueError('a study is already active')
            ident=uuid.uuid4().hex; event=threading.Event(); self.cancel[ident]=event
            job={'id':ident,'status':'queued','created':time.time(),'config':config,'progress':{},'result':None,'error':None}
            self.jobs[ident]=job; self.persist(job)
        def worker():
            def progress(value):
                with self.job_lock: job['progress']=value; self.persist(job)
            try:
                from .experiments import run_study
                with self.job_lock: job['status']='running'; self.persist(job)
                result=run_study(self.root/'studies'/ident,config,progress=progress,cancelled=event.is_set)
                with self.job_lock: job.update(status='cancelled' if event.is_set() else 'completed',result=result)
            except Exception as exc:
                with self.job_lock: job.update(status='cancelled' if event.is_set() else 'failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.job_lock: job['finished']=time.time(); self.persist(job)
        threading.Thread(target=worker,daemon=True,name=f'study-{ident[:8]}').start()
        return dict(job)
    def start_enhancement(self,kind,data):
        from copy import deepcopy
        from .stateio import digest,save_state
        steps=integer(data.get('steps',128 if kind=='pretrain' else 24),'steps',8,4096 if kind=='pretrain' else 64)
        updates=integer(data.get('updates',32),'updates',1,512)
        batch=integer(data.get('batch_size',128),'batch_size',8,1024)
        candidates=integer(data.get('candidates',2),'candidates',1,4)
        workers=integer(data.get('workers',2),'workers',1,min(4,__import__('os').cpu_count() or 1))
        with self.lock:
            if kind=='pretrain' and self.runtime.agent.neural_graft is None:
                raise ValueError('enable neural graft in training controls first')
            parent=digest(self.runtime.agent.export_state())
            clone=deepcopy(self.runtime)
        with self.job_lock:
            if any(j['status'] in ('queued','running') for j in self.jobs.values()):
                raise ValueError('a background job is already active')
            ident=uuid.uuid4().hex; event=threading.Event(); self.cancel[ident]=event
            job={'id':ident,'kind':kind,'status':'queued','created':time.time(),'progress':{},'result':None,'error':None}
            self.jobs[ident]=job; self.persist(job)
        def progress(value):
            with self.job_lock: job['progress']=value; self.persist(job)
        def worker():
            try:
                with self.job_lock: job['status']='running'; self.persist(job)
                if kind=='pretrain':
                    from .active_training import pretrain_runtime
                    result=pretrain_runtime(clone,steps=steps,updates=updates,batch_size=batch,apply=False,progress=progress,cancelled=event.is_set,workers=workers)
                    candidate=result.pop('candidate_state',None)
                else:
                    from .live_programming import run_live_rsi
                    result=run_live_rsi(clone.agent,task=data.get('task'),steps=steps,candidates=candidates,
                        seed=integer(data.get('seed',7),'seed',0,2**31-1),progress=progress,cancelled=event.is_set)
                    candidate=clone.agent.export_state()
                with self.lock:
                    applied=False
                    if not event.is_set() and candidate is not None:
                        if digest(self.runtime.agent.export_state())==parent:
                            if kind=='pretrain': self.runtime.agent.neural_graft.import_state(candidate)
                            else: self.runtime.agent.import_state(candidate)
                            applied=True
                            save_state(self.root/'training'/f'{ident}.pt',self.runtime.agent.export_state())
                        else: result['apply_reason']='live parent changed; candidate retained without application'
                        save_state(self.root/'training'/f'{ident}-candidate.pt',candidate)
                    result['applied']=applied
                with self.job_lock:
                    job.update(status='cancelled' if event.is_set() else 'completed',result=result)
            except Exception as exc:
                with self.job_lock: job.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                with self.job_lock: job['finished']=time.time(); self.persist(job)
        threading.Thread(target=worker,daemon=True,name=f'{kind}-{ident[:8]}').start()
        return deepcopy(job)

    def get(self,path):
        if path=='/api/jobs':
            with self.job_lock: return {'jobs':json.loads(json.dumps(list(self.jobs.values()),default=str))}
        with self.lock:
            if path=='/api/training/status':
                from .compute import compute_summary
                agent=self.runtime.agent
                return {'compute':{**compute_summary(),'threads':self.runtime.config.compute_threads},
                        'graft':agent.neural_graft.summary() if agent.neural_graft is not None else {'enabled':False},
                        'active_training':self.runtime.config.active_training,'programmed_prior':agent.programmed_prior_status}
            if path=='/api/connectomes/inventory':
                from .connectome_assets import get_source_inventory
                return {'sources':get_source_inventory(),'active_sources':self.runtime.agent.source_provenance}
            if path=='/api/state': return self.runtime.snapshot()
            if path=='/api/capabilities':
                from .capabilities import catalog
                return catalog()
            if path=='/api/tensors':
                return {'tensors':{k:{'shape':list(v.shape),'count':v.numel(),'min':float(v.detach().min()),'max':float(v.detach().max())} for k,v in self.runtime.agent.editable_tensors().items()}}
            if path=='/api/checkpoints':
                return {'checkpoints':sorted(p.stem for p in self.root.glob('*.pt') if p.is_file())}
            if path=='/api/observatory/structure':
                from .observatory import NeuralObservatory
                return NeuralObservatory(self.runtime.agent).get_structure_payload()
            if path=='/api/observatory/activity':
                from .observatory import NeuralObservatory
                return NeuralObservatory(self.runtime.agent).get_live_activity_payload()
            if path=='/api/connectomes':
                from .connectome_datasets import list_connectome_datasets
                return {'status': 'ok', 'total_datasets': 8, 'datasets': list_connectome_datasets()}
            if path=='/api/dormant/status':
                agent=self.runtime.agent
                return agent.dormant_lifecycle.summary() if hasattr(agent, 'dormant_lifecycle') else {'status': 'inactive'}
            if path=='/api/continual_backprop/status':
                agent=self.runtime.agent
                return agent.continual_backprop.get_metrics() if (hasattr(agent, 'continual_backprop') and agent.continual_backprop) else {'enabled': False}
            if path=='/api/fast_weight/status':
                agent=self.runtime.agent
                return agent.fast_weight_memory.get_metrics() if (hasattr(agent, 'fast_weight_memory') and agent.fast_weight_memory) else {'enabled': False}
            if path=='/api/self_model/status':
                agent=self.runtime.agent
                return agent.self_model.summary() if (hasattr(agent, 'self_model') and agent.self_model) else {'enabled': False}
            if path=='/api/uncertainty/status':
                base=self.runtime.uncertainty_status() if hasattr(self.runtime,'uncertainty_status') else {'status':'active'}
                return {**base, 'tripartite_decomposer': True}
            if path=='/api/tandem/status':
                return self.last_tandem_receipt or {'status':'idle','mode':'tandem_max_sections_1_to_5','total_neurons':self.runtime.config.total_neuron_count()}
            if path=='/api/reusable_skills':
                library=self.omnibus_runner.skill_library if self.omnibus_runner is not None else self.runtime.skill_library
                return {**library.summary(),'consumer':'current runtime or retained omnibus library'}
            if path=='/api/omnibus/status':
                if getattr(self, 'omnibus_runner', None) is not None:
                    return self.omnibus_runner.summary()
                return {'status': 'idle', 'total_neurons': self.runtime.config.total_neuron_count()}
            if path=='/api/neural-comm/status':
                agent = self.runtime.agent
                fabric = getattr(agent, 'comm_fabric', None)
                return {
                    'enabled': bool(getattr(self.runtime.config, 'neural_comm_enabled', False)),
                    'mode': getattr(self.runtime.config, 'neural_comm_mode', 'continuous'),
                    'dim': getattr(self.runtime.config, 'neural_comm_dim', 16),
                    'precision_bits': getattr(self.runtime.config, 'neural_comm_precision_bits', 4),
                    'neurowave_enabled': bool(getattr(self.runtime.config, 'neurowave_enabled', False)),
                    'telemetry': fabric.get_telemetry_dict() if fabric is not None else {},
                    'non_claim': 'Abstract digital numerical channel; not literal physical radio or biological membrane rhythm.',
                }
            if path=='/api/sim-worker/status':
                return self.sim_worker.status()
            if path=='/api/world4d/status':
                if hasattr(self.runtime.world, 'snapshot') and getattr(self.runtime.config, 'task', '') == 'world4d_tesseract_bypass':
                    return {'active': True, **self.runtime.world.snapshot()}
                return {'active': False, 'task': getattr(self.runtime.config, 'task', '')}
            if path=='/api/tool-arena/status':
                if hasattr(self.runtime.world, 'snapshot') and getattr(self.runtime.config, 'task', '') == 'digital_tool_arena':
                    return {'active': True, **self.runtime.world.snapshot()}
                return {'active': False, 'task': getattr(self.runtime.config, 'task', '')}
            if path=='/api/supermix/status':
                return {
                    'adapters': ['Poseidon', 'Expanse', 'NeuroGraft', 'Tether', 'Faultline', 'GhostInTheMachine'],
                    'status': 'integrated',
                    'safety_boundary': 'in-memory strict compliance; zero unconstrained host access',
                }
            if path=='/api/benchmark-suite/catalog':
                from .benchmark_suite import BenchmarkRegistry
                return {'benchmarks': BenchmarkRegistry().list_benchmarks()}
        raise KeyError('unknown API endpoint')
    def post(self,path,data):
        if path=='/api/training/pretrain': return self.start_enhancement('pretrain',data)
        if path=='/api/rsi/live': return self.start_enhancement('rsi',data)
        if path=='/api/studies': return self.start_study(data.get('config',{}))
        if path=='/api/jobs/cancel':
            with self.job_lock:
                ident=safe_name(data.get('id')); job=self.jobs[ident]
                if ident in self.cancel: self.cancel[ident].set(); job['cancel_requested']=True; self.persist(job)
                return dict(job)
        with self.lock:
            if path=='/api/training/config':
                active=data.get('active',self.runtime.config.active_training)
                if type(active) is not bool: raise ValueError('active must be boolean')
                programmed=data.get('programmed_connectome',self.runtime.config.programmed_connectome)
                if type(programmed) is not bool: raise ValueError('programmed_connectome must be boolean')
                dendritic=data.get('dendritic_ltc',getattr(self.runtime.config,'dendritic_ltc',False))
                if type(dendritic) is not bool: raise ValueError('dendritic_ltc must be boolean')
                oscillatory=data.get('oscillatory_coherence',getattr(self.runtime.config,'oscillatory_coherence',False))
                if type(oscillatory) is not bool: raise ValueError('oscillatory_coherence must be boolean')
                dpo=data.get('verifiable_dpo',getattr(self.runtime.config,'verifiable_dpo',False))
                if type(dpo) is not bool: raise ValueError('verifiable_dpo must be boolean')
                dvgem=data.get('dvgem_memory',getattr(self.runtime.config,'dvgem_memory',False))
                if type(dvgem) is not bool: raise ValueError('dvgem_memory must be boolean')
                tether=data.get('tether_coordinator',getattr(self.runtime.config,'tether_coordinator',False))
                if type(tether) is not bool: raise ValueError('tether_coordinator must be boolean')
                criticality=data.get('criticality_homeostat',getattr(self.runtime.config,'criticality_homeostat',False))
                if type(criticality) is not bool: raise ValueError('criticality_homeostat must be boolean')
                vsa=data.get('holographic_vsa',getattr(self.runtime.config,'holographic_vsa',False))
                if type(vsa) is not bool: raise ValueError('holographic_vsa must be boolean')
                threads=integer(data.get('threads',self.runtime.config.compute_threads),'threads',1,__import__('os').cpu_count() or 1)
                config=dataclasses.replace(self.runtime.config,neural_graft=True,active_training=active,programmed_connectome=programmed,dendritic_ltc=dendritic,oscillatory_coherence=oscillatory,verifiable_dpo=dpo,dvgem_memory=dvgem,tether_coordinator=tether,criticality_homeostat=criticality,holographic_vsa=vsa,compute_threads=threads)
                from .compute import configure_torch
                configure_torch(threads)
                self.runtime.config=config; self.runtime.agent.config=config
                self.runtime.agent._setup_neural_graft()
                if hasattr(self.runtime.agent,'_setup_frontier_extensions'):
                    self.runtime.agent._setup_frontier_extensions()
                return self.runtime.snapshot()
            if path=='/api/patch/evaluate':
                from .live_programming import screen_live_patch
                return screen_live_patch(self.runtime.agent,data.get('packet',{}),task=data.get('task'),
                    steps=integer(data.get('steps',24),'steps',8,64),seeds=data.get('seeds',[19,23]),
                    min_gain=float(data.get('min_gain',0.)))
            if path=='/api/experience/preview':
                from .experience import encode_experience_state
                from .experience_planner import ExperienceMPC
                horizon=integer(data.get('horizon',self.runtime.config.experience_horizon),'horizon',1,5)
                use_memory=data.get('memory',self.runtime.config.surprise_memory)
                if type(use_memory) is not bool: raise ValueError('memory must be boolean')
                value_guided=data.get('value_guided',getattr(self.runtime.config,'value_guided_planning',False))
                if type(value_guided) is not bool: raise ValueError('value_guided must be boolean')
                algo=str(data.get('planner_algorithm',getattr(self.runtime.config,'planner_algorithm','experience_mpc')))
                if algo not in ('experience_mpc','recurrent_latent','hybrid'):
                    raise ValueError('planner_algorithm must be experience_mpc, recurrent_latent, or hybrid')
                agent=self.runtime.agent
                state=encode_experience_state(self.runtime.obs,agent.last_features)
                baseline=self.runtime.last_action.copy()
                proposed,diagnostics=ExperienceMPC(agent.experience_model,horizon=horizon,value_guided=value_guided).plan(
                    state,baseline,memory=agent.surprise_memory if use_memory else None)
                extra_diag={}
                if algo in ('recurrent_latent','hybrid') and getattr(agent,'recurrent_planner',None) is not None:
                    import numpy as _np
                    rec_plan=agent.recurrent_planner.plan(self.runtime.obs,baseline,horizon=horizon)
                    rec_act=_np.asarray(rec_plan['action'],dtype=_np.float32)
                    extra_diag['recurrent_latent_preview']={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in rec_plan.items()}
                    if algo=='recurrent_latent' and rec_plan.get('used_plan'):
                        proposed=rec_act
                    elif algo=='hybrid':
                        proposed=_np.clip(0.55*proposed+0.45*rec_act,-1.0,1.0).astype(_np.float32)
                return {**diagnostics,**extra_diag,'planner_algorithm':algo,'horizon':horizon,'source_step':self.runtime.total_steps,
                        'baseline':baseline.tolist(),'proposed_action':proposed.tolist(),
                        'action_order':['drive','turn','reach','height','grip'],
                        'baseline_source':'repeat previous executed action; current observation with last neural context',
                        'scope':'pure model forecast; no actor/world advance; final runtime drive governor applies on execution'}
            if path=='/api/experience/mode':
                mode=data.get('mode')
                if mode not in ('off','model','memory'):
                    raise ValueError('experience mode must be off, model, or memory')
                horizon=integer(data.get('horizon',self.runtime.config.experience_horizon),'horizon',1,5)
                value_guided=data.get('value_guided',getattr(self.runtime.config,'value_guided_planning',False))
                if type(value_guided) is not bool: raise ValueError('value_guided must be boolean')
                algo=str(data.get('planner_algorithm',getattr(self.runtime.config,'planner_algorithm','experience_mpc')))
                if algo not in ('experience_mpc','recurrent_latent','hybrid'):
                    raise ValueError('planner_algorithm must be experience_mpc, recurrent_latent, or hybrid')
                config=dataclasses.replace(self.runtime.config,experience_planning=mode!='off',surprise_memory=mode!='model',experience_horizon=horizon,value_guided_planning=value_guided,planner_algorithm=algo)
                self.runtime.config=config; self.runtime.agent.config=config
                if hasattr(self.runtime.agent,'_setup_frontier_extensions'):
                    self.runtime.agent._setup_frontier_extensions()
                return self.runtime.snapshot()
            if path=='/api/omnibus/run':
                from .omnibus_mode import OmnibusRunner
                steps = integer(data.get('steps_per_task', 25), 'steps_per_task', 1, 1000)
                neurons = integer(data.get('total_neurons', self.runtime.config.total_neuron_count()), 'total_neurons', 48, 30000)
                tasks = data.get('tasks')
                self.omnibus_runner = OmnibusRunner(total_neurons=neurons, seed=self.runtime.config.seed, steps_per_task=steps, curriculum_tasks=tasks, output_dir=self.root)
                receipt = self.omnibus_runner.run_full_curriculum()
                return {'status': 'completed', 'receipt': receipt.to_dict()}
            if path=='/api/omnibus/step':
                from .omnibus_mode import OmnibusRunner
                if getattr(self, 'omnibus_runner', None) is None:
                    steps = integer(data.get('steps_per_task', 25), 'steps_per_task', 1, 1000)
                    neurons = integer(data.get('total_neurons', self.runtime.config.total_neuron_count()), 'total_neurons', 48, 30000)
                    tasks = data.get('tasks')
                    self.omnibus_runner = OmnibusRunner(total_neurons=neurons, seed=self.runtime.config.seed, steps_per_task=steps, curriculum_tasks=tasks, output_dir=self.root)
                out = self.omnibus_runner.step()
                return {'step_result': out, 'summary': self.omnibus_runner.summary()}
            if path=='/api/omnibus/reset':
                self.omnibus_runner = None
                return {'status': 'reset'}
            if path=='/api/step':
                learn=data.get('learn',False)
                if type(learn) is not bool: raise ValueError('learn must be boolean')
                act=data.get('action')
                if act is not None:
                    if not isinstance(act,(list,tuple)) or len(act)!=5:
                        raise ValueError('action must be a 5-element list')
                    act=[float(x) for x in act]
                    return self.runtime.step(integer(data.get('steps',1),'steps',1,1000),learn=learn,action=act)
                return self.runtime.step(integer(data.get('steps',1),'steps',1,1000),learn=learn)
            if path=='/api/reset':
                cfg=data.get('config')
                if cfg is not None:
                    if not isinstance(cfg,dict): raise ValueError('config must be object')
                    cfg=dict(cfg)
                    if isinstance(cfg.get('connectome_sources'),list): cfg['connectome_sources']=tuple(cfg['connectome_sources'])
                    from .runtime import LabRuntime
                    self.runtime=LabRuntime(SynapseConfig(**cfg),output_root=self.root)
                    self.receipts.clear()
                    return self.runtime.snapshot()
                return self.runtime.reset(task=data.get('task'),seed=data.get('seed'))
            if path in ('/api/save','/api/load'):
                return getattr(self.runtime,path.rsplit('/',1)[1])(safe_name(data.get('name')))
            if path=='/api/invoke':
                from .capabilities import invoke
                return invoke(data.get('id'),data.get('payload',{}),runtime=self.runtime)
            if path=='/api/patch/make':
                from .patches import make_patch
                return make_patch(self.runtime.agent,data.get('edits',[]),data.get('description',''))
            if path=='/api/patch/preview':
                from .patches import preview_patch
                return preview_patch(self.runtime.agent,data.get('packet',{}))
            if path=='/api/patch/apply':
                from .patches import apply_patch
                receipt=apply_patch(self.runtime.agent,data.get('packet',{})); ident=uuid.uuid4().hex; self.receipts[ident]=receipt
                while len(self.receipts)>8: self.receipts.pop(next(iter(self.receipts)))
                return {'receipt_id':ident,'receipt':receipt}
            if path=='/api/patch/rollback':
                from .patches import rollback
                ident=data.get('receipt_id'); receipt=self.receipts[ident]
                rollback(self.runtime.agent,receipt); del self.receipts[ident]
                return {'rolled_back':ident}
            if path=='/api/autoprogram':
                from .auto_programmer import execute_closed_loop_repair
                return execute_closed_loop_repair(
                    self.runtime.agent,
                    task=data.get('task', self.runtime.config.task),
                    num_diagnostic_episodes=integer(data.get('diagnostic_episodes', 3), 'diagnostic_episodes', 1, 10),
                    num_eval_episodes=integer(data.get('eval_episodes', 3), 'eval_episodes', 1, 10),
                    min_gain=float(data.get('min_gain', -0.05)),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/transplant':
                from .transplantation import extract_donor_subcircuit, run_transplantation_experiment
                donor = extract_donor_subcircuit(self.runtime.agent, source_task=data.get('donor_task', 'forage'))
                return run_transplantation_experiment(
                    self.runtime.agent,
                    donor=donor,
                    test_task=data.get('test_task', 'delayed_match'),
                    episodes=integer(data.get('episodes', 3), 'episodes', 1, 10),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/metamorphic':
                from .metamorphic import run_metamorphic_verification_suite
                return run_metamorphic_verification_suite(self.runtime.agent)
            if path=='/api/spectral':
                from .spectral import compute_connectome_harmonics, project_spectral_activity
                harmonics = compute_connectome_harmonics(self.runtime.agent.graph, num_modes=8)
                proj = project_spectral_activity(self.runtime.agent.brain.r, harmonics)
                return {'harmonics': harmonics['eigenvalues'], 'projection': proj}
            if path=='/api/hyperprogram':
                from .hypernetwork import WeightSpaceHypernetwork, compile_hypernetwork_patch
                hypernet = WeightSpaceHypernetwork()
                diag = {
                    'mean_return': float(data.get('return', 0.2)),
                    'total_steps': 30,
                    'mean_graph_activity': float(self.runtime.agent.brain.r.mean()),
                    'mean_approach_valence': float(self.runtime.last.get('mbon', [0, 0])[0]) if self.runtime.last.get('mbon') else 0.0,
                    'mean_avoid_valence': float(self.runtime.last.get('mbon', [0, 0])[1]) if self.runtime.last.get('mbon') else 0.0,
                    'action_mean': self.runtime.last_action.tolist(),
                    'task': self.runtime.config.task,
                }
                return compile_hypernetwork_patch(self.runtime.agent, hypernet, diag, target_surface=data.get('target', 'motor.weight'), max_edits=16)
            if path=='/api/flagship':
                from .flagship import FlagshipExperimentRunner
                runner = FlagshipExperimentRunner(seed=data.get('seed', 42))
                return runner.run_full_experiment(
                    training_steps=data.get('training_steps', 15),
                    evolution_generations=data.get('generations', 3),
                    eval_episodes=data.get('eval_episodes', 3)
                )
            if path=='/api/atlas':
                from .circuit_atlas import CausalCircuitAtlas
                return CausalCircuitAtlas(self.runtime.agent).build_full_atlas()
            if path=='/api/controlled_study':
                from .controlled_study import ControlledSixArmStudy
                study = ControlledSixArmStudy(
                    steps_per_stage=integer(data.get('steps_per_stage', 15), 'steps_per_stage', 5, 200),
                    eval_episodes_per_task=integer(data.get('eval_episodes_per_task', 1), 'eval_episodes_per_task', 1, 5),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                return study.run_study()
            if path=='/api/diffusion_dpo':
                from .tandem_max import run_diffusion_and_dpo_assay
                return run_diffusion_and_dpo_assay(
                    self.runtime,
                    horizon=integer(data.get('horizon', 5), 'horizon', 2, 8),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/dvgem_tether':
                from .tandem_max import run_dvgem_and_tether_assay
                return run_dvgem_and_tether_assay(
                    self.runtime,
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/causal_9arm':
                from .tandem_max import run_causal_9arm_and_scorecard
                return run_causal_9arm_and_scorecard(
                    self.runtime,
                    steps_per_arm=integer(data.get('steps_per_arm', 12), 'steps_per_arm', 5, 60),
                    episodes_per_arm=integer(data.get('episodes_per_arm', 1), 'episodes_per_arm', 1, 5),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/training/intensive':
                from .tandem_max import run_intensive_compute_boost
                return run_intensive_compute_boost(
                    self.runtime,
                    steps=integer(data.get('steps', 64), 'steps', 16, 1024),
                    updates=integer(data.get('updates', 24), 'updates', 4, 256),
                    batch_size=integer(data.get('batch_size', 256), 'batch_size', 16, 1024),
                    online_steps=integer(data.get('online_steps', 12), 'online_steps', 1, 128),
                    workers=data.get('workers'),
                )
            if path=='/api/criticality_vsa':
                from .tandem_max import run_criticality_and_vsa_assay
                return run_criticality_and_vsa_assay(
                    self.runtime,
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/frontier/innovations':
                from .tandem_max import run_frontier_innovations_assay
                return run_frontier_innovations_assay(
                    self.runtime,
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1),
                )
            if path=='/api/tandem/max':
                from .tandem_max import run_tandem_max_orchestration
                return run_tandem_max_orchestration(self, data)
            if path=='/api/neural-comm/config':
                enabled = data.get('enabled', self.runtime.config.neural_comm_enabled)
                mode = data.get('mode', self.runtime.config.neural_comm_mode)
                dim = integer(data.get('dim', self.runtime.config.neural_comm_dim), 'dim', 1, 128)
                bits = integer(data.get('precision_bits', self.runtime.config.neural_comm_precision_bits), 'precision_bits', 1, 32)
                neurowave = data.get('neurowave_enabled', self.runtime.config.neurowave_enabled)
                config = dataclasses.replace(
                    self.runtime.config,
                    neural_comm_enabled=bool(enabled),
                    neural_comm_mode=str(mode),
                    neural_comm_dim=dim,
                    neural_comm_precision_bits=bits,
                    neurowave_enabled=bool(neurowave)
                )
                self.runtime.config = config
                self.runtime.agent.config = config
                if hasattr(self.runtime.agent, '_setup_frontier_extensions'):
                    self.runtime.agent._setup_frontier_extensions()
                return self.runtime.snapshot()
            if path=='/api/neural-comm/bench':
                from .neural_comm_bench import SancfMatrixBenchmark
                bench = SancfMatrixBenchmark(
                    num_steps=integer(data.get('steps', 15), 'steps', 5, 50),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                matrix_res = bench.run_benchmark()
                neurowave_res = bench.run_neurowave_assay()
                return {
                    'sancf_matrix': [dataclasses.asdict(arm) for arm in matrix_res.arm_results],
                    'neurowave': [dataclasses.asdict(m) for m in neurowave_res.modes],
                    'controls': [dataclasses.asdict(c) for c in neurowave_res.controls],
                }
            if path=='/api/sim-worker/control':
                action = str(data.get('action', 'status'))
                target_sps = data.get('target_sps')
                if action == 'start':
                    self.sim_worker.start()
                elif action == 'pause':
                    self.sim_worker.pause()
                elif action == 'stop':
                    self.sim_worker.stop()
                elif action == 'set_target_sps':
                    self.sim_worker.set_target_sps(float(target_sps) if target_sps is not None else None)
                return self.sim_worker.status()
            if path=='/api/motif-discovery/run':
                from .causal_connectome_assay import CausalConnectomeAssay
                assay = CausalConnectomeAssay(
                    num_nodes=integer(data.get('nodes', 300), 'nodes', 50, 1000),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                c_res = assay.run_full_assay()
                return {
                    'status': 'completed',
                    'motifs_count': len(c_res.motifs),
                    'interventions_count': len(c_res.interventions),
                    'disclaimer': c_res.disclaimer,
                }
            if path=='/api/causal-world/run':
                from .causal_experiment_assay import CausalExperimentAssay
                cw_assay = CausalExperimentAssay(
                    state_dim=64,
                    action_dim=5,
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                cw_res = cw_assay.run_assay()
                return {
                    'status': 'completed',
                    'passed': cw_res.passed,
                    'disentanglement_score': cw_res.disentanglement_score,
                    'passive_drift_prediction_error': cw_res.passive_drift_prediction_error,
                }
            if path=='/api/relational-memory/run':
                from .memory_interference_assay import MemoryInterferenceAssay
                m_assay = MemoryInterferenceAssay(
                    dim=integer(data.get('dim', 256), 'dim', 64, 1024),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                m_res = m_assay.run_assay(max_pairs=integer(data.get('max_pairs', 8), 'max_pairs', 1, 16))
                return {
                    'status': 'completed',
                    'max_clean_capacity': m_res.max_clean_capacity,
                    'distractor_resistance': m_res.distractor_resistance_score,
                    'capacity_curve': [dataclasses.asdict(cp) for cp in m_res.capacity_curve],
                }
            if path=='/api/meta-plasticity/run':
                from .plasticity_rule_learner import PlasticityRuleLearner
                learner = PlasticityRuleLearner(seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1))
                l_res = learner.train_and_evaluate(meta_epochs=integer(data.get('epochs', 3), 'epochs', 1, 10))
                return {
                    'status': 'completed',
                    'learned_rule_score': l_res.learned_rule_score,
                    'hebbian_baseline_score': l_res.hebbian_baseline_score,
                    'passed': l_res.learned_rule_score >= l_res.hebbian_baseline_score - 0.05,
                }
            if path=='/api/verifier-rsi/run':
                from .verifier_rsi import RecursiveImprovementEngine
                rsi = RecursiveImprovementEngine(base_config=self.runtime.config.to_dict(), seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1))
                prop = rsi.propose_mutation()
                rec = rsi.evaluate_proposal(prop)
                return {
                    'status': 'completed',
                    'proposal': dataclasses.asdict(prop),
                    'receipt': dataclasses.asdict(rec),
                    'lineage': rsi.lineage.export_lineage(),
                }
            if path=='/api/event-driven/run':
                from .energy_efficiency_assay import EnergyEfficiencyAssay
                eff_assay = EnergyEfficiencyAssay(
                    num_neurons=integer(data.get('neurons', 200), 'neurons', 50, 1000),
                    num_steps=integer(data.get('steps', 10), 'steps', 5, 50),
                    seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1)
                )
                eff_res = eff_assay.run_assay()
                return {
                    'status': 'completed',
                    'trials': [dataclasses.asdict(t) for t in eff_res.trials],
                    'recommended_threshold': eff_res.recommended_threshold,
                    'summary': eff_res.summary,
                }
            if path=='/api/benchmark-suite/run':
                from .benchmark_suite import BenchmarkRunner, ResearchScorecard
                runner = BenchmarkRunner()
                b_results = runner.run_all(seed=integer(data.get('seed', 42), 'seed', 0, 2**31-1))
                scorecard = ResearchScorecard().generate_scorecard(b_results)
                return {
                    'status': 'completed',
                    'overall_pass_rate': scorecard.overall_pass_rate,
                    'scorecard_md': scorecard.summary_markdown,
                    'results_count': len(scorecard.results),
                }
            if path=='/api/ablation-engine/run':
                from .ablation_engine import FactorialAblationEngine
                ablation = FactorialAblationEngine(
                    base_config=self.runtime.config.to_dict(),
                    num_seeds=integer(data.get('seeds', 3), 'seeds', 1, 10)
                )
                a_res = ablation.run_study()
                return {
                    'status': 'completed',
                    'summary': a_res.summary_findings,
                    'conditions': [dataclasses.asdict(c) for c in a_res.conditions],
                }
            if path=='/api/checkpoint-migration/run':
                from .checkpoint_migration import CheckpointMigrationSimulator
                mig = CheckpointMigrationSimulator()
                state_snap = self.runtime.agent.export_state()
                mig_res = mig.simulate_migration(state_snap)
                return {
                    'status': 'completed',
                    'migration_report': dataclasses.asdict(mig_res),
                }
        raise KeyError('unknown API endpoint')

def create_server(host='127.0.0.1',port=8765,runtime=None,config=None,output_root=None):
    if host not in ('127.0.0.1','localhost'): raise ValueError('server must bind to IPv4 loopback')
    app=Workbench(runtime,config,output_root); web=Path(__file__).parent/'web'
    class Handler(BaseHTTPRequestHandler):
        timeout=60
        def log_message(self,*args): pass
        def send(self,status,data,content='application/json'):
            raw=json.dumps(data,allow_nan=False,default=str).encode() if content=='application/json' else data
            self.send_response(status); self.send_header('Content-Type',content); self.send_header('Content-Length',str(len(raw)))
            self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
            self.end_headers(); self.wfile.write(raw)
        def local(self):
            host=self.headers.get('Host',''); parsed=urlparse('http://'+host)
            if parsed.hostname not in ('127.0.0.1','localhost') or parsed.port != self.server.server_port: return False
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlparse(origin)
                if parsed.scheme!='http' or parsed.netloc!=host: return False
            return True
        def do_GET(self):
            try:
                if not self.local(): return self.send(403,{'error':'loopback host and same origin required'})
                path=urlparse(self.path).path
                if path.startswith('/api/'): return self.send(200,app.get(path))
                names={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path not in names: return self.send(404,{'error':'not found'})
                file=web/names[path]; self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or 'text/plain')
            except KeyError as exc: self.send(404,{'error':str(exc)})
            except Exception as exc: self.send(400,{'error':f'{type(exc).__name__}: {exc}'})
        def do_POST(self):
            try:
                if not self.local(): return self.send(403,{'error':'loopback host and same origin required'})
                if self.headers.get('Content-Type','').split(';')[0]!='application/json': return self.send(415,{'error':'application/json required'})
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY: return self.send(413,{'error':'body must be between 1 byte and 1 MiB'})
                data=json.loads(self.rfile.read(size),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if not isinstance(data,dict): raise ValueError('JSON body must be an object')
                self.send(200,app.post(urlparse(self.path).path,data))
            except KeyError as exc: self.send(404,{'error':str(exc)})
            except Exception as exc: self.send(400,{'error':f'{type(exc).__name__}: {exc}'})
    server=ThreadingHTTPServer((host,port),Handler); server.workbench=app; server.daemon_threads=True
    return server

def serve(host='127.0.0.1',port=8765,**kwargs):
    server=create_server(host,port,**kwargs)
    print(f'SYNAPSE workbench: http://{host}:{server.server_port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
