"""Synchronous bounded CPU laboratory with complete mid-episode restoration."""
from pathlib import Path
from copy import deepcopy
from collections import deque
from dataclasses import replace
import re
import numpy as np
import torch
from .config import ROOT
from .agent import IntegratedAgent
from .stateio import save_state,load_state
from .complex_world import ComplexWorld, ALL_TASKS as TASKS
from .actions import ACTION_NAMES, govern_action, validate_action

class LabRuntime:
    def __init__(self,config,output_root=None):
        self.config=config; self.output_root=Path(output_root or ROOT/'state')
        self.agent=IntegratedAgent(config)
        if config.task == 'world4d_tesseract_bypass':
            from .world4d import World4DEnvironment
            self.world = World4DEnvironment(seed=config.seed)
        elif config.task == 'digital_tool_arena':
            from .digital_tool_arena import DigitalToolArena
            self.world = DigitalToolArena(seed=config.seed)
        else:
            self.world=ComplexWorld(seed=config.seed,curriculum=config.task)
        from .reusable_skills import ReusableSkillLibrary
        self.skill_library=ReusableSkillLibrary(seed=config.seed)
        self._init_monitors()
        self.obs=self.world.observe(); self.total_steps=0; self.recent=[]; self.last={}; self.last_action=np.zeros(5,np.float32)

    def _init_monitors(self):
        from .plasticity_monitor import PlasticityMonitor
        from .uncertainty_calibration import TripartiteUncertaintyDecomposer
        self.plasticity_monitor=PlasticityMonitor(num_features=64,buffer_size=128)
        with torch.random.fork_rng(devices=[]):
            self.uncertainty_decomposer=TripartiteUncertaintyDecomposer(state_dim=32,action_dim=5,ensemble_size=3,seed=self.config.seed)
        self._prev_uncertainty_transition=None
        self.uncertainty_updates=0
        self.uncertainty_last_loss=None

    def _export_monitors(self):
        ood=self.uncertainty_decomposer.ood_detector
        return {
            'plasticity':{
                'activity_buffer':[x.copy() for x in self.plasticity_monitor.activity_buffer],
                'step_counter':int(self.plasticity_monitor.step_counter),
                'total_collapse_alarms':int(self.plasticity_monitor.total_collapse_alarms),
                'last_report':self.plasticity_monitor.history[-1].to_dict() if self.plasticity_monitor.history else None,
            },
            'uncertainty':{
                'ensemble':deepcopy(self.uncertainty_decomposer.ensemble.state_dict()),
                'optimizer':deepcopy(self.uncertainty_decomposer.optimizer.state_dict()),
                'ood_storage':[x.copy() for x in ood.storage],
                'ood_mean':None if ood.mean is None else ood.mean.copy(),
                'ood_inv_cov':None if ood.inv_cov is None else ood.inv_cov.copy(),
                'ood_fitted':bool(ood._fitted),
                'ood_rng':deepcopy(ood.rng.bit_generator.state),
                'calibration_scale':float(self.uncertainty_decomposer.calibration_scale),
                'prev_transition':deepcopy(self._prev_uncertainty_transition),
                'updates':int(self.uncertainty_updates),
                'last_loss':deepcopy(self.uncertainty_last_loss),
            }
        }

    def _import_monitors(self,state):
        self._init_monitors()
        if not isinstance(state,dict):
            return
        pm=state.get('plasticity')
        if isinstance(pm,dict):
            from .plasticity_monitor import PlasticityReport
            self.plasticity_monitor.activity_buffer=[np.asarray(x,dtype=np.float32).copy() for x in pm.get('activity_buffer',[])]
            self.plasticity_monitor.step_counter=int(pm.get('step_counter',0))
            self.plasticity_monitor.total_collapse_alarms=int(pm.get('total_collapse_alarms',0))
            if isinstance(pm.get('last_report'),dict):
                self.plasticity_monitor.history=[PlasticityReport(**pm['last_report'])]
        unc=state.get('uncertainty')
        if isinstance(unc,dict):
            if 'ensemble' in unc:
                self.uncertainty_decomposer.ensemble.load_state_dict(deepcopy(unc['ensemble']))
            if 'optimizer' in unc:
                self.uncertainty_decomposer.optimizer.load_state_dict(deepcopy(unc['optimizer']))
            ood=self.uncertainty_decomposer.ood_detector
            ood.storage=[np.asarray(x,dtype=np.float32).copy() for x in unc.get('ood_storage',[])]
            ood.mean=None if unc.get('ood_mean') is None else np.asarray(unc['ood_mean'],dtype=np.float32).copy()
            ood.inv_cov=None if unc.get('ood_inv_cov') is None else np.asarray(unc['ood_inv_cov'],dtype=np.float32).copy()
            ood._fitted=bool(unc.get('ood_fitted',False))
            if 'ood_rng' in unc:
                ood.rng.bit_generator.state=deepcopy(unc['ood_rng'])
            self.uncertainty_decomposer.calibration_scale=float(unc.get('calibration_scale',1.0))
            self._prev_uncertainty_transition=deepcopy(unc.get('prev_transition'))
            self.uncertainty_updates=int(unc.get('updates',0))
            self.uncertainty_last_loss=deepcopy(unc.get('last_loss'))

    def step(self,steps=1,learn=False,camera=None,action=None):
        if type(steps) is not int or not 1<=steps<=10000: raise ValueError('steps must be integer 1..10000')
        if type(learn) is not bool: raise ValueError('learn must be boolean')
        manual = None if action is None else validate_action(action)
        for _ in range(steps):
            previous_observation=self.obs.copy()
            # Manual actions still observe the current state through the actor;
            # the actual motor command is then bound before any outcome arrives.
            out=self.agent.act(self.obs,camera=camera,learn=learn,
                               planner_mode='off' if manual is not None else None)
            self.last=out['telemetry']
            proposed = out['action'] if manual is None else manual
            act=govern_action(proposed,self.agent.homeostasis.get_max_allowed_thrust())
            self.agent.pending_transition['action']=act.copy()
            self.agent.pending_transition['manual']=manual is not None
            self.last_action=act.copy()
            self.last.update(action=act.tolist(),manual_teleop=manual is not None,
                             action_source='manual' if manual is not None else 'actor',
                             action_coordinates=list(ACTION_NAMES))
            features=np.asarray(out['features'],dtype=np.float32)
            pooled=features[:32]
            if learn and self._prev_uncertainty_transition is not None:
                prev_z,prev_act,prev_rew=self._prev_uncertainty_transition
                with torch.random.fork_rng(devices=[]):
                    self.uncertainty_last_loss=self.uncertainty_decomposer.train_batch(
                        prev_z[None],prev_act[None],np.array([prev_rew],dtype=np.float32),pooled[None]
                    )
                self.uncertainty_updates+=1
            with torch.no_grad():
                unc_est=self.uncertainty_decomposer.evaluate_uncertainty(pooled,act)
            self.obs,reward,done,info=self.world.step(act)
            if hasattr(self.agent, 'homeostasis'):
                self.agent.homeostasis.step(
                    applied_action=act,
                    reward=float(reward),
                    hazard_dist=info.get('min_hazard_dist', 10.0),
                    consumed_food=float(reward) > 0.3,
                    in_friction_zone=info.get('in_friction_zone', False)
                )
            if learn:
                self.plasticity_monitor.record_activity(features,step=self.total_steps+1)
                plast_report=self.plasticity_monitor.evaluate_plasticity().to_dict()
                self.last['experience_update']=self.agent.observe_transition(self.obs,reward,done,action=act)
                self.agent.feedback(reward,done)
                self.skill_library.record_transition(previous_observation,act,reward,self.obs,self.total_steps)
                if (self.total_steps+1)%32==0: self.skill_library.discover_skills()
                self._prev_uncertainty_transition=None if done else (pooled.copy(),act.copy(),float(reward))
            else:
                # Query outcomes cannot be learned accidentally by a later call.
                self.agent.pending_transition=None
                plast_report=self.plasticity_monitor.history[-1].to_dict() if self.plasticity_monitor.history else self.plasticity_monitor.evaluate_plasticity().to_dict()
            self.last['plasticity_monitor']=plast_report
            self.last['uncertainty']={
                **unc_est.to_dict(),
                'ensemble_size':int(self.uncertainty_decomposer.ensemble_size),
                'training_updates':int(self.uncertainty_updates),
                'last_loss':deepcopy(self.uncertainty_last_loss),
                'ood_support_samples':len(self.uncertainty_decomposer.ood_detector.storage),
            }
            self.last['homeostasis']=self.agent.homeostasis.summary()
            self.total_steps+=1
            self.recent=(self.recent+[{'step':self.total_steps,'reward':float(reward),'done':bool(done)}])[-128:]
            if done:
                self.agent.reset_episode()
                self.obs=self.world.reset()
                self._prev_uncertainty_transition=None
        return self.snapshot()
    def uncertainty_status(self):
        unc=deepcopy(self.last.get('uncertainty')) if isinstance(self.last,dict) and 'uncertainty' in self.last else {
            'epistemic':0.0,'aleatoric':0.0,'ood_score':0.0,'composite_penalty':0.0,'confidence':1.0,
            'ensemble_size':int(self.uncertainty_decomposer.ensemble_size),
            'training_updates':int(self.uncertainty_updates),
            'last_loss':deepcopy(self.uncertainty_last_loss),
            'ood_support_samples':len(self.uncertainty_decomposer.ood_detector.storage),
        }
        plast=deepcopy(self.last.get('plasticity_monitor')) if isinstance(self.last,dict) and 'plasticity_monitor' in self.last else self.plasticity_monitor.evaluate_plasticity().to_dict()
        return {
            'status':'active',
            'ensemble_members':int(self.uncertainty_decomposer.ensemble_size),
            'uncertainty':unc,
            'plasticity_monitor':plast,
            'dendritic_ltc':deepcopy(self.last.get('dendritic_ltc',{'enabled':bool(self.config.dendritic_ltc)})),
            'oscillatory_coherence':deepcopy(self.last.get('oscillatory_coherence',{'enabled':bool(self.config.oscillatory_coherence)})),
            'verifiable_dpo':deepcopy(self.last.get('verifiable_dpo',{'enabled':bool(self.config.verifiable_dpo)})),
            'dvgem':deepcopy(self.last.get('dvgem',{'enabled':bool(self.config.dvgem_memory)})),
            'tether_coordinator':deepcopy(self.last.get('tether_coordinator',{'enabled':bool(self.config.tether_coordinator)})),
            'criticality_homeostat':deepcopy(self.last.get('criticality_homeostat',{'enabled':bool(self.config.criticality_homeostat)})),
            'holographic_vsa':deepcopy(self.last.get('holographic_vsa',{'enabled':bool(self.config.holographic_vsa)})),
            'sancf':deepcopy(self.last.get('sancf',{'enabled':bool(self.config.neural_comm_enabled)})),
            'overflow_mini_brain':deepcopy(self.last.get('overflow_mini_brain',{'enabled':bool(self.config.overflow_mini_brain)})),
            'event_driven':deepcopy(self.last.get('event_driven',{'enabled':bool(self.config.event_driven_backend)})),
        }
    def snapshot(self):
        b=self.agent.brain; g=self.agent.graph; r=b.r.detach().numpy()
        indices=np.linspace(0,g.n-1,min(8000,g.n)).astype(int)
        edge_indices=np.linspace(0,len(g.base_pre)-1,min(3200,len(g.base_pre))).astype(int)
        mode=('memory' if self.config.surprise_memory else 'model') if self.config.experience_planning else 'off'
        experience={'model':self.agent.experience_model.summary(),'memory':self.agent.surprise_memory.summary(),'mode':mode,'planner':deepcopy(self.last.get('experience_planner',{}))}
        result={'config':self.config.to_dict(),'steps':self.total_steps,'world':self.world.snapshot(),'action':self.last_action.tolist(),'telemetry':deepcopy(self.last),'recent':deepcopy(self.recent),'experience':experience,'graph':{'nodes':[{'id':int(i),'origin':int(g.origin[i]),'role':str(g.role[i]),'activity':float(r[i])} for i in indices],'edges':[[int(g.base_pre[i]),int(g.base_post[i]),float(g.base_w[i])] for i in edge_indices],'neurons':g.n,'graft_edges':len(g.graft_pre)},'status':{'bridge_training':'experimental supervised preparation' if self.agent.bridge_updates else 'untrained','readout':self.config.readout,'camera_live':bool(self.last.get('camera_consumed',False)),'experience_planning':self.config.experience_planning}}
        from .compute import compute_summary
        result['compute']={**compute_summary(),'threads':self.config.compute_threads}
        result['training']={'graft':self.agent.neural_graft.summary() if self.agent.neural_graft is not None else {'enabled':False},'programmed_prior':deepcopy(self.agent.programmed_prior_status)}
        result['graph']['sources']=deepcopy(self.agent.source_provenance)
        indices_set=set(indices.tolist())
        graft_rows=[i for i,(pre,post) in enumerate(zip(g.graft_pre,g.graft_post)) if pre in indices_set and post in indices_set and bool(self.agent.plasticity.active_mask[i])]
        result['graph']['graft_links']=[[int(g.graft_pre[i]),int(g.graft_post[i]),float(b.graft_w[i].detach())] for i in graft_rows[:2400]]
        if not hasattr(self, '_cached_observatory_struct') or self._cached_observatory_struct is None:
            try:
                from .observatory import NeuralObservatory
                obs = NeuralObservatory(self.agent)
                self._cached_observatory_struct = obs.get_structure_payload()
            except Exception:
                self._cached_observatory_struct = {}
        result['graph']['lod_clusters'] = self._cached_observatory_struct.get('lod_clusters', [])
        result['graph']['streamlines'] = self._cached_observatory_struct.get('streamlines', [])
        result['graph']['layout_scope']='schematic source/role layout; coordinates are not measured anatomy'
        return result
    def _path(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name): raise ValueError('checkpoint name must use 1..64 letters digits underscores or hyphens')
        return self.output_root/(name+'.pt')
    def _world_state(self):
        if not hasattr(self.world, 'history'):
            return {'task': getattr(self.world, 'snapshot', lambda: {})().get('task', 'custom'), 'rng': self.world.rng.bit_generator.state}
        d=deepcopy(self.world.__dict__); d['rng']=self.world.rng.bit_generator.state
        if hasattr(self.world, 'rng_complex'):
            d['rng_complex']=self.world.rng_complex.bit_generator.state
        d['history']={k:list(v) for k,v in d['history'].items()}
        # Explicit object references preserve mutations through st and objects.
        def encode(v):
            if isinstance(v,dict):
                for i,obj in enumerate(getattr(self.world,'objects',[])):
                    if v is obj: return {'__object_ref__':i}
                return {k:encode(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [encode(x) for x in v]
            return v
        d['st']=encode(d['st']); return d
    def save(self,name):
        path=self._path(name)
        identity=save_state(path,{'schema':1,'config':self.config.to_dict(),'agent':self.agent.export_state(),'world':self._world_state(),'obs':self.obs,'steps':self.total_steps,'recent':self.recent,'last':self.last,'last_action':self.last_action,'skill_library':self.skill_library.export_state(),'monitors':self._export_monitors()})
        return {'name':name,'digest':identity,'steps':self.total_steps,'restore_semantics':'exact mid-episode'}
    def load(self,name):
        data=load_state(self._path(name))
        if data.get('schema')!=1 or not self.config.matches_saved(data.get('config')): raise ValueError('runtime checkpoint/config mismatch')
        self.agent.import_state(data['agent']); d=data['world']
        if 'history' in d and hasattr(self.world, 'history'):
            rng=np.random.default_rng(); rng.bit_generator.state=d['rng']; d['rng']=rng
            if 'rng_complex' in d and hasattr(self.world, 'rng_complex'):
                rng_c = np.random.default_rng()
                rng_c.bit_generator.state = d['rng_complex']
                d['rng_complex'] = rng_c
            d['history']={k:deque(v,maxlen=50) for k,v in d['history'].items()}
            def decode(v):
                if isinstance(v,dict):
                    if set(v)=={'__object_ref__'}:
                        ref_idx = v['__object_ref__']
                        objs = d.get('objects', [])
                        return objs[ref_idx] if 0 <= ref_idx < len(objs) else v
                    return {k:decode(x) for k,x in v.items()}
                if isinstance(v,list): return [decode(x) for x in v]
                return v
            d['st']=decode(d['st']); self.world.__dict__.clear(); self.world.__dict__.update(d)
        elif 'rng' in d and hasattr(self.world, 'rng'):
            self.world.rng.bit_generator.state = d['rng']
        if 'skill_library' in data: self.skill_library.import_state(data['skill_library'])
        self._import_monitors(data.get('monitors'))
        self.obs=data['obs']; self.total_steps=data['steps']; self.recent=data['recent']; self.last=data['last']; self.last_action=data['last_action']
        return self.snapshot()
    def reset(self,task=None,seed=None):
        self._cached_observatory_struct = None
        cfg=replace(self.config,task=task or self.config.task,seed=self.config.seed if seed is None else seed)
        if seed is not None:
            self.config=cfg; self.agent=IntegratedAgent(cfg)
            if cfg.task == 'world4d_tesseract_bypass':
                from .world4d import World4DEnvironment
                self.world = World4DEnvironment(seed=cfg.seed)
            elif cfg.task == 'digital_tool_arena':
                from .digital_tool_arena import DigitalToolArena
                self.world = DigitalToolArena(seed=cfg.seed)
            else:
                self.world = ComplexWorld(seed=cfg.seed, curriculum=cfg.task)
            self._init_monitors()
        else:
            self.config=cfg; self.agent.config=cfg
            if cfg.task == 'world4d_tesseract_bypass':
                from .world4d import World4DEnvironment
                self.world = World4DEnvironment(seed=cfg.seed)
            elif cfg.task == 'digital_tool_arena':
                from .digital_tool_arena import DigitalToolArena
                self.world = DigitalToolArena(seed=cfg.seed)
            else:
                self.world.curriculum=cfg.task; self.world.reset(None if cfg.task=='auto' else cfg.task)
            self.agent.reset_episode()
            self._prev_uncertainty_transition=None
        self.obs=self.world.observe(); self.total_steps=0; self.recent=[]; self.last={}; self.last_action=np.zeros(5,np.float32)
        return self.snapshot()
