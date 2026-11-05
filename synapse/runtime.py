"""Synchronous bounded CPU laboratory with complete mid-episode restoration."""
from pathlib import Path
from copy import deepcopy
from collections import deque
from dataclasses import replace
import re
import numpy as np
from .config import ROOT
from .agent import IntegratedAgent
from .stateio import save_state,load_state
from graftbrain.world import World,TASKS

class LabRuntime:
    def __init__(self,config,output_root=None):
        self.config=config; self.output_root=Path(output_root or ROOT/'state')
        self.agent=IntegratedAgent(config); self.world=World(seed=config.seed,curriculum=config.task)
        self.obs=self.world.observe(); self.total_steps=0; self.recent=[]; self.last={}; self.last_action=np.zeros(5,np.float32)
    def step(self,steps=1,learn=False):
        if type(steps) is not int or not 1<=steps<=10000: raise ValueError('steps must be integer 1..10000')
        for _ in range(steps):
            out=self.agent.act(self.obs,learn=learn); self.last=out['telemetry']; self.last_action=out['action']
            self.obs,reward,done,info=self.world.step(out['action'])
            if learn: self.agent.feedback(reward,done)
            self.total_steps+=1
            self.recent=(self.recent+[{'step':self.total_steps,'reward':float(reward),'done':bool(done)}])[-128:]
            if done: self.agent.reset_episode(); self.obs=self.world.reset()
        return self.snapshot()
    def snapshot(self):
        b=self.agent.brain; g=self.agent.graph; r=b.r.detach().numpy()
        indices=np.linspace(0,g.n-1,min(256,g.n)).astype(int)
        edge_indices=np.linspace(0,len(g.base_pre)-1,min(400,len(g.base_pre))).astype(int)
        return {'config':self.config.to_dict(),'steps':self.total_steps,'world':self.world.snapshot(),'action':self.last_action.tolist(),'telemetry':deepcopy(self.last),'recent':deepcopy(self.recent),'graph':{'nodes':[{'id':int(i),'origin':int(g.origin[i]),'role':str(g.role[i]),'activity':float(r[i])} for i in indices],'edges':[[int(g.base_pre[i]),int(g.base_post[i]),float(g.base_w[i])] for i in edge_indices],'neurons':g.n,'graft_edges':len(g.graft_pre)},'status':{'bridge_training':'untrained','readout':self.config.readout,'camera_live':False}}
    def _path(self,name):
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name): raise ValueError('checkpoint name must use 1..64 letters digits underscores or hyphens')
        return self.output_root/(name+'.pt')
    def _world_state(self):
        d=deepcopy(self.world.__dict__); d['rng']=self.world.rng.bit_generator.state
        d['history']={k:list(v) for k,v in d['history'].items()}
        # Explicit object references preserve mutations through st and objects.
        def encode(v):
            if isinstance(v,dict):
                if 'id' in v and 'kind' in v: return {'__object_ref__':v['id']}
                return {k:encode(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [encode(x) for x in v]
            return v
        d['st']=encode(d['st']); return d
    def save(self,name):
        path=self._path(name)
        identity=save_state(path,{'schema':1,'config':self.config.to_dict(),'agent':self.agent.export_state(),'world':self._world_state(),'obs':self.obs,'steps':self.total_steps,'recent':self.recent,'last':self.last,'last_action':self.last_action})
        return {'name':name,'digest':identity,'steps':self.total_steps,'restore_semantics':'exact mid-episode'}
    def load(self,name):
        data=load_state(self._path(name))
        if data.get('schema')!=1 or data['config']!=self.config.to_dict(): raise ValueError('runtime checkpoint/config mismatch')
        self.agent.import_state(data['agent']); d=data['world']; rng=np.random.default_rng(); rng.bit_generator.state=d['rng']; d['rng']=rng
        d['history']={k:deque(v,maxlen=50) for k,v in d['history'].items()}
        def decode(v):
            if isinstance(v,dict):
                if set(v)=={'__object_ref__'}: return d['objects'][v['__object_ref__']]
                return {k:decode(x) for k,x in v.items()}
            if isinstance(v,list): return [decode(x) for x in v]
            return v
        d['st']=decode(d['st']); self.world.__dict__.clear(); self.world.__dict__.update(d)
        self.obs=data['obs']; self.total_steps=data['steps']; self.recent=data['recent']; self.last=data['last']; self.last_action=data['last_action']
        return self.snapshot()
    def reset(self,task=None,seed=None):
        cfg=replace(self.config,task=task or self.config.task,seed=self.config.seed if seed is None else seed)
        if seed is not None:
            self.config=cfg; self.agent=IntegratedAgent(cfg); self.world=World(seed=cfg.seed,curriculum=cfg.task)
        else:
            self.config=cfg; self.agent.config=cfg; self.world.curriculum=cfg.task; self.world.reset(None if cfg.task=='auto' else cfg.task); self.agent.reset_episode()
        self.obs=self.world.observe(); self.total_steps=0; self.recent=[]; self.last={}; self.last_action=np.zeros(5,np.float32)
        return self.snapshot()
