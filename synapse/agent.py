"""CPU coupled graph/Tidal actor. Bridge and motor heads start untrained.
Features v1: graph 0:32, donor 32:48, memory 48:56, compass 56:64.
Resource mapping is an explicit synthetic adapter, not measured animal physiology.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import sources
from .config import ARTIFACTS
from graftbrain.brain import GraftedBrain
from graftbrain.connectomes import make_proxy
from graftbrain.graft import build
from graftbrain.neuro_modules import MushroomBody, CentralComplexCompass, CuriosityRND
from graftbrain.plasticity import GraftPlasticity, PlasticityConfig
from poseidon.core import load_checkpoint
from .neuromodulation import NeuromodulatoryEngine, NeuromodulatorState
from .astrocyte import AstrocyteNetwork
from .predictive_coding import PredictiveCodingEngine

class AcquiringMushroomBody(MushroomBody):
    def forward(self,cue_vec,adapt=True):
        out=super().forward(cue_vec,adapt=False)
        if adapt:
            raw=F.linear(cue_vec,self.w_claw)
            values,indices=torch.topk(raw,self.k_sparse)
            kc=torch.zeros_like(raw).scatter_(0,indices,F.relu(values))
            self.kc_act.copy_(kc.detach())
            # Presynaptic eligibility permits learning from neutral MBON weights.
            self.elig.mul_(.85).add_(kc.detach().expand(self.n_mbon,-1))
        return out

class IntegratedAgent:
    feature_schema={'version':1,'dimension':64,'graph':[0,32],'donor':[32,48],'memory':[48,56],'context':[56,64]}
    def __init__(self,config):
        self.config=config
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.graph=build([make_proxy(p,config.nodes_per_source,config.seed) for p in ('flywire','manc','optic-lobe')],seed=config.seed)
            self.brain=GraftedBrain(self.graph,microsteps=config.microsteps,noise=0,seed=config.seed)
            self.plasticity=GraftPlasticity(self.graph,155,PlasticityConfig(mode='adaptive'),seed=config.seed)
            self.mb=AcquiringMushroomBody(4)
            self.compass=CentralComplexCompass()
            self.curiosity=CuriosityRND(155)
            self.neuromod=NeuromodulatoryEngine()
            self.astrocytes=AstrocyteNetwork(num_domains=3)
            self.predictive=PredictiveCodingEngine()
            self.tidal,self.receipt=load_checkpoint(ARTIFACTS/'tidal_core.pt')
            h=self.tidal.config.hidden_size
            self.encoder=nn.Linear(155+16+2,len(self.brain.sensory_idx))
            self.donor_to_graph=nn.Linear(h+6,len(self.brain.sensory_idx),bias=False)
            self.graph_to_resource=nn.Linear(32,16,bias=False)
            self.motor=nn.Linear(64,5,bias=False)
            self.dormant=nn.ModuleList([nn.Linear(32,32,bias=False) for _ in range(2)])
            self.native=nn.Linear(32,32,bias=False)
            nn.init.zeros_(self.native.weight)
        for p in self.tidal.parameters(): p.requires_grad_(False)
        self.modules={'brain':self.brain,'plasticity':self.plasticity,'mb':self.mb,'curiosity':self.curiosity,'tidal':self.tidal,'encoder':self.encoder,'donor_to_graph':self.donor_to_graph,'graph_to_resource':self.graph_to_resource,'motor':self.motor,'dormant':self.dormant,'native':self.native}
        self.opt=torch.optim.Adam(list(self.encoder.parameters())+list(self.motor.parameters())+list(self.donor_to_graph.parameters())+list(self.graph_to_resource.parameters())+[self.brain.graft_w]+list(self.plasticity.router.parameters()),lr=3e-4)
        self.rng=np.random.default_rng(config.seed)
        self.preparation_updates=0; self.bridge_updates=0; self.episodic=[]; self.tool_memory=[]
        self.reset_episode()
    def reset_episode(self):
        self.brain.reset_state(); self.mb.elig.zero_(); self.mb.kc_act.zero_(); self.compass.reset()
        if hasattr(self, 'neuromod'): self.neuromod.reset()
        if hasattr(self, 'astrocytes'): self.astrocytes.reset()
        if hasattr(self, 'predictive'): self.predictive.reset()
        self.hidden=torch.zeros(1,self.tidal.config.hidden_size); self.intent=torch.zeros(6)
        self.evoked=torch.zeros(self.brain.n); self.ticks=0; self.last_compass=torch.zeros(16)
        self.last_features=np.zeros(64,np.float32); self.last_telemetry={}; self.last_obs=np.zeros(155,np.float32)
        self.resources=np.array([1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,.5,1.,0.,0.,0.,0.],np.float32)
        self.expert_usage=torch.zeros(2); self.consolidated=torch.zeros(32); self.consolidations=0
    def _pooled(self,r): return F.adaptive_avg_pool1d(r.reshape(1,1,-1),32).flatten()
    def semantic_frame(self,obs):
        # health/energy/hydration/stamina/exposure/threat/food/water/shelter,
        # weather/temperature/daylight/terrain/scent/last_intent/time.
        depth=obs[:33]; rgb=obs[33:132].reshape(3,33)
        frame=self.resources.copy()
        frame[1]=np.clip(frame[1]-.0005*(1+abs(obs[136])),0,1)
        frame[3]=np.clip(1-abs(obs[136])*.3,0,1)
        frame[5]=float(depth.max()); frame[6]=float((rgb[1]*depth).max()); frame[7]=float((rgb[2]*depth).max())
        frame[8]=float(1-depth.mean()); frame[12]=float(depth.mean()); frame[13]=float(frame[6:8].mean())
        frame[14]=float(self.intent.argmax())/5; frame[15]=float(obs[154]); self.resources=frame
        return torch.from_numpy(frame)
    def act(self,obs,camera=None,learn=False):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(155,) or not np.isfinite(obs).all(): raise ValueError('observation must be finite shape (155,)')
        if camera is not None and not np.isfinite(camera).all(): raise ValueError('camera must be finite')
        o=torch.from_numpy(obs.copy())
        with torch.no_grad():
            compass=self.compass.step(float(obs[137])*2.5,obs[:132]); self.last_compass=compass
            mbon=self.mb(o[149:153],adapt=learn)
            if self.ticks%self.config.cognition_interval==0:
                graph_context=self._pooled(self.brain.r)
                frame=(self.semantic_frame(obs)+.05*torch.tanh(self.graph_to_resource(graph_context))).clamp(0,1)
                out=self.tidal(torch.zeros(1,self.tidal.config.hash_buckets),frame[None],memory=self.hidden)
                self.hidden=out['memory'].detach(); self.intent=out['action'][0].softmax(-1).detach(); self.routing=out['routing'][0].tolist()
            current=2*torch.tanh(self.encoder(torch.cat([o,compass,mbon]))+.3*self.donor_to_graph(torch.cat([self.hidden[0],self.intent])))
            pred_info=self.predictive.step(o,self.hidden,acetylcholine=self.neuromod.state.acetylcholine if hasattr(self,'neuromod') else 0.5) if hasattr(self,'predictive') else {'steer_bias':0.0}
            gates=self.plasticity.gates(o); effective=self.plasticity.effective_weights(self.brain.graft_w,gates)
            if hasattr(self,'astrocytes'):
                effective=self.astrocytes.modulate_graft_weights(effective,torch.as_tensor(self.graph.graft_pre,dtype=torch.long),self.brain.n)
            previous=self.brain.r.clone()
            baseline=self.brain(torch.zeros_like(current),effective,adapt=False,advance=False) if self.config.readout!='absolute' else None
            r=self.brain(current,effective,adapt=learn)
            if hasattr(self,'astrocytes'):
                self.astrocytes.step(r.detach())
            if learn: self.plasticity.observe(r)
            if self.config.readout=='absolute':
                read=r; delta_norm=0.0; baseline_norm=0.0
            else:
                delta=r-baseline
                delta_norm=float(delta.norm())
                baseline_norm=float(baseline.norm())
                if self.config.readout=='persistent_evoked':
                    gate=torch.sigmoid(delta.abs().mean()*8)
                    self.evoked=.9*self.evoked+delta
                    read=gate*self.evoked+(1.-gate)*baseline
                else:
                    self.evoked=delta
                    read=self.evoked
            pooled=self._pooled(read)
            # Dormant candidates participate only when novelty recruits one; unused weights persist.
            novelty=float((r-previous).abs().mean()); expert=int(self.intent.argmax())%2
            recruited=novelty>.02
            if recruited:
                pooled=pooled+.05*torch.tanh(self.dormant[expert](pooled)); self.expert_usage[expert]+=1
            if learn: self.consolidated.mul_(.99).add_(pooled*.01)
            if learn and recruited and self.ticks%self.config.cognition_interval==0:
                self.native.weight.mul_(.995).add_(self.dormant[expert].weight*.005)
                self.consolidations+=1
            pooled=pooled+.01*self.consolidated+.05*torch.tanh(self.native(pooled))
            features=torch.cat([pooled,self.hidden[0,:16],mbon,self.intent,compass[:8]])
            # Every action is gated by actual graph activity; no action from a world oracle.
            gains=self.neuromod.compute_gains() if hasattr(self, 'neuromod') else {'motor_vigor':1.0}
            action=torch.tanh(self.motor(features))*torch.sigmoid(pooled.mean()*4)*gains['motor_vigor']
            if 'steer_bias' in pred_info and abs(pred_info['steer_bias'])>1e-4:
                action[1]=torch.clamp(action[1]+pred_info['steer_bias'],-1.0,1.0)
        bonus=self.curiosity.compute_bonus(o,train=learn)
        self.ticks+=1; self.last_obs=obs.copy(); self.last_features=features.numpy().copy()
        self.last_telemetry={'ticks':self.ticks,'graph_mean':float(r.mean()),'graph_delta':novelty,'tidal_intent':self.intent.tolist(),'tidal_routing':self.routing,'resources':self.resources.tolist(),'compass':float(self.compass.bump_angle),'mbon':mbon.tolist(),'curiosity':bonus,'recruited_expert':expert if recruited else None,'expert_usage':self.expert_usage.tolist(),'consolidations':self.consolidations,'readout':self.config.readout,'evoked_delta_norm':delta_norm,'evoked_baseline_norm':baseline_norm,'bridge_training':'experimental supervised preparation' if self.bridge_updates else 'untrained','camera_consumed':False,'neuromodulators':self.neuromod.state.to_dict() if hasattr(self, 'neuromod') else {},'astrocytes':self.astrocytes.summary() if hasattr(self, 'astrocytes') else {},'predictive_coding':self.predictive.state.to_dict() if hasattr(self, 'predictive') else {}}
        return {'action':action.numpy().copy(),'features':self.last_features.copy(),'telemetry':deepcopy(self.last_telemetry)}
    def feedback(self,reward,done=False):
        self.mb.dopaminergic_update(float(reward)); self.brain.hebbian(float(reward)); self.brain.apply_hebbian()
        if hasattr(self, 'neuromod'):
            threat=float(self.resources[5]) if hasattr(self, 'resources') and len(self.resources)>5 else 0.0
            contrast=float(np.std(self.last_obs[:33])) if len(self.last_obs)>=33 else 0.0
            depletion=float(1.0-self.resources[1]) if hasattr(self, 'resources') and len(self.resources)>1 else 0.0
            self.neuromod.step(reward=float(reward), sensory_contrast=contrast, threat_level=threat, energy_depletion=depletion)
        self.episodic=(self.episodic+[{'reward':float(reward),'intent':self.intent.tolist(),'done':bool(done)}])[-128:]
    def prepare(self,features,targets,steps=1):
        x=torch.as_tensor(features,dtype=torch.float32); y=torch.as_tensor(targets,dtype=torch.float32)
        if x.ndim!=2 or x.shape[1]!=64 or y.shape!=(len(x),5) or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('finite feature [N,64] and motor target [N,5] required')
        if not 1<=steps<=1000: raise ValueError('steps outside bound')
        for _ in range(steps):
            loss=F.mse_loss(torch.tanh(self.motor(x)),y); self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.preparation_updates+=steps
        return {'loss':float(loss.detach()),'updates':self.preparation_updates,'scope':'supervised motor readout only; semantic bridge remains untrained'}
    def prepare_observations(self,observations,targets,steps=1):
        """Supervise the actual coupled actor; caller supplies motor demonstrations.

        Tidal parameters stay frozen; gradients propagate through its computation
        to graph_to_resource. This is preparation, never an evaluation oracle.
        """
        x=torch.as_tensor(observations,dtype=torch.float32)
        y=torch.as_tensor(targets,dtype=torch.float32)
        if x.ndim!=2 or x.shape[1]!=155 or y.shape!=(len(x),5) or not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('expected finite [N,155] observations and [N,5] actions')
        if type(steps) is not int or not 1<=steps<=1000: raise ValueError('steps outside bound')
        state=self.export_state(); gradients={}
        for _ in range(steps):
            losses=[]
            self.plasticity.begin_gradient_update()
            for o,target in zip(x,y):
                preliminary=2*torch.tanh(self.encoder(torch.cat([o,self.last_compass.detach(),self.mb(o[149:153],adapt=False)])))
                preliminary_rates=self.brain(preliminary,adapt=False,advance=False)
                pooled=self._pooled(preliminary_rates)
                frame=torch.tensor(state['transient']['resources'])+.05*torch.tanh(self.graph_to_resource(pooled))
                donor=self.tidal(torch.zeros(1,self.tidal.config.hash_buckets),frame.clamp(0,1)[None])
                hidden=donor['memory'][0]; intent=donor['action'][0].softmax(-1)
                compass=self.last_compass.detach(); mbon=self.mb(o[149:153],adapt=False)
                current=2*torch.tanh(self.encoder(torch.cat([o,compass,mbon]))+.3*self.donor_to_graph(torch.cat([hidden,intent])))
                gates=self.plasticity.gates(o); weights=self.plasticity.effective_weights(self.brain.graft_w,gates)
                self.plasticity.capture_gradient(weights,gates)
                r=self.brain(current,weights,adapt=False,advance=False)
                self.plasticity.observe(r.detach())
                features=torch.cat([self._pooled(r),hidden[:16],mbon,intent,compass[:8]])
                losses.append(F.mse_loss(torch.tanh(self.motor(features))*torch.sigmoid(self._pooled(r).mean()*4),target))
            loss=torch.stack(losses).mean(); self.opt.zero_grad(); loss.backward()
            self.plasticity.finish_gradient_update()
            gradients={name:float(value.grad.norm()) for name,value in self.editable_tensors().items() if value.grad is not None}
            nn.utils.clip_grad_norm_([p for group in self.opt.param_groups for p in group['params']],1.)
            self.opt.step()
            self.plasticity.after_update(self.brain.graft_w,self.opt,self.brain.elig,self.brain.heb)
        self.preparation_updates+=steps; self.bridge_updates+=steps
        return {'loss':float(loss.detach()),'updates':self.preparation_updates,'gradient_norms':gradients,'scope':'supervised coupled actor; frozen pretrained Tidal; no performance promotion'}
    def editable_tensors(self):
        out={}
        for name,module in self.modules.items():
            if name in ('tidal','curiosity'): continue
            out.update({name+'.'+k:v for k,v in module.named_parameters()})
        out['mb.w_mbon']=self.mb.w_mbon
        return out
    def export_state(self):
        return deepcopy({'schema':1,'config':self.config.to_dict(),'modules':{k:v.state_dict() for k,v in self.modules.items()},'optimizer':self.opt.state_dict(),'curiosity_optimizer':self.curiosity.opt.state_dict(),'rng':self.rng.bit_generator.state,'transient':{k:getattr(self,k) for k in ('hidden','intent','evoked','ticks','last_compass','last_features','last_telemetry','last_obs','resources','expert_usage','consolidated','episodic','tool_memory','preparation_updates','bridge_updates','consolidations')},'compass_angle':self.compass.bump_angle,'neuromod':self.neuromod.state.to_dict() if hasattr(self, 'neuromod') else None,'astrocytes':self.astrocytes.export_state() if hasattr(self, 'astrocytes') else None,'predictive':self.predictive.export_state() if hasattr(self, 'predictive') else None,'routing':getattr(self,'routing',[])})
    def import_state(self,state):
        if state.get('schema')!=1 or state['config']!=self.config.to_dict(): raise ValueError('agent state/config mismatch')
        for k,v in state['modules'].items(): self.modules[k].load_state_dict(v)
        self.opt.load_state_dict(state['optimizer']); self.curiosity.opt.load_state_dict(state['curiosity_optimizer']); self.rng.bit_generator.state=state['rng']
        for k,v in deepcopy(state['transient']).items(): setattr(self,k,v)
        self.compass.bump_angle=state['compass_angle']; self.routing=state['routing']
        if hasattr(self, 'neuromod') and state.get('neuromod'):
            self.neuromod.state = NeuromodulatorState(**state['neuromod'])
        if hasattr(self, 'astrocytes') and state.get('astrocytes'):
            self.astrocytes.import_state(state['astrocytes'])
        if hasattr(self, 'predictive') and state.get('predictive'):
            self.predictive.import_state(state['predictive'])
    def status(self):
        return {'feature_schema':self.feature_schema,'graph':self.graph.describe(),'tidal_parameters':sum(p.numel() for p in self.tidal.parameters()),'bridge_training':'experimental supervised preparation' if self.bridge_updates else 'untrained','motor_preparation_updates':self.preparation_updates,'camera':'inactive; image not consumed','evidence_scope':'synthetic graph and heuristic resource adapter; no integrated performance claim','telemetry':deepcopy(self.last_telemetry)}
