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
from .vision_encoder import TrainableVisualEncoder
from .experience import ExperienceWorldModel, encode_experience_state
from .surprise_memory import SurpriseEpisodicMemory
from .homeostasis import HomeostaticEngine
from .neurogenesis import ConnectomeNeurogenesisEngine
from .mimomix_reasoner import MiMoMixCognitiveCore, MiMoMixReasoningTrace
from .dormant_lifecycle import DormantLifecycleManager
from .fast_weight_memory import UniversalFastWeightMemory
from .latent_reasoner import ContinuousLatentReasoner
from .actions import TURN, govern_action
from .self_model import EfferenceCopyComparator

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
        from .compute import configure_torch
        self.compute=configure_torch(config.compute_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            sources = getattr(config, 'connectome_sources', ('flywire', 'manc', 'optic-lobe'))
            if config.connectome_mode=='proxy':
                substrates=[make_proxy(p,n,config.seed) for p,n in zip(sources,config.source_node_counts())]
            else:
                from .connectome_assets import build_runtime_connectomes
                substrates=build_runtime_connectomes(config,strict=config.connectome_mode=='strict_authentic')
            self.graph=build(substrates,seed=config.seed)
            self.source_provenance=[deepcopy(getattr(c,'provenance',{'representation':'synthetic_proxy','dataset':name}))
                                    for c,name in zip(substrates,sources)]
            self.brain=GraftedBrain(self.graph,microsteps=config.microsteps,noise=0,seed=config.seed)
            self.plasticity=GraftPlasticity(self.graph,155,PlasticityConfig(mode='adaptive'),seed=config.seed)
            self.mb=AcquiringMushroomBody(4)
            self.compass=CentralComplexCompass()
            self.curiosity=CuriosityRND(155)
            self.neuromod=NeuromodulatoryEngine()
            self.astrocytes=AstrocyteNetwork(num_domains=len(sources))
            self.predictive=PredictiveCodingEngine()
            self.vision_encoder=TrainableVisualEncoder(seed=config.seed)
            self.vis_to_sensory=nn.Linear(32,len(self.brain.sensory_idx),bias=False)
            # A small seeded projection gives camera input a measurable causal path.
            visual_generator=torch.Generator(device='cpu').manual_seed(config.seed+1907)
            nn.init.normal_(self.vis_to_sensory.weight,std=0.02,generator=visual_generator)
            self.tidal,self.receipt=load_checkpoint(ARTIFACTS/'tidal_core.pt')
            h=self.tidal.config.hidden_size
            self.encoder=nn.Linear(155+16+2,len(self.brain.sensory_idx))
            self.donor_to_graph=nn.Linear(h+6,len(self.brain.sensory_idx),bias=False)
            self.graph_to_resource=nn.Linear(32,16,bias=False)
            self.motor=nn.Linear(64,5,bias=False)
            self.dormant_lifecycle=DormantLifecycleManager(num_slots=2,feature_dim=32,seed=config.seed)
            self.dormant=self.dormant_lifecycle.modules
            self.native=nn.Linear(32,32,bias=False)
            nn.init.zeros_(self.native.weight)
            self.mimomix=MiMoMixCognitiveCore(latent_dim=64,action_dim=5)
            self.latent_reasoner=(ContinuousLatentReasoner(latent_dim=64,action_dim=5)
                                  if getattr(config,'latent_reasoning',False) else None)
        for p in self.tidal.parameters(): p.requires_grad_(False)
        self.modules={'brain':self.brain,'plasticity':self.plasticity,'mb':self.mb,'curiosity':self.curiosity,'tidal':self.tidal,'encoder':self.encoder,'donor_to_graph':self.donor_to_graph,'graph_to_resource':self.graph_to_resource,'motor':self.motor,'dormant':self.dormant,'native':self.native,'vision_encoder':self.vision_encoder,'vis_to_sensory':self.vis_to_sensory,'predictive_network':self.predictive,'mimomix':self.mimomix}
        self.opt=torch.optim.Adam(self._optimization_parameters(),lr=3e-4)
        self.rng=np.random.default_rng(config.seed)
        self.experience_model=ExperienceWorldModel(state_dim=16,action_dim=5,ensemble_size=3,seed=config.seed,context_dim=2)
        self.surprise_memory=SurpriseEpisodicMemory()
        self.homeostasis=HomeostaticEngine(seed=config.seed)
        self.neurogenesis=ConnectomeNeurogenesisEngine(num_neurons=self.brain.n,seed=config.seed)
        if getattr(config,'continual_backprop',False):
            from .continual_backprop import ContinualBackpropEngine
            self.continual_backprop=ContinualBackpropEngine(num_neurons=self.brain.n,seed=config.seed)
        else:
            self.continual_backprop=None
        self.self_model=(EfferenceCopyComparator(sensory_dim=155,action_dim=5,seed=config.seed)
                         if getattr(config,'self_model',False) else None)
        self.fast_weight_memory=(UniversalFastWeightMemory(in_dim=32,out_dim=32,seed=config.seed)
                                 if getattr(config,'fast_weight_memory',False) else None)
        self.dendritic_ltc=None
        self.oscillatory_coherence=None
        self.recurrent_planner=None
        self.diffusion_repair=None
        self.dpo_engine=None
        self._setup_frontier_extensions()
        self.preparation_updates=0; self.bridge_updates=0; self.episodic=[]; self.tool_memory=[]
        self._camera_consumed_any=False
        self.neural_graft=None
        self.programmed_prior_status={'enabled':False}
        self._setup_neural_graft()
        self.reset_episode()

    def _setup_frontier_extensions(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed + 4099)
            if getattr(self.config, 'dendritic_ltc', False) and self.dendritic_ltc is None:
                from .dendritic_ltc import DendriticLiquidSubGraft
                self.dendritic_ltc = DendriticLiquidSubGraft(seed=self.config.seed)
                self.modules['dendritic_ltc'] = self.dendritic_ltc
            if getattr(self.config, 'oscillatory_coherence', False) and self.oscillatory_coherence is None:
                from .oscillatory_coherence import OscillatoryCoherenceEngine
                sources = getattr(self.config, 'connectome_sources', ('flywire', 'manc', 'optic-lobe'))
                self.oscillatory_coherence = OscillatoryCoherenceEngine(source_names=sources)
            if getattr(self.config, 'planner_algorithm', 'experience_mpc') in ('recurrent_latent', 'hybrid') and self.recurrent_planner is None:
                from .multistep_planner import MultistepRecurrentPlanner
                self.recurrent_planner = MultistepRecurrentPlanner(default_horizon=self.config.experience_horizon, seed=self.config.seed)
            if getattr(self.config, 'planner_algorithm', 'experience_mpc') == 'hybrid' and self.diffusion_repair is None:
                from .diffusion_repair import DiffusionRepairSidecar
                self.diffusion_repair = DiffusionRepairSidecar(feature_dim=5, seed=self.config.seed)
            if getattr(self.config, 'verifiable_dpo', False) and self.dpo_engine is None:
                from .preference_dpo import MachineVerifiableDPOEngine
                self.dpo_engine = MachineVerifiableDPOEngine(obs_dim=155, action_dim=5, seed=self.config.seed)

    def _setup_neural_graft(self):
        if not self.config.neural_graft:
            return
        from .neural_graft import NeuralResidualGraft
        if self.neural_graft is None:
            self.neural_graft=NeuralResidualGraft(seed=self.config.seed)
            self.modules['neural_graft']=self.neural_graft.network
        if self.config.programmed_connectome and self.neural_graft.connectome is None:
            try:
                from .connectome_assets import load_programmed_graft
                prior=load_programmed_graft(max_nodes=128,seed=self.config.seed)
                self.neural_graft.bind_connectome(prior['pre'],prior['post'],prior['weights'],prior['n'],prior['provenance'])
                self.programmed_prior_status={'enabled':True,'available':True,'nodes':prior['n'],'provenance':prior['provenance']}
            except (FileNotFoundError,ImportError) as exc:
                self.programmed_prior_status={'enabled':True,'available':False,'reason':str(exc)}
    def _optimization_parameters(self):
        # Preserve the historical order so old optimizer moments can be migrated.
        return (list(self.encoder.parameters())+list(self.motor.parameters())+
                list(self.donor_to_graph.parameters())+list(self.graph_to_resource.parameters())+
                [self.brain.graft_w]+list(self.plasticity.router.parameters())+
                list(self.vision_encoder.parameters())+list(self.vis_to_sensory.parameters()))

    def _restore_optimizer(self, saved, rebuild=False):
        """Bind moments to live tensors, extending legacy visual-free optimizers."""
        if rebuild:
            self.opt=torch.optim.Adam(self._optimization_parameters(),lr=3e-4)
        restored=deepcopy(saved)
        current=self.opt.state_dict()
        old_groups=restored.get('param_groups', [])
        if len(old_groups)!=len(current['param_groups']):
            raise ValueError('optimizer parameter groups differ')
        for old,new in zip(old_groups,current['param_groups']):
            if len(old['params'])>len(new['params']):
                raise ValueError('optimizer has unsupported parameters')
            # Existing parameter indices precede the newly registered visual paths.
            old['params']=list(new['params'])
        self.opt.load_state_dict(restored)
        # Resized structural parameters cannot retain moments of the old shape.
        for parameter,moments in list(self.opt.state.items()):
            if any(isinstance(value,torch.Tensor) and value.ndim>0 and value.shape!=parameter.shape
                   for value in moments.values()):
                del self.opt.state[parameter]

    def reset_episode(self):
        self.brain.reset_state(); self.mb.elig.zero_(); self.mb.kc_act.zero_(); self.compass.reset()
        if hasattr(self, 'neuromod'): self.neuromod.reset()
        if hasattr(self, 'astrocytes'): self.astrocytes.reset()
        if hasattr(self, 'predictive'): self.predictive.reset()
        if getattr(self, 'dendritic_ltc', None) is not None: self.dendritic_ltc.reset_state()
        if getattr(self, 'oscillatory_coherence', None) is not None: self.oscillatory_coherence.reset()
        self._camera_consumed_any=False
        self.hidden=torch.zeros(1,self.tidal.config.hidden_size); self.intent=torch.zeros(6)
        self.evoked=torch.zeros(self.brain.n); self.ticks=0; self.last_compass=torch.zeros(16)
        self.last_features=np.zeros(64,np.float32); self.last_telemetry={}; self.last_obs=np.zeros(155,np.float32)
        self.resources=np.array([1.,1.,1.,1.,0.,0.,0.,0.,0.,0.,.5,1.,0.,0.,0.,0.],np.float32)
        self.expert_usage=torch.zeros(2); self.consolidated=torch.zeros(32); self.consolidations=0
        self.pending_transition=None
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
    def act(self,obs,camera=None,learn=False,planner_mode=None):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=(155,) or not np.isfinite(obs).all(): raise ValueError('observation must be finite shape (155,)')
        if camera is not None and not np.isfinite(camera).all(): raise ValueError('camera must be finite')
        mode=planner_mode
        if mode is None:
            mode=('memory' if self.config.surprise_memory else 'model') if self.config.experience_planning else 'off'
        if not isinstance(mode,str) or mode not in {'off','model','memory'}: raise ValueError('planner_mode must be off, model, or memory')
        
        vis_latent=None
        vis_info={}
        camera_consumed=False
        if camera is not None and hasattr(self, 'vision_encoder'):
            with torch.no_grad():
                vis_info=self.vision_encoder.encode_frame(camera)
            vis_latent=vis_info['latent']
            camera_consumed=True
            self._camera_consumed_any=True
            
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
            if vis_latent is not None and hasattr(self, 'vis_to_sensory'):
                current = current + 0.3 * torch.tanh(self.vis_to_sensory(vis_latent.to(current.device)))
            # These neural corrections have no trained objective in this actor.
            pred_info=(self.predictive.step(o,self.hidden,acetylcholine=self.neuromod.state.acetylcholine)
                       if self.config.experimental_cognition else {'steer_bias':0.0})
            gates=self.plasticity.gates(o); effective=self.plasticity.effective_weights(self.brain.graft_w,gates)
            if hasattr(self,'astrocytes'):
                effective=self.astrocytes.modulate_graft_weights(effective,torch.as_tensor(self.graph.graft_pre,dtype=torch.long),self.brain.n)
            previous=self.brain.r.clone()
            baseline=self.brain(torch.zeros_like(current),effective,adapt=False,advance=False) if self.config.readout!='absolute' else None
            r=self.brain(current,effective,adapt=learn)
            if hasattr(self,'astrocytes'):
                self.astrocytes.step(r.detach())
            if learn: self.plasticity.observe(r)
            gate_alpha=0.0
            if self.config.readout=='absolute':
                read=r; delta_norm=0.0; baseline_norm=0.0
            else:
                delta=r-baseline
                delta_norm=float(delta.norm())
                baseline_norm=float(baseline.norm())
                if self.config.readout=='persistent_evoked':
                    gate=torch.sigmoid(delta.abs().mean()*8)
                    gate_alpha=float(gate)
                    self.evoked=.9*self.evoked+delta
                    read=gate*self.evoked+(1.-gate)*baseline
                else:
                    self.evoked=delta
                    read=self.evoked
                    gate_alpha=1.0
            pooled=self._pooled(read)
            dendritic_info = None
            if getattr(self, 'dendritic_ltc', None) is not None:
                apical_ctx = torch.cat([self.hidden[0, :16], self.intent, mbon])
                ach_val = float(self.neuromod.state.acetylcholine) if hasattr(self, 'neuromod') else 0.5
                ca_val = float(self.astrocytes.domains[0].calcium) if (hasattr(self, 'astrocytes') and self.astrocytes.domains) else 0.1
                dend_delta, dendritic_info = self.dendritic_ltc.step(
                    pooled, apical_ctx, ach_precision=ach_val, astrocyte_ca=ca_val, learn=learn
                )
                pooled = pooled + 0.05 * torch.tanh(dend_delta)
            coherence_info = None
            if getattr(self, 'oscillatory_coherence', None) is not None:
                ach_val = float(self.neuromod.state.acetylcholine) if hasattr(self, 'neuromod') else 0.5
                da_val = float(self.neuromod.state.dopamine) if hasattr(self, 'neuromod') else 0.5
                surp_val = float(self.predictive.state.sensory_surprisal) if hasattr(self, 'predictive') else 0.0
                coherence_info = self.oscillatory_coherence.step(
                    r.detach().numpy(), self.graph.origin, surprisal=surp_val, ach=ach_val, dopamine=da_val, learn=learn
                )
            if self.fast_weight_memory is not None:
                fmem_out, _ = self.fast_weight_memory(pooled.unsqueeze(0))
                pooled = pooled + 0.05 * torch.tanh(fmem_out.squeeze(0))
            # Dormant candidates participate only when novelty recruits one; unused weights persist.
            novelty=float((r-previous).abs().mean()); preferred_expert=int(self.intent.argmax())%2
            if getattr(self.config, 'dormant_lifecycle', True):
                recruited, expert, gain = self.dormant_lifecycle.step_novelty(novelty, preferred_expert, learn=learn)
                if gain > 0:
                    pooled = pooled + 0.05 * gain * torch.tanh(self.dormant[expert](pooled))
                    if learn:
                        self.expert_usage[expert] += 1
                        self.dormant_lifecycle.evaluate_contribution(pooled, expert)
            else:
                recruited=novelty>.02; expert=preferred_expert
                if recruited:
                    pooled=pooled+.05*torch.tanh(self.dormant[expert](pooled))
                    if learn: self.expert_usage[expert]+=1
            if learn: self.consolidated.mul_(.99).add_(pooled*.01)
            if learn and recruited and self.ticks%self.config.cognition_interval==0:
                if getattr(self.config, 'dormant_lifecycle', True):
                    self.dormant_lifecycle.step_consolidation(self.native, recruited_slot=expert)
                else:
                    self.native.weight.mul_(.995).add_(self.dormant[expert].weight*.005)
                self.consolidations+=1
            pooled=pooled+.01*self.consolidated+.05*torch.tanh(self.native(pooled))
            features=torch.cat([pooled,self.hidden[0,:16],mbon,self.intent,compass[:8]])
            if learn and self.config.structural_learning and hasattr(self, 'neurogenesis'):
                self.neurogenesis.record_activity(r)
                if self.ticks % 16 == 0 or learn:
                    da = self.neuromod.state.dopamine if hasattr(self, 'neuromod') else 1.0
                    ca = self.astrocytes.domains[0].calcium if (hasattr(self, 'astrocytes') and self.astrocytes.domains) else 1.0
                    thr = self.homeostasis.state.threat if hasattr(self, 'homeostasis') else 0.0
                    self.neurogenesis.evaluate_structural_remodeling(self, dopamine=da, astrocyte_calcium=ca, threat=thr)
            if hasattr(self, 'homeostasis') and hasattr(self, 'neuromod'):
                da, sht, oa, ach = self.homeostasis.modulate_neuromodulators(
                    self.neuromod.state.dopamine, self.neuromod.state.serotonin,
                    self.neuromod.state.octopamine, self.neuromod.state.acetylcholine
                )
                self.neuromod.state.dopamine = da; self.neuromod.state.serotonin = sht
                self.neuromod.state.octopamine = oa; self.neuromod.state.acetylcholine = ach
            # Every action is gated by actual graph activity; no action from a world oracle.
            gains=self.neuromod.compute_gains() if hasattr(self, 'neuromod') else {'motor_vigor':1.0}
            action=torch.tanh(self.motor(features))*torch.sigmoid(pooled.mean()*4)*gains['motor_vigor']
            if 'steer_bias' in pred_info and abs(pred_info['steer_bias'])>1e-4:
                action[TURN]=torch.clamp(action[TURN]+pred_info['steer_bias'],-1.0,1.0)
            fe = float(self.predictive.state.free_energy) if hasattr(self, 'predictive') else 0.0
            surprisal = float(self.predictive.state.sensory_surprisal) if hasattr(self, 'predictive') else 0.0
            thr = float(self.homeostasis.state.threat) if hasattr(self, 'homeostasis') else 0.0
            mimomix_trace = MiMoMixReasoningTrace(system_engaged='disabled; untrained experimental correction')
            if self.config.experimental_cognition:
                mimomix_delta, mimomix_trace = self.mimomix.deliberate(latent_state=features, free_energy=fe, surprisal=surprisal, threat=thr)
                action = action + mimomix_delta
            latent_reasoner_trace = None
            if self.latent_reasoner is not None and getattr(self.config, 'latent_reasoning', False):
                if fe > 0.35 or surprisal > 0.25 or thr > 0.25:
                    refined_act, latent_reasoner_trace = self.latent_reasoner.forward_continuous_thinking(features)
                    action = 0.8 * action + 0.2 * refined_act.squeeze(0)
        bonus=self.curiosity.compute_bonus(o,train=learn)
        self.ticks+=1; self.last_obs=obs.copy(); self.last_features=features.numpy().copy()
        experience_state=encode_experience_state(obs,self.last_features)
        executed=np.clip(action.numpy(),-1.,1.).astype(np.float32,copy=True)
        graft_receipt=None
        if self.neural_graft is not None:
            executed,graft_receipt=self.neural_graft.propose(obs,self.last_features,executed,learn=learn)
        planner_info={'mode':mode,'enabled':mode!='off','horizon':self.config.experience_horizon,
                      'scope':'experimental short-horizon planning; heuristic uncertainty'}
        if mode!='off':
            from .experience_planner import ExperienceMPC
            value_guided = getattr(self.config, 'value_guided_planning', False)
            planner=ExperienceMPC(self.experience_model,horizon=self.config.experience_horizon,value_guided=value_guided)
            planned,diagnostics=planner.plan(experience_state,executed,
                memory=self.surprise_memory if mode=='memory' else None)
            candidate=np.asarray(planned,dtype=np.float32)
            if candidate.shape!=(5,) or not np.isfinite(candidate).all():
                raise ValueError('planner must return a finite five-dimensional action')
            if getattr(self.config, 'planner_algorithm', 'experience_mpc') in ('recurrent_latent', 'hybrid') and getattr(self, 'recurrent_planner', None) is not None and self.recurrent_planner.training_steps >= 4:
                rec_out = self.recurrent_planner.plan(obs, executed, horizon=self.config.experience_horizon)
                rec_act = np.asarray(rec_out['action'], dtype=np.float32)
                if getattr(self.config, 'planner_algorithm', 'experience_mpc') == 'recurrent_latent':
                    if rec_out.get('used_plan'):
                        candidate = rec_act
                    planner_info['recurrent_latent'] = {k: v for k, v in rec_out.items() if k != 'action'}
                else:
                    if rec_out.get('used_plan'):
                        candidate = 0.65 * candidate + 0.35 * rec_act
                    if getattr(self, 'diffusion_repair', None) is not None:
                        traj = np.vstack([executed[None, :], candidate[None, :], rec_act[None, :]])
                        repaired = self.diffusion_repair.repair_trajectory_plan(traj, hazard_zones=[])
                        candidate = repaired[1].astype(np.float32)
                    planner_info['hybrid_recurrent'] = {k: v for k, v in rec_out.items() if k != 'action'}
            # Enforce the conservative action trust region at the acting boundary.
            executed=np.clip(candidate,np.maximum(-1.,executed-.35),np.minimum(1.,executed+.35))
            planner_info.update(diagnostics)
            planner_info['mode']=mode
            planner_info['algorithm']=getattr(self.config, 'planner_algorithm', 'experience_mpc')
        executed=govern_action(executed,self.homeostasis.get_max_allowed_thrust())
        self.pending_transition={'state':np.array(experience_state,copy=True),'action':executed.copy(),'graft_receipt':graft_receipt}
        self.last_telemetry={'ticks':self.ticks,'graph_mean':float(r.mean()),'graph_delta':novelty,'tidal_intent':self.intent.tolist(),'tidal_routing':self.routing,'resources':self.resources.tolist(),'compass':float(self.compass.bump_angle),'mbon':mbon.tolist(),'curiosity':bonus,'recruited_expert':expert if recruited else None,'expert_usage':self.expert_usage.tolist(),'consolidations':self.consolidations,'readout':self.config.readout,'evoked_delta_norm':delta_norm,'evoked_baseline_norm':baseline_norm,'bridge_training':'experimental supervised preparation' if self.bridge_updates else 'untrained','camera_consumed':camera_consumed,'visual_latents':vis_latent.tolist() if vis_latent is not None else [],'visual_brightness':vis_info.get('mean_brightness',0.0),'visual_contrast':vis_info.get('contrast',0.0),'neuromodulators':self.neuromod.state.to_dict() if hasattr(self, 'neuromod') else {},'astrocytes':self.astrocytes.summary() if hasattr(self, 'astrocytes') else {},'predictive_coding':self.predictive.state.to_dict() if hasattr(self, 'predictive') else {}}
        self.last_telemetry['evoked_channels']={'channel_m_norm':baseline_norm,'channel_d_norm':delta_norm,'gate_alpha':gate_alpha,'persistent_norm':float(self.evoked.norm()) if hasattr(self,'evoked') else 0.0}
        self.last_telemetry['dormant_lifecycle']=self.dormant_lifecycle.summary() if hasattr(self,'dormant_lifecycle') else {}
        if self.continual_backprop is not None:
            self.last_telemetry['continual_backprop']={'enabled':True,**self.continual_backprop.get_metrics()}
        if self.fast_weight_memory is not None:
            self.last_telemetry['fast_weight_memory']={'enabled':True,**self.fast_weight_memory.get_metrics()}
        if latent_reasoner_trace is not None:
            self.last_telemetry['latent_reasoner']={'engaged':True,'recurrent_depth':latent_reasoner_trace.recurrent_depth,'ponder_cost':latent_reasoner_trace.ponder_cost,'accumulated_halt':latent_reasoner_trace.accumulated_halt}
        self.last_telemetry['experience_planner']=planner_info
        self.last_telemetry['experience_model']=self.experience_model.summary()
        self.last_telemetry['surprise_memory']=self.surprise_memory.summary()
        if hasattr(self, 'homeostasis'): self.last_telemetry['homeostasis']=self.homeostasis.summary()
        if hasattr(self, 'neurogenesis'): self.last_telemetry['neurogenesis']=self.neurogenesis.summary()
        if getattr(self, 'self_model', None) is not None: self.last_telemetry['self_model']=self.self_model.summary()
        if dendritic_info is not None: self.last_telemetry['dendritic_ltc']=dendritic_info
        if coherence_info is not None: self.last_telemetry['oscillatory_coherence']=coherence_info
        if getattr(self, 'dpo_engine', None) is not None:
            self.last_telemetry['verifiable_dpo']={'enabled':True,'pairs':len(self.dpo_engine.preference_buffer),'last_margin':getattr(self.dpo_engine,'last_margin',0.0)}
        if hasattr(self, 'mimomix'): self.last_telemetry['mimomix']={'system_engaged':mimomix_trace.system_engaged,'ponder_steps':mimomix_trace.ponder_steps,'ponder_cost':mimomix_trace.ponder_cost,'confidence':mimomix_trace.confidence,'action_delta_l2':mimomix_trace.action_delta_l2}
        return {'action':executed.copy(),'features':self.last_features.copy(),'telemetry':deepcopy(self.last_telemetry)}

    def observe_transition(self,next_obs,reward,done=False,action=None):
        """Learn once from an explicitly supplied real outcome and executed action.

        The next observation retains the current graph/donor context as conditioning
        inputs; the model does not predict neural dynamics or advance the actor
        speculatively. Explicit actions support shared/off-policy support streams.
        """
        observation=np.asarray(next_obs,dtype=np.float32)
        if observation.shape!=(155,) or not np.isfinite(observation).all():
            raise ValueError('next observation must be finite shape (155,)')
        if isinstance(reward,(bool,np.bool_)) or not np.isscalar(reward) or not np.isfinite(reward):
            raise ValueError('reward must be finite')
        if type(done) not in (bool,np.bool_): raise ValueError('done must be boolean')
        if self.pending_transition is None:
            raise RuntimeError('observe_transition requires a preceding action')
        executed=np.asarray(self.pending_transition['action'] if action is None else action,dtype=np.float32)
        if executed.shape!=(5,) or not np.isfinite(executed).all() or np.any(np.abs(executed)>1):
            raise ValueError('executed action must be finite shape (5,) within [-1,1]')
        state=np.array(self.pending_transition['state'],copy=True)
        next_state=encode_experience_state(observation,self.last_features)
        diagnostics=self.experience_model.update(state,executed,next_state,float(reward),done=bool(done))
        if self.neural_graft is not None:
            receipt=self.pending_transition.get('graft_receipt')
            if receipt is not None and receipt.get('learning'):
                diagnostics['neural_graft']=self.neural_graft.observe(observation,float(reward),bool(done),executed,
                    receipt=receipt,manual=bool(self.pending_transition.get('manual',False)))
            else:
                diagnostics['neural_graft']=self.neural_graft.record_transition(self.last_obs,self.last_features,
                    executed,executed,observation,float(reward),bool(done))
            if self.config.active_training and self.neural_graft.samples%16==0:
                fit=self.neural_graft.train_replay(updates=self.config.graft_updates,batch_size=self.config.graft_batch_size,actor=True)
                diagnostics['graft_fit']={k:v for k,v in fit.items() if k not in ('cpu_seconds','wall_seconds')}
        if self.config.surprise_memory:
            diagnostics={**diagnostics,'memory':self.surprise_memory.update(state,executed,float(reward),float(diagnostics['prediction_error']))}
        if getattr(self, 'self_model', None) is not None and getattr(self, 'last_obs', None) is not None:
            self_receipt = self.self_model.process_transition(self.last_obs, executed, observation, learn=True)
            diagnostics['self_model'] = self_receipt.to_dict()
        if self.fast_weight_memory is not None:
            pooled_f = torch.as_tensor(self.last_features[:32], dtype=torch.float32).unsqueeze(0)
            self.fast_weight_memory.inner_update(pooled_f, surprisal=float(abs(diagnostics['prediction_error'])))
        if self.continual_backprop is not None:
            self.continual_backprop.update_utility(self.brain.r.detach().numpy())
            if self.ticks % 16 == 0:
                candidates = self.continual_backprop.identify_replacement_candidates()
                if candidates:
                    self.continual_backprop.replace_neurons(self, candidates, optimizer=self.opt)
        if getattr(self, 'dendritic_ltc', None) is not None:
            diagnostics['dendritic_ltc'] = self.dendritic_ltc.update_from_outcome(float(reward), float(diagnostics.get('prediction_error', 0.0)))
        if getattr(self, 'recurrent_planner', None) is not None and getattr(self, 'last_obs', None) is not None:
            diagnostics['recurrent_planner'] = self.recurrent_planner.train_step(
                self.last_obs[None, :], executed[None, :], np.array([float(reward)], np.float32),
                observation[None, :], np.array([float(done)], np.float32)
            )
        if getattr(self, 'dpo_engine', None) is not None and getattr(self, 'last_obs', None) is not None:
            if abs(float(reward)) > 0.02:
                winner = executed if float(reward) >= 0 else np.clip(-0.5 * executed, -1.0, 1.0)
                loser = np.clip(-0.5 * executed, -1.0, 1.0) if float(reward) >= 0 else executed
                self.dpo_engine.record_preference(self.last_obs, winner, loser, 'return', abs(float(reward)))
                if len(self.dpo_engine.preference_buffer) >= 4 and self.ticks % 8 == 0:
                    with torch.random.fork_rng(devices=[]):
                        dpo_fit = self.dpo_engine.train_dpo_step(batch_size=min(8, len(self.dpo_engine.preference_buffer)))
                    self.dpo_engine.last_margin = float(dpo_fit.get('implicit_reward_margin', 0.0))
                    diagnostics['verifiable_dpo'] = dpo_fit
        self.pending_transition=None
        return diagnostics
    def feedback(self,reward,done=False):
        self.mb.dopaminergic_update(float(reward)); self.brain.hebbian(float(reward)); self.brain.apply_hebbian()
        # LabRuntime advances physiology exactly once per real transition.
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
    def prepare_observations(self,observations,targets,steps=1,cameras=None):
        """Supervise the actual coupled actor; caller supplies motor demonstrations.

        Tidal parameters stay frozen; gradients propagate through its computation
        to graph_to_resource. This is preparation, never an evaluation oracle.
        """
        x=torch.as_tensor(observations,dtype=torch.float32)
        y=torch.as_tensor(targets,dtype=torch.float32)
        if x.ndim!=2 or x.shape[1]!=155 or y.shape!=(len(x),5) or not len(x) or not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('expected finite [N,155] observations and [N,5] actions')
        if type(steps) is not int or not 1<=steps<=1000: raise ValueError('steps outside bound')
        frames=None
        if cameras is not None:
            frames=np.asarray(cameras,dtype=np.float32)
            if frames.ndim!=4 or len(frames)!=len(x) or not np.isfinite(frames).all():
                raise ValueError('cameras must contain one finite RGB frame per observation')
        state=self.export_state(); gradients={}
        for _ in range(steps):
            losses=[]
            self.plasticity.begin_gradient_update()
            for index,(o,target) in enumerate(zip(x,y)):
                preliminary=2*torch.tanh(self.encoder(torch.cat([o,self.last_compass.detach(),self.mb(o[149:153],adapt=False)])))
                preliminary_rates=self.brain(preliminary,adapt=False,advance=False)
                pooled=self._pooled(preliminary_rates)
                frame=torch.tensor(state['transient']['resources'])+.05*torch.tanh(self.graph_to_resource(pooled))
                donor=self.tidal(torch.zeros(1,self.tidal.config.hash_buckets),frame.clamp(0,1)[None])
                hidden=donor['memory'][0]; intent=donor['action'][0].softmax(-1)
                compass=self.last_compass.detach(); mbon=self.mb(o[149:153],adapt=False)
                current=2*torch.tanh(self.encoder(torch.cat([o,compass,mbon]))+.3*self.donor_to_graph(torch.cat([hidden,intent])))
                if frames is not None:
                    visual=self.vision_encoder.encode_frame(frames[index])['latent']
                    current=current+.3*torch.tanh(self.vis_to_sensory(visual))
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

    def reset_edit_moments(self,names):
        surfaces=self.editable_tensors()
        for name in names:
            parameter=surfaces[name]
            optimizer=self.neural_graft.optimizer if name.startswith('neural_graft.') and self.neural_graft is not None else self.opt
            optimizer.state.pop(parameter,None)
        if self.neural_graft is not None and any(name.startswith('neural_graft.') for name in names):
            self.neural_graft.policy_version+=1
            self.neural_graft.pending=None
    def _restore_graft_topology(self,pre,post,initial_weights):
        """Synchronize fixed-budget candidate identity with executing endpoints.

        Endpoint changes retain each slot's origin direction and site, so routing
        and fixed directional budgets remain meaningful. Vendor code is unchanged.
        """
        pre=np.asarray(pre); post=np.asarray(post); initial=np.asarray(initial_weights,dtype=np.float32)
        shape=tuple(self.brain.graft_w.shape)
        if (pre.shape!=shape or post.shape!=shape or initial.shape!=shape
                or pre.dtype.kind not in 'iu' or post.dtype.kind not in 'iu'
                or not np.isfinite(initial).all()
                or np.any(pre<0) or np.any(pre>=self.graph.n)
                or np.any(post<0) or np.any(post>=self.graph.n)):
            raise ValueError('invalid graft topology')
        graph=deepcopy(self.graph)
        graph.graft_pre=pre.astype(np.int64,copy=True)
        graph.graft_post=post.astype(np.int64,copy=True)
        graph.graft_w0=initial.copy()
        template=GraftPlasticity(graph,155,self.plasticity.config,seed=self.config.seed)
        if (not torch.equal(template.bucket_ids,self.plasticity.bucket_ids)
                or not torch.equal(template.bucket_keys,self.plasticity.bucket_keys)
                or not torch.equal(template.bucket_budgets,self.plasticity.bucket_budgets)):
            raise ValueError('rewiring must preserve directional candidate budgets')
        # Keep module/Parameter objects and optimizer bindings intact.
        with torch.no_grad():
            self.brain.graft_pre.copy_(template.candidate_pre)
            self.brain.graft_post.copy_(template.candidate_post)
            for name in ('candidate_pre','candidate_post','original_weights'):
                getattr(self.plasticity,name).copy_(getattr(template,name))
        self.plasticity._candidate_digest=template._candidate_digest
        self.graph.graft_pre=graph.graft_pre
        self.graph.graft_post=graph.graft_post
        self.graph.graft_w0=graph.graft_w0

    def rewire_graft_slots(self,pre,post,weights,slots):
        """Commit neurogenesis to live topology and reset replaced-slot credit."""
        slots=np.asarray(slots,dtype=np.int64)
        values=np.asarray(weights,dtype=np.float32)
        if (slots.ndim!=1 or np.any(slots<0) or np.any(slots>=len(self.brain.graft_w))
                or values.shape!=tuple(self.brain.graft_w.shape) or not np.isfinite(values).all()):
            raise ValueError('invalid graft slot replacement')
        initial=self.graph.graft_w0.copy()
        initial[slots]=values[slots]
        self._restore_graft_topology(pre,post,initial)
        indices=torch.as_tensor(slots,dtype=torch.long)
        with torch.no_grad():
            self.brain.graft_w.copy_(torch.from_numpy(values))
            for tensor in (self.brain.elig,self.brain.heb,self.plasticity.coactivity_ema,
                           self.plasticity.gradient_ema,self.plasticity.gradient_evidence_count,
                           self.plasticity.edge_age,self.plasticity._pending_gradient):
                tensor[indices]=0
            for value in self.opt.state.get(self.brain.graft_w,{}).values():
                if isinstance(value,torch.Tensor) and value.shape==self.brain.graft_w.shape:
                    value[indices]=0

    def export_state(self):
        state=deepcopy({'schema':1,'config':self.config.to_dict(),'topology':{'pre':self.brain.graft_pre.numpy(),'post':self.brain.graft_post.numpy(),'initial_weights':self.plasticity.original_weights.numpy()},'modules':{k:v.state_dict() for k,v in self.modules.items()},'optimizer':self.opt.state_dict(),'curiosity_optimizer':self.curiosity.opt.state_dict(),'rng':self.rng.bit_generator.state,'transient':{k:getattr(self,k) for k in ('hidden','intent','evoked','ticks','last_compass','last_features','last_telemetry','last_obs','resources','expert_usage','consolidated','episodic','tool_memory','preparation_updates','bridge_updates','consolidations','pending_transition','_camera_consumed_any')},'compass_angle':self.compass.bump_angle,'neuromod':self.neuromod.state.to_dict() if hasattr(self, 'neuromod') else None,'neuromod_context':{'v_predicted':self.neuromod.v_predicted,'history':self.neuromod.history},'astrocytes':self.astrocytes.export_state() if hasattr(self, 'astrocytes') else None,'predictive':self.predictive.export_state() if hasattr(self, 'predictive') else None,'routing':getattr(self,'routing',[]),'experience_model':self.experience_model.export_state(),'surprise_memory':self.surprise_memory.export_state(),'homeostasis':self.homeostasis.export_state() if hasattr(self,'homeostasis') else None,'neurogenesis':self.neurogenesis.export_state() if hasattr(self,'neurogenesis') else None,'dormant_lifecycle':self.dormant_lifecycle.export_state() if hasattr(self,'dormant_lifecycle') else None,'continual_backprop':self.continual_backprop.export_state() if (hasattr(self,'continual_backprop') and self.continual_backprop is not None) else None,'fast_weight_memory':self.fast_weight_memory.export_state() if (hasattr(self,'fast_weight_memory') and self.fast_weight_memory is not None) else None,'latent_reasoner':self.latent_reasoner.state_dict() if (hasattr(self,'latent_reasoner') and self.latent_reasoner is not None) else None,'self_model':self.self_model.export_state() if (hasattr(self,'self_model') and self.self_model is not None) else None})
        state['neural_graft']=self.neural_graft.export_state() if self.neural_graft is not None else None
        state['dendritic_ltc']=self.dendritic_ltc.export_state() if getattr(self,'dendritic_ltc',None) is not None else None
        state['oscillatory_coherence']=self.oscillatory_coherence.export_state() if getattr(self,'oscillatory_coherence',None) is not None else None
        state['recurrent_planner']=self.recurrent_planner.export_state() if getattr(self,'recurrent_planner',None) is not None else None
        state['dpo_engine']=self.dpo_engine.export_state() if getattr(self,'dpo_engine',None) is not None else None
        state['programmed_prior_status']=deepcopy(self.programmed_prior_status)
        state['source_provenance']=deepcopy(self.source_provenance)
        return state
    def import_state(self,state):
        backup=self.export_state()
        try:
            self._import_state(state)
        except Exception:
            self._import_state(backup)
            raise

    def _import_state(self,state):
        if state.get('schema')!=1 or not self.config.matches_saved(state.get('config')): raise ValueError('agent state/config mismatch')
        brain_state=state['modules']['brain']
        plasticity_state=state['modules'].get('plasticity',{})
        topology=state.get('topology',{'pre':brain_state['graft_pre'].numpy(),
            'post':brain_state['graft_post'].numpy(),
            'initial_weights':plasticity_state.get('original_weights',torch.from_numpy(self.graph.graft_w0)).numpy()})
        if (not np.array_equal(topology['pre'],brain_state['graft_pre'].numpy())
                or not np.array_equal(topology['post'],brain_state['graft_post'].numpy())):
            raise ValueError('checkpoint topology disagrees with executing endpoints')
        rebuild_optimizer=self.brain.graft_w.shape!=brain_state['graft_w'].shape
        if rebuild_optimizer:
            self.brain.graft_w=nn.Parameter(torch.zeros_like(brain_state['graft_w']))
        for name in ('graft_pre','graft_post'):
            if getattr(self.brain,name).shape!=brain_state[name].shape:
                self.brain.register_buffer(name,torch.zeros_like(brain_state[name]))
        self._restore_graft_topology(topology['pre'],topology['post'],topology['initial_weights'])
        for k,v in state['modules'].items():
            if k == 'brain':
                for buf_name in ('r', 'activity_ema', 'elig', 'heb', 'graft_pre', 'graft_post', 'base_pre', 'base_post', 'base_w', 'graft_site', 'bias', 'sensory_idx', 'motor_idx', 'probe_idx'):
                    if buf_name in v and getattr(self.brain,buf_name).shape!=v[buf_name].shape:
                        self.brain.register_buffer(buf_name, torch.zeros_like(v[buf_name]))
                if 'graft_w' in v and self.brain.graft_w.shape!=v['graft_w'].shape:
                    self.brain.graft_w = nn.Parameter(torch.zeros_like(v['graft_w']))
                    rebuild_optimizer=True
                self.brain.n = len(v['r'])
            if k in self.modules:
                self.modules[k].load_state_dict(v)
        live_parameters=self._optimization_parameters()
        optimizer_parameters=[p for group in self.opt.param_groups for p in group['params']]
        rebuild_optimizer=rebuild_optimizer or len(live_parameters)!=len(optimizer_parameters) or any(a is not b for a,b in zip(live_parameters,optimizer_parameters))
        self._restore_optimizer(state['optimizer'],rebuild_optimizer)
        self.curiosity.opt.load_state_dict(state['curiosity_optimizer']); self.rng.bit_generator.state=state['rng']
        self.pending_transition=None; self._camera_consumed_any=False
        for k,v in deepcopy(state['transient']).items(): setattr(self,k,v)
        if 'experience_model' in state:
            self.experience_model.import_state(state['experience_model'])
        else:
            self.experience_model=ExperienceWorldModel(state_dim=16,action_dim=5,ensemble_size=3,seed=self.config.seed,context_dim=2)
        if 'surprise_memory' in state:
            self.surprise_memory.import_state(state['surprise_memory'])
        else:
            self.surprise_memory=SurpriseEpisodicMemory()
        self.compass.bump_angle=state['compass_angle']; self.routing=state['routing']
        if hasattr(self, 'neuromod') and state.get('neuromod'):
            self.neuromod.state = NeuromodulatorState(**state['neuromod'])
            context=state.get('neuromod_context',{})
            self.neuromod.v_predicted=float(context.get('v_predicted',0.))
            self.neuromod.history=deepcopy(context.get('history',[]))
        if hasattr(self, 'astrocytes') and state.get('astrocytes'):
            self.astrocytes.import_state(state['astrocytes'])
        if hasattr(self, 'predictive') and state.get('predictive'):
            self.predictive.import_state(state['predictive'])
        if hasattr(self, 'homeostasis') and state.get('homeostasis'):
            self.homeostasis.import_state(state['homeostasis'])
        if hasattr(self, 'neurogenesis') and state.get('neurogenesis'):
            self.neurogenesis.import_state(state['neurogenesis'])
        if hasattr(self, 'dormant_lifecycle') and state.get('dormant_lifecycle') and hasattr(self, 'dormant_lifecycle'):
            self.dormant_lifecycle.import_state(state['dormant_lifecycle'])
        if hasattr(self, 'continual_backprop') and self.continual_backprop is not None and state.get('continual_backprop'):
            self.continual_backprop.import_state(state['continual_backprop'])
        if hasattr(self, 'fast_weight_memory') and self.fast_weight_memory is not None and state.get('fast_weight_memory'):
            self.fast_weight_memory.import_state(state['fast_weight_memory'])
        if hasattr(self, 'latent_reasoner') and self.latent_reasoner is not None and state.get('latent_reasoner'):
            self.latent_reasoner.load_state_dict(state['latent_reasoner'])
        if hasattr(self, 'self_model') and self.self_model is not None and state.get('self_model'):
            self.self_model.import_state(state['self_model'])
        if self.neural_graft is not None:
            if state.get('neural_graft') is not None:
                self.neural_graft.import_state(state['neural_graft'])
                self.modules['neural_graft']=self.neural_graft.network
            else:
                from .neural_graft import NeuralResidualGraft
                self.neural_graft=NeuralResidualGraft(seed=self.config.seed)
                self.modules['neural_graft']=self.neural_graft.network
        if getattr(self,'dendritic_ltc',None) is not None and state.get('dendritic_ltc') is not None:
            self.dendritic_ltc.import_state(state['dendritic_ltc'])
        if getattr(self,'oscillatory_coherence',None) is not None and state.get('oscillatory_coherence') is not None:
            self.oscillatory_coherence.import_state(state['oscillatory_coherence'])
        if getattr(self,'recurrent_planner',None) is not None and state.get('recurrent_planner') is not None:
            self.recurrent_planner.import_state(state['recurrent_planner'])
        if getattr(self,'dpo_engine',None) is not None and state.get('dpo_engine') is not None:
            self.dpo_engine.import_state(state['dpo_engine'])
        self.programmed_prior_status=deepcopy(state.get('programmed_prior_status',self.programmed_prior_status))
        self.source_provenance=deepcopy(state.get('source_provenance',self.source_provenance))

    def status(self):
        return {'feature_schema':self.feature_schema,'graph':self.graph.describe(),'tidal_parameters':sum(p.numel() for p in self.tidal.parameters()),'bridge_training':'experimental supervised preparation' if self.bridge_updates else 'untrained','motor_preparation_updates':self.preparation_updates,'camera':'active; visual encoder fused' if getattr(self, '_camera_consumed_any', False) else 'inactive; image not consumed','experience_planning':self.config.experience_planning,'experience_model':self.experience_model.summary(),'surprise_memory':self.surprise_memory.summary(),'evidence_scope':'synthetic graph and heuristic resource adapter; no integrated performance claim','telemetry':deepcopy(self.last_telemetry)}
