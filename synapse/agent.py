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
        self.dvgem_memory=None
        self.tether_coordinator=None
        self.criticality_homeostat=None
        self.holographic_vsa=None
        self.apical_engine=None
        self.spectral_reservoir=None
        self.hyperbolic_manifold=None
        self.spiking_engine=None
        self.glymphatic_engine=None
        self.kuramoto_engine=None
        self.archimedes_engine=None
        self.memory_carrier_nexus=None
        self.cross_graft_engine=None
        self.comm_fabric=None
        self.graft_bus=None
        self.sancf_fabric=None
        self.aether_bridge=None
        self.overflow_mini_brain=None
        self.grafted_diffusion=None
        self.selective_ssm=None
        self.multitimescale=None
        self.meta_plasticity=None
        self.causal_world=None
        self.intervention_planner=None
        self.relational_memory=None
        self.event_backend=None
        self.graph_diffusion=None
        self.active_inference=None
        self.thermodynamics=None
        self.metaplasticity_bcm=None
        self.transfer_entropy=None
        self.neurowave_x2=None
        self.sensory_fabric=None
        self.wpac=None
        self.wpac_programmer=None
        self.mixed_affect=None
        self.equilibrium_regulator=None
        self.synthetic_glands=None
        self.receptor_bank=None
        self.modulatory_compositor=None
        self.pulse_field=None
        self.wave_solver=None
        self.phase_protocol=None
        self.pulse_router=None
        self.fatigue_controller=None
        self.dream_engine=None
        self.replay_scheduler=None
        self.knowledge_pool=None
        self.knowledge_graft=None
        self.donor_hub=None
        self.donor_bridge=None
        self.exchange_controller=None
        self.evo_controller=None
        self.audio_gateway=None
        self.auditory_encoder=None
        self.chat_gateway=None
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
            if getattr(self.config, 'dvgem_memory', False) and self.dvgem_memory is None:
                from .dvgem_tether import DecisionValueEpisodicMemory
                self.dvgem_memory = DecisionValueEpisodicMemory(seed=self.config.seed)
            if getattr(self.config, 'tether_coordinator', False) and self.tether_coordinator is None:
                from .dvgem_tether import TetherSuccessorCoordinator
                self.tether_coordinator = TetherSuccessorCoordinator(seed=self.config.seed)
            if getattr(self.config, 'criticality_homeostat', False) and self.criticality_homeostat is None:
                from .criticality_vsa import CriticalityAvalancheEngine
                self.criticality_homeostat = CriticalityAvalancheEngine(seed=self.config.seed)
            if getattr(self.config, 'holographic_vsa', False) and self.holographic_vsa is None:
                from .criticality_vsa import HolographicVSAWorkingMemory
                self.holographic_vsa = HolographicVSAWorkingMemory(seed=self.config.seed)
            if getattr(self.config, 'apical_amplification', False) and getattr(self, 'apical_engine', None) is None:
                from .apical_amplification import ApicalAmplificationEngine
                self.apical_engine = ApicalAmplificationEngine(num_units=64, seed=self.config.seed)
                self.modules['apical_engine'] = self.apical_engine
            if getattr(self.config, 'spectral_reservoir', False) and getattr(self, 'spectral_reservoir', None) is None:
                from .spectral_reservoir import SpectralReservoirEngine
                self.spectral_reservoir = SpectralReservoirEngine(seed=self.config.seed)
                self.modules['spectral_reservoir'] = self.spectral_reservoir
            if getattr(self.config, 'hyperbolic_manifold', False) and getattr(self, 'hyperbolic_manifold', None) is None:
                from .hyperbolic_manifold import HyperbolicManifoldEngine
                self.hyperbolic_manifold = HyperbolicManifoldEngine(seed=self.config.seed)
            if getattr(self.config, 'spiking_neuromorphic', False) and getattr(self, 'spiking_engine', None) is None:
                from .spiking_neuromorphic import SpikingNeuromorphicEngine
                self.spiking_engine = SpikingNeuromorphicEngine(num_neurons=min(64, self.brain.n), seed=self.config.seed)
                self.modules['spiking_engine'] = self.spiking_engine
            if getattr(self.config, 'glymphatic_clearance', False) and getattr(self, 'glymphatic_engine', None) is None:
                from .glymphatic_clearance import GlymphaticClearanceEngine
                self.glymphatic_engine = GlymphaticClearanceEngine(seed=self.config.seed)
                self.modules['glymphatic_engine'] = self.glymphatic_engine
            if getattr(self.config, 'quantum_kuramoto', False) and getattr(self, 'kuramoto_engine', None) is None:
                from .quantum_kuramoto import QuantumKuramotoEngine
                self.kuramoto_engine = QuantumKuramotoEngine(num_oscillators=min(32, self.brain.n), seed=self.config.seed)
                self.modules['kuramoto_engine'] = self.kuramoto_engine
            if getattr(self.config, 'archimedes_buoyancy', False) and getattr(self, 'archimedes_engine', None) is None:
                from .archimedes_buoyancy import ArchimedesBuoyancyEngine
                self.archimedes_engine = ArchimedesBuoyancyEngine(seed=self.config.seed)
                self.modules['archimedes_engine'] = self.archimedes_engine
            if getattr(self.config, 'memory_carrier_nexus', False) and getattr(self, 'memory_carrier_nexus', None) is None:
                from .memory_carrier import MemoryCarrierNexus
                self.memory_carrier_nexus = MemoryCarrierNexus(dimension=min(64, self.brain.n), seed=self.config.seed)
                self.modules['memory_carrier_nexus'] = self.memory_carrier_nexus
            from .graft_bus import (
                NeuralGraftCommunicationBus,
                SANCFBusChannel,
                NeuroWaveX2BusChannel,
                CrossGraftBusChannel,
                AetherBridgeBusChannel,
            )
            if getattr(self, 'graft_bus', None) is None:
                self.graft_bus = NeuralGraftCommunicationBus(
                    max_aux_current=getattr(self.config, 'neural_comm_max_aux_current', 0.5),
                    context_dim=min(64, self.brain.n),
                    seed=self.config.seed,
                )
                self.modules['graft_bus'] = self.graft_bus

            if getattr(self.config, 'cross_connectome_graft', False) and getattr(self, 'cross_graft_engine', None) is None:
                from .cross_graft import CrossConnectomeGraft
                self.cross_graft_engine = CrossConnectomeGraft(
                    node_dim=min(64, self.brain.n),
                    gamma=0.2,
                    seed=self.config.seed,
                    target_dim=self.brain.n,
                )
                self.modules['cross_graft_engine'] = self.cross_graft_engine
                self.graft_bus.register_channel("cross_graft", CrossGraftBusChannel(self.cross_graft_engine), gain=1.0)

            if getattr(self.config, 'neural_comm_enabled', False) and getattr(self, 'sancf_fabric', None) is None:
                from .neural_comm import NeuralCommFabric
                from .neural_comm_channels import CommChannelSpec
                comm_spec = CommChannelSpec(
                    mode=getattr(self.config, 'neural_comm_mode', 'continuous'),
                    message_dim=getattr(self.config, 'neural_comm_dim', 16),
                    precision_bits=getattr(self.config, 'neural_comm_precision_bits', 32),
                    packet_loss_probability=getattr(self.config, 'neural_comm_drop_rate', 0.0),
                    max_latency_steps=getattr(self.config, 'neural_comm_latency_steps', 0),
                    seed=self.config.seed,
                )
                self.sancf_fabric = NeuralCommFabric(
                    self.graph,
                    spec=comm_spec,
                    enabled=True,
                    max_aux_current=self.config.neural_comm_max_aux_current,
                    topology_strategy=getattr(self.config, 'neural_comm_topology', 'hybrid_causal'),
                    seed=self.config.seed,
                )
                self.graft_bus.register_channel("sancf", SANCFBusChannel(self.sancf_fabric), gain=1.0)

            if getattr(self.config, 'neurowave_x2_enabled', False) and self.neurowave_x2 is None:
                from .neurowave_x2 import NeuroWaveX2Config, NeuroWaveX2Fabric
                nw_cfg = NeuroWaveX2Config(
                    enabled=True,
                    residual_gain=float(getattr(self.config, 'neurowave_x2_gamma', 0.0)),
                    protocol=str(getattr(self.config, 'neurowave_x2_protocol', 'adaptive')),
                    message_dimension=int(getattr(self.config, 'neurowave_x2_dimension', 16)),
                    precision_bits=int(getattr(self.config, 'neurowave_x2_precision_bits', 32)),
                    spectral_modes=int(getattr(self.config, 'neurowave_x2_spectral_modes', 8)),
                    spectral_basis=str(getattr(self.config, 'neurowave_x2_spectral_basis', 'hybrid')),
                    phase_enabled=bool(getattr(self.config, 'neurowave_x2_phase', True)),
                    structural_plasticity=bool(getattr(self.config, 'neurowave_x2_plasticity', False)),
                    development_mode=bool(getattr(self.config, 'neurowave_x2_development', False)),
                    development_level=int(getattr(self.config, 'neurowave_x2_development_level', 0)),
                )
                num_src = getattr(self.graph, 'num_sources', 8) or 8
                self.neurowave_x2 = NeuroWaveX2Fabric(
                    config=nw_cfg,
                    num_regions=num_src,
                    neurons_per_region=max(16, self.brain.n // max(1, num_src)),
                    total_neurons=self.brain.n,
                    seed=self.config.seed,
                )
                self.graft_bus.register_channel("neurowave_x2", NeuroWaveX2BusChannel(self.neurowave_x2), gain=1.0)

            if getattr(self.config, 'aether_graft_bridge', False) and getattr(self, 'aether_bridge', None) is None:
                from .aether_graft_bridge import AetherGraftBridge, GraftBridgeSpec
                ab_spec = GraftBridgeSpec(
                    enabled=True,
                    coupling_gain=float(getattr(self.config, 'aether_bridge_gamma', 0.0)),
                    num_modes=int(getattr(self.config, 'aether_bridge_modes', 16)),
                    negative_control=str(getattr(self.config, 'aether_bridge_control', 'none')),
                    seed=self.config.seed,
                )
                self.aether_bridge = AetherGraftBridge(
                    spec=ab_spec,
                    donor_dim=min(64, self.brain.n),
                    recipient_dim=self.brain.n,
                )
                self.modules['aether_bridge'] = self.aether_bridge
                self.graft_bus.register_channel("aether_bridge", AetherBridgeBusChannel(self.aether_bridge), gain=1.0)

            # Route unified communication fabric
            if self.graft_bus.enabled:
                self.comm_fabric = self.graft_bus
            elif getattr(self, 'sancf_fabric', None) is not None:
                self.comm_fabric = self.sancf_fabric
            elif getattr(self, 'neurowave_x2', None) is not None:
                self.comm_fabric = self.neurowave_x2
            if getattr(self.config, 'overflow_mini_brain', False) and self.overflow_mini_brain is None:
                from .grafted_sections import OverflowMiniBrain
                self.overflow_mini_brain = OverflowMiniBrain(num_neurons=256, sensory_dim=155, motor_dim=5)
                self.modules['overflow_mini_brain'] = self.overflow_mini_brain
            if getattr(self.config, 'grafted_diffusion', False) and self.grafted_diffusion is None:
                from .grafted_sections import GraftedMiniDiffusionModel
                self.grafted_diffusion = GraftedMiniDiffusionModel(action_dim=5, cond_dim=64, num_steps=5)
                self.modules['grafted_diffusion'] = self.grafted_diffusion
            if getattr(self.config, 'selective_state_space', False) and self.selective_ssm is None:
                from .grafted_sections import SelectiveStateSpaceSection
                self.selective_ssm = SelectiveStateSpaceSection(d_model=64, d_state=16)
                self.modules['selective_ssm'] = self.selective_ssm
            if getattr(self.config, 'multitimescale_learning', False) and self.multitimescale is None:
                from .adaptation_coordinator import AdaptationCoordinator
                self.multitimescale = AdaptationCoordinator(num_neurons=self.brain.n, seed=self.config.seed)
            if getattr(self.config, 'meta_plasticity', False) and self.meta_plasticity is None:
                from .meta_plasticity import ParameterizedPlasticityRule
                self.meta_plasticity = ParameterizedPlasticityRule()
            if getattr(self.config, 'causal_world_model', False) and self.causal_world is None:
                from .causal_world_model import CausalWorldModel
                from .intervention_planner import InterventionPlanner
                self.causal_world = CausalWorldModel(state_dim=64, action_dim=5, seed=self.config.seed)
                self.intervention_planner = InterventionPlanner(self.causal_world)
            if getattr(self.config, 'relational_memory', False) and self.relational_memory is None:
                from .relational_memory import RelationalMemoryBank
                self.relational_memory = RelationalMemoryBank(dim=256)
            if getattr(self.config, 'event_driven_backend', False) and self.event_backend is None:
                from .event_driven_backend import EventDrivenSparseBackend
                self.event_backend = EventDrivenSparseBackend(delta_threshold=0.05)
            if getattr(self.config, 'graph_diffusion', False) and self.graph_diffusion is None:
                from .graph_diffusion import ChebyshevGraphDiffusion
                self.graph_diffusion = ChebyshevGraphDiffusion(
                    num_nodes=self.brain.n,
                    pre=torch.as_tensor(self.graph.base_pre),
                    post=torch.as_tensor(self.graph.base_post),
                    weights=torch.as_tensor(self.brain.base_w),
                    order=4,
                    seed=self.config.seed,
                )
                self.modules['graph_diffusion'] = self.graph_diffusion
            if getattr(self.config, 'active_inference_planner', False) and self.active_inference is None:
                from .active_inference import ActiveInferenceEngine
                self.active_inference = ActiveInferenceEngine(
                    observation_dim=155,
                    action_dim=5,
                    hidden_dim=32,
                    seed=self.config.seed,
                )
            if getattr(self.config, 'thermodynamics', False) and self.thermodynamics is None:
                from .thermodynamics import NeuralThermodynamicsEngine
                self.thermodynamics = NeuralThermodynamicsEngine(num_states=8)
            if getattr(self.config, 'metaplasticity_bcm', False) and self.metaplasticity_bcm is None:
                from .metaplasticity_bcm import BcmMetaplasticityEngine
                self.metaplasticity_bcm = BcmMetaplasticityEngine(
                    num_neurons=self.brain.n,
                    target_activity=0.25,
                )
            if getattr(self.config, 'transfer_entropy', False) and self.transfer_entropy is None:
                from .transfer_entropy import TransferEntropyEngine
                self.transfer_entropy = TransferEntropyEngine(buffer_size=64)

            # SONAPSE-NEUROENDOCRINE / SENSE / EVO Architecture Integrations
            if (getattr(self.config, 'neuroendocrine', False) or getattr(self.config, 'affective_core', False) or getattr(self.config, 'synthetic_glands', False)) and self.wpac is None:
                from .neuroendocrine import (
                    SensoryEventFabric,
                    WeightProgrammedAffectiveCore,
                    AffectiveWeightProgrammer,
                    MixedAffectiveState,
                    AffectiveEquilibriumRegulator,
                    SyntheticGlandBank,
                    SyntheticReceptorBank,
                    ModulatoryCompositor,
                )
                self.sensory_fabric = SensoryEventFabric()
                self.wpac = WeightProgrammedAffectiveCore(num_neurons=256, sensory_dim=64, seed=self.config.seed)
                self.wpac_programmer = AffectiveWeightProgrammer()
                self.wpac_programmer.program_all_motifs(self.wpac)
                self.mixed_affect = MixedAffectiveState(hidden_dim=256, seed=self.config.seed)
                self.equilibrium_regulator = AffectiveEquilibriumRegulator(seed=self.config.seed)
                self.synthetic_glands = SyntheticGlandBank(wpac_dim=256, seed=self.config.seed)
                self.receptor_bank = SyntheticReceptorBank(seed=self.config.seed)
                self.modulatory_compositor = ModulatoryCompositor(
                    enabled=True,
                    receptor_bank=self.receptor_bank,
                    seed=self.config.seed,
                )
                self.modules['wpac'] = self.wpac
                self.modules['mixed_affect'] = self.mixed_affect
                self.modules['synthetic_glands'] = self.synthetic_glands

            if getattr(self.config, 'neural_pulses', False) and self.wave_solver is None:
                from .neural_pulses import (
                    NeuromodulatoryPulseField,
                    GraphWaveSolver,
                    NeuralPhaseResetProtocol,
                    BoundedPulseRouter,
                )
                self.pulse_field = NeuromodulatoryPulseField(num_regions=8)
                self.wave_solver = GraphWaveSolver(num_regions=8, seed=self.config.seed)
                self.phase_protocol = NeuralPhaseResetProtocol(num_regions=8, seed=self.config.seed)
                self.pulse_router = BoundedPulseRouter(num_regions=8, target_dim=64, enabled=True, seed=self.config.seed)
                self.modules['wave_solver'] = self.wave_solver
                self.modules['pulse_router'] = self.pulse_router

            if getattr(self.config, 'sleep_replay', False) and self.fatigue_controller is None:
                from .sleep_replay import (
                    FatigueController,
                    DreamReplayEngine,
                    ReplayScheduler,
                )
                self.fatigue_controller = FatigueController(seed=self.config.seed)
                self.dream_engine = DreamReplayEngine(latent_dim=64, seed=self.config.seed)
                self.replay_scheduler = ReplayScheduler(capacity=1000, seed=self.config.seed)
                self.modules['dream_engine'] = self.dream_engine

            if getattr(self.config, 'shared_knowledge', False) and self.knowledge_pool is None:
                from .shared_knowledge import (
                    NeuralKnowledgePool,
                    KnowledgeGraftAdapter,
                )
                self.knowledge_pool = NeuralKnowledgePool(dim=64, seed=self.config.seed)
                self.knowledge_graft = KnowledgeGraftAdapter(knowledge_dim=64, target_dim=32, enabled=True, seed=self.config.seed)
                self.modules['knowledge_graft'] = self.knowledge_graft

            if (getattr(self.config, 'cross_connectome_graft', False) or getattr(self.config, 'aether_graft_bridge', False)) and self.donor_hub is None:
                from .donor_grafts import PoseidonHubDonorAdapter, PretrainedGraftBridge
                self.donor_hub = PoseidonHubDonorAdapter(donor_dim=128, obs_dim=64, seed=self.config.seed)
                self.donor_bridge = PretrainedGraftBridge(donor_dim=128, recipient_dim=64, enabled=True, seed=self.config.seed)
                self.modules['donor_bridge'] = self.donor_bridge

            if getattr(self.config, 'cross_model_exchange', False) and self.exchange_controller is None:
                from .neural_exchange import SpontaneousNeuralExchangeController
                self.exchange_controller = SpontaneousNeuralExchangeController(feature_dim=64, seed=self.config.seed)

            if getattr(self.config, 'evo_compat', False) and self.evo_controller is None:
                from .evo_compat import EvolutionaryGrowthController
                self.evo_controller = EvolutionaryGrowthController(seed=self.config.seed)

            if (getattr(self.config, 'live_microphone', False) or getattr(self.config, 'live_chat', False)) and self.chat_gateway is None:
                from .multimodal_live import (
                    LiveAudioGateway,
                    AuditorySpikeEncoder,
                    ConversationalChatGateway,
                )
                self.audio_gateway = LiveAudioGateway(enabled=getattr(self.config, 'live_microphone', False))
                self.auditory_encoder = AuditorySpikeEncoder(seed=self.config.seed)
                self.chat_gateway = ConversationalChatGateway(embed_dim=64, seed=self.config.seed)

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
        if getattr(self, 'dvgem_memory', None) is not None: self.dvgem_memory.reset_state()
        if getattr(self, 'tether_coordinator', None) is not None: self.tether_coordinator.reset()
        if getattr(self, 'criticality_homeostat', None) is not None: self.criticality_homeostat.reset()
        if getattr(self, 'holographic_vsa', None) is not None: self.holographic_vsa.reset()
        if getattr(self, 'apical_engine', None) is not None: self.apical_engine.reset()
        if getattr(self, 'spectral_reservoir', None) is not None: self.spectral_reservoir.reset()
        if getattr(self, 'spiking_engine', None) is not None: self.spiking_engine.reset()
        if getattr(self, 'glymphatic_engine', None) is not None: self.glymphatic_engine.reset()
        if getattr(self, 'kuramoto_engine', None) is not None: self.kuramoto_engine.reset()
        if getattr(self, 'archimedes_engine', None) is not None: self.archimedes_engine.reset()
        if getattr(self, 'memory_carrier_nexus', None) is not None: self.memory_carrier_nexus.reset()
        if getattr(self, 'cross_graft_engine', None) is not None: self.cross_graft_engine.reset()
        if getattr(self, 'comm_fabric', None) is not None: self.comm_fabric.reset()
        if getattr(self, 'neurowave_x2', None) is not None: self.neurowave_x2.reset()
        if getattr(self, 'overflow_mini_brain', None) is not None: self.overflow_mini_brain.reset_state()
        if getattr(self, 'selective_ssm', None) is not None: self.selective_ssm.reset_state()
        if getattr(self, 'event_backend', None) is not None: self.event_backend.reset()
        if getattr(self, 'wpac', None) is not None: self.wpac.reset_state()
        if getattr(self, 'mixed_affect', None) is not None: self.mixed_affect.reset_state()
        if getattr(self, 'equilibrium_regulator', None) is not None: self.equilibrium_regulator.reset()
        if getattr(self, 'synthetic_glands', None) is not None: self.synthetic_glands.reset()
        if getattr(self, 'pulse_field', None) is not None: self.pulse_field.clear()
        if getattr(self, 'wave_solver', None) is not None: self.wave_solver.reset()
        if getattr(self, 'phase_protocol', None) is not None: self.phase_protocol.reset()
        if getattr(self, 'pulse_router', None) is not None: self.pulse_router.reset()
        if getattr(self, 'fatigue_controller', None) is not None: self.fatigue_controller.reset()
        if getattr(self, 'chat_gateway', None) is not None: self.chat_gateway.clear()
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
            if getattr(self, 'graft_bus', None) is not None:
                pre_ctx = torch.cat([o[:min(32, len(o))], self.hidden[0, :16], self.intent])
                self.graft_bus.set_context(pre_ctx)
            previous=self.brain.r.clone()
            baseline=self.brain(torch.zeros_like(current),effective,adapt=False,advance=False) if self.config.readout!='absolute' else None
            r=self.brain(
                current,
                effective,
                adapt=learn,
                comm_fabric=getattr(self, 'comm_fabric', None),
                phase_angles=getattr(self.oscillatory_coherence, 'theta_phases', None) if getattr(self, 'oscillatory_coherence', None) is not None else None,
            )
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
            criticality_info = None
            if getattr(self, 'criticality_homeostat', None) is not None:
                pooled, criticality_info = self.criticality_homeostat.step(r.detach().numpy(), pooled, learn=learn)
            vsa_info = None
            if getattr(self, 'holographic_vsa', None) is not None:
                vsa_delta, vsa_info = self.holographic_vsa.step(pooled, obs)
                pooled = pooled + 0.04 * torch.tanh(vsa_delta)
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
            criticality_info = None
            if getattr(self, 'criticality_homeostat', None) is not None:
                pooled, criticality_info = self.criticality_homeostat.step(r.detach().numpy(), pooled, learn=learn)
            vsa_info = None
            if getattr(self, 'holographic_vsa', None) is not None:
                vsa_res, vsa_info = self.holographic_vsa.step(pooled, obs, learn=learn)
                pooled = pooled + 0.03 * torch.tanh(vsa_res)
            graph_diff_info = None
            if getattr(self, 'graph_diffusion', None) is not None:
                gdiff_res, graph_diff_info = self.graph_diffusion(r, learn=learn)
                pooled = pooled + 0.04 * torch.tanh(gdiff_res[:32])
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
            if getattr(self, 'selective_ssm', None) is not None:
                features = features + 0.05 * torch.tanh(self.selective_ssm.step(features))
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
            if getattr(self, 'modulatory_compositor', None) is not None and getattr(self, 'synthetic_glands', None) is not None:
                wpac_in = pooled.repeat(2) if len(pooled) < 64 else pooled[:64]
                wpac_h = self.wpac.step(sensory_input=wpac_in) if self.wpac is not None else None
                if getattr(self, 'mixed_affect', None) is not None and wpac_h is not None:
                    self.mixed_affect.step(wpac_hidden=wpac_h)
                g_out = self.synthetic_glands.step(wpac_hidden=wpac_h)
                comp_state, target_gains = self.modulatory_compositor.composite(
                    legacy_state=self.neuromod.state,
                    gland_deltas=g_out['deltas'],
                    gland_levels=g_out['levels'],
                )
                if 'motor_vigor' in gains:
                    gains['motor_vigor'] = float(gains['motor_vigor'] * target_gains.get('excitability', 1.0))
            action=torch.tanh(self.motor(features))*torch.sigmoid(pooled.mean()*4)*gains['motor_vigor']
            if getattr(self, 'overflow_mini_brain', None) is not None:
                of_curr, of_act, _ = self.overflow_mini_brain(o, motor_feedback=action)
                action = action + 0.05 * of_act
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
        base_act=np.clip(action.numpy(),-1.,1.).astype(np.float32,copy=True)
        executed=base_act.copy()
        graft_receipt=None
        if self.neural_graft is not None:
            executed,graft_receipt=self.neural_graft.propose(obs,self.last_features,executed,learn=learn)
        if getattr(self, 'active_inference', None) is not None:
            if getattr(self.config, 'planner_algorithm', '') == 'active_inference':
                executed, act_inf_info = self.active_inference.select_action(executed, self.last_features)
            else:
                _, act_inf_info = self.active_inference.evaluate_expected_free_energy(
                    np.array([executed]), self.last_features
                )
            self.last_telemetry['active_inference'] = act_inf_info
        graft_act=executed.copy()
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
        planner_act=executed.copy()
        dvgem_diag=None
        if getattr(self, 'dvgem_memory', None) is not None:
            v_res = float(min(self.homeostasis.state.energy, self.homeostasis.state.health)) if hasattr(self, 'homeostasis') else 1.0
            executed, dvgem_diag = self.dvgem_memory.query_and_gate(obs, executed, viability_reserve=v_res)
        tether_diag=None
        # Holographic VSA transition recording is performed in observe_transition.
        if getattr(self, 'tether_coordinator', None) is not None:
            proposals = {'base_actor': base_act, 'neural_graft': graft_act, 'planner': planner_act, 'dvgem': executed.copy()}
            executed, tether_diag = self.tether_coordinator.coordinate_specialists(
                proposals, self.last_features, r.detach().numpy(), self.graph.origin
            )
            executed = np.clip(executed, np.maximum(-1., planner_act - .35), np.minimum(1., planner_act + .35))
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
        if dvgem_diag is not None: self.last_telemetry['dvgem']=dvgem_diag
        if tether_diag is not None: self.last_telemetry['tether_coordinator']=tether_diag
        if criticality_info is not None: self.last_telemetry['criticality_homeostat']=criticality_info
        if vsa_info is not None: self.last_telemetry['holographic_vsa']=vsa_info
        if getattr(self, 'apical_engine', None) is not None:
            _, apical_telem = self.apical_engine(
                r[:64],
                apical_context=torch.tensor(self.last_features[:32], dtype=torch.float32),
                neuromod_ach=getattr(self.neuromod.state, 'acetylcholine', 1.0) if hasattr(self, 'neuromod') else 1.0,
                neuromod_da=getattr(self.neuromod.state, 'dopamine', 1.0) if hasattr(self, 'neuromod') else 1.0,
            )
            self.last_telemetry['apical_amplification'] = apical_telem.to_dict()
        if getattr(self, 'spectral_reservoir', None) is not None:
            res_readout, res_telem = self.spectral_reservoir.step(
                torch.tensor(self.last_features[:32], dtype=torch.float32),
                learn=learn,
                target=torch.tensor(executed[:5], dtype=torch.float32) if learn else None,
            )
            self.last_telemetry['spectral_reservoir'] = res_telem.to_dict()
        if getattr(self, 'hyperbolic_manifold', None) is not None:
            hyp_telem = self.hyperbolic_manifold.step(active_node_ids=list(range(min(4, self.brain.n))))
            self.last_telemetry['hyperbolic_manifold'] = hyp_telem.to_dict()
        if getattr(self, 'spiking_engine', None) is not None:
            _, _, spk_telem = self.spiking_engine(
                r[:min(64, self.brain.n)],
                neuromod_ach=getattr(self.neuromod.state, 'acetylcholine', 1.0) if hasattr(self, 'neuromod') else 1.0
            )
            self.last_telemetry['spiking_neuromorphic'] = spk_telem.to_dict()
        if getattr(self, 'glymphatic_engine', None) is not None:
            act_norm = float(r.norm().item()) if isinstance(r, torch.Tensor) else float(np.linalg.norm(r))
            is_sleep = bool(getattr(self, 'in_sleep_phase', False))
            _, glymph_telem = self.glymphatic_engine.step(act_norm, is_quiescent_sleep=is_sleep)
            self.last_telemetry['glymphatic_clearance'] = glymph_telem.to_dict()
        if getattr(self, 'kuramoto_engine', None) is not None:
            _, kur_telem = self.kuramoto_engine.step()
            self.last_telemetry['quantum_kuramoto'] = kur_telem.to_dict()
        if getattr(self, 'archimedes_engine', None) is not None:
            pos = (float(self.last_features[0]), float(self.last_features[1])) if len(self.last_features) >= 2 else (0.0, 0.0)
            vel = (float(self.last_features[2]), float(self.last_features[3])) if len(self.last_features) >= 4 else (0.0, 0.0)
            _, _, arch_telem = self.archimedes_engine.step(pos, vel, vertical_thrust=float(executed[0]) if len(executed) > 0 else 0.0)
            self.last_telemetry['archimedes_buoyancy'] = arch_telem.to_dict()
        if getattr(self, 'memory_carrier_nexus', None) is not None:
            k_vec = torch.tensor(self.last_features[:min(64, self.brain.n)], dtype=torch.float32)
            if learn:
                v_vec = torch.tensor(executed[:min(64, len(executed))], dtype=torch.float32)
                self.memory_carrier_nexus.store(k_vec, v_vec, carrier_idx=self.ticks % 16)
            _, _ = self.memory_carrier_nexus.retrieve(k_vec, carrier_idx=self.ticks % 16)
            self.last_telemetry['memory_carrier_nexus'] = self.memory_carrier_nexus.telemetry.to_dict()
        if getattr(self, 'cross_graft_engine', None) is not None:
            self.last_telemetry['cross_connectome_graft'] = self.cross_graft_engine.telemetry.to_dict()
        if getattr(self, 'aether_bridge', None) is not None:
            self.last_telemetry['aether_graft_bridge'] = self.aether_bridge.telemetry.to_dict()
        if getattr(self, 'graft_bus', None) is not None:
            self.last_telemetry['graft_bus'] = self.graft_bus.get_telemetry_dict()
        if getattr(self, 'dpo_engine', None) is not None:
            self.last_telemetry['verifiable_dpo']={'enabled':True,'pairs':len(self.dpo_engine.preference_buffer),'last_margin':getattr(self.dpo_engine,'last_margin',0.0)}
        if getattr(self, 'sancf_fabric', None) is not None:
            self.last_telemetry['sancf'] = self.sancf_fabric.get_telemetry_dict()
        elif getattr(self, 'comm_fabric', None) is not None and hasattr(self.comm_fabric, 'get_telemetry_dict'):
            self.last_telemetry['sancf'] = self.comm_fabric.get_telemetry_dict()
        if getattr(self, 'neurowave_x2', None) is not None:
            self.last_telemetry['neurowave_x2'] = self.neurowave_x2.get_telemetry_dict()
        if getattr(self, 'graph_diffusion', None) is not None and graph_diff_info is not None:
            self.last_telemetry['graph_diffusion'] = graph_diff_info
        if getattr(self, 'thermodynamics', None) is not None:
            t_st = self.thermodynamics.step(r.detach().numpy(), task_utility=0.75, learn=learn)
            self.last_telemetry['thermodynamics'] = t_st.to_dict()
        if getattr(self, 'metaplasticity_bcm', None) is not None:
            bcm_delta, bcm_st = self.metaplasticity_bcm.step(
                r.detach().numpy(),
                graft_weights=self.brain.graft_w,
                graft_pre=self.brain.graft_pre,
                graft_post=self.brain.graft_post,
                learn=learn,
            )
            if learn and bcm_delta is not None and self.brain.graft_w is not None:
                with torch.no_grad():
                    self.brain.graft_w.add_(0.01 * bcm_delta.clamp(-0.1, 0.1))
            self.last_telemetry['metaplasticity_bcm'] = bcm_st.to_dict()
        if getattr(self, 'transfer_entropy', None) is not None:
            te_st = self.transfer_entropy.step(r.detach().numpy(), self.graph.origin, learn=learn)
            self.last_telemetry['transfer_entropy'] = te_st.to_dict()
        if getattr(self, 'overflow_mini_brain', None) is not None:
            self.last_telemetry['overflow_mini_brain'] = {'enabled': True, 'state_norm': float(torch.norm(self.overflow_mini_brain.state).item())}
        if getattr(self, 'event_backend', None) is not None:
            from dataclasses import asdict
            _, _, ev_m = self.event_backend.detect_events(r)
            self.last_telemetry['event_driven'] = asdict(ev_m)
        if getattr(self, 'grafted_diffusion', None) is not None:
            diff_proposal = self.grafted_diffusion.sample_action(features)
            self.last_telemetry['grafted_diffusion'] = {'proposal': diff_proposal.tolist()}
        if getattr(self, 'modulatory_compositor', None) is not None:
            self.last_telemetry['neuroendocrine'] = self.modulatory_compositor.get_telemetry()
        if getattr(self, 'mixed_affect', None) is not None:
            self.last_telemetry['affective_state'] = self.mixed_affect.get_state_dict()
        if getattr(self, 'synthetic_glands', None) is not None:
            self.last_telemetry['synthetic_glands'] = {name: self.synthetic_glands.get_level(name) for name in ['dopamine', 'serotonin', 'adrenaline', 'cortisol', 'oxytocin', 'adenosine']}
        if getattr(self, 'pulse_field', None) is not None:
            self.last_telemetry['neural_pulses'] = {'active_count': len(self.pulse_field.active_pulses), 'pulses': self.pulse_field.get_active_pulses()[:4]}
        if getattr(self, 'fatigue_controller', None) is not None:
            f_res = self.fatigue_controller.step(threat_level=thr)
            self.last_telemetry['sleep_replay'] = {'state': f_res['state'], 'sleep_pressure': f_res['sleep_pressure'], 'dream_cycles': f_res['dream_cycles']}
        if getattr(self, 'knowledge_pool', None) is not None:
            self.last_telemetry['shared_knowledge'] = {'records_count': self.knowledge_pool.size()}
        if getattr(self, 'chat_gateway', None) is not None:
            self.last_telemetry['chat_gateway'] = {'message_count': len(self.chat_gateway.history)}
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
        if getattr(self, 'dvgem_memory', None) is not None and getattr(self, 'last_obs', None) is not None:
            v_res = float(min(self.homeostasis.state.energy, self.homeostasis.state.health)) if hasattr(self, 'homeostasis') else 1.0
            diagnostics['dvgem'] = self.dvgem_memory.record_transition(self.last_obs, executed, float(reward), observation, viability_reserve=v_res)
        if getattr(self, 'holographic_vsa', None) is not None and getattr(self, 'last_obs', None) is not None:
            diagnostics['holographic_vsa'] = self.holographic_vsa.record_transition(
                self.last_features[:32], self.last_obs, executed, float(reward)
            )
        # Holographic VSA transition recording is performed in observe_transition.
        if getattr(self, 'tether_coordinator', None) is not None:
            diagnostics['tether_coordinator'] = self.tether_coordinator.update_successor_td(
                state, executed, float(reward), next_state, executed, done=bool(done)
            )
        if getattr(self, 'multitimescale', None) is not None:
            diagnostics['multitimescale'] = self.multitimescale.step(float(reward), novelty=float(abs(diagnostics.get('prediction_error', 0.0))))
        if getattr(self, 'causal_world', None) is not None:
            f_t = torch.as_tensor(self.last_features, dtype=torch.float32)
            a_t = torch.as_tensor(executed, dtype=torch.float32)
            next_f = torch.as_tensor(next_state[:64] if len(next_state) >= 64 else np.pad(next_state, (0, 64 - len(next_state))), dtype=torch.float32)
            diagnostics['causal_world'] = self.causal_world.observe(f_t, a_t, next_f)
        if getattr(self, 'relational_memory', None) is not None:
            diagnostics['relational_memory'] = self.relational_memory.bind_step(state, executed, next_state)
        if getattr(self, 'active_inference', None) is not None:
            self.active_inference.update_beliefs(float(abs(reward)))
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

    def __deepcopy__(self, memo):
        cloned = IntegratedAgent(self.config)
        cloned.import_state(self.export_state())
        memo[id(self)] = cloned
        return cloned

    def export_state(self):
        state=deepcopy({'schema':1,'config':self.config.to_dict(),'topology':{'pre':self.brain.graft_pre.numpy(),'post':self.brain.graft_post.numpy(),'initial_weights':self.plasticity.original_weights.numpy()},'modules':{k:v.state_dict() for k,v in self.modules.items()},'optimizer':self.opt.state_dict(),'curiosity_optimizer':self.curiosity.opt.state_dict(),'rng':self.rng.bit_generator.state,'transient':{k:getattr(self,k) for k in ('hidden','intent','evoked','ticks','last_compass','last_features','last_telemetry','last_obs','resources','expert_usage','consolidated','episodic','tool_memory','preparation_updates','bridge_updates','consolidations','pending_transition','_camera_consumed_any')},'compass_angle':self.compass.bump_angle,'neuromod':self.neuromod.state.to_dict() if hasattr(self, 'neuromod') else None,'neuromod_context':{'v_predicted':self.neuromod.v_predicted,'history':self.neuromod.history},'astrocytes':self.astrocytes.export_state() if hasattr(self, 'astrocytes') else None,'predictive':self.predictive.export_state() if hasattr(self, 'predictive') else None,'routing':getattr(self,'routing',[]),'experience_model':self.experience_model.export_state(),'surprise_memory':self.surprise_memory.export_state(),'homeostasis':self.homeostasis.export_state() if hasattr(self,'homeostasis') else None,'neurogenesis':self.neurogenesis.export_state() if hasattr(self,'neurogenesis') else None,'dormant_lifecycle':self.dormant_lifecycle.export_state() if hasattr(self,'dormant_lifecycle') else None,'continual_backprop':self.continual_backprop.export_state() if (hasattr(self,'continual_backprop') and self.continual_backprop is not None) else None,'fast_weight_memory':self.fast_weight_memory.export_state() if (hasattr(self,'fast_weight_memory') and self.fast_weight_memory is not None) else None,'latent_reasoner':self.latent_reasoner.state_dict() if (hasattr(self,'latent_reasoner') and self.latent_reasoner is not None) else None,'self_model':self.self_model.export_state() if (hasattr(self,'self_model') and self.self_model is not None) else None})
        state['neural_graft']=self.neural_graft.export_state() if self.neural_graft is not None else None
        state['dendritic_ltc']=self.dendritic_ltc.export_state() if getattr(self,'dendritic_ltc',None) is not None else None
        state['oscillatory_coherence']=self.oscillatory_coherence.export_state() if getattr(self,'oscillatory_coherence',None) is not None else None
        state['recurrent_planner']=self.recurrent_planner.export_state() if getattr(self,'recurrent_planner',None) is not None else None
        state['dpo_engine']=self.dpo_engine.export_state() if getattr(self,'dpo_engine',None) is not None else None
        state['dvgem_memory']=self.dvgem_memory.export_state() if getattr(self,'dvgem_memory',None) is not None else None
        state['tether_coordinator']=self.tether_coordinator.export_state() if getattr(self,'tether_coordinator',None) is not None else None
        state['criticality_homeostat']=self.criticality_homeostat.export_state() if getattr(self,'criticality_homeostat',None) is not None else None
        state['holographic_vsa']=self.holographic_vsa.export_state() if getattr(self,'holographic_vsa',None) is not None else None
        if getattr(self, 'apical_engine', None) is not None: state['apical_engine'] = self.apical_engine.export_state()
        if getattr(self, 'spectral_reservoir', None) is not None: state['spectral_reservoir'] = self.spectral_reservoir.export_state()
        if getattr(self, 'hyperbolic_manifold', None) is not None: state['hyperbolic_manifold'] = self.hyperbolic_manifold.export_state()
        if getattr(self, 'spiking_engine', None) is not None: state['spiking_engine'] = self.spiking_engine.export_state()
        if getattr(self, 'glymphatic_engine', None) is not None: state['glymphatic_engine'] = self.glymphatic_engine.export_state()
        if getattr(self, 'kuramoto_engine', None) is not None: state['kuramoto_engine'] = self.kuramoto_engine.export_state()
        if getattr(self, 'archimedes_engine', None) is not None: state['archimedes_engine'] = self.archimedes_engine.export_state()
        if getattr(self, 'memory_carrier_nexus', None) is not None: state['memory_carrier_nexus'] = self.memory_carrier_nexus.export_state()
        if getattr(self, 'cross_graft_engine', None) is not None: state['cross_graft_engine'] = self.cross_graft_engine.export_state()
        if getattr(self, 'aether_bridge', None) is not None: state['aether_bridge'] = self.aether_bridge.export_state()
        if getattr(self, 'graft_bus', None) is not None: state['graft_bus'] = self.graft_bus.export_state()
        state['comm_fabric']=self.comm_fabric.export_state() if getattr(self,'comm_fabric',None) is not None else None
        state['neurowave_x2']=self.neurowave_x2.export_state() if getattr(self,'neurowave_x2',None) is not None else None
        state['overflow_mini_brain']=self.overflow_mini_brain.state_dict() if getattr(self,'overflow_mini_brain',None) is not None else None
        state['selective_ssm']=self.selective_ssm.state_dict() if getattr(self,'selective_ssm',None) is not None else None
        state['grafted_diffusion']=self.grafted_diffusion.export_state() if getattr(self,'grafted_diffusion',None) is not None else None
        state['event_backend']=self.event_backend.export_state() if getattr(self,'event_backend',None) is not None else None
        state['multitimescale']=self.multitimescale.export_state() if getattr(self,'multitimescale',None) is not None else None
        state['relational_memory']=self.relational_memory.export_state() if getattr(self,'relational_memory',None) is not None else None
        state['graph_diffusion']=self.graph_diffusion.export_state() if getattr(self,'graph_diffusion',None) is not None else None
        state['active_inference']=self.active_inference.export_state() if getattr(self,'active_inference',None) is not None else None
        state['thermodynamics']=self.thermodynamics.export_state() if getattr(self,'thermodynamics',None) is not None else None
        state['metaplasticity_bcm']=self.metaplasticity_bcm.export_state() if getattr(self,'metaplasticity_bcm',None) is not None else None
        state['transfer_entropy']=self.transfer_entropy.export_state() if getattr(self,'transfer_entropy',None) is not None else None
        if getattr(self, 'wpac', None) is not None: state['wpac'] = self.wpac.state_dict()
        if getattr(self, 'mixed_affect', None) is not None: state['mixed_affect'] = self.mixed_affect.state_dict()
        if getattr(self, 'synthetic_glands', None) is not None: state['synthetic_glands'] = self.synthetic_glands.state_dict()
        if getattr(self, 'wave_solver', None) is not None: state['wave_solver'] = self.wave_solver.state_dict()
        if getattr(self, 'phase_protocol', None) is not None: state['phase_protocol'] = self.phase_protocol.state_dict()
        if getattr(self, 'fatigue_controller', None) is not None: state['fatigue_controller'] = self.fatigue_controller.state_dict()
        if getattr(self, 'knowledge_pool', None) is not None: state['knowledge_pool'] = self.knowledge_pool.get_summary()
        state['torch_rng']=torch.random.get_rng_state()
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
        if getattr(self,'dvgem_memory',None) is not None and state.get('dvgem_memory') is not None:
            self.dvgem_memory.import_state(state['dvgem_memory'])
        if getattr(self,'tether_coordinator',None) is not None and state.get('tether_coordinator') is not None:
            self.tether_coordinator.import_state(state['tether_coordinator'])
        if getattr(self,'criticality_homeostat',None) is not None and state.get('criticality_homeostat') is not None:
            self.criticality_homeostat.import_state(state['criticality_homeostat'])
        if getattr(self,'holographic_vsa',None) is not None and state.get('holographic_vsa') is not None:
            self.holographic_vsa.import_state(state['holographic_vsa'])
        if getattr(self, 'apical_engine', None) is not None and state.get('apical_engine') is not None:
            self.apical_engine.import_state(state['apical_engine'])
        if getattr(self, 'spectral_reservoir', None) is not None and state.get('spectral_reservoir') is not None:
            self.spectral_reservoir.import_state(state['spectral_reservoir'])
        if getattr(self, 'hyperbolic_manifold', None) is not None and state.get('hyperbolic_manifold') is not None:
            self.hyperbolic_manifold.import_state(state['hyperbolic_manifold'])
        if getattr(self, 'spiking_engine', None) is not None and state.get('spiking_engine') is not None:
            self.spiking_engine.import_state(state['spiking_engine'])
        if getattr(self, 'glymphatic_engine', None) is not None and state.get('glymphatic_engine') is not None:
            self.glymphatic_engine.import_state(state['glymphatic_engine'])
        if getattr(self, 'kuramoto_engine', None) is not None and state.get('kuramoto_engine') is not None:
            self.kuramoto_engine.import_state(state['kuramoto_engine'])
        if getattr(self, 'archimedes_engine', None) is not None and state.get('archimedes_engine') is not None:
            self.archimedes_engine.import_state(state['archimedes_engine'])
        if getattr(self, 'memory_carrier_nexus', None) is not None and state.get('memory_carrier_nexus') is not None:
            self.memory_carrier_nexus.import_state(state['memory_carrier_nexus'])
        if getattr(self, 'cross_graft_engine', None) is not None and state.get('cross_graft_engine') is not None:
            self.cross_graft_engine.import_state(state['cross_graft_engine'])
        if getattr(self, 'aether_bridge', None) is not None and state.get('aether_bridge') is not None:
            self.aether_bridge.import_state(state['aether_bridge'])
        if getattr(self, 'graft_bus', None) is not None and state.get('graft_bus') is not None:
            self.graft_bus.import_state(state['graft_bus'])
        if getattr(self,'comm_fabric',None) is not None and state.get('comm_fabric') is not None:
            self.comm_fabric.import_state(state['comm_fabric'])
        if getattr(self,'neurowave_x2',None) is not None and state.get('neurowave_x2') is not None:
            self.neurowave_x2.import_state(state['neurowave_x2'])
        if getattr(self,'overflow_mini_brain',None) is not None and state.get('overflow_mini_brain') is not None:
            self.overflow_mini_brain.load_state_dict(state['overflow_mini_brain'])
        if getattr(self,'selective_ssm',None) is not None and state.get('selective_ssm') is not None:
            self.selective_ssm.load_state_dict(state['selective_ssm'])
        if getattr(self,'grafted_diffusion',None) is not None and state.get('grafted_diffusion') is not None:
            self.grafted_diffusion.import_state(state['grafted_diffusion'])
        if getattr(self,'event_backend',None) is not None and state.get('event_backend') is not None:
            self.event_backend.import_state(state['event_backend'])
        if getattr(self,'multitimescale',None) is not None and state.get('multitimescale') is not None:
            self.multitimescale.import_state(state['multitimescale'])
        if getattr(self,'relational_memory',None) is not None and state.get('relational_memory') is not None:
            self.relational_memory.import_state(state['relational_memory'])
        if getattr(self,'graph_diffusion',None) is not None and state.get('graph_diffusion') is not None:
            self.graph_diffusion.import_state(state['graph_diffusion'])
        if getattr(self,'active_inference',None) is not None and state.get('active_inference') is not None:
            self.active_inference.import_state(state['active_inference'])
        if getattr(self,'thermodynamics',None) is not None and state.get('thermodynamics') is not None:
            self.thermodynamics.import_state(state['thermodynamics'])
        if getattr(self,'metaplasticity_bcm',None) is not None and state.get('metaplasticity_bcm') is not None:
            self.metaplasticity_bcm.import_state(state['metaplasticity_bcm'])
        if getattr(self,'transfer_entropy',None) is not None and state.get('transfer_entropy') is not None:
            self.transfer_entropy.import_state(state['transfer_entropy'])
        if getattr(self, 'wpac', None) is not None and state.get('wpac') is not None:
            self.wpac.load_state_dict(state['wpac'])
        if getattr(self, 'mixed_affect', None) is not None and state.get('mixed_affect') is not None:
            self.mixed_affect.load_state_dict(state['mixed_affect'])
        if getattr(self, 'synthetic_glands', None) is not None and state.get('synthetic_glands') is not None:
            self.synthetic_glands.load_state_dict(state['synthetic_glands'])
        if getattr(self, 'wave_solver', None) is not None and state.get('wave_solver') is not None:
            self.wave_solver.load_state_dict(state['wave_solver'])
        if getattr(self, 'phase_protocol', None) is not None and state.get('phase_protocol') is not None:
            self.phase_protocol.load_state_dict(state['phase_protocol'])
        if getattr(self, 'fatigue_controller', None) is not None and state.get('fatigue_controller') is not None:
            self.fatigue_controller.load_state_dict(state['fatigue_controller'])
        if 'torch_rng' in state:
            torch.random.set_rng_state(state['torch_rng'])
        self.programmed_prior_status=deepcopy(state.get('programmed_prior_status',self.programmed_prior_status))
        self.source_provenance=deepcopy(state.get('source_provenance',self.source_provenance))

    def status(self):
        return {'feature_schema':self.feature_schema,'graph':self.graph.describe(),'tidal_parameters':sum(p.numel() for p in self.tidal.parameters()),'bridge_training':'experimental supervised preparation' if self.bridge_updates else 'untrained','motor_preparation_updates':self.preparation_updates,'camera':'active; visual encoder fused' if getattr(self, '_camera_consumed_any', False) else 'inactive; image not consumed','experience_planning':self.config.experience_planning,'experience_model':self.experience_model.summary(),'surprise_memory':self.surprise_memory.summary(),'evidence_scope':'synthetic graph and heuristic resource adapter; no integrated performance claim','telemetry':deepcopy(self.last_telemetry)}
