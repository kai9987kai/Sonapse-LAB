from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Tuple
import os

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SYNAPSE_HOME', str(PACKAGE.parent))).resolve()
VENDOR = PACKAGE / '_vendor'
ARTIFACTS = ROOT / 'artifacts'

@dataclass(frozen=True)
class SynapseConfig:
    seed: int = 7
    nodes_per_source: int = 100
    microsteps: int = 2
    cognition_interval: int = 8
    readout: str = 'absolute'
    task: str = 'forage'
    experience_planning: bool = False
    surprise_memory: bool = True
    experience_horizon: int = 1
    experimental_cognition: bool = False
    continual_backprop: bool = False
    fast_weight_memory: bool = False
    dormant_lifecycle: bool = True
    latent_reasoning: bool = False
    value_guided_planning: bool = False
    self_model: bool = False
    frontier_all: bool = False
    planner_algorithm: str = 'experience_mpc'
    connectome_sources: Tuple[str, ...] = ('flywire', 'manc', 'optic-lobe')
    total_neurons: int | None = None
    neural_graft: bool = False
    active_training: bool = False
    programmed_connectome: bool = False
    compute_threads: int = 1
    graft_batch_size: int = 64
    graft_updates: int = 2
    connectome_mode: str = 'proxy'
    structural_learning: bool = True
    dendritic_ltc: bool = False
    oscillatory_coherence: bool = False
    verifiable_dpo: bool = False
    dvgem_memory: bool = False
    tether_coordinator: bool = False
    criticality_homeostat: bool = False
    holographic_vsa: bool = False
    apical_amplification: bool = False
    spectral_reservoir: bool = False
    hyperbolic_manifold: bool = False
    spiking_neuromorphic: bool = False
    glymphatic_clearance: bool = False
    quantum_kuramoto: bool = False
    archimedes_buoyancy: bool = False
    memory_carrier_nexus: bool = False
    cross_connectome_graft: bool = False
    aether_graft_bridge: bool = False
    aether_bridge_gamma: float = 0.0
    aether_bridge_modes: int = 16
    aether_bridge_control: str = 'none'
    graft_bus_enabled: bool = False

    # SANCF (Sonapse Adaptive Neural Communication Fabric) & Advanced Research Features
    neural_comm_enabled: bool = False
    neural_comm_mode: str = 'continuous'
    neural_comm_dim: int = 16
    neural_comm_precision_bits: int = 4
    neural_comm_every: int = 1
    neural_comm_max_messages: int = 4
    neural_comm_max_bits: int = 128
    neural_comm_max_aux_current: float = 0.5
    neural_comm_topology: str = 'hybrid_causal'
    neural_comm_drop_rate: float = 0.0
    neural_comm_latency_steps: int = 0
    neurowave_enabled: bool = False
    overflow_mini_brain: bool = False
    grafted_diffusion: bool = False
    selective_state_space: bool = False
    multitimescale_learning: bool = False
    meta_plasticity: bool = False
    causal_world_model: bool = False
    relational_memory: bool = False
    verifier_rsi: bool = False
    event_driven_backend: bool = False
    graph_diffusion: bool = False
    active_inference_planner: bool = False
    thermodynamics: bool = False
    metaplasticity_bcm: bool = False
    transfer_entropy: bool = False

    # Sonapse NeuroWave-X2 Advanced Neural Communication Platform (default OFF)
    neurowave_x2_enabled: bool = False
    neurowave_x2_gamma: float = 0.0
    neurowave_x2_protocol: str = 'adaptive'
    neurowave_x2_dimension: int = 16
    neurowave_x2_precision_bits: int = 32
    neurowave_x2_spectral_modes: int = 8
    neurowave_x2_spectral_basis: str = 'hybrid'
    neurowave_x2_phase: bool = True
    neurowave_x2_plasticity: bool = False
    neurowave_x2_development: bool = False
    neurowave_x2_development_level: int = 0
    neurowave_x2_dendritic: bool = False

    # SONAPSE-NEUROENDOCRINE / SENSE / EVO compatibility flags (default OFF for zero-gain identity)
    neuroendocrine: bool = False
    synthetic_glands: bool = False
    affective_core: bool = False
    neural_pulses: bool = False
    sleep_replay: bool = False
    shared_knowledge: bool = False
    live_microphone: bool = False
    live_chat: bool = False
    virtual_ecology_v2: bool = False
    cross_model_exchange: bool = False
    evo_compat: bool = False
    # Kept outside legacy config exports when absent so old checkpoint schemas
    # remain byte-for-byte the same under the default-off path.
    dsag: object | None = None

    def __post_init__(self):
        if self.dsag is not None:
            from .dsag.config import DSAGSpecV1
            if isinstance(self.dsag, dict):
                object.__setattr__(self, 'dsag', DSAGSpecV1.from_dict(self.dsag))
            elif not isinstance(self.dsag, DSAGSpecV1):
                raise ValueError('dsag must be a DSAGSpecV1 or its mapping')
        bool_keys = (
            'experience_planning', 'surprise_memory', 'experimental_cognition',
            'continual_backprop', 'fast_weight_memory', 'dormant_lifecycle',
            'latent_reasoning', 'value_guided_planning', 'self_model', 'frontier_all',
            'neural_graft', 'active_training', 'programmed_connectome', 'structural_learning',
            'dendritic_ltc', 'oscillatory_coherence', 'verifiable_dpo',
            'dvgem_memory', 'tether_coordinator', 'criticality_homeostat', 'holographic_vsa',
            'neural_comm_enabled', 'neurowave_enabled', 'overflow_mini_brain',
            'grafted_diffusion', 'selective_state_space', 'multitimescale_learning',
            'meta_plasticity', 'causal_world_model', 'relational_memory',
            'verifier_rsi', 'event_driven_backend', 'graph_diffusion', 'active_inference_planner', 'thermodynamics', 'metaplasticity_bcm', 'transfer_entropy',
            'neurowave_x2_enabled', 'neurowave_x2_phase', 'neurowave_x2_plasticity', 'neurowave_x2_development', 'neurowave_x2_dendritic', 'cross_connectome_graft', 'aether_graft_bridge', 'graft_bus_enabled', 'apical_amplification', 'spectral_reservoir', 'hyperbolic_manifold', 'spiking_neuromorphic', 'glymphatic_clearance', 'quantum_kuramoto', 'archimedes_buoyancy', 'memory_carrier_nexus',
            'neuroendocrine', 'synthetic_glands', 'affective_core', 'neural_pulses', 'sleep_replay', 'shared_knowledge', 'live_microphone', 'live_chat', 'virtual_ecology_v2', 'cross_model_exchange', 'evo_compat'
        )
        for key in bool_keys:
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')

        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')

        int_ranges = [
            ('nodes_per_source', 16, 10000),
            ('microsteps', 1, 8),
            ('cognition_interval', 1, 128),
            ('experience_horizon', 1, 5),
            ('compute_threads', 1, 32),
            ('graft_batch_size', 8, 1024),
            ('graft_updates', 1, 32),
            ('neural_comm_dim', 1, 128),
            ('neural_comm_precision_bits', 1, 32),
            ('neural_comm_every', 1, 16),
            ('neural_comm_max_messages', 1, 64),
            ('neural_comm_max_bits', 1, 4096),
            ('neural_comm_latency_steps', 0, 16),
            ('aether_bridge_modes', 1, 64),
        ]
        for key, low, high in int_ranges:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')

        if not 0.0 <= self.neural_comm_max_aux_current <= 5.0:
            raise ValueError('neural_comm_max_aux_current must be in [0.0, 5.0]')
        if not 0.0 <= self.neural_comm_drop_rate <= 1.0:
            raise ValueError('neural_comm_drop_rate must be in [0.0, 1.0]')
        if not 0.0 <= self.neurowave_x2_gamma <= 5.0:
            raise ValueError('neurowave_x2_gamma must be in [0.0, 5.0]')
        if not 1 <= self.neurowave_x2_dimension <= 512:
            raise ValueError('neurowave_x2_dimension must be in [1, 512]')
        if not 0.0 <= self.aether_bridge_gamma <= 5.0:
            raise ValueError('aether_bridge_gamma must be in [0.0, 5.0]')
        if self.aether_bridge_control not in {'none', 'scrambled', 'shuffled_modes', 'phase_inverted', 'zero_donor'}:
            raise ValueError('unsupported aether_bridge_control mode')

        if self.readout not in {'absolute', 'evoked', 'persistent_evoked'}:
            raise ValueError('unsupported graph readout')
        if self.planner_algorithm not in {'experience_mpc', 'recurrent_latent', 'hybrid'}:
            raise ValueError('unsupported planner algorithm')
        if self.connectome_mode not in {'proxy', 'prefer_authentic', 'strict_authentic'}:
            raise ValueError('unsupported connectome mode')
        if self.active_training and not self.neural_graft:
            raise ValueError('active_training requires neural_graft')
        if self.programmed_connectome and not self.neural_graft:
            raise ValueError('programmed_connectome requires neural_graft')

        valid_tasks = {
            'auto', 'forage', 'maze', 'push', 'grasp_place', 'peg_insert',
            'delayed_match', 'go_nogo', 'button_sequence', 'webcam_chase',
            'ecosystem', 'predator_evasion', 'dynamic_arena', 'puzzle_chamber',
            'apex_megasite', 'stealth_labyrinth', 'kinetic_foundry',
            'quantum_bifurcation', 'cybernetic_sanctuary', 'continental_biosphere',
            'chrono_tidepool_citadel', 'world4d_tesseract_bypass', 'digital_tool_arena',
            'hyper_boreal_tundra', 'subterranean_abyss', 'vortex_maelstrom'
        }
        if self.task not in valid_tasks:
            raise ValueError('unknown task')
        if not isinstance(self.connectome_sources, tuple) or len(self.connectome_sources) == 0:
            raise ValueError('connectome_sources must be a non-empty tuple of dataset names')

        if self.total_neurons is not None:
            count = len(self.connectome_sources)
            if type(self.total_neurons) is not int or not 16 * count <= self.total_neurons <= 10000 * count:
                raise ValueError(f'total_neurons must be an integer in [{16*count},{10000*count}]')

    def source_node_counts(self):
        """Allocate an exact total while preserving legacy per-source configs."""
        count = len(self.connectome_sources)
        if self.total_neurons is None:
            return (self.nodes_per_source,) * count
        base, remainder = divmod(self.total_neurons, count)
        return tuple(base + (index < remainder) for index in range(count))

    def total_neuron_count(self) -> int:
        return int(sum(self.source_node_counts()))

    @classmethod
    def frontier_apex(cls, total_neurons: int = 9000, seed: int = 7, task: str = 'forage', **kwargs) -> 'SynapseConfig':
        """Omnibus configuration activating all frontier cognitive, neuromorphic, and SANCF communication features."""
        cfg_dict = dict(
            seed=seed,
            total_neurons=total_neurons,
            task=task,
            frontier_all=True,
            experience_planning=True,
            surprise_memory=True,
            experience_horizon=2,
            experimental_cognition=True,
            continual_backprop=True,
            fast_weight_memory=True,
            dormant_lifecycle=True,
            latent_reasoning=True,
            value_guided_planning=True,
            self_model=True,
            planner_algorithm='experience_mpc',
            readout='absolute',
            neural_graft=True,
            active_training=True,
            programmed_connectome=True,
            dendritic_ltc=True,
            oscillatory_coherence=True,
            verifiable_dpo=True,
            dvgem_memory=True,
            tether_coordinator=True,
            criticality_homeostat=True,
            holographic_vsa=True,
            apical_amplification=True,
            spectral_reservoir=True,
            hyperbolic_manifold=True,
            spiking_neuromorphic=True,
            glymphatic_clearance=True,
            quantum_kuramoto=True,
            archimedes_buoyancy=True,
            memory_carrier_nexus=True,
            cross_connectome_graft=True,
            neural_comm_enabled=True,
            neural_comm_mode='continuous',
            neural_comm_dim=24,
            neurowave_enabled=True,
            overflow_mini_brain=True,
            grafted_diffusion=True,
            selective_state_space=True,
            multitimescale_learning=True,
            meta_plasticity=True,
            causal_world_model=True,
            relational_memory=True,
            verifier_rsi=True,
            event_driven_backend=True,
            graph_diffusion=True,
            active_inference_planner=True,
            thermodynamics=True,
            metaplasticity_bcm=True,
            transfer_entropy=True,
            compute_threads=min(4, os.cpu_count() or 1),
            connectome_mode='prefer_authentic',
        )
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def max_tandem(cls, total_neurons: int = 9000, seed: int = 7, task: str = 'continental_biosphere', **kwargs) -> 'SynapseConfig':
        """Maximum configuration with every optional subsystem enabled."""
        from .dsag.config import DSAGSpecV1

        cfg_dict = dict(
            seed=seed,
            total_neurons=total_neurons,
            microsteps=4,
            cognition_interval=4,
            task=task,
            frontier_all=True,
            experience_planning=True,
            surprise_memory=True,
            experience_horizon=5,
            experimental_cognition=True,
            continual_backprop=True,
            fast_weight_memory=True,
            dormant_lifecycle=True,
            latent_reasoning=True,
            value_guided_planning=True,
            self_model=True,
            planner_algorithm='hybrid',
            readout='persistent_evoked',
            neural_graft=True,
            active_training=True,
            programmed_connectome=True,
            dendritic_ltc=True,
            oscillatory_coherence=True,
            verifiable_dpo=True,
            dvgem_memory=True,
            tether_coordinator=True,
            criticality_homeostat=True,
            holographic_vsa=True,
            neural_comm_enabled=True,
            neural_comm_mode='continuous',
            neural_comm_dim=32,
            neurowave_enabled=True,
            overflow_mini_brain=True,
            grafted_diffusion=True,
            selective_state_space=True,
            multitimescale_learning=True,
            meta_plasticity=True,
            causal_world_model=True,
            relational_memory=True,
            verifier_rsi=True,
            event_driven_backend=True,
            graph_diffusion=True,
            active_inference_planner=True,
            thermodynamics=True,
            metaplasticity_bcm=True,
            transfer_entropy=True,
            neurowave_x2_enabled=True,
            neurowave_x2_gamma=0.25,
            neurowave_x2_protocol='adaptive',
            neurowave_x2_dimension=32,
            neurowave_x2_precision_bits=32,
            neurowave_x2_spectral_modes=16,
            neurowave_x2_spectral_basis='hybrid',
            neurowave_x2_phase=True,
            neurowave_x2_plasticity=True,
            neurowave_x2_development=True,
            neurowave_x2_development_level=6,
            neurowave_x2_dendritic=True,
            cross_connectome_graft=True,
            aether_graft_bridge=True,
            aether_bridge_gamma=0.25,
            aether_bridge_modes=16,
            aether_bridge_control='none',
            graft_bus_enabled=True,
            apical_amplification=True,
            spectral_reservoir=True,
            hyperbolic_manifold=True,
            spiking_neuromorphic=True,
            glymphatic_clearance=True,
            quantum_kuramoto=True,
            archimedes_buoyancy=True,
            memory_carrier_nexus=True,
            compute_threads=min(32, max(1, os.cpu_count() or 4)),
            graft_batch_size=256,
            graft_updates=4,
            connectome_mode='prefer_authentic',
            structural_learning=True,
            neuroendocrine=True,
            synthetic_glands=True,
            affective_core=True,
            neural_pulses=True,
            sleep_replay=True,
            shared_knowledge=True,
            live_microphone=True,
            live_chat=True,
            virtual_ecology_v2=True,
            cross_model_exchange=True,
            evo_compat=True,
            dsag=DSAGSpecV1(
                enabled=True,
                observe_only=False,
                gain=0.025,
                max_gain=0.1,
                ramp_steps=32,
                seed=seed,
                plasticity_enabled=True,
                phase_gate_enabled=True,
                richer_messages=True,
                structural_growth_enabled=True,
            ),
        )
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def scale_12000(cls, seed: int = 7, task: str = 'continental_biosphere', **kwargs) -> 'SynapseConfig':
        return cls.max_tandem(total_neurons=12082, seed=seed, task=task, **kwargs)

    @classmethod
    def frontier_12k(cls, seed: int = 7, task: str = 'continental_biosphere', **kwargs) -> 'SynapseConfig':
        return cls.scale_12000(seed=seed, task=task, **kwargs)

    def to_dict(self):
        d = asdict(self)
        d['connectome_sources'] = list(self.connectome_sources)
        if self.dsag is None:
            d.pop('dsag')
        return d

    def max_tandem_feature_audit(self):
        """Return an explicit inventory of enabled Max Mode subsystems."""
        boolean_features = [
            field.name for field in fields(self)
            if type(getattr(self, field.name)) is bool
        ]
        enabled = [name for name in boolean_features if getattr(self, name)]
        disabled = [name for name in boolean_features if not getattr(self, name)]
        dsag_features = (
            'enabled', 'plasticity_enabled', 'phase_gate_enabled',
            'richer_messages', 'dual_soma', 'axo_branches',
            'structural_growth_enabled',
        )
        dsag = self.dsag
        if dsag is None:
            disabled.append('dsag_v1')
        else:
            enabled.extend(f'dsag_v1.{name}' for name in dsag_features if getattr(dsag, name))
            disabled.extend(f'dsag_v1.{name}' for name in dsag_features if not getattr(dsag, name))
        return {
            'schema': 'max-tandem-feature-audit-v1',
            'enabled_count': len(enabled),
            'enabled_features': enabled,
            'disabled_features': disabled,
            'complete': not disabled,
        }

    def matches_saved(self, saved):
        """Only new optional settings may be omitted by legacy checkpoints."""
        if not isinstance(saved, dict):
            return False
        normalized = {
            'dsag': None,
            'experience_planning': False, 'surprise_memory': True, 'experience_horizon': 1,
            'experimental_cognition': False, 'continual_backprop': False,
            'fast_weight_memory': False, 'dormant_lifecycle': True,
            'latent_reasoning': False, 'value_guided_planning': False,
            'self_model': False,
            'frontier_all': False,
            'planner_algorithm': 'experience_mpc',
            'total_neurons': None,
            'neural_graft': False, 'active_training': False, 'programmed_connectome': False,
            'compute_threads': 1, 'graft_batch_size': 64, 'graft_updates': 2, 'connectome_mode': 'proxy',
            'structural_learning': True,
            'dendritic_ltc': False, 'oscillatory_coherence': False, 'verifiable_dpo': False,
            'dvgem_memory': False, 'tether_coordinator': False, 'criticality_homeostat': False, 'holographic_vsa': False, 'apical_amplification': False, 'spectral_reservoir': False, 'hyperbolic_manifold': False, 'spiking_neuromorphic': False, 'glymphatic_clearance': False, 'quantum_kuramoto': False, 'archimedes_buoyancy': False, 'memory_carrier_nexus': False, 'cross_connectome_graft': False, 'aether_graft_bridge': False, 'aether_bridge_gamma': 0.0, 'aether_bridge_modes': 16, 'aether_bridge_control': 'none', 'graft_bus_enabled': False,
            'neural_comm_enabled': False, 'neural_comm_mode': 'continuous', 'neural_comm_dim': 16,
            'neural_comm_precision_bits': 4, 'neural_comm_every': 1, 'neural_comm_max_messages': 4,
            'neural_comm_max_bits': 128, 'neural_comm_max_aux_current': 0.5,
            'neural_comm_topology': 'hybrid_causal', 'neural_comm_drop_rate': 0.0,
            'neural_comm_latency_steps': 0, 'neurowave_enabled': False,
            'overflow_mini_brain': False, 'grafted_diffusion': False, 'selective_state_space': False,
            'multitimescale_learning': False, 'meta_plasticity': False, 'causal_world_model': False,
            'relational_memory': False, 'verifier_rsi': False, 'event_driven_backend': False, 'graph_diffusion': False, 'active_inference_planner': False, 'thermodynamics': False, 'metaplasticity_bcm': False, 'transfer_entropy': False,
            'neuroendocrine': False, 'synthetic_glands': False, 'affective_core': False,
            'neural_pulses': False, 'sleep_replay': False, 'shared_knowledge': False,
            'live_microphone': False, 'live_chat': False, 'virtual_ecology_v2': False,
            'cross_model_exchange': False, 'evo_compat': False,
            'connectome_sources': ('flywire', 'manc', 'optic-lobe'),
            **saved
        }
        if isinstance(normalized.get('connectome_sources'), list):
            normalized['connectome_sources'] = tuple(normalized['connectome_sources'])
        try:
            restored = type(self)(**normalized)
        except (TypeError, ValueError):
            return False
        return restored.to_dict() == self.to_dict()
