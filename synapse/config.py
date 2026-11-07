from dataclasses import dataclass, asdict
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

    def __post_init__(self):
        for key in ('experience_planning', 'surprise_memory', 'experimental_cognition',
                    'continual_backprop', 'fast_weight_memory', 'dormant_lifecycle',
                    'latent_reasoning', 'value_guided_planning', 'self_model', 'frontier_all',
                    'neural_graft','active_training','programmed_connectome','structural_learning',
                    'dendritic_ltc', 'oscillatory_coherence', 'verifiable_dpo',
                    'dvgem_memory', 'tether_coordinator'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')
        for key, low, high in [('nodes_per_source', 16, 10000), ('microsteps', 1, 8), ('cognition_interval', 1, 128), ('experience_horizon', 1, 5), ('compute_threads',1,32), ('graft_batch_size',8,1024), ('graft_updates',1,32)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        if self.readout not in {'absolute', 'evoked', 'persistent_evoked'}:
            raise ValueError('unsupported graph readout')
        if self.planner_algorithm not in {'experience_mpc', 'recurrent_latent', 'hybrid'}:
            raise ValueError('unsupported planner algorithm')
        if self.connectome_mode not in {'proxy','prefer_authentic','strict_authentic'}:
            raise ValueError('unsupported connectome mode')
        if self.active_training and not self.neural_graft:
            raise ValueError('active_training requires neural_graft')
        if self.programmed_connectome and not self.neural_graft:
            raise ValueError('programmed_connectome requires neural_graft')
        valid_tasks = {'auto', 'forage', 'maze', 'push', 'grasp_place', 'peg_insert', 'delayed_match', 'go_nogo', 'button_sequence', 'webcam_chase', 'ecosystem', 'predator_evasion', 'dynamic_arena', 'puzzle_chamber', 'apex_megasite', 'stealth_labyrinth', 'kinetic_foundry', 'quantum_bifurcation', 'cybernetic_sanctuary', 'continental_biosphere', 'chrono_tidepool_citadel'}
        if self.task not in valid_tasks:
            raise ValueError('unknown task')
        if not isinstance(self.connectome_sources, tuple) or len(self.connectome_sources) == 0:
            raise ValueError('connectome_sources must be a non-empty tuple of dataset names')
        if self.total_neurons is not None:
            count=len(self.connectome_sources)
            if type(self.total_neurons) is not int or not 16*count <= self.total_neurons <= 10000*count:
                raise ValueError(f'total_neurons must be an integer in [{16*count},{10000*count}]')

    def source_node_counts(self):
        """Allocate an exact total while preserving legacy per-source configs."""
        count=len(self.connectome_sources)
        if self.total_neurons is None:
            return (self.nodes_per_source,)*count
        base,remainder=divmod(self.total_neurons,count)
        return tuple(base+(index<remainder) for index in range(count))

    def total_neuron_count(self) -> int:
        return int(sum(self.source_node_counts()))

    @classmethod
    def frontier_apex(cls, total_neurons: int = 4000, seed: int = 7, task: str = 'forage', **kwargs) -> 'SynapseConfig':
        """Omnibus configuration activating all frontier cognitive and neuromorphic features."""
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
            compute_threads=min(4,os.cpu_count() or 1),
            connectome_mode='prefer_authentic',
        )
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def max_tandem(cls, total_neurons: int = 4000, seed: int = 7, task: str = 'continental_biosphere', **kwargs) -> 'SynapseConfig':
        """Maximum tandem configuration activating every optional feature, hybrid planning, and full CPU compute."""
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
            compute_threads=min(32, max(1, os.cpu_count() or 4)),
            graft_batch_size=256,
            graft_updates=4,
            connectome_mode='prefer_authentic',
            structural_learning=True,
        )
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    def to_dict(self):
        d = asdict(self)
        d['connectome_sources'] = list(self.connectome_sources)
        return d

    def matches_saved(self, saved):
        """Only new optional settings may be omitted by legacy checkpoints."""
        if not isinstance(saved, dict):
            return False
        normalized = {'experience_planning': False, 'surprise_memory': True, 'experience_horizon': 1,
                      'experimental_cognition': False, 'continual_backprop': False,
                      'fast_weight_memory': False, 'dormant_lifecycle': True,
                      'latent_reasoning': False, 'value_guided_planning': False,
                      'self_model': False,
                      'frontier_all': False,
                      'planner_algorithm': 'experience_mpc',
                      'total_neurons': None,
                      'neural_graft':False,'active_training':False,'programmed_connectome':False,
                      'compute_threads':1,'graft_batch_size':64,'graft_updates':2,'connectome_mode':'proxy',
                      'structural_learning':True,
                      'dendritic_ltc':False,'oscillatory_coherence':False,'verifiable_dpo':False,
                      'dvgem_memory':False,'tether_coordinator':False,
                      'connectome_sources': ('flywire', 'manc', 'optic-lobe'), **saved}
        if isinstance(normalized.get('connectome_sources'), list):
            normalized['connectome_sources'] = tuple(normalized['connectome_sources'])
        try:
            restored = type(self)(**normalized)
        except (TypeError, ValueError):
            return False
        return restored.to_dict() == self.to_dict()
