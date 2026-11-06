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
    planner_algorithm: str = 'experience_mpc'
    connectome_sources: Tuple[str, ...] = ('flywire', 'manc', 'optic-lobe')
    total_neurons: int | None = None

    def __post_init__(self):
        for key in ('experience_planning', 'surprise_memory', 'experimental_cognition',
                    'continual_backprop', 'fast_weight_memory', 'dormant_lifecycle',
                    'latent_reasoning', 'value_guided_planning', 'self_model'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')
        for key, low, high in [('nodes_per_source', 16, 5000), ('microsteps', 1, 8), ('cognition_interval', 1, 128), ('experience_horizon', 1, 5)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        if self.readout not in {'absolute', 'evoked', 'persistent_evoked'}:
            raise ValueError('unsupported graph readout')
        if self.planner_algorithm not in {'experience_mpc', 'recurrent_latent', 'hybrid'}:
            raise ValueError('unsupported planner algorithm')
        if self.task not in {'auto', 'forage', 'maze', 'push', 'grasp_place', 'peg_insert', 'delayed_match', 'go_nogo', 'button_sequence', 'webcam_chase', 'ecosystem', 'predator_evasion', 'dynamic_arena', 'puzzle_chamber'}:
            raise ValueError('unknown task')
        if not isinstance(self.connectome_sources, tuple) or len(self.connectome_sources) == 0:
            raise ValueError('connectome_sources must be a non-empty tuple of dataset names')
        if self.total_neurons is not None:
            count=len(self.connectome_sources)
            if type(self.total_neurons) is not int or not 16*count <= self.total_neurons <= 5000*count:
                raise ValueError(f'total_neurons must be an integer in [{16*count},{5000*count}]')

    def source_node_counts(self):
        """Allocate an exact total while preserving legacy per-source configs."""
        count=len(self.connectome_sources)
        if self.total_neurons is None:
            return (self.nodes_per_source,)*count
        base,remainder=divmod(self.total_neurons,count)
        return tuple(base+(index<remainder) for index in range(count))

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
                      'planner_algorithm': 'experience_mpc',
                      'total_neurons': None,
                      'connectome_sources': ('flywire', 'manc', 'optic-lobe'), **saved}
        if isinstance(normalized.get('connectome_sources'), list):
            normalized['connectome_sources'] = tuple(normalized['connectome_sources'])
        try:
            restored = type(self)(**normalized)
        except (TypeError, ValueError):
            return False
        return restored.to_dict() == self.to_dict()
