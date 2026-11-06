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
    connectome_sources: Tuple[str, ...] = ('flywire', 'manc', 'optic-lobe')

    def __post_init__(self):
        for key in ('experience_planning', 'surprise_memory'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')
        for key, low, high in [('nodes_per_source', 16, 5000), ('microsteps', 1, 8), ('cognition_interval', 1, 128)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        if self.readout not in {'absolute', 'evoked', 'persistent_evoked'}:
            raise ValueError('unsupported graph readout')
        if self.task not in {'auto', 'forage', 'maze', 'push', 'grasp_place', 'peg_insert', 'delayed_match', 'go_nogo', 'button_sequence', 'webcam_chase', 'ecosystem', 'predator_evasion', 'dynamic_arena', 'puzzle_chamber'}:
            raise ValueError('unknown task')
        if not isinstance(self.connectome_sources, tuple) or len(self.connectome_sources) == 0:
            raise ValueError('connectome_sources must be a non-empty tuple of dataset names')

    def to_dict(self):
        d = asdict(self)
        d['connectome_sources'] = list(self.connectome_sources)
        return d

    def matches_saved(self, saved):
        """Only new optional settings may be omitted by legacy checkpoints."""
        if not isinstance(saved, dict):
            return False
        normalized = {'experience_planning': False, 'surprise_memory': True, 'connectome_sources': ('flywire', 'manc', 'optic-lobe'), **saved}
        if isinstance(normalized.get('connectome_sources'), list):
            normalized['connectome_sources'] = tuple(normalized['connectome_sources'])
        try:
            restored = type(self)(**normalized)
        except (TypeError, ValueError):
            return False
        return restored.to_dict() == self.to_dict()
