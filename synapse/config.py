from dataclasses import dataclass, asdict
from pathlib import Path
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

    def __post_init__(self):
        for key in ('experience_planning', 'surprise_memory'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be a boolean')
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')
        for key, low, high in [('nodes_per_source',16,1500),('microsteps',1,8),('cognition_interval',1,128)]:
            value = getattr(self,key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        if self.readout not in {'absolute','evoked','persistent_evoked'}:
            raise ValueError('unsupported graph readout')
        if self.task not in {'auto','forage','maze','push','grasp_place','peg_insert','delayed_match','go_nogo','button_sequence','webcam_chase'}:
            raise ValueError('unknown task')

    def to_dict(self): return asdict(self)

    def matches_saved(self, saved):
        """Only new optional settings may be omitted by legacy checkpoints."""
        if not isinstance(saved, dict):
            return False
        normalized = {'experience_planning': False, 'surprise_memory': True, **saved}
        try:
            restored = type(self)(**normalized)
        except (TypeError, ValueError):
            return False
        return restored.to_dict() == self.to_dict()
