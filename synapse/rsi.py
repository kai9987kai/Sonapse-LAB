"""Versioned finite improver programs; evaluator and query are not arguments."""
from dataclasses import dataclass, asdict, replace
import hashlib,json
import numpy as np

@dataclass(frozen=True)
class Policy:
    version: int=1
    generation: int=0
    parent: str='root'
    operator: str='gradient'
    rate: float=.5
    steps: int=16
    ridge: float=.1
    replay: int=1
    probe: str='forward'
    support_fraction: float=1.
    ranking: str='accuracy_then_cost'
    mutation_scale: float=1.

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def to_dict(self): return {**asdict(self),'identity':self.identity}

    @classmethod
    def from_dict(cls,data):
        data=dict(data); expected=data.pop('identity',None); p=cls(**data)
        if expected and p.identity!=expected: raise ValueError('policy identity mismatch')
        return p


def propose(parent,index,seed):
    """Execute inherited proposal policy: its scale and ranking affect successors."""
    rng=np.random.default_rng(seed+index*1009+int(parent.identity[:8],16))
    direction=float(rng.choice([.5,2.]))**parent.mutation_scale
    operator=['gradient','direct','ridge'][(index+parent.generation)%3]
    return replace(parent,generation=parent.generation+1,parent=parent.identity,
        operator=operator,rate=float(np.clip(parent.rate*direction,.05,2)),
        steps=int(np.clip(parent.steps+(4 if direction>1 else -4),4,48)),
        replay=int(rng.integers(1,3)),probe=str(rng.choice(['forward','reverse','balanced'])),
        support_fraction=float(rng.choice([.5,.75,1.])),ridge=float(np.clip(parent.ridge*direction,.001,10)),
        mutation_scale=float(rng.choice([.5,1.,1.5])),
        ranking=str(rng.choice(['accuracy_then_cost','accuracy_then_norm'])))
