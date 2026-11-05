"""Multi-model donor subcircuit extraction, representation alignment and graft transplantation."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .config import SynapseConfig
from .stateio import digest
from graftbrain.world import World

class DonorSubcircuit:
    """Portable neural subcircuit containing graft weights, site gains, and alignment metadata."""
    def __init__(
        self,
        graft_weights: torch.Tensor,
        site_gains: Optional[torch.Tensor],
        source_task: str,
        donor_digest: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.graft_weights = graft_weights.clone().detach()
        self.site_gains = site_gains.clone().detach() if site_gains is not None else None
        self.source_task = source_task
        self.donor_digest = donor_digest
        self.metadata = metadata or {}
        self.identity = hashlib.sha256(
            self.graft_weights.numpy().tobytes() + source_task.encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'identity': self.identity,
            'source_task': self.source_task,
            'donor_digest': self.donor_digest,
            'graft_shape': list(self.graft_weights.shape),
            'graft_norm': float(self.graft_weights.norm()),
            'has_site_gains': self.site_gains is not None,
            'metadata': self.metadata,
        }


def extract_donor_subcircuit(agent: IntegratedAgent, source_task: str = 'forage') -> DonorSubcircuit:
    """Extract functional graft connections and plasticity parameters from a donor agent."""
    graft_w = agent.brain.graft_w.clone().detach()
    site_gains = None
    if hasattr(agent.plasticity, 'router') and hasattr(agent.plasticity.router, 'w_gate'):
        site_gains = agent.plasticity.router.w_gate.clone().detach()
    
    donor_digest = digest(agent.export_state())
    metadata = {
        'neurons': agent.graph.n,
        'graft_edges': len(agent.graph.graft_pre),
        'timestamp': time.time(),
    }
    return DonorSubcircuit(
        graft_weights=graft_w,
        site_gains=site_gains,
        source_task=source_task,
        donor_digest=donor_digest,
        metadata=metadata,
    )


def fit_representation_alignment(
    donor_features: np.ndarray,
    recipient_features: np.ndarray,
    ridge: float = 0.1,
) -> np.ndarray:
    """Compute ridge projection map M aligning donor feature space into recipient space.
    
    Formula: M = (X_donor^T X_donor + ridge * I)^(-1) X_donor^T X_recipient
    """
    X = np.asarray(donor_features, dtype=np.float64)
    Y = np.asarray(recipient_features, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError("Feature matrices must have matching sample counts")
    
    dim_in = X.shape[1]
    reg = ridge * np.eye(dim_in, dtype=np.float64)
    M = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return M


def transplant_subcircuit(
    recipient: IntegratedAgent,
    donor: DonorSubcircuit,
    mode: str = 'correct',
    seed: int = 42,
) -> Dict[str, Any]:
    """Stitch donor subcircuit into recipient host under declared experimental arm.
    
    Supported arms (controls required by spec):
    - 'correct': Direct transplantation of donor graft parameters.
    - 'random': Random parameter initialization with matched L2 norm.
    - 'shuffled': Coordinate-permuted donor weights.
    - 'irrelevant': Inverted/orthogonalized donor parameters.
    - 'disabled': Zeroed graft transfer (control).
    - 'permuted_host': Recipient target site coordinates randomly permuted.
    """
    surfaces = recipient.editable_tensors()
    target_tensor = surfaces['brain.graft_w']
    donor_w = donor.graft_weights.clone()

    rng = np.random.default_rng(seed)

    if mode == 'correct':
        # Align length if recipient graft count differs
        if len(donor_w) >= len(target_tensor):
            applied_w = donor_w[:len(target_tensor)]
        else:
            applied_w = target_tensor.clone()
            applied_w[:len(donor_w)] = donor_w

    elif mode == 'random':
        norm = float(donor_w.norm())
        rand = torch.from_numpy(rng.normal(0, 1.0, size=target_tensor.shape)).float()
        applied_w = rand * (norm / (float(rand.norm()) + 1e-8))

    elif mode == 'shuffled':
        perm = rng.permutation(len(donor_w))
        shuffled = donor_w[perm]
        applied_w = shuffled[:len(target_tensor)]

    elif mode == 'irrelevant':
        applied_w = -donor_w[:len(target_tensor)]

    elif mode == 'disabled':
        applied_w = torch.zeros_like(target_tensor)

    elif mode == 'permuted_host':
        applied_w = donor_w[:len(target_tensor)]
        perm = rng.permutation(len(applied_w))
        applied_w = applied_w[perm]

    else:
        raise ValueError(f"Unknown transplantation mode: {mode}")

    with torch.no_grad():
        target_tensor.copy_(applied_w)

    return {
        'status': 'ok',
        'mode': mode,
        'donor_identity': donor.identity,
        'source_task': donor.source_task,
        'recipient_digest': digest(recipient.export_state()),
        'transplanted_norm': float(applied_w.norm()),
        'scope': f'graft transplantation under {mode} control',
    }


def run_transplantation_experiment(
    recipient: IntegratedAgent,
    donor: DonorSubcircuit,
    test_task: str = 'delayed_match',
    episodes: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full factorial comparison across all transplantation control arms."""
    results = {}
    saved_state = copy.deepcopy(recipient.export_state())

    arms = ['correct', 'random', 'shuffled', 'irrelevant', 'disabled', 'permuted_host']
    for arm in arms:
        recipient.import_state(copy.deepcopy(saved_state))
        trans_receipt = transplant_subcircuit(recipient, donor, mode=arm, seed=seed)
        
        # Test performance on the target task
        world = World(seed=seed, curriculum=test_task)
        ep_returns = []
        for ep in range(episodes):
            obs = world.reset(test_task if test_task != 'auto' else None)
            recipient.reset_episode()
            ret = 0.0
            for _ in range(40):
                out = recipient.act(obs, learn=False)
                obs, r, done, _ = world.step(out['action'])
                ret += r
                if done:
                    break
            ep_returns.append(ret)
        
        results[arm] = {
            'mean_return': float(np.mean(ep_returns)),
            'returns': ep_returns,
            'norm': trans_receipt['transplanted_norm'],
        }

    # Restore initial state
    recipient.import_state(saved_state)

    gain_over_disabled = results['correct']['mean_return'] - results['disabled']['mean_return']
    gain_over_random = results['correct']['mean_return'] - results['random']['mean_return']
    gain_over_shuffled = results['correct']['mean_return'] - results['shuffled']['mean_return']

    return {
        'status': 'ok',
        'task': test_task,
        'donor_task': donor.source_task,
        'arms': results,
        'comparisons': {
            'correct_minus_disabled': gain_over_disabled,
            'correct_minus_random': gain_over_random,
            'correct_minus_shuffled': gain_over_shuffled,
        },
        'causal_donor_supported': gain_over_shuffled > 0 and gain_over_disabled > 0,
        'scope': 'factorial donor transplantation control assay',
    }
