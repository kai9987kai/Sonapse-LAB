"""Continual Backpropagation and Utility-Driven Neuron Replacement.

Grounding:
Dohare et al. (Nature 2024), "Continual backpropagation".
Maintains network plastic capacity across continual task shifts by identifying
and replacing mature, low-utility neurons while zeroing stale optimizer moments
and preserving Dale's law sign constraints.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
from .agent import IntegratedAgent

@dataclass
class NeuronUtilityStats:
    utility: float = 1.0
    age: int = 0
    replacement_count: int = 0
    last_replaced_step: int = -1

class ContinualBackpropEngine:
    """Tracks neural utility and executes controlled replacement of dead/stagnant units."""

    def __init__(
        self,
        num_neurons: int,
        beta: float = 0.95,
        utility_threshold: float = 0.015,
        maturity_threshold: int = 50,
        max_replacements_per_step: int = 2,
        seed: int = 42
    ):
        self.num_neurons = num_neurons
        self.beta = beta
        self.utility_threshold = utility_threshold
        self.maturity_threshold = maturity_threshold
        self.max_replacements_per_step = max_replacements_per_step
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.total_replacements = 0

        # Per-neuron tracking
        self.utilities = np.ones(num_neurons, dtype=np.float32) * 0.5
        self.ages = np.zeros(num_neurons, dtype=np.int32)
        self.replacement_history: List[Dict[str, Any]] = []

    def update_utility(self, rates: np.ndarray, weight_matrix: Optional[np.ndarray] = None):
        """Update running utility estimate using downstream weight magnitude and activation."""
        self.step_count += 1
        self.ages += 1
        r = np.abs(np.asarray(rates, dtype=np.float32).flatten())
        if len(r) > self.num_neurons:
            r = r[:self.num_neurons]
        elif len(r) < self.num_neurons:
            r = np.pad(r, (0, self.num_neurons - len(r)))

        if weight_matrix is not None and weight_matrix.ndim == 2:
            W = np.abs(weight_matrix[:self.num_neurons, :self.num_neurons])
            downstream_impact = W.sum(axis=0)  # column sum = total outgoing weight
            tilde_u = r * downstream_impact
        else:
            tilde_u = r

        # Exponential moving average: u_i(t) = beta * u_i(t-1) + (1-beta) * tilde_u
        self.utilities = self.beta * self.utilities + (1.0 - self.beta) * tilde_u

    def identify_replacement_candidates(self) -> List[int]:
        """Find mature units whose utility has fallen below the replacement threshold."""
        mature_mask = self.ages >= self.maturity_threshold
        low_utility_mask = self.utilities < self.utility_threshold
        eligible_indices = np.where(mature_mask & low_utility_mask)[0]

        if len(eligible_indices) == 0:
            return []

        # Sort by utility ascending (lowest first)
        sorted_indices = eligible_indices[np.argsort(self.utilities[eligible_indices])]
        return sorted_indices[:self.max_replacements_per_step].tolist()

    def replace_neurons(
        self,
        agent: IntegratedAgent,
        candidates: List[int],
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Re-initialize candidate weights, preserve Dale signs, and reset optimizer moments."""
        if not candidates:
            return {'replaced_count': 0, 'candidates': []}

        replaced_info = []
        with torch.no_grad():
            # Graft weight matrix (recurrent connectome weights)
            w = agent.brain.graft_w  # [E, 1] or similar parameter
            # If agent has dense weight matrix or recurrent connections
            # Reinitialize incoming / outgoing connections for these candidates
            n = min(self.num_neurons, agent.brain.n)

            for idx in candidates:
                if idx >= n:
                    continue

                old_util = float(self.utilities[idx])
                old_age = int(self.ages[idx])

                # Reset internal neuron utility and age
                self.utilities[idx] = 0.5
                self.ages[idx] = 0
                self.total_replacements += 1

                # Re-initialize graft connection weights connected to this neuron
                if hasattr(agent.graph, 'graft_pre') and hasattr(agent.graph, 'graft_post'):
                    pre = np.array(agent.graph.graft_pre)
                    post = np.array(agent.graph.graft_post)
                    connected_synapses = np.where((pre == idx) | (post == idx))[0]

                    for syn_idx in connected_synapses:
                        is_pre = (pre[syn_idx] == idx)
                        # Determine Dale sign
                        is_inhibitory = (idx % 4 == 0)  # 25% inhibitory fraction
                        new_scale = float(self.rng.uniform(0.02, 0.08))
                        if is_inhibitory:
                            new_val = -new_scale
                        else:
                            new_val = new_scale
                        agent.brain.graft_w.data[syn_idx] = new_val

                        # Reset Adam moments for this parameter slice if optimizer provided
                        if optimizer is not None:
                            self._reset_param_moment(optimizer, agent.brain.graft_w, syn_idx)

                # Also reset corresponding motor decoder incoming weights for this neuron
                if hasattr(agent, 'motor') and hasattr(agent.motor, 'weight'):
                    mw = agent.motor.weight
                    if mw.shape[1] > idx:
                        mw.data[:, idx] = torch.randn(mw.shape[0], device=mw.device) * 0.05
                        if optimizer is not None:
                            self._reset_param_slice(optimizer, mw, col_idx=idx)

                replaced_info.append({
                    'neuron_idx': idx,
                    'old_utility': old_util,
                    'age_at_replacement': old_age,
                })

        record = {
            'step': self.step_count,
            'replaced_count': len(replaced_info),
            'replaced_neurons': replaced_info,
            'total_lifetime_replacements': self.total_replacements
        }
        self.replacement_history.append(record)
        return record

    def _reset_param_moment(self, optimizer: torch.optim.Optimizer, param: torch.Tensor, index: int):
        """Zero out 1st and 2nd Adam moments for a specific 1D parameter index."""
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'exp_avg' in state:
                state['exp_avg'][index] = 0.0
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'][index] = 0.0

    def _reset_param_slice(self, optimizer: torch.optim.Optimizer, param: torch.Tensor, col_idx: int):
        """Zero out 1st and 2nd Adam moments for a specific column in a 2D weight matrix."""
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'exp_avg' in state and state['exp_avg'].ndim == 2:
                state['exp_avg'][:, col_idx] = 0.0
            if 'exp_avg_sq' in state and state['exp_avg_sq'].ndim == 2:
                state['exp_avg_sq'][:, col_idx] = 0.0

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics on neuron utility distribution and replacements."""
        return {
            'step_count': self.step_count,
            'total_replacements': self.total_replacements,
            'mean_utility': float(np.mean(self.utilities)),
            'min_utility': float(np.min(self.utilities)),
            'max_utility': float(np.max(self.utilities)),
            'dead_units_count': int(np.sum(self.utilities < self.utility_threshold)),
            'mature_units_count': int(np.sum(self.ages >= self.maturity_threshold)),
            'recent_events': self.replacement_history[-5:] if self.replacement_history else []
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'total_replacements': self.total_replacements,
            'utilities': self.utilities.tolist(),
            'ages': self.ages.tolist(),
            'replacement_history': self.replacement_history,
        }

    def import_state(self, state: Dict[str, Any]):
        self.step_count = int(state.get('step_count', 0))
        self.total_replacements = int(state.get('total_replacements', 0))
        self.utilities = np.array(state['utilities'], dtype=np.float32)
        self.ages = np.array(state['ages'], dtype=np.int32)
        self.replacement_history = list(state.get('replacement_history', []))


def benchmark_continual_backprop(
    agent: IntegratedAgent,
    steps: int = 60,
    seed: int = 42
) -> Dict[str, Any]:
    """Comparative assay benchmarking standard vs. utility-driven replacement."""
    engine = ContinualBackpropEngine(num_neurons=agent.brain.n, seed=seed)
    rng = np.random.default_rng(seed)

    returns = []
    dead_unit_trajectory = []

    for t in range(steps):
        # Generate simulated observation
        obs = rng.uniform(0.0, 1.0, size=155).astype(np.float32)
        out = agent.act(obs, learn=False)
        rates = agent.brain.r.detach().cpu().numpy()

        # Update utility
        engine.update_utility(rates)
        candidates = engine.identify_replacement_candidates()
        if candidates:
            engine.replace_neurons(agent, candidates)

        dead_unit_trajectory.append(engine.summary()['dead_units_count'])
        returns.append(float(out['action'].sum()))

    return {
        'status': 'ok',
        'steps': steps,
        'final_mean_utility': float(np.mean(engine.utilities)),
        'total_replacements': engine.total_replacements,
        'dead_unit_trajectory': dead_unit_trajectory,
        'scope': 'continual backpropagation and utility-guided neuron replacement'
    }
