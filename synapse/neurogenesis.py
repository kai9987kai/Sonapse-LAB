"""v93 Supermix-Archimedes Connectome Neurogenesis & Synaptic Plasticity Engine.

Adapted from Kai9987kai's Supermix Archimedes (v93 / mimomix) architecture.
Implements continuous activity-dependent structural plasticity:
1. **Apoptosis**: Pruning stagnant, weak or inactive synapses below softplus threshold.
2. **Synaptogenesis**: Sprouting new synaptic connections between pairs of neurons
   exhibiting high pre-post firing rate covariance.
3. **Sign-aware Growth**: Reuses consistent frozen-source polarity when available;
   otherwise retains the replaced slot's sign. No measured cell-type guarantee.
4. **Neuromodulatory & Astrocytic Gating**: Synaptic growth is facilitated by elevated Dopamine (positive RPE)
   and Astrocytic [Ca2+] spillover, and suppressed under high stress/threat.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class NeurogenesisStats:
    step: int = 0
    synapses_born: int = 0
    synapses_pruned: int = 0
    active_synapses: int = 0
    mean_weight: float = 0.0
    synaptogenesis_events: int = 0
    apoptosis_events: int = 0
    recent_events: List[Dict[str, Any]] = field(default_factory=list)


class ConnectomeNeurogenesisEngine:
    """Orchestrates activity-dependent structural remodeling of the grafted connectome substrate."""

    def __init__(
        self,
        num_neurons: int,
        prune_threshold: float = 0.02,
        growth_rate: float = 0.05,
        covariance_decay: float = 0.95,
        max_new_synapses_per_event: int = 8,
        seed: int = 42,
    ):
        self.num_neurons = num_neurons
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        self.covariance_decay = covariance_decay
        self.max_new_synapses = max_new_synapses_per_event
        self.rng = np.random.default_rng(seed)

        # Running rate covariance accumulator [N, N]
        self.rate_mean = np.zeros(num_neurons, dtype=np.float32)
        self.rate_cov = np.zeros((num_neurons, num_neurons), dtype=np.float32)
        self.stats = NeurogenesisStats()
        self.activity_steps = 0

    def record_activity(self, rates: np.ndarray | torch.Tensor) -> None:
        """Accumulates neural firing rate covariance across microsteps."""
        if isinstance(rates, torch.Tensor):
            r = rates.detach().cpu().numpy().flatten()
        else:
            r = np.asarray(rates, dtype=np.float32).flatten()

        n = min(len(r), self.num_neurons)
        r = r[:n]

        # Online mean and covariance update (Welford-style decay)
        delta = r - self.rate_mean[:n]
        self.rate_mean[:n] += 0.05 * delta
        self.rate_cov[:n, :n] = (
            self.covariance_decay * self.rate_cov[:n, :n]
            + (1.0 - self.covariance_decay) * np.outer(delta, delta)
        )
        self.activity_steps += 1

    def evaluate_structural_remodeling(
        self,
        agent: Any,
        dopamine: float = 1.0,
        astrocyte_calcium: float = 1.0,
        threat: float = 0.0,
    ) -> Dict[str, Any]:
        """Executes one round of Supermix Archimedes apoptosis & synaptogenesis."""
        self.stats.step += 1
        event_record: Dict[str, Any] = {
            'step': self.stats.step,
            'pruned': 0,
            'sprouted': 0,
            'dopamine': float(dopamine),
            'calcium': float(astrocyte_calcium),
        }

        brain = getattr(agent, 'brain', None)
        graph = getattr(agent, 'graph', None)
        if brain is None or graph is None:
            return event_record

        # Current graft weights and edges
        g_w = brain.graft_w.detach().cpu().numpy().copy()
        pre = np.array(graph.graft_pre, dtype=np.int64)
        post = np.array(graph.graft_post, dtype=np.int64)
        n_edges = len(g_w)
        changed_slots = []

        # -------------------------------------------------------------
        # 1. APOPTOSIS (Synaptic Pruning)
        # -------------------------------------------------------------
        # Prune synapses with near-zero transmission efficacy
        alive_mask = np.abs(g_w) >= self.prune_threshold
        pruned_count = int(np.sum(~alive_mask))

        if pruned_count > 0:
            # Weaken pruned synapses toward zero
            g_w[~alive_mask] *= 0.1
            self.stats.synapses_pruned += pruned_count
            self.stats.apoptosis_events += 1
            event_record['pruned'] = pruned_count

        # -------------------------------------------------------------
        # 2. SYNAPTOGENESIS (Sprouting between covarying neurons)
        # -------------------------------------------------------------
        # Neurogenesis gating: promoted by Dopamine RPE and Astrocyte [Ca2+]
        growth_drive = float(np.clip(dopamine * 0.7 + astrocyte_calcium * 0.3 - threat * 0.8, 0.0, 2.0))

        if growth_drive > 0.4 and self.activity_steps > 10:
            # Find candidate pairs with high positive rate covariance
            existing_pairs = set(zip(pre.tolist(), post.tolist()))
            n = min(self.num_neurons, self.rate_cov.shape[0])

            # Zero out diagonal (no self-loops)
            cov_mat = self.rate_cov[:n, :n].copy()
            np.fill_diagonal(cov_mat, -1.0)

            # Find top covarying candidate pairs
            flat_indices = np.argsort(cov_mat.ravel())[::-1]
            sprouted = 0

            for idx in flat_indices:
                if sprouted >= self.max_new_synapses:
                    break
                p_i, p_j = divmod(int(idx), n)
                cov_val = cov_mat[p_i, p_j]
                if cov_val < 0.01:
                    break

                # If connection already exists in graft, strengthen it
                if (p_i, p_j) in existing_pairs:
                    continue

                # Sprout new synaptic coupling within existing graft weight slots
                # (Find least used edge or weakest slot to repurpose)
                compatible=np.arange(n_edges)
                if hasattr(agent,'plasticity'):
                    # Birth telemetry represents an executing edge. Replacing an
                    # active weak slot preserves the fixed directional budget.
                    compatible=compatible[agent.plasticity.active_mask.detach().cpu().numpy()]
                if hasattr(graph,'origin'):
                    compatible=compatible[(graph.origin[pre[compatible]]==graph.origin[p_i])
                                          & (graph.origin[post[compatible]]==graph.origin[p_j])]
                if not len(compatible):
                    continue
                weakest_idx = int(compatible[np.argmin(np.abs(g_w[compatible]))])
                if np.abs(g_w[weakest_idx]) < self.prune_threshold * 1.5:
                    old_sign=float(np.sign(g_w[weakest_idx])) or 1.0
                    pre[weakest_idx] = p_i
                    post[weakest_idx] = p_j
                    source_signs=np.sign(np.asarray(graph.base_w)[np.asarray(graph.base_pre)==p_i]) if hasattr(graph,'base_w') else np.array([])
                    source_signs=source_signs[source_signs!=0]
                    sign=float(source_signs[0]) if len(source_signs) and np.all(source_signs==source_signs[0]) else old_sign
                    g_w[weakest_idx] = sign * (0.05 + 0.03 * growth_drive)
                    changed_slots.append(weakest_idx)
                    existing_pairs.add((p_i, p_j))
                    sprouted += 1

            if sprouted > 0:
                self.stats.synapses_born += sprouted
                self.stats.synaptogenesis_events += 1
                event_record['sprouted'] = sprouted

        # Apply updated weights and topology back to brain
        if changed_slots and hasattr(agent,'rewire_graft_slots'):
            agent.rewire_graft_slots(pre,post,g_w,changed_slots)
        else:
            with torch.no_grad():
                brain.graft_w.copy_(torch.tensor(g_w, dtype=torch.float32))
                if hasattr(brain,'graft_pre'):
                    brain.graft_pre.copy_(torch.as_tensor(pre))
                    brain.graft_post.copy_(torch.as_tensor(post))
                graph.graft_pre = pre
                graph.graft_post = post

        # Update stats
        self.stats.active_synapses = int(np.sum(np.abs(g_w) >= self.prune_threshold))
        self.stats.mean_weight = float(np.mean(np.abs(g_w)))
        self.stats.recent_events.append(event_record)
        if len(self.stats.recent_events) > 32:
            self.stats.recent_events.pop(0)

        return event_record

    def export_state(self) -> Dict[str, Any]:
        return {
            'rate_mean': self.rate_mean.copy(),
            'rate_cov': self.rate_cov.copy(),
            'activity_steps': self.activity_steps,
            'rng': self.rng.bit_generator.state,
            'stats': asdict(self.stats),
        }

    def import_state(self, d: Dict[str, Any]) -> None:
        if 'rate_mean' in d:
            self.rate_mean = np.array(d['rate_mean'], dtype=np.float32)
        if 'rate_cov' in d:
            self.rate_cov = np.array(d['rate_cov'], dtype=np.float32)
        self.activity_steps = d.get('activity_steps', 0)
        if 'rng' in d:
            self.rng.bit_generator.state = d['rng']
        if 'stats' in d:
            st = d['stats']
            self.stats = NeurogenesisStats(**st)

    def summary(self) -> Dict[str, Any]:
        """Returns JSON-serializable telemetry dictionary."""
        return {
            'step': self.stats.step,
            'active_synapses': self.stats.active_synapses,
            'synapses_born': self.stats.synapses_born,
            'synapses_pruned': self.stats.synapses_pruned,
            'mean_weight': round(self.stats.mean_weight, 5),
            'synaptogenesis_events': self.stats.synaptogenesis_events,
            'apoptosis_events': self.stats.apoptosis_events,
            'last_event': self.stats.recent_events[-1] if self.stats.recent_events else None,
        }
