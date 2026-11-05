"""Activity-dependent structural plasticity, synaptogenesis and synaptic pruning.
Implements developmental connectome topological evolution under strict wiring economy.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .patches import make_patch

def audit_structural_topology(agent: IntegratedAgent) -> Dict[str, Any]:
    """Audit the topological wiring properties of the agent's connectome."""
    graph = agent.graph
    graft_w = agent.brain.graft_w.detach()
    n_neurons = graph.n
    n_base = len(graph.base_pre)
    n_graft = len(graph.graft_pre)
    active_grafts = int((graft_w.abs() > 0.01).sum())

    # Origin distribution
    origins, counts = np.unique(graph.origin, return_counts=True)
    origin_map = {int(o): int(c) for o, c in zip(origins, counts)}

    # Inter-neuropil bridge count
    inter_neuropil = 0
    intra_neuropil = 0
    for pre, post in zip(graph.graft_pre, graph.graft_post):
        if graph.origin[pre] != graph.origin[post]:
            inter_neuropil += 1
        else:
            intra_neuropil += 1

    return {
        'status': 'ok',
        'neurons': n_neurons,
        'base_synapses': n_base,
        'graft_synapses': n_graft,
        'active_grafts': active_grafts,
        'inter_neuropil_bridges': inter_neuropil,
        'intra_neuropil_bridges': intra_neuropil,
        'neuropil_neuron_counts': origin_map,
        'synaptic_density': float((n_base + n_graft) / (n_neurons * n_neurons)),
        'scope': 'developmental connectome structural topology audit',
    }


def prune_dormant_grafts(
    agent: IntegratedAgent,
    prune_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Zero out sub-threshold graft weights that fail to justify metabolic maintenance."""
    graft_w = agent.brain.graft_w
    surfaces = agent.editable_tensors()
    edits = []

    indices = []
    values = []
    pruned_count = 0

    with torch.no_grad():
        for i in range(len(graft_w)):
            val = float(graft_w[i].detach())
            if 0.0 < abs(val) < prune_threshold:
                indices.append([i])
                values.append(0.0)
                pruned_count += 1

    if indices:
        edits.append({
            'tensor': 'brain.graft_w',
            'indices': indices,
            'values': values,
            'shape': list(graft_w.shape),
        })

    patch = make_patch(agent, edits, description=f"Structural pruning of {pruned_count} dormant graft synapses")
    return {
        'status': 'ok',
        'pruned_synapses': pruned_count,
        'remaining_active': int((graft_w.abs() >= prune_threshold).sum()),
        'patch_packet': patch,
        'scope': 'activity-dependent synaptic pruning',
    }


def sprout_correlated_synapses(
    agent: IntegratedAgent,
    coactivity_matrix: np.ndarray,
    sprout_threshold: float = 0.60,
    initial_weight: float = 0.05,
    max_sprouts: int = 8,
) -> Dict[str, Any]:
    """Hebbian synaptogenesis: identify co-active neuron pairs across neuropil boundaries
    and propose initial synaptic connections.
    """
    graph = agent.graph
    graft_w = agent.brain.graft_w
    n = graph.n

    existing_edges = set(zip(graph.graft_pre.tolist(), graph.graft_post.tolist()))
    candidates = []

    # Search for highly correlated cross-neuropil neuron pairs
    for i in range(min(n, coactivity_matrix.shape[0])):
        for j in range(min(n, coactivity_matrix.shape[1])):
            if i != j and graph.origin[i] != graph.origin[j]:
                if (i, j) not in existing_edges:
                    corr = float(coactivity_matrix[i, j])
                    if corr > sprout_threshold:
                        candidates.append((i, j, corr))

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = candidates[:max_sprouts]

    sprouted_edges = [
        {'pre': pre, 'post': post, 'correlation': corr, 'initial_weight': initial_weight}
        for pre, post, corr in selected
    ]

    return {
        'status': 'ok',
        'candidates_found': len(candidates),
        'sprouted_count': len(sprouted_edges),
        'sprouted_edges': sprouted_edges,
        'scope': 'Hebbian activity-dependent synaptogenesis',
    }
