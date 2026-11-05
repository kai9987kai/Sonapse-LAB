"""Biological connectome metabolic efficiency and Pareto-optimal weight sparsification.
Models synaptic wiring cost, firing rate homeostatic energy, and sweeps metabolic Pareto frontiers.
"""
from __future__ import annotations
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .patches import make_patch, evaluate_patch
from graftbrain.world import World

def compute_metabolic_cost(
    agent: IntegratedAgent,
    rates: Optional[torch.Tensor] = None,
    alpha_wire: float = 1.0,
    beta_spike: float = 2.0,
    gamma_homeostasis: float = 0.5,
    target_rate: float = 0.15,
) -> Dict[str, float]:
    """Compute the multi-factor biological metabolic energy cost of the agent."""
    graph = agent.graph
    graft_w = agent.brain.graft_w.detach()
    r = rates if rates is not None else agent.brain.r.detach()

    # 1. Wiring cost: inter-neuropil grafts vs local synapses
    # Inter-neuropil axons span larger biological distances (e.g. optic lobe -> thoracic ganglion MANC)
    wire_cost = 0.0
    for idx, (pre, post) in enumerate(zip(graph.graft_pre, graph.graft_post)):
        pre_origin = graph.origin[pre]
        post_origin = graph.origin[post]
        distance = 2.5 if pre_origin != post_origin else 1.0
        wire_cost += float(abs(graft_w[idx])) * distance

    # 2. Spiking / firing rate metabolic consumption
    spike_energy = float((r ** 2).sum())

    # 3. Homeostatic energy penalty: deviations from physiological target rate
    homeostatic_penalty = float(((r - target_rate) ** 2).sum())

    total_cost = alpha_wire * wire_cost + beta_spike * spike_energy + gamma_homeostasis * homeostatic_penalty

    return {
        'total_metabolic_cost': total_cost,
        'wiring_cost': wire_cost,
        'spike_energy': spike_energy,
        'homeostatic_penalty': homeostatic_penalty,
        'active_graft_synapses': int((graft_w.abs() > 0.01).sum()),
        'total_graft_synapses': len(graft_w),
        'mean_rate': float(r.mean()),
    }


def compute_metabolic_pareto_front(
    agent: IntegratedAgent,
    task: str = 'forage',
    num_eval_episodes: int = 2,
    thresholds: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Sweep synaptic pruning thresholds to discover the Pareto-optimal frontier
    balancing behavioral return against metabolic energy consumption.
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]

    original_state = copy.deepcopy(agent.export_state())
    results: List[Dict[str, Any]] = []

    def evaluate_current(eval_agent: IntegratedAgent) -> Tuple[float, Dict[str, float]]:
        world = World(seed=seed, curriculum=task)
        ep_returns = []
        recorded_rates = []
        for _ in range(num_eval_episodes):
            obs = world.reset(task if task != 'auto' else None)
            eval_agent.reset_episode()
            ret = 0.0
            for _ in range(30):
                out = eval_agent.act(obs, learn=False)
                recorded_rates.append(eval_agent.brain.r.clone().detach())
                obs, r, done, _ = world.step(out['action'])
                ret += r
                if done:
                    break
            ep_returns.append(ret)
        avg_rates = torch.stack(recorded_rates).mean(dim=0) if recorded_rates else eval_agent.brain.r.detach()
        cost = compute_metabolic_cost(eval_agent, rates=avg_rates)
        return float(np.mean(ep_returns)), cost

    # Evaluate baseline
    base_return, base_cost = evaluate_current(agent)
    results.append({
        'pruning_threshold': 0.0,
        'task_return': base_return,
        'metabolic_cost': base_cost['total_metabolic_cost'],
        'wiring_cost': base_cost['wiring_cost'],
        'active_synapses': base_cost['active_graft_synapses'],
        'pareto_efficient': True,
    })

    # Sweep pruning levels
    graft_w = agent.brain.graft_w
    for tau in thresholds[1:]:
        agent.import_state(copy.deepcopy(original_state))
        with torch.no_grad():
            mask = graft_w.abs() < tau
            graft_w[mask] = 0.0
        
        ret, cost = evaluate_current(agent)
        results.append({
            'pruning_threshold': float(tau),
            'task_return': ret,
            'metabolic_cost': cost['total_metabolic_cost'],
            'wiring_cost': cost['wiring_cost'],
            'active_synapses': cost['active_graft_synapses'],
            'pareto_efficient': False,
        })

    # Restore agent
    agent.import_state(original_state)

    # Classify Pareto frontier: a point is Pareto optimal if no other point has both
    # higher return and lower metabolic cost.
    for i, p1 in enumerate(results):
        dominated = False
        for j, p2 in enumerate(results):
            if i != j:
                if p2['task_return'] >= p1['task_return'] and p2['metabolic_cost'] <= p1['metabolic_cost']:
                    if p2['task_return'] > p1['task_return'] or p2['metabolic_cost'] < p1['metabolic_cost']:
                        dominated = True
                        break
        p1['pareto_efficient'] = not dominated

    pareto_points = [p for p in results if p['pareto_efficient']]
    best_tradeoff = max(results, key=lambda p: p['task_return'] / (p['metabolic_cost'] + 1e-6))

    return {
        'status': 'ok',
        'task': task,
        'frontier_points': results,
        'pareto_optimal_count': len(pareto_points),
        'best_efficiency_threshold': best_tradeoff['pruning_threshold'],
        'baseline_metabolic_cost': base_cost['total_metabolic_cost'],
        'scope': 'biological connectome metabolic Pareto optimization',
    }
