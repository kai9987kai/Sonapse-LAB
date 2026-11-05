"""Pearlian causal mediation analysis for weight-space programs and connectome grafts.
Decomposes the total effect of a weight program into:
- Natural Direct Effect (NDE): direct impact on motor readout without changing connectome dynamics.
- Natural Indirect Effect (NIE): mediated impact transmitted through recurrent connectome dynamics.
"""
from __future__ import annotations
import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .patches import apply_patch, rollback, preview_patch
from graftbrain.world import World

def run_causal_mediation_assay(
    agent: IntegratedAgent,
    patch_packet: Dict[str, Any],
    task: str = 'forage',
    num_episodes: int = 2,
    max_steps: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute Pearlian causal mediation analysis comparing baseline and patched models.
    
    1. Condition 0 (Baseline): Base weights theta_0 -> produces rates R_0 and actions A_0.
    2. Condition 1 (Patched): Patched weights theta_1 -> produces rates R_1 and actions A_1.
    3. Condition 2 (Counterfactual Mediation): Patched weights theta_1, but intervening
       on connectome pooled rates to force them to match R_0.
    """
    preview_patch(agent, patch_packet)
    world = World(seed=seed, curriculum=task)
    saved_state = copy.deepcopy(agent.export_state())

    # --- Phase 1: Baseline run (theta_0) ---
    base_actions: List[np.ndarray] = []
    base_rates: List[torch.Tensor] = []
    base_returns: List[float] = []

    for ep in range(num_episodes):
        obs = world.reset(task if task != 'auto' else None)
        agent.reset_episode()
        ep_ret = 0.0
        for _ in range(max_steps):
            out = agent.act(obs, learn=False)
            base_actions.append(out['action'])
            base_rates.append(agent.brain.r.clone().detach())
            obs, r, done, _ = world.step(out['action'])
            ep_ret += r
            if done:
                break
        base_returns.append(ep_ret)

    # --- Phase 2: Patched run (theta_1) ---
    agent.import_state(copy.deepcopy(saved_state))
    receipt = apply_patch(agent, patch_packet)
    patched_actions: List[np.ndarray] = []
    patched_rates: List[torch.Tensor] = []
    patched_returns: List[float] = []

    world = World(seed=seed, curriculum=task)
    for ep in range(num_episodes):
        obs = world.reset(task if task != 'auto' else None)
        agent.reset_episode()
        ep_ret = 0.0
        for _ in range(max_steps):
            out = agent.act(obs, learn=False)
            patched_actions.append(out['action'])
            patched_rates.append(agent.brain.r.clone().detach())
            obs, r, done, _ = world.step(out['action'])
            ep_ret += r
            if done:
                break
        patched_returns.append(ep_ret)

    # --- Phase 3: Counterfactual Mediation Interventions ---
    # Patched agent with connectome rate intervention R <- R_0
    counterfactual_actions: List[np.ndarray] = []
    min_steps = min(len(base_actions), len(patched_actions))

    world = World(seed=seed, curriculum=task)
    step_idx = 0
    for ep in range(num_episodes):
        obs = world.reset(task if task != 'auto' else None)
        agent.reset_episode()
        for _ in range(max_steps):
            if step_idx >= min_steps:
                break
            # Force connectome rate to baseline rate R_0
            with torch.no_grad():
                agent.brain.r.copy_(base_rates[step_idx])
            out = agent.act(obs, learn=False)
            counterfactual_actions.append(out['action'])
            obs, _, done, _ = world.step(out['action'])
            step_idx += 1
            if done:
                break

    # Restore agent to original baseline state
    agent.import_state(saved_state)

    # Compute effects on action space
    N = min(len(base_actions), len(patched_actions), len(counterfactual_actions))
    A_base = np.asarray(base_actions[:N])
    A_patch = np.asarray(patched_actions[:N])
    A_cf = np.asarray(counterfactual_actions[:N])

    # Total Effect: ||A_patch - A_base||
    total_effect = float(np.mean(np.linalg.norm(A_patch - A_base, axis=1)))

    # Natural Direct Effect: ||A_cf - A_base|| (effect of weight change when rates held to baseline)
    natural_direct_effect = float(np.mean(np.linalg.norm(A_cf - A_base, axis=1)))

    # Natural Indirect Effect: ||A_patch - A_cf|| (effect of rate change mediated through connectome)
    natural_indirect_effect = float(np.mean(np.linalg.norm(A_patch - A_cf, axis=1)))

    # Mediation Ratio: proportion of effect mediated through connectome rates
    denom = natural_direct_effect + natural_indirect_effect + 1e-8
    mediation_ratio = float(natural_indirect_effect / denom)

    # Connectome rate shift: ||R_1 - R_0||
    R_base_mat = torch.stack(base_rates[:N]).numpy()
    R_patch_mat = torch.stack(patched_rates[:N]).numpy()
    rate_shift_norm = float(np.mean(np.linalg.norm(R_patch_mat - R_base_mat, axis=1)))

    return {
        'status': 'ok',
        'task': task,
        'steps_analyzed': N,
        'baseline_return': float(np.mean(base_returns)),
        'patched_return': float(np.mean(patched_returns)),
        'total_effect': total_effect,
        'natural_direct_effect': natural_direct_effect,
        'natural_indirect_effect': natural_indirect_effect,
        'mediation_ratio': mediation_ratio,
        'connectome_rate_shift_l2': rate_shift_norm,
        'connectome_mediated': mediation_ratio > 0.40,
        'scope': 'Pearlian causal mediation analysis in weight space',
    }
