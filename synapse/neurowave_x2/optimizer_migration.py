"""Optimizer State Migration for Sonapse NeuroWave-X2 Transactional Topology Mutations.

Preserves momentum and variance moments (exp_avg, exp_avg_sq) for existing parameters,
initializes moments to zero for newly introduced parameters, and cleans up removed parameters.
Supports Adam, AdamW, and SGD.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def migrate_optimizer_state_for_edges(
    optimizer: optim.Optimizer,
    param_old: nn.Parameter,
    param_new: nn.Parameter,
    retained_indices: List[int],
    new_indices: List[int],
) -> None:
    """Migrates optimizer momentum moments from param_old to param_new across a weight mutation.
    
    Args:
        optimizer: PyTorch optimizer (Adam, AdamW, SGD)
        param_old: the parameter before mutation (e.g., shape (num_old_edges, ...))
        param_new: the parameter after mutation (e.g., shape (num_new_edges, ...))
        retained_indices: indices in param_new that map 1-to-1 to param_old[:len(retained_indices)]
        new_indices: newly added edge indices in param_new that should be initialized to zero moments
    """
    if param_old not in optimizer.state:
        # No accumulated moments yet; initialize entry for param_new
        optimizer.state[param_new] = {}
        return

    old_state = optimizer.state[param_old]
    new_state: Dict[str, Any] = {}

    for k, val in old_state.items():
        if isinstance(val, torch.Tensor) and val.dim() > 0:
            # Tensor moment (e.g., exp_avg, exp_avg_sq in Adam, or momentum_buffer in SGD)
            new_moment = torch.zeros_like(param_new.data)
            # Copy retained moments
            if len(retained_indices) > 0:
                n_copy = min(len(retained_indices), val.shape[0])
                new_moment[retained_indices[:n_copy]] = val[:n_copy].to(param_new.device)
            # Newly added indices remain zeroed out
            new_state[k] = new_moment
        elif isinstance(val, torch.Tensor):
            new_state[k] = val.clone()
        else:
            # Scalar value (e.g., step count)
            new_state[k] = val

    # Assign new state and remove old parameter key
    optimizer.state[param_new] = new_state
    del optimizer.state[param_old]

    # Update optimizer param_groups to point to param_new instead of param_old
    for group in optimizer.param_groups:
        for idx, p in enumerate(group["params"]):
            if p is param_old:
                group["params"][idx] = param_new
                break


def clone_optimizer_state(optimizer: optim.Optimizer) -> Dict[int, Dict[str, Any]]:
    """Deep-clones optimizer state dictionary for rollback capabilities."""
    cloned: Dict[int, Dict[str, Any]] = {}
    for p, state_dict in optimizer.state.items():
        p_id = id(p)
        cloned[p_id] = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                cloned[p_id][k] = v.clone()
            else:
                cloned[p_id][k] = v
    return cloned


def restore_optimizer_state(optimizer: optim.Optimizer, saved_state: Dict[int, Dict[str, Any]]) -> None:
    """Restores cloned optimizer state during transaction rollback."""
    optimizer.state.clear()
    for group in optimizer.param_groups:
        for p in group["params"]:
            p_id = id(p)
            if p_id in saved_state:
                optimizer.state[p] = {}
                for k, v in saved_state[p_id].items():
                    if isinstance(v, torch.Tensor):
                        optimizer.state[p][k] = v.clone()
                    else:
                        optimizer.state[p][k] = v
