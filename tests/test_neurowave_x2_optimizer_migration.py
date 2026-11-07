import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from synapse.neurowave_x2.optimizer_migration import (
    migrate_optimizer_state_for_edges,
    clone_optimizer_state,
    restore_optimizer_state,
)


def test_migrate_optimizer_state_for_edges():
    param_old = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    opt = optim.Adam([param_old], lr=1e-3)

    # Perform a dummy step so optimizer.state has exp_avg, exp_avg_sq
    loss = (param_old ** 2).sum()
    loss.backward()
    opt.step()

    assert param_old in opt.state
    assert "exp_avg" in opt.state[param_old]
    old_exp_avg = opt.state[param_old]["exp_avg"].clone()

    # Mutation: expand to 3 edges (add 1 new edge)
    param_new = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    retained = [0, 1]
    new_idx = [2]

    migrate_optimizer_state_for_edges(opt, param_old, param_new, retained, new_idx)

    assert param_old not in opt.state
    assert param_new in opt.state
    new_exp_avg = opt.state[param_new]["exp_avg"]
    assert new_exp_avg.shape == (3, 2)
    # Retained edges preserve prior moments
    assert torch.allclose(new_exp_avg[:2], old_exp_avg)
    # New edge has zero moments
    assert torch.allclose(new_exp_avg[2], torch.zeros(2))
    # Param group now points to param_new
    assert opt.param_groups[0]["params"][0] is param_new


def test_clone_and_restore_optimizer_state():
    param = nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = optim.Adam([param], lr=1e-3)

    loss = (param ** 2).sum()
    loss.backward()
    opt.step()

    cloned = clone_optimizer_state(opt)
    p_id = id(param)
    assert p_id in cloned
    orig_step = cloned[p_id]["step"]

    # Step again to alter state
    loss = (param ** 2).sum()
    loss.backward()
    opt.step()
    assert opt.state[param]["step"] > orig_step

    # Restore
    restore_optimizer_state(opt, cloned)
    assert opt.state[param]["step"] == orig_step
