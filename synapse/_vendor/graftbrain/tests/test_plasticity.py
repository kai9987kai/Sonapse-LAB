"""Behavioral contracts for bounded candidate graft adaptation."""
from copy import deepcopy
from importlib import import_module

import numpy as np
import pytest
import torch

from graftbrain.graft import GraftedGraph


def engine_api():
    module = import_module("graftbrain.plasticity")
    return module.PlasticityConfig, module.GraftPlasticity


def graph_fixture(edges_per_direction=20, empty_site=False):
    forward = np.arange(edges_per_direction) % 4
    pre = np.concatenate((forward, forward + 4)).astype(np.int64)
    post = np.concatenate((forward + 4, forward)).astype(np.int64)
    count = len(pre)
    sites = [{"id": 0, "kind": "bridge"}] if count else []
    if empty_site:
        sites.append({"id": len(sites), "kind": "empty"})
    return GraftedGraph(
        ["A", "B"], np.array([0, 4, 8]), np.full(8, "central"),
        np.full(8, "x"), np.repeat([0, 1], 4), np.array([0, 4]),
        np.array([1, 5]), np.array([0.2, -0.3], np.float32), pre, post,
        np.where(np.arange(count) % 2, -0.1, 0.1).astype(np.float32),
        np.zeros(count, dtype=np.int64), sites,
    )


def canonical_support(module):
    module.active_mask.zero_()
    module.active_mask[:12] = True
    module.active_mask[20:32] = True


def test_legacy_is_neutral_and_preserves_shared_torch_rng():
    Config, Engine = engine_api()
    graph = graph_fixture()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    module = Engine(graph, 2, Config(), seed=90)
    assert torch.equal(torch.get_rng_state(), before)
    weights = torch.tensor(graph.graft_w0, requires_grad=True)
    gains = module.gates(torch.tensor([2.0, -1.0]))
    assert gains.tolist() == [1.0]
    assert torch.equal(module.effective_weights(weights, gains), weights)
    assert module.router is None
    assert not module.last_gates.requires_grad


@pytest.mark.parametrize("mode", ["static", "routing", "structural", "adaptive"])
def test_sparse_modes_share_seeded_directional_budget_and_dormant_isolation(mode):
    Config, Engine = engine_api()
    graph = graph_fixture()
    module = Engine(graph, 2, Config(mode=mode), seed=7)
    control = Engine(graph, 2, Config(mode="static"), seed=7)
    assert torch.equal(module.active_mask, control.active_mask)
    assert module.active_mask[:20].sum().item() == 12
    assert module.active_mask[20:].sum().item() == 12
    weights = torch.ones(40, requires_grad=True)
    effective = module.effective_weights(weights, module.gates(torch.zeros(2)))
    effective.sum().backward()
    assert torch.equal(weights.grad, module.active_mask.float())
    rates = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    module.observe(rates)
    assert torch.all(module.coactivity_ema[~module.active_mask] > 0)
    assert not module.coactivity_ema.requires_grad


def test_router_learns_different_context_gains_and_keeps_detached_telemetry():
    Config, Engine = engine_api()
    torch.manual_seed(13)
    before = torch.get_rng_state().clone()
    module = Engine(graph_fixture(), 2, Config(mode="routing"), seed=5)
    assert torch.equal(torch.get_rng_state(), before)
    optimizer = torch.optim.Adam(module.router.parameters(), lr=0.1)
    for _ in range(60):
        positive = module.gates(torch.tensor([1.0, 0.0]))
        negative = module.gates(torch.tensor([-1.0, 0.0]))
        loss = ((positive - 1.5) ** 2).mean() + ((negative - 0.5) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(module.router.weight.grad).all()
        assert module.router.weight.grad.abs().sum() > 0
        optimizer.step()
    assert module.gates(torch.tensor([1.0, 0.0])).item() > 1.4
    assert module.gates(torch.tensor([-1.0, 0.0])).item() < 0.6
    assert module.last_gates.grad_fn is None
    assert module.regularization(torch.ones(1)).item() == 0.0
    assert module.regularization(torch.full((1,), 2.0)).item() == pytest.approx(0.001)


def test_rewire_preserves_directional_budgets_and_resets_only_changed_traces():
    Config, Engine = engine_api()
    graph = graph_fixture(empty_site=True)
    base_before = graph.base_w.copy()
    module = Engine(graph, 2, Config(mode="adaptive", interval=1, warmup=0, turnover=0.25))
    canonical_support(module)
    module.coactivity_ema.copy_(torch.arange(40))
    weights = torch.nn.Parameter(torch.full((40,), 0.25))
    optimizer = torch.optim.Adam([weights], lr=0.01, amsgrad=True)
    weights.sum().backward()
    optimizer.step()
    with torch.no_grad():
        weights.fill_(0.25)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            optimizer.state[weights][key].fill_(3)
    elig, heb = torch.ones(40), torch.full((40,), 2.0)
    event = module.after_update(weights, optimizer, elig, heb)
    assert set(event["pruned"]) == {0, 1, 2, 20, 21, 22}
    assert set(event["grown"]) == {17, 18, 19, 37, 38, 39}
    changed = sorted(event["pruned"] + event["grown"])
    unchanged = sorted(set(range(40)) - set(changed))
    assert not set(event["pruned"]) & set(event["grown"])
    assert module.active_mask[:20].sum().item() == 12
    assert module.active_mask[20:].sum().item() == 12
    assert torch.equal(weights[event["grown"]], torch.tensor(graph.graft_w0[event["grown"]]))
    assert torch.all(weights[event["pruned"]] == 0.25)
    assert torch.all(elig[changed] == 0) and torch.all(heb[changed] == 0)
    assert torch.all(elig[unchanged] == 1) and torch.all(heb[unchanged] == 2)
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.all(optimizer.state[weights][key][changed] == 0)
        assert torch.all(optimizer.state[weights][key][unchanged] == 3)
    assert np.array_equal(graph.base_w, base_before)
    stats = module.site_stats()
    assert stats[0]["candidates"] == 40 and stats[0]["active"] == 24
    assert stats[0]["swaps"] == 6 and stats[1]["active"] == 0


def test_equal_score_order_is_candidate_index_and_schedule_waits_for_warmup():
    Config, Engine = engine_api()
    module = Engine(graph_fixture(), 2, Config(mode="structural", interval=2, warmup=2, turnover=0.25))
    canonical_support(module)
    weights = torch.nn.Parameter(torch.ones(40))
    traces = torch.ones(40)
    for _ in range(3):
        assert module.after_update(weights, None, traces, traces) is None
    event = module.after_update(weights, None, traces, traces)
    assert event["update"] == 4
    assert event["pruned"] == [0, 1, 2, 20, 21, 22]
    assert event["grown"] == [12, 13, 14, 32, 33, 34]


@pytest.mark.parametrize("mode", ["legacy", "static", "routing"])
def test_nonstructural_modes_never_rewire(mode):
    Config, Engine = engine_api()
    module = Engine(graph_fixture(), 2, Config(mode=mode, interval=1, warmup=0, turnover=1))
    before = module.active_mask.clone()
    weights = torch.nn.Parameter(torch.ones(40))
    for _ in range(3):
        assert module.after_update(weights, None, torch.ones(40), torch.ones(40)) is None
    assert torch.equal(module.active_mask, before)
    assert module.summary()["updates"] == 3


@pytest.mark.parametrize("edges,density", [(0, 0.6), (1, 0.1), (20, 1.0)])
def test_empty_tiny_and_full_buckets_are_safe_noops(edges, density):
    Config, Engine = engine_api()
    graph = graph_fixture(edges)
    module = Engine(graph, 2, Config(mode="adaptive", density=density, interval=1, warmup=0))
    weights = torch.nn.Parameter(torch.tensor(graph.graft_w0))
    gates = module.gates(torch.zeros(2))
    assert torch.isfinite(module.regularization(gates))
    assert module.after_update(weights, None, torch.zeros(len(weights)), torch.zeros(len(weights))) is None
    assert module.summary()["gate_mean"] == (1.0 if edges else 0.0)
    if edges == 1:
        assert module.active_mask.tolist() == [True, True]


def test_roundtrip_reproduces_next_rewire_effective_weights_and_telemetry():
    Config, Engine = engine_api()
    graph = graph_fixture()
    config = Config(mode="adaptive", interval=2, warmup=0, turnover=0.25)
    module = Engine(graph, 2, config, seed=8)
    weights = torch.nn.Parameter(torch.tensor(graph.graft_w0))
    elig, heb = torch.ones(40), torch.ones(40)
    module.observe(torch.arange(8, dtype=torch.float32) / 8)
    assert module.after_update(weights, None, elig, heb) is None
    state = deepcopy(module.state_dict())
    restored = Engine(graph, 2, config, seed=99)
    restored.load_state_dict(state)
    cloned_weights = torch.nn.Parameter(weights.detach().clone())
    event = module.after_update(weights, None, elig, heb)
    same_event = restored.after_update(cloned_weights, None, elig.clone(), heb.clone())
    assert event == same_event
    obs = torch.tensor([0.2, -0.4])
    assert torch.equal(module.effective_weights(weights, module.gates(obs)),
                       restored.effective_weights(cloned_weights, restored.gates(obs)))
    assert module.summary() == restored.summary()
    assert module.site_stats() == restored.site_stats()
    after_event = Engine(graph, 2, config)
    after_event.load_state_dict(deepcopy(module.state_dict()))
    assert after_event.summary()["last_event"] == event


def test_restore_rejects_changed_candidate_identity_or_configuration_before_mutation():
    Config, Engine = engine_api()
    source = Engine(graph_fixture(), 2, Config(mode="adaptive"))
    graph = graph_fixture()
    graph.graft_post[0] = 5
    wrong_graph = Engine(graph, 2, Config(mode="adaptive"))
    before = wrong_graph.active_mask.clone()
    with pytest.raises(RuntimeError, match="candidate"):
        wrong_graph.load_state_dict(source.state_dict())
    assert torch.equal(wrong_graph.active_mask, before)
    wrong_config = Engine(graph_fixture(), 2, Config(mode="adaptive", density=0.5))
    with pytest.raises(RuntimeError, match="config"):
        wrong_config.load_state_dict(source.state_dict())


@pytest.mark.parametrize("kwargs", [
    {"mode": "unknown"}, {"density": 0}, {"density": 1.1}, {"density": float("nan")},
    {"interval": 0}, {"interval": 1.5}, {"warmup": -1}, {"warmup": 0.5},
    {"turnover": -0.1}, {"turnover": 1.1}, {"router_lr": 0}, {"router_lr": float("inf")},
])
def test_configuration_rejects_invalid_bounds(kwargs):
    Config, _ = engine_api()
    with pytest.raises(ValueError):
        Config(**kwargs)
