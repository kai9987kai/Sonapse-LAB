"""Loss-credit, edge residency, support interventions and compatibility contracts."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from graftbrain.plasticity import GraftPlasticity, PlasticityConfig
from test_plasticity import graph_fixture


def make_engine(**kwargs):
    settings = dict(mode="structural", density=0.5, interval=1, warmup=0, turnover=0.5)
    settings.update(kwargs)
    graph = graph_fixture(edges_per_direction=4)
    engine = GraftPlasticity(graph, 2, PlasticityConfig(**settings), seed=17)
    engine.active_mask.copy_(torch.tensor([True, True, False, False] * 2))
    engine.edge_age.zero_()
    engine.edge_age[engine.active_mask] = engine.config.min_age
    return graph, engine


def score(engine, weights, coefficients, gains=None):
    gains = engine.gates(torch.zeros(2)) if gains is None else gains
    effective = engine.effective_weights(weights, gains)
    engine.capture_gradient(effective, gains)
    engine.begin_gradient_update()
    (effective * coefficients).sum().backward()
    engine.finish_gradient_update()
    return effective


@pytest.mark.parametrize("settings", [
    {"growth_rule": "bad"}, {"growth_init": "bad"}, {"min_age": -1},
    {"min_age": 0.5}, {"min_age": True},
])
def test_credit_configuration_validates_before_construction(settings):
    with pytest.raises(ValueError):
        PlasticityConfig(**settings)


def test_min_age_above_int64_range_is_rejected_before_construction():
    with pytest.raises(ValueError, match="min_age"):
        PlasticityConfig(min_age=2**63)


def test_maximum_min_age_remains_mature_after_first_update():
    _, engine = make_engine(min_age=2**63 - 1)
    weights = torch.nn.Parameter(torch.ones(8))
    event = engine.after_update(weights, None, torch.ones(8), torch.ones(8))
    assert event is not None
    assert event["pruned"] == [0, 4]
    assert event["grown"] == [2, 6]
    assert engine.edge_age.tolist() == [0, 2**63 - 1, 0, 0, 0, 2**63 - 1, 0, 0]


def test_maximum_edge_age_saturates_and_roundtrips_without_rewiring():
    _, engine = make_engine(min_age=2**63 - 1)
    weights = torch.nn.Parameter(torch.ones(8))
    for _ in range(2):
        assert engine.after_update(weights, None, torch.ones(8), torch.ones(8), allow_rewire=False) is None
    _, restored = make_engine(min_age=2**63 - 1)
    restored.load_state_dict(deepcopy(engine.state_dict()))
    assert engine.edge_age.tolist() == [2**63 - 1, 2**63 - 1, 0, 0, 2**63 - 1, 2**63 - 1, 0, 0]
    assert torch.equal(restored.edge_age, engine.edge_age)
    expected = engine.after_update(weights, None, torch.ones(8), torch.ones(8))
    actual = restored.after_update(torch.nn.Parameter(torch.ones(8)), None, torch.ones(8), torch.ones(8))
    assert expected == actual
    assert actual["grown"] == [2, 6]


def test_shadow_gradient_matches_latent_finite_difference_without_dormant_signal():
    _, engine = make_engine(growth_rule="gradient")
    weights = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) / 10)
    gain = torch.tensor([1.5], requires_grad=True)
    coefficients = torch.tensor([1., -2., 3., -4., 5., -6., 7., -8.])
    effective = engine.effective_weights(weights, gain)
    engine.capture_gradient(effective, gain)
    engine.begin_gradient_update()
    current = torch.tanh((effective * coefficients).sum())
    current.backward()
    engine.finish_gradient_update()
    dormant = ~engine.active_mask
    assert torch.equal(effective.detach()[dormant], torch.zeros(4))
    assert torch.equal(weights.grad[dormant], torch.zeros(4))
    # Perturb a latent raw coefficient immediately before the masked current.
    edge, epsilon = 2, 1e-3
    baseline = effective.detach().clone()
    positive, negative = baseline.clone(), baseline.clone()
    positive[edge] += epsilon * gain.detach()[0]
    negative[edge] -= epsilon * gain.detach()[0]
    derivative = (torch.tanh((positive * coefficients).sum())
                  - torch.tanh((negative * coefficients).sum())) / (2 * epsilon)
    assert engine.gradient_ema[edge].item() == pytest.approx(0.1 * abs(derivative.item()), rel=0.002)
    assert int(engine.gradient_update_count) == 1
    assert engine.gradient_ema.grad_fn is None
    assert gain.grad is not None


def test_rollout_gradients_sum_signed_before_one_ema_and_are_order_independent():
    outputs = []
    for reverse in (False, True):
        _, engine = make_engine()
        weights = torch.nn.Parameter(torch.ones(8))
        gain = torch.tensor([1.5])
        terms = []
        for coefficient in ([2., -1.] if not reverse else [-1., 2.]):
            effective = engine.effective_weights(weights, gain)
            engine.capture_gradient(effective, gain)
            terms.append(coefficient * effective.sum())
        engine.begin_gradient_update()
        sum(terms).backward()
        assert torch.count_nonzero(engine.gradient_ema) == 0
        engine.finish_gradient_update()
        engine.finish_gradient_update()  # One learner update cannot finalize twice.
        assert torch.allclose(engine.gradient_ema, torch.full((8,), 0.15))
        assert int(engine.gradient_update_count) == 1
        outputs.append(engine.gradient_ema.clone())
    assert torch.allclose(*outputs)


def test_gradient_growth_uses_loss_ranking_and_skips_missing_or_nonpositive_evidence():
    _, engine = make_engine(growth_rule="gradient", growth_init="zero")
    weights = torch.nn.Parameter(torch.ones(8))
    traces = torch.ones(8)
    assert engine.after_update(weights, None, traces, traces) is None
    assert int(engine.swap_count) == 0
    # Coactivity ties prefer 2/6; loss sensitivity requires 3/7.
    coefficients = torch.tensor([1., 1., 0., 9., 1., 1., 0., -8.])
    score(engine, weights, coefficients)
    before = engine.effective_weights(weights).sum().item()
    event = engine.after_update(weights, None, traces, traces)
    assert event["pruned"] == [0, 4]
    assert event["grown"] == [3, 7]
    assert torch.equal(weights[[3, 7]], torch.zeros(2))
    # Immediate loss of dropped current is independent of zero-growth insertion.
    assert engine.effective_weights(weights).sum().item() == before - 2
    assert int(engine.edge_age[3]) == 0


def test_maturation_reduces_swaps_and_disabled_rewiring_still_advances_age():
    _, engine = make_engine(min_age=3, turnover=1., growth_init="zero")
    weights = torch.nn.Parameter(torch.ones(8))
    traces = torch.ones(8)
    event = engine.after_update(weights, None, traces, traces)
    assert event["grown"] == [2, 3, 6, 7]
    assert torch.equal(engine.edge_age, torch.zeros(8, dtype=torch.long))
    assert engine.after_update(weights, None, traces, traces, allow_rewire=False) is None
    assert engine.summary()["protected_active"] == 4
    assert engine.after_update(weights, None, traces, traces) is None
    assert engine.after_update(weights, None, traces, traces)["swaps"] == 4
    assert engine.active_mask[:4].sum().item() == 2
    assert engine.active_mask[4:].sum().item() == 2


def test_zero_growth_adds_no_immediate_current_and_clears_changed_optimizer_history():
    _, engine = make_engine(growth_init="zero")
    weights = torch.nn.Parameter(torch.ones(8))
    optimizer = torch.optim.Adam([weights], amsgrad=True)
    weights.sum().backward()
    optimizer.step()
    with torch.no_grad():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            optimizer.state[weights][name].fill_(3)
    elig, heb = torch.ones(8), torch.full((8,), 2.)
    event = engine.after_update(weights, optimizer, elig, heb)
    assert event["grown"] == [2, 6]
    changed = event["pruned"] + event["grown"]
    assert engine.effective_weights(weights)[event["grown"]].sum().item() == 0
    assert torch.all(elig[changed] == 0) and torch.all(heb[changed] == 0)
    assert torch.all(elig[[1, 3, 5, 7]] == 1)
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.all(optimizer.state[weights][name][changed] == 0)
        assert torch.all(optimizer.state[weights][name][[1, 3, 5, 7]] == 3)


def test_random_growth_is_private_and_checkpoint_reproduces_next_choice():
    _, engine = make_engine(growth_rule="random")
    weights = torch.nn.Parameter(torch.ones(8))
    traces = torch.ones(8)
    torch.manual_seed(921)
    global_before = torch.get_rng_state().clone()
    engine.after_update(weights, None, traces, traces)
    state = deepcopy(engine.state_dict())
    _, restored = make_engine(growth_rule="random")
    restored.load_state_dict(state)
    second_weights = torch.nn.Parameter(weights.detach().clone())
    expected = engine.after_update(weights, None, traces, traces)
    actual = restored.after_update(second_weights, None, traces.clone(), traces.clone())
    assert expected == actual
    assert torch.equal(global_before, torch.get_rng_state())
    assert engine.summary() == restored.summary()


def test_frozen_capture_is_noop_and_nonfinite_evidence_never_grows():
    _, engine = make_engine(growth_rule="gradient")
    weights = torch.nn.Parameter(torch.ones(8))
    before = deepcopy(engine.state_dict())
    with torch.no_grad():
        gains = engine.gates(torch.zeros(2))
        engine.capture_gradient(engine.effective_weights(weights, gains), gains)
    engine.begin_gradient_update()
    engine.finish_gradient_update()
    for name in ("gradient_ema", "gradient_update_count", "gradient_evidence_count"):
        assert torch.equal(before[name], engine.state_dict()[name])
    score(engine, weights, torch.tensor([1., 1., float("nan"), 0., 1., 1., float("inf"), 0.]))
    assert torch.isfinite(engine.gradient_ema).all()
    assert engine.summary()["gradient_nonfinite"] == 2
    assert engine.after_update(weights, None, torch.ones(8), torch.ones(8)) is None


def test_perturbation_preserves_budgets_and_does_not_advance_learning_rng_or_swaps():
    _, engine = make_engine(growth_rule="random", growth_init="zero", min_age=20)
    weights = torch.nn.Parameter(torch.ones(8))
    elig, heb = torch.ones(8), torch.ones(8)
    saved_rng = engine.rng_state.clone()
    saved_global = torch.get_rng_state().clone()
    saved_source = deepcopy(engine.state_dict())
    _, clone = make_engine(growth_rule="random", growth_init="zero", min_age=20)
    clone.load_state_dict(saved_source)
    event = engine.perturb_support(weights, None, elig, heb, fraction=0.5, seed=312)
    clone_event = clone.perturb_support(torch.nn.Parameter(torch.ones(8)), None,
                                       torch.ones(8), torch.ones(8), fraction=0.5, seed=312)
    assert event == clone_event
    assert event["intervention"] is True and event["swaps"] == 2
    assert torch.equal(engine.rng_state, saved_rng)
    assert torch.equal(saved_global, torch.get_rng_state())
    assert engine.active_mask[:4].sum().item() == 2
    assert engine.active_mask[4:].sum().item() == 2
    assert int(engine.swap_count) == 0 and int(engine.rewire_count) == 0
    assert engine.summary()["interventions"] == 1
    assert engine.summary()["intervention_swaps"] == 2
    changed = event["pruned"] + event["grown"]
    assert torch.all(engine.edge_age[changed] == 0)
    assert torch.all(weights[event["grown"]] == 0)
    assert torch.all(elig[changed] == 0) and torch.all(heb[changed] == 0)


def legacy_checkpoint(engine):
    state = deepcopy(engine.state_dict())
    old_buffers = {"candidate_pre", "candidate_post", "candidate_site", "original_weights",
                   "bucket_ids", "bucket_keys", "bucket_budgets", "active_mask", "coactivity_ema",
                   "last_gates", "site_swaps", "update_count", "last_rewire_update", "rewire_count",
                   "swap_count", "rng_state", "_extra_state"}
    state = {key: value for key, value in state.items() if key in old_buffers or key.startswith("router.")}
    metadata = state["_extra_state"]
    metadata["version"] = 1
    for name in ("growth_rule", "growth_init", "min_age"):
        metadata["config"].pop(name)
    return state


def test_version_one_migrates_only_default_policy_and_preserves_next_decision():
    _, engine = make_engine()
    weights = torch.nn.Parameter(torch.ones(8))
    engine.coactivity_ema.copy_(torch.arange(8))
    old = legacy_checkpoint(engine)
    _, restored = make_engine()
    restored.load_state_dict(old)
    expected = engine.after_update(weights, None, torch.ones(8), torch.ones(8))
    actual = restored.after_update(torch.nn.Parameter(torch.ones(8)), None, torch.ones(8), torch.ones(8))
    assert expected == actual
    assert torch.count_nonzero(restored.gradient_ema) == 0
    assert restored.get_extra_state()["config"] == asdict(restored.config)
    _, changed_policy = make_engine(growth_rule="gradient")
    with pytest.raises(RuntimeError, match="config"):
        changed_policy.load_state_dict(old)


@pytest.mark.parametrize("name,value", [
    ("edge_age", -1), ("gradient_ema", float("nan")),
    ("gradient_ema", -1), ("gradient_evidence_count", -1),
])
def test_invalid_checkpoint_credit_rejected_before_any_mask_copy(name, value):
    _, engine = make_engine()
    saved = deepcopy(engine.state_dict())
    saved[name][0] = value
    # A valid different support would otherwise overwrite the target.
    saved["active_mask"] = ~saved["active_mask"]
    before = engine.active_mask.clone()
    with pytest.raises(RuntimeError, match="invalid"):
        engine.load_state_dict(saved)
    assert torch.equal(before, engine.active_mask)


def test_finite_hook_values_that_overflow_rollout_sum_are_reported_and_skipped():
    _, engine = make_engine(growth_rule="gradient")
    weights = torch.nn.Parameter(torch.zeros(8))
    terms = []
    for _ in range(2):
        gains = engine.gates(torch.zeros(2))
        effective = engine.effective_weights(weights, gains)
        engine.capture_gradient(effective, gains)
        terms.append(effective.sum() * 3e38)
    engine.begin_gradient_update()
    sum(terms).backward()
    engine.finish_gradient_update()
    assert engine.summary()["gradient_nonfinite"] == 8
    assert torch.count_nonzero(engine.gradient_evidence_count) == 0
    assert torch.count_nonzero(engine.gradient_ema) == 0


def test_large_finite_gradients_keep_cohort_telemetry_finite():
    graph = graph_fixture(edges_per_direction=4)
    engine = GraftPlasticity(graph, 2, PlasticityConfig(mode="static", density=.5))
    score(engine, torch.nn.Parameter(torch.zeros(8)), torch.full((8,), 1e38))
    assert engine.summary()["incumbent_gradient_mean"] == pytest.approx(1e38)


@pytest.mark.parametrize("mode", ["static", "routing", "structural", "adaptive"])
def test_cohort_diagnostics_count_edges_once_and_initial_support_is_incumbent(mode):
    graph = graph_fixture(edges_per_direction=4)
    engine = GraftPlasticity(graph, 2, PlasticityConfig(mode=mode, density=.5), seed=17)
    weights = torch.nn.Parameter(torch.ones(8))
    terms = []
    for _ in range(3):
        gains = engine.gates(torch.zeros(2))
        effective = engine.effective_weights(weights, gains)
        engine.capture_gradient(effective, gains)
        terms.append(effective.sum())
    engine.begin_gradient_update()
    sum(terms).backward()
    engine.finish_gradient_update()
    stats = engine.summary()
    assert stats["newborn_gradient_count"] == 0
    assert stats["incumbent_gradient_count"] == 4
    assert stats["incumbent_gradient_mean"] == pytest.approx(3.)
    assert stats["gradient_coverage"] == 1.
    assert stats["dormant_gradient_coverage"] == 1.
    assert engine.site_stats()[0]["incumbent_gradient_count"] == 4


def test_insufficient_mature_edges_reduce_turnover_without_pruning_newborns():
    _, engine = make_engine(min_age=4, turnover=1.)
    engine.edge_age.copy_(torch.tensor([4, 0, 0, 0, 0, 4, 0, 0]))
    weights = torch.nn.Parameter(torch.ones(8))
    event = engine.after_update(weights, None, torch.ones(8), torch.ones(8))
    assert event["pruned"] == [0, 5]
    assert event["grown"] == [2, 6]
    assert event["swaps"] == 2
    assert event["protected_active"] == 2
    assert engine.active_mask[:4].sum().item() == 2
    assert engine.active_mask[4:].sum().item() == 2


def test_gradient_scores_ages_and_cohorts_restore_the_same_next_decision():
    _, engine = make_engine(growth_rule="gradient", min_age=3, growth_init="zero")
    weights = torch.nn.Parameter(torch.ones(8))
    score(engine, weights, torch.tensor([1., 1., 2., -9., 1., 1., 3., 8.]))
    engine.after_update(weights, None, torch.ones(8), torch.ones(8), allow_rewire=False)
    _, restored = make_engine(growth_rule="gradient", min_age=3, growth_init="zero")
    restored.load_state_dict(deepcopy(engine.state_dict()))
    next_weights = torch.nn.Parameter(weights.detach().clone())
    assert engine.summary() == restored.summary()
    expected = engine.after_update(weights, None, torch.ones(8), torch.ones(8))
    actual = restored.after_update(next_weights, None, torch.ones(8), torch.ones(8))
    assert expected == actual
    assert actual["grown"] == [3, 7]
    assert torch.equal(engine.edge_age, restored.edge_age)
