import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from poseidon.atlas import _digest, _model_digest
from poseidon.core import CoreConfig, CoreRuntime, TidalCore, save_checkpoint
from poseidon.world import TidePool


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    with torch.random.fork_rng():
        torch.manual_seed(73)
        model = TidalCore(CoreConfig(hash_buckets=32, hidden_size=16))
    path = tmp_path_factory.mktemp("helm-core") / "core.pt"
    save_checkpoint(path, model, receipt={})
    return CoreRuntime(path)


def fit_small(core, **kwargs):
    from poseidon.helm import HelmConfig, HelmCritic
    settings = dict(reservoir_size=8, bootstrap_heads=2, horizons=(1, 2),
                    scarcity_levels=(1.0,), **kwargs)
    return HelmCritic.fit(core, train_seeds=[111, 112], selection_seeds=[211, 212],
                         calibration_seeds=[311, 312], anchors_per_episode=2,
                         max_steps=4, config=HelmConfig(**settings))


@pytest.fixture(scope="module")
def fitted(core):
    return fit_small(core)


def rehash(payload):
    payload.pop("sha256", None)
    payload["sha256"] = _digest(payload)
    return payload


def test_module_provides_deterministic_compact_history_critic(core):
    assert importlib.util.find_spec("poseidon.helm") is not None, "Helm critic is missing"
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    before_model = _model_digest(core)
    a, ar = fit_small(core)
    b, br = fit_small(core)
    assert a.artifact == b.artifact
    assert ar == br
    assert random.getstate() == before_python
    assert np.random.get_state()[0] == before_numpy[0]
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert np.random.get_state()[2:] == before_numpy[2:]
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert _model_digest(core) == before_model


def test_all_families_and_controls_use_disjoint_forecast_partitions(fitted):
    critic, receipt = fitted
    data = critic.artifact
    assert set(data["readouts"]) == {"observation", "history", "no_innovation"}
    assert set(data["calibration"]) == {"selected", "observation", "no_innovation", "yoked", "ungated"}
    assert data["partition"]["selection_seeds"] == [211, 212]
    assert data["partition"]["calibration_seeds"] == [311, 312]
    assert receipt["selection"]["family"] in data["readouts"]
    for mode, calibration in data["calibration"].items():
        assert calibration["episode_count"] == 2
        assert calibration["episode_seeds"] == [311, 312]
        assert set(calibration["radii"]) == {"1", "2"}
        assert calibration["metric"] == "episode-max-horizon-advantage-overestimation"
    assert data["yoked_errors"]["episode_seeds"] == [111, 112]
    assert data["yoked_errors"]["source"] == "frozen-training-only-realized-errors"


def test_roundtrip_and_reset_reproduce_multistep_decisions(core, fitted, tmp_path):
    from poseidon.helm import HelmCritic
    critic, _ = fitted
    target = tmp_path / "helm.json"
    assert Path(critic.save(target)).is_file()
    loaded = HelmCritic.load(core, target)
    assert loaded.digest == critic.digest
    env = TidePool(seed=701, max_steps=4)
    critic.reset(scarcity=1.0, max_steps=4)
    loaded.reset(scarcity=1.0, max_steps=4)
    while not env.done:
        a = critic.plan(env.observe())
        b = loaded.plan(env.observe())
        assert a == b
        env.step(a["action"])
    critic.reset(scarcity=1.0, max_steps=4)
    first = critic.plan(TidePool(seed=701, max_steps=4).observe())
    assert first["step"] == 0
    assert first["history"]["previous_executed_action"] is None
    assert first["history"]["innovation_norm"] == 0


def test_plan_reads_actual_executed_action_and_only_completed_transition(core, fitted):
    critic, _ = fitted
    env = TidePool(seed=702, max_steps=4)
    critic.reset(scarcity=1.0, max_steps=4)
    old = env.observe()
    first = critic.plan(old)
    external_action = (first["action"] + 1) % 6
    observed, _, _, _ = env.step(external_action)
    expected = np.asarray(observed) - np.clip(core.predict_transition(old, external_action), 0, 1)
    decision = critic.plan(observed)
    assert decision["history"]["previous_executed_action"] == external_action
    assert np.allclose(decision["history"]["innovation"], expected)


def test_planning_never_constructs_simulator(core, fitted, monkeypatch):
    import poseidon.helm as helm
    critic, _ = fitted
    obs = TidePool(seed=703, max_steps=4).observe()
    def forbidden(*args, **kwargs):
        raise AssertionError("Simulator access at inference")
    monkeypatch.setattr(helm, "TidePool", forbidden)
    critic.reset(scarcity=1.0, max_steps=4)
    assert 0 <= critic.plan(obs)["action"] < 6


def test_unknown_regime_and_insufficient_horizon_abstain(fitted):
    critic, _ = fitted
    critic.reset(scarcity=4.0, max_steps=4)
    unknown = critic.plan(TidePool(seed=704, scarcity=4.0, max_steps=4).observe())
    assert not unknown["supported"]
    assert unknown["action"] == unknown["policy_action"]
    assert unknown["fallback_reason"] == "outside_fitted_regime"
    critic.reset(scarcity=1.0, max_steps=1)
    near_end = critic.plan(TidePool(seed=705, max_steps=1).observe())
    assert near_end["action"] == near_end["policy_action"]
    assert near_end["fallback_reason"] == "insufficient_remaining_horizon"


def test_act_advances_exactly_once_and_records_decision(fitted):
    critic, _ = fitted
    critic.reset(max_steps=4)
    action = critic.act(TidePool(seed=706, max_steps=4).observe())
    assert critic.last_decision["action"] == action
    assert critic.last_decision["step"] == 0
    assert critic.step == 1


def test_changed_core_weights_including_data_mutation_invalidate_inference(core, fitted):
    from poseidon.helm import HelmCritic
    original = next(core.model.parameters()).detach().clone()
    parameter = next(core.model.parameters())
    try:
        parameter.data.add_(0.01)
        with pytest.raises(ValueError, match="model.*changed|model mismatch"):
            fitted[0].plan(TidePool().observe())
        with pytest.raises(ValueError, match="model mismatch"):
            HelmCritic(core, fitted[0].artifact)
    finally:
        parameter.data.copy_(original)


@pytest.mark.parametrize("mutation", ["checksum", "readout_shape", "overlap", "yoked_seed", "radius", "unknown"])
def test_artifact_rejects_tampering_even_when_resigned(core, fitted, mutation):
    from poseidon.helm import HelmCritic
    payload = fitted[0].artifact
    if mutation == "checksum":
        payload["selection"]["family"] = "history"
        payload["sha256"] = "0" * 64
    elif mutation == "readout_shape":
        payload["readouts"]["history"][0].pop()
    elif mutation == "overlap":
        payload["partition"]["selection_seeds"] = payload["partition"]["train_seeds"][:]
    elif mutation == "yoked_seed":
        payload["yoked_errors"]["episode_seeds"] = [311, 312]
    elif mutation == "radius":
        payload["calibration"]["yoked"]["radii"]["1"] += 0.1
    else:
        payload["unexpected"] = True
    if mutation != "checksum":
        rehash(payload)
    with pytest.raises(ValueError):
        HelmCritic(core, payload)


@pytest.mark.parametrize("kwargs", [{"reservoir_size": 7}, {"horizons": (4, 4)},
                                   {"horizons": (16, 4)}, {"quantile": float("nan")},
                                   {"seed": True}, {"bootstrap_heads": 1}])
def test_config_bounds_are_strict(kwargs):
    from poseidon.helm import HelmConfig
    with pytest.raises(ValueError):
        HelmConfig(**kwargs)


def test_partition_overlap_and_cooperative_cancel(core):
    from poseidon.helm import HelmCritic, HelmConfig
    with pytest.raises(ValueError, match="disjoint"):
        HelmCritic.fit(core, train_seeds=[1], selection_seeds=[1], calibration_seeds=[3])
    events = []
    class Cancelled(Exception):
        pass
    def progress(event):
        events.append(event)
        raise Cancelled("stop requested")
    before = _model_digest(core)
    with pytest.raises(Cancelled):
        HelmCritic.fit(core, train_seeds=[1], selection_seeds=[2], calibration_seeds=[3],
                       max_steps=4, config=HelmConfig(reservoir_size=8, horizons=(1, 2)),
                       progress=progress)
    assert events and events[0]["phase"] == "collect_train"
    assert _model_digest(core) == before


def test_stable_json_rejects_duplicate_keys_and_bound(tmp_path):
    from poseidon.helm import stable_json
    path = tmp_path / "data.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        stable_json(path)
    path.write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        stable_json(path, max_bytes=3)
