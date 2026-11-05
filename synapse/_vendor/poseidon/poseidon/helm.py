"""Frozen, history-conditioned return readouts for CPU-local synthetic control.

Fitting alone branches TidePool. Decision features contain current observations,
preceding executed actions and already realized prediction errors. Calibration
radii describe held-out episode errors; they are not safety guarantees.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Callable, Sequence

import numpy as np
import torch

from . import world as _world
from .atlas import _canonical, _digest, _integer, _model_digest, _number, _observation, _quantile, _seeds
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION, teacher_action

HELM_SCHEMA = "poseidon-helm-critic-v1"
HELM_MODES = ("selected", "observation", "no_innovation", "yoked", "ungated")
_FAMILIES = ("observation", "history", "no_innovation")
_METRIC = "episode-max-horizon-advantage-overestimation"
_PATH_POLICY = "local-rng:teacher-0.7/core-0.2/random-0.1"
MAX_HELM_ARTIFACT_BYTES = 32 * 1024 * 1024
_LIMITS = [
    "Synthetic TidePool returns under frozen core continuation, with terminal truncation.",
    "Frozen reservoir and training-episode bootstrap ridge heads; no online weight learning.",
    "Selection and descriptive episode-max calibration use disjoint episode seeds.",
    "Calibration radii and ensemble disagreement are not safety or coverage guarantees.",
    "Yoked errors are a frozen training-only schedule with independent held-out calibration.",
    "Inference receives observations and configured episode limits, never simulator state or seeds.",
    "Ungated bypasses descriptive margins only; support, admissibility and remaining horizons still apply.",
    "This opt-in critic does not change or promote the incumbent neural model.",
]


def _stable_bytes(path, max_bytes):
    _integer(max_bytes, "max_bytes", 1, 512 * 1024 * 1024)
    target = Path(path)
    with target.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > max_bytes:
            raise ValueError("File exceeds size limit")
        content = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    current = target.stat()
    stamp = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if len(content) > max_bytes:
        raise ValueError("File exceeds size limit")
    if (stamp(before) != stamp(after) or stamp(after) != stamp(current) or
            before.st_ctime_ns != after.st_ctime_ns or len(content) != before.st_size):
        raise ValueError("File changed during stable read")
    return content


def stable_json(path, *, max_bytes=128 * 1024 * 1024):
    """Parse only the bounded stable bytes read, rejecting ambiguous/nonfinite JSON."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    try:
        return json.loads(_stable_bytes(path, max_bytes).decode("utf-8-sig"),
                          object_pairs_hook=unique, parse_constant=bad_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded JSON document") from exc


def _file_digest(path, max_bytes=512 * 1024 * 1024):
    return hashlib.sha256(_stable_bytes(path, max_bytes)).hexdigest()


def _source_digest():
    return _file_digest(__file__, 1024 * 1024)


def _world_digest():
    return _file_digest(_world.__file__, 1024 * 1024)


@dataclass(frozen=True)
class HelmConfig:
    reservoir_size: int = 48
    horizons: tuple[int, int] = (4, 16)
    bootstrap_heads: int = 5
    seed: int = 1701
    ridge_alpha: float = 0.05
    leak: float = 0.35
    recurrent_radius: float = 0.85
    input_scale: float = 0.6
    support_radius: float = 0.25
    quantile: float = 0.90
    override_margin: float = 0.03
    selection_improvement: float = 0.02
    min_harvest_resource: float = 0.04
    max_samples: int = 4096
    scarcity_levels: tuple[float, ...] = (1.0, 2.5, 4.0)

    def __post_init__(self):
        _integer(self.reservoir_size, "reservoir_size", 8, 64)
        _integer(self.bootstrap_heads, "bootstrap_heads", 2, 8)
        _integer(self.seed, "seed", 0, 2**32 - 1)
        _integer(self.max_samples, "max_samples", 1, 4096)
        if not isinstance(self.horizons, (list, tuple)) or len(self.horizons) != 2:
            raise ValueError("horizons must contain two distinct increasing values")
        horizons = tuple(_integer(h, "horizon", 1, 32) for h in self.horizons)
        if horizons[0] >= horizons[1]:
            raise ValueError("horizons must contain two distinct increasing values")
        object.__setattr__(self, "horizons", horizons)
        for key, low, high in (("ridge_alpha", 1e-6, 1000), ("leak", 0.01, 1),
                               ("recurrent_radius", 0.01, 0.99), ("input_scale", 0.01, 2),
                               ("support_radius", 1e-8, 1), ("quantile", 0.5, 1),
                               ("override_margin", 0, 5), ("selection_improvement", 0, 0.5),
                               ("min_harvest_resource", 0, 0.5)):
            _number(getattr(self, key), key, low, high)
        if not isinstance(self.scarcity_levels, (list, tuple)) or not 1 <= len(self.scarcity_levels) <= 8:
            raise ValueError("scarcity_levels requires 1 to 8 profiles")
        profiles = tuple(_number(x, "scarcity", 0.5, 4) for x in self.scarcity_levels)
        if len(set(profiles)) != len(profiles):
            raise ValueError("scarcity profiles must be distinct")
        object.__setattr__(self, "scarcity_levels", profiles)


def _partitions(train, selection, calibration):
    groups = [_seeds(values, name) for values, name in (
        (train, "train_seeds"), (selection, "selection_seeds"), (calibration, "calibration_seeds"))]
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("Train, selection and calibration episodes must be disjoint")
    return groups


def _reservoir(config):
    rng = np.random.default_rng(config.seed)
    n = config.reservoir_size
    inputs = rng.normal(0, config.input_scale / math.sqrt(OBS_SIZE + N_ACTIONS + OBS_SIZE),
                        (n, OBS_SIZE + N_ACTIONS + OBS_SIZE))
    recurrent = rng.normal(0, 1, (n, n))
    recurrent *= config.recurrent_radius / np.max(np.abs(recurrent).sum(axis=1))
    bias = rng.uniform(-0.1, 0.1, n)
    return {"input_weights": inputs.tolist(), "recurrent_weights": recurrent.tolist(), "bias": bias.tolist()}


def _advance(state, observation, action, innovation, matrices, config):
    onehot = np.zeros(N_ACTIONS)
    if action is not None:
        onehot[action] = 1
    drive = np.concatenate((observation, onehot, innovation))
    proposed = np.tanh(matrices[0] @ drive + matrices[1] @ state + matrices[2])
    return (1 - config.leak) * state + config.leak * proposed


def _features(observation, state=None):
    return np.concatenate(([1.0], observation, [] if state is None else state))


def _matrices(reservoir):
    values = tuple(np.asarray(reservoir[key], dtype=np.float64) for key in (
        "input_weights", "recurrent_weights", "bias"))
    for value in values:
        value.setflags(write=False)
    return values


def _notify(progress, phase, **fields):
    if progress is not None:
        progress({"phase": phase, **fields})


@torch.inference_mode()
def _policies(core, observations):
    result = core.model([""] * len(observations), torch.tensor(observations, dtype=torch.float32))["action"]
    if not torch.isfinite(result).all():
        raise ValueError("Core returned nonfinite actions")
    return result.argmax(-1).tolist()


def _prediction(core, observation, action):
    values = np.asarray(core.predict_transition(observation, action), dtype=np.float64)
    if values.shape != (OBS_SIZE,) or not np.isfinite(values).all():
        raise ValueError("Core returned nonfinite or malformed transition predictions")
    return np.clip(values, 0, 1)


def _returns(snapshot, core, config, progress, phase, episode, anchor):
    envs = [TidePool.from_snapshot(snapshot) for _ in range(N_ACTIONS)]
    totals = np.zeros(N_ACTIONS)
    targets = {}
    for tick in range(max(config.horizons)):
        live = [a for a, env in enumerate(envs) if not env.done]
        if tick == 0:
            actions = live
        else:
            actions = _policies(core, [envs[a].observe() for a in live]) if live else []
        for branch, action in zip(live, actions):
            _, reward, _, _ = envs[branch].step(action)
            totals[branch] += reward
        if tick + 1 in config.horizons:
            targets[str(tick + 1)] = totals.tolist()
        if tick % 8 == 0:
            _notify(progress, phase, episode=episode, anchor=anchor, branch_step=tick + 1)
    return targets


def _collect(core, seeds, anchors, max_steps, config, progress, phase):
    episodes = []
    # When possible, labels use complete horizons rather than artificial end truncation.
    span = max(1, max_steps - max(config.horizons) + 1)
    ticks = {min(span - 1, i * span // anchors) for i in range(anchors)}
    for index, seed in enumerate(seeds):
        scarcity = config.scarcity_levels[index % len(config.scarcity_levels)]
        env = TidePool(seed=seed, scarcity=scarcity, max_steps=max_steps)
        rng = random.Random(seed ^ 0x48454C4D)
        record = {"seed": seed, "scarcity": scarcity, "observations": [], "actions": [], "errors": [], "anchors": []}
        innovation = np.zeros(OBS_SIZE)
        _notify(progress, phase, episode=index, episode_count=len(seeds), completed=0)
        while not env.done:
            obs = env.observe()
            record["observations"].append(obs)
            record["errors"].append(innovation.tolist())
            if env.tick in ticks:
                targets = _returns(env.snapshot(), core, config, progress, phase, index, env.tick)
                record["anchors"].append({"tick": env.tick, "returns": targets,
                                           "policy_action": _policies(core, [obs])[0]})
            draw = rng.random()
            action = teacher_action(obs) if draw < 0.7 else _policies(core, [obs])[0] if draw < 0.9 else rng.randrange(N_ACTIONS)
            forecast = _prediction(core, obs, action)
            observed, _, _, _ = env.step(action)
            innovation = np.asarray(observed) - forecast
            record["actions"].append(action)
            if env.tick % 8 == 0:
                _notify(progress, phase, episode=index, episode_count=len(seeds), step=env.tick)
        episodes.append(record)
        _notify(progress, phase, episode=index + 1, episode_count=len(seeds), completed=index + 1)
    return episodes


def _episode_rows(episodes, config, matrices, schedule):
    rows = []
    for ep in episodes:
        states = {key: np.zeros(config.reservoir_size) for key in ("history", "no_innovation", "yoked")}
        anchors = {a["tick"]: a for a in ep["anchors"]}
        for tick, obs in enumerate(ep["observations"]):
            action = None if tick == 0 else ep["actions"][tick - 1]
            error = np.asarray(ep["errors"][tick])
            yoke = np.zeros(OBS_SIZE) if tick == 0 else schedule[(tick - 1) % len(schedule)]
            for family, value in (("history", error), ("no_innovation", np.zeros(OBS_SIZE)), ("yoked", yoke)):
                states[family] = _advance(states[family], obs, action, value, matrices, config)
            if tick in anchors:
                anchor = anchors[tick]
                rows.append({"episode_seed": ep["seed"], "scarcity": ep["scarcity"],
                             "features": {"observation": _features(obs),
                                          **{key: _features(obs, value) for key, value in states.items()}},
                             "returns": anchor["returns"], "policy_action": anchor["policy_action"]})
    return rows


def _fit_readouts(rows, seeds, config, progress):
    targets = np.asarray([[row["returns"][str(h)] for h in config.horizons] for row in rows])
    rng = np.random.default_rng(config.seed ^ 0xB007)
    draws = [rng.choice(seeds, len(seeds), replace=True).tolist() for _ in range(config.bootstrap_heads)]
    result = {}
    for family in _FAMILIES:
        features = np.asarray([row["features"][family] for row in rows])
        heads = []
        for head, sample in enumerate(draws):
            _notify(progress, "fit_readouts", family=family, head=head, head_count=config.bootstrap_heads)
            indices = [i for seed in sample for i, row in enumerate(rows) if row["episode_seed"] == seed]
            x, y = features[indices], targets[indices].reshape(len(indices), -1)
            penalty = np.eye(x.shape[1]) * config.ridge_alpha
            penalty[0, 0] = config.ridge_alpha * 0.01
            solved = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            heads.append(solved.reshape(x.shape[1], len(config.horizons), N_ACTIONS).transpose(1, 0, 2).tolist())
        result[family] = heads
    return result


def _forecasts(readouts, features, horizons):
    weights = np.asarray(readouts, dtype=np.float64)
    predictions = np.einsum("bhfa,f->bha", weights, features)
    for i, horizon in enumerate(horizons):
        predictions[:, i, :] = np.clip(predictions[:, i, :], -8 * horizon, 2 * horizon)
    return predictions.mean(axis=0), predictions.std(axis=0)


def _selection(rows, seeds, readouts, config):
    errors, episode_errors = {}, {}
    for family in _FAMILIES:
        values = []
        for seed in seeds:
            local = []
            for row in rows:
                if row["episode_seed"] != seed:
                    continue
                forecast, _ = _forecasts(readouts[family], row["features"][family], config.horizons)
                actual = np.asarray([row["returns"][str(h)] for h in config.horizons])
                local.append(float(np.mean(((forecast - actual) / np.asarray(config.horizons)[:, None]) ** 2)))
            values.append(float(np.mean(local)))
        episode_errors[family] = values
        errors[family] = float(np.mean(values))
    baseline = errors["observation"]
    best = min(_FAMILIES, key=lambda family: (errors[family], _FAMILIES.index(family)))
    improvement = (baseline - errors[best]) / max(baseline, 1e-12)
    family = best if best != "observation" and improvement > config.selection_improvement else "observation"
    return {"family": family, "metric": "episode-balanced-horizon-normalized-return-mse",
            "errors": errors, "episode_errors": episode_errors, "episode_seeds": seeds[:],
            "improvement_threshold": config.selection_improvement, "relative_improvement": improvement}


def _mode_family(mode, selected):
    if mode in ("selected", "ungated"):
        return selected
    return "history" if mode == "yoked" else mode


def _calibrate(rows, seeds, readouts, selected, config):
    result = {}
    for mode in HELM_MODES:
        family = _mode_family(mode, selected)
        feature_key = "yoked" if mode == "yoked" else family
        maxima = {str(h): [] for h in config.horizons}
        for seed in seeds:
            episode = {str(h): 0.0 for h in config.horizons}
            for row in rows:
                if row["episode_seed"] != seed:
                    continue
                forecast, _ = _forecasts(readouts[family], row["features"][feature_key], config.horizons)
                incumbent = row["policy_action"]
                for i, h in enumerate(config.horizons):
                    actual = np.asarray(row["returns"][str(h)])
                    overestimate = (forecast[i] - forecast[i, incumbent]) - (actual - actual[incumbent])
                    episode[str(h)] = max(episode[str(h)], float(overestimate.max()))
            for h in maxima:
                maxima[h].append(episode[h])
        result[mode] = {"family": family, "metric": _METRIC, "quantile": config.quantile,
                        "episode_count": len(seeds), "episode_seeds": seeds[:], "anchor_count": len(rows),
                        "episode_max_errors": maxima,
                        "radii": {h: float(_quantile(values, config.quantile)) for h, values in maxima.items()}}
    return result


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid {name} fields")


def _array(value, shape, name, bound=1e6):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} array") from exc
    if array.shape != shape or not np.isfinite(array).all() or np.any(np.abs(array) > bound):
        raise ValueError(f"Invalid {name} shape, bounds or finiteness")
    # Reject JSON booleans and strings that NumPy would silently convert to numbers.
    def numeric(items):
        if isinstance(items, list):
            return all(numeric(item) for item in items)
        return type(items) in (int, float)
    if not numeric(value):
        raise ValueError(f"Invalid {name} numeric values")
    return array


def validate_helm_artifact(payload, core=None):
    """Validate detached artifact structure; core=None supports weightless receipts."""
    try:
        encoded = _canonical(payload).encode()
        if len(encoded) > MAX_HELM_ARTIFACT_BYTES:
            raise ValueError("Helm artifact exceeds size limit")
        data = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Helm artifact must contain bounded finite JSON") from exc
    expected = data.pop("sha256", None) if isinstance(data, dict) else None
    if expected != _digest(data):
        raise ValueError("Helm checksum mismatch")
    _keys(data, ("schema", "world_version", "world_sha256", "source_sha256", "checkpoint_sha256",
                 "model_sha256", "config", "partition", "reservoir", "readouts", "support",
                 "yoked_errors", "selection", "calibration", "limits"), "Helm artifact")
    if data["schema"] != HELM_SCHEMA or data["world_version"] != WORLD_VERSION:
        raise ValueError("Helm schema/world version mismatch")
    for key in ("world_sha256", "source_sha256", "checkpoint_sha256", "model_sha256"):
        digest = data[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid Helm {key}")
    if data["world_sha256"] != _world_digest() or data["source_sha256"] != _source_digest():
        raise ValueError("Helm source/world fingerprint mismatch")
    if core is not None:
        if data["checkpoint_sha256"] != _file_digest(core.path):
            raise ValueError("Helm checkpoint mismatch")
        if data["model_sha256"] != _model_digest(core):
            raise ValueError("Helm in-memory model mismatch")
    try:
        config = HelmConfig(**data["config"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid Helm configuration") from exc
    if set(data["config"]) != set(asdict(config)):
        raise ValueError("Incomplete Helm configuration")
    part = data["partition"]
    _keys(part, ("train_seeds", "selection_seeds", "calibration_seeds", "anchors_per_episode", "max_steps", "path_policy"), "partition")
    groups = _partitions(part["train_seeds"], part["selection_seeds"], part["calibration_seeds"])
    anchors = _integer(part["anchors_per_episode"], "anchors_per_episode", 1, 64)
    steps = _integer(part["max_steps"], "max_steps", 1, 1024)
    if part["path_policy"] != _PATH_POLICY:
        raise ValueError("Invalid fitted path policy")
    n = config.reservoir_size
    _keys(data["reservoir"], ("input_weights", "recurrent_weights", "bias"), "reservoir")
    _array(data["reservoir"]["input_weights"], (n, 38), "reservoir inputs")
    _array(data["reservoir"]["recurrent_weights"], (n, n), "recurrent weights")
    _array(data["reservoir"]["bias"], (n,), "reservoir bias")
    if data["reservoir"] != _reservoir(config):
        raise ValueError("Frozen reservoir does not match deterministic configuration")
    _keys(data["readouts"], _FAMILIES, "readouts")
    _keys(data["support"], ("observation", "history", "no_innovation", "yoked"), "support")
    count = None
    for family in data["support"]:
        features = 17 if family == "observation" else 17 + n
        bank = data["support"][family]
        _keys(bank, ("features", "scarcity", "episode_seeds"), "support bank")
        size = len(bank["features"]) if isinstance(bank["features"], list) else 0
        if not 1 <= size <= min(config.max_samples, len(groups[0]) * anchors):
            raise ValueError("Support sample count exceeds bounds")
        if count is not None and size != count:
            raise ValueError("Support banks must be aligned")
        count = size
        array = _array(bank["features"], (size, features), "support features", 1)
        if not np.all(array[:, 0] == 1) or np.any(array[:, 1:17] < 0):
            raise ValueError("Invalid support feature convention")
        _array(bank["scarcity"], (size,), "support scarcity", 4)
        if any(x not in config.scarcity_levels for x in bank["scarcity"]):
            raise ValueError("Unknown support scarcity profile")
        if not isinstance(bank["episode_seeds"], list) or len(bank["episode_seeds"]) != size:
            raise ValueError("Invalid support episode identity")
        if any(type(seed) is not int or seed not in groups[0] for seed in bank["episode_seeds"]):
            raise ValueError("Support must come from training episodes")
        if set(bank["episode_seeds"]) != set(groups[0]):
            raise ValueError("Missing training episode support")
        if family != "yoked":
            _array(data["readouts"][family], (config.bootstrap_heads, 2, features, N_ACTIONS), "return readouts")
    yoke = data["yoked_errors"]
    _keys(yoke, ("source", "episode_seeds", "values"), "yoked errors")
    if yoke["source"] != "frozen-training-only-realized-errors" or yoke["episode_seeds"] != groups[0]:
        raise ValueError("Yoked schedule must bind training-only episodes")
    length = len(yoke["values"]) if isinstance(yoke["values"], list) else 0
    if not 1 <= length <= min(16384, len(groups[0]) * steps):
        raise ValueError("Yoked schedule exceeds bounds")
    _array(yoke["values"], (length, OBS_SIZE), "yoked error schedule", 1)
    selection = data["selection"]
    _keys(selection, ("family", "metric", "errors", "episode_errors", "episode_seeds", "improvement_threshold", "relative_improvement"), "selection")
    if selection["metric"] != "episode-balanced-horizon-normalized-return-mse" or selection["episode_seeds"] != groups[1]:
        raise ValueError("Invalid selection identity or metric")
    _keys(selection["errors"], _FAMILIES, "selection errors")
    _keys(selection["episode_errors"], _FAMILIES, "selection episode errors")
    for family in _FAMILIES:
        values = _array(selection["episode_errors"][family], (len(groups[1]),), "selection errors", 1000)
        if np.any(values < 0) or not math.isclose(_number(selection["errors"][family], "selection MSE", 0, 1000), float(values.mean()), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Invalid selection aggregate")
    if selection["improvement_threshold"] != config.selection_improvement:
        raise ValueError("Selection threshold mismatch")
    best = min(_FAMILIES, key=lambda f: (selection["errors"][f], _FAMILIES.index(f)))
    baseline = selection["errors"]["observation"]
    improvement = (baseline - selection["errors"][best]) / max(baseline, 1e-12)
    chosen = best if best != "observation" and improvement > config.selection_improvement else "observation"
    if selection["family"] != chosen or not math.isclose(selection["relative_improvement"], improvement, abs_tol=1e-12):
        raise ValueError("Invalid family selection")
    _keys(data["calibration"], HELM_MODES, "calibration modes")
    horizon_keys = {str(h) for h in config.horizons}
    for mode in HELM_MODES:
        cal = data["calibration"][mode]
        _keys(cal, ("family", "metric", "quantile", "episode_count", "episode_seeds", "anchor_count", "episode_max_errors", "radii"), "calibration")
        if cal["family"] != _mode_family(mode, chosen) or cal["metric"] != _METRIC or cal["quantile"] != config.quantile:
            raise ValueError("Calibration family/metric mismatch")
        if cal["episode_seeds"] != groups[2] or cal["episode_count"] != len(groups[2]):
            raise ValueError("Calibration episodes mismatch")
        _integer(cal["anchor_count"], "calibration anchors", len(groups[2]), len(groups[2]) * anchors)
        _keys(cal["episode_max_errors"], horizon_keys, "calibration errors")
        _keys(cal["radii"], horizon_keys, "calibration radii")
        for h in horizon_keys:
            errors = _array(cal["episode_max_errors"][h], (len(groups[2]),), "episode maxima", 16 * int(h))
            if np.any(errors < 0):
                raise ValueError("Calibration maxima must be nonnegative")
            radius = _number(cal["radii"][h], "calibration radius", 0, 16 * int(h))
            if radius != float(_quantile(errors.tolist(), config.quantile)):
                raise ValueError("Calibration radius does not match episode maxima")
    if data["limits"] != _LIMITS:
        raise ValueError("Helm evidence limitations mismatch")
    data["sha256"] = expected
    return data


class HelmCritic:
    def __init__(self, core, payload):
        self._payload = validate_helm_artifact(payload, core)
        self.core = core
        self.config = HelmConfig(**self._payload["config"])
        self._matrices = _matrices(self._payload["reservoir"])
        self._readouts = {key: np.asarray(value) for key, value in self._payload["readouts"].items()}
        self._schedule = np.asarray(self._payload["yoked_errors"]["values"])
        self._support = {key: np.asarray(bank["features"]) for key, bank in self._payload["support"].items()}
        for value in list(self._readouts.values()) + [self._schedule] + list(self._support.values()):
            value.setflags(write=False)
        self.reset()

    @property
    def artifact(self):
        return copy.deepcopy(self._payload)

    @property
    def digest(self):
        return self._payload["sha256"]

    @classmethod
    @torch.inference_mode()
    def fit(cls, core, *, train_seeds, selection_seeds, calibration_seeds,
            anchors_per_episode=8, max_steps=96, config=None, progress=None):
        config = config or HelmConfig()
        if not isinstance(config, HelmConfig):
            raise ValueError("config must be HelmConfig")
        groups = _partitions(train_seeds, selection_seeds, calibration_seeds)
        anchors = _integer(anchors_per_episode, "anchors_per_episode", 1, 64)
        steps = _integer(max_steps, "max_steps", 1, 1024)
        if len(groups[0]) * anchors > config.max_samples or sum(map(len, groups)) * anchors * max(config.horizons) * N_ACTIONS > 2_000_000:
            raise ValueError("Helm fit exceeds anchor/branch workload bounds")
        if sum(map(len, groups)) * steps > 100000 or len(groups[0]) * steps > 16384:
            raise ValueError("Helm fit exceeds episode/error-schedule bounds")
        if core.model.training:
            raise ValueError("Core must be in evaluation mode for frozen fitting")
        binding = {"checkpoint_sha256": _file_digest(core.path), "model_sha256": _model_digest(core),
                   "world_sha256": _world_digest(), "source_sha256": _source_digest()}
        train = _collect(core, groups[0], anchors, steps, config, progress, "collect_train")
        errors = [error for ep in train for error in ep["errors"][1:]] or [[0.0] * OBS_SIZE]
        rng = np.random.default_rng(config.seed ^ 0x70CE)
        schedule = np.asarray(errors, dtype=np.float64)[rng.permutation(len(errors))]
        reservoir = _reservoir(config)
        matrices = _matrices(reservoir)
        train_rows = _episode_rows(train, config, matrices, schedule)
        readouts = _fit_readouts(train_rows, groups[0], config, progress)
        selected_episodes = _collect(core, groups[1], anchors, steps, config, progress, "collect_selection")
        selection_rows = _episode_rows(selected_episodes, config, matrices, schedule)
        selected = _selection(selection_rows, groups[1], readouts, config)
        calibration_episodes = _collect(core, groups[2], anchors, steps, config, progress, "collect_calibration")
        calibration_rows = _episode_rows(calibration_episodes, config, matrices, schedule)
        calibration = _calibrate(calibration_rows, groups[2], readouts, selected["family"], config)
        support = {family: {"features": [row["features"][family].tolist() for row in train_rows],
                             "scarcity": [row["scarcity"] for row in train_rows],
                             "episode_seeds": [row["episode_seed"] for row in train_rows]}
                   for family in ("observation", "history", "no_innovation", "yoked")}
        payload = {"schema": HELM_SCHEMA, "world_version": WORLD_VERSION, **binding,
                   "config": asdict(config),
                   "partition": {"train_seeds": groups[0], "selection_seeds": groups[1], "calibration_seeds": groups[2],
                                 "anchors_per_episode": anchors, "max_steps": steps, "path_policy": _PATH_POLICY},
                   "reservoir": reservoir, "readouts": readouts, "support": support,
                   "yoked_errors": {"source": "frozen-training-only-realized-errors", "episode_seeds": groups[0], "values": schedule.tolist()},
                   "selection": selected, "calibration": calibration, "limits": _LIMITS[:]}
        payload["sha256"] = _digest(payload)
        _notify(progress, "validate_candidate", completed=1)
        critic = cls(core, payload)
        receipt = {"schema": HELM_SCHEMA + "-fit", "artifact_sha256": critic.digest,
                   "training_anchors": len(train_rows), "selection_anchors": len(selection_rows), "calibration_anchors": len(calibration_rows),
                   "training_samples": len(train_rows) * N_ACTIONS,
                   "selection_samples": len(selection_rows) * N_ACTIONS,
                   "calibration_samples": len(calibration_rows) * N_ACTIONS,
                   "partition": copy.deepcopy(payload["partition"]), "selection": copy.deepcopy(selected),
                   "calibration": copy.deepcopy(calibration), "config": asdict(config), **binding, "limits": _LIMITS[:]}
        _notify(progress, "completed", completed=1, artifact_sha256=critic.digest)
        return critic, receipt

    def reset(self, scarcity=1.0, max_steps=96):
        scarcity = _number(scarcity, "scarcity", 0.5, 4)
        max_steps = _integer(max_steps, "max_steps", 1, 10000)
        self.scarcity, self.max_steps = scarcity, max_steps
        self.step = 0
        self._states = {key: np.zeros(self.config.reservoir_size) for key in ("history", "no_innovation", "yoked")}
        self._previous_observation = None
        self.last_decision = None

    def _assert_bindings(self):
        # Full digest also catches .data writes, which bypass PyTorch's version counter.
        if _model_digest(self.core) != self._payload["model_sha256"]:
            raise ValueError("Helm bound in-memory model changed")
        if _file_digest(self.core.path) != self._payload["checkpoint_sha256"]:
            raise ValueError("Helm bound checkpoint changed")
        if _source_digest() != self._payload["source_sha256"] or _world_digest() != self._payload["world_sha256"]:
            raise ValueError("Helm bound source/world changed")

    @torch.inference_mode()
    def plan(self, observation: Sequence[float], *, mode="selected"):
        obs = _observation(observation)
        if mode not in HELM_MODES:
            raise ValueError("Unknown Helm mode")
        self._assert_bindings()
        scaled_action = obs[14] * (N_ACTIONS - 1)
        if not math.isclose(scaled_action, round(scaled_action), abs_tol=1e-5):
            raise ValueError("Observable last action must encode an executed action")
        previous_action = None if self._previous_observation is None else int(round(scaled_action))
        innovation = np.zeros(OBS_SIZE) if previous_action is None else np.asarray(obs) - _prediction(self.core, self._previous_observation, previous_action)
        yoke_index = None if previous_action is None else (self.step - 1) % len(self._schedule)
        yoked = np.zeros(OBS_SIZE) if yoke_index is None else self._schedule[yoke_index]
        states = {family: _advance(self._states[family], obs, previous_action, value, self._matrices, self.config)
                  for family, value in (("history", innovation), ("no_innovation", np.zeros(OBS_SIZE)), ("yoked", yoked))}
        family = _mode_family(mode, self._payload["selection"]["family"])
        feature_key = "yoked" if mode == "yoked" else family
        feature = _features(obs, None if feature_key == "observation" else states[feature_key])
        forecast, disagreement = _forecasts(self._readouts[family], feature, self.config.horizons)
        incumbent = _policies(self.core, [obs])[0]
        advantages = forecast - forecast[:, incumbent, None]
        bank = self._payload["support"][feature_key]
        mask = np.isclose(np.asarray(bank["scarcity"]), self.scarcity, atol=1e-9, rtol=0)
        regime_supported = bool(mask.any())
        scale = np.ones(len(feature))
        scale[17:] = 2  # Reservoir range is [-1,1]; observation range is [0,1].
        distance = float(np.min(np.sqrt(np.mean(((self._support[feature_key][mask] - feature) / scale) ** 2, axis=1)))) if regime_supported else 1.0
        supported = regime_supported and distance <= self.config.support_radius
        candidates = []
        for action in range(N_ACTIONS):
            reason = None
            if action == 1 and obs[6] < self.config.min_harvest_resource:
                reason = "depleted_food"
            elif action == 2 and obs[7] < self.config.min_harvest_resource:
                reason = "depleted_water"
            elif action in (4, 5) and (obs[3] < 0.12 or obs[1] < 0.08 or obs[2] < 0.08):
                reason = "insufficient_travel_reserves"
            candidates.append({"action": action, "label": ACTIONS[action], "admissible": reason is None, "reason": reason})
        valid = [c["action"] for c in candidates if c["admissible"]]
        # Rank by the worst horizon-normalized point advantage; stable action tie-break.
        scores = np.min(advantages / np.asarray(self.config.horizons)[:, None], axis=0)
        proposal = max(valid, key=lambda a: (float(scores[a]), a == incumbent, -a)) if valid else incumbent
        cal = self._payload["calibration"][mode]
        radii = {str(h): float(cal["radii"][str(h)]) for h in self.config.horizons}
        margins = {str(h): float(advantages[i, proposal] - radii[str(h)] - self.config.override_margin)
                   for i, h in enumerate(self.config.horizons)}
        margin = min(margins.values())
        remaining = min(self.max_steps - self.step, int(round(self.max_steps * (1 - obs[15]))))
        chosen, reason = incumbent, None
        if remaining < max(self.config.horizons):
            reason = "insufficient_remaining_horizon"
        elif not regime_supported:
            reason = "outside_fitted_regime"
        elif not supported:
            reason = "outside_fitted_support"
        elif proposal == incumbent:
            reason = "policy_preferred"
        elif not candidates[proposal]["admissible"]:
            reason = "inadmissible_action"
        elif mode == "ungated":
            chosen, reason = proposal, None
        elif margin > 0:
            chosen, reason = proposal, None
        else:
            reason = "insufficient_calibrated_margin"
        decision = {"action": int(chosen), "action_name": ACTIONS[chosen], "policy_action": int(incumbent),
                    "proposed_action": int(proposal), "overridden": chosen != incumbent, "override_accepted": chosen != incumbent,
                    "mode": mode, "family": family, "forecasts": {str(h): forecast[i].tolist() for i, h in enumerate(self.config.horizons)},
                    "advantages": {str(h): advantages[i].tolist() for i, h in enumerate(self.config.horizons)}, "radii": radii,
                    "disagreement": {str(h): disagreement[i].tolist() for i, h in enumerate(self.config.horizons)},
                    "horizon_margins": margins, "margin": float(margin), "support_distance": distance, "supported": supported,
                    "regime_supported": regime_supported, "fallback_reason": reason, "step": self.step, "remaining_steps": remaining,
                    "history": {"state_norm": float(np.linalg.norm(states[feature_key])) if feature_key != "observation" else 0.0,
                                "innovation_norm": float(np.linalg.norm(yoked if mode == "yoked" else innovation if family == "history" else np.zeros(OBS_SIZE))),
                                "innovation": (yoked if mode == "yoked" else innovation).tolist(),
                                "previous_executed_action": previous_action, "yoked_index": yoke_index if mode == "yoked" else None},
                    "candidates": candidates, "artifact_sha256": self.digest, "backend": "helm-critic-v1"}
        self._states = states
        self._previous_observation = obs[:]
        self.step += 1
        self.last_decision = copy.deepcopy(decision)
        return decision

    def act(self, observation):
        return self.plan(observation)["action"]

    def save(self, path):
        self._assert_bindings()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = _canonical(self._payload).encode()
        if len(content) > MAX_HELM_ARTIFACT_BYTES:
            raise ValueError("Helm artifact exceeds size limit")
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(target.resolve())

    @classmethod
    def load(cls, core, path):
        return cls(core, stable_json(path, max_bytes=MAX_HELM_ARTIFACT_BYTES))
