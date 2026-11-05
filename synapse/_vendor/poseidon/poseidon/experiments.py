"""Frozen paired controller experiments and model-independent action replay.

The atlas sees observations only. Simulator state belongs to the evaluator, and
is retained so every executed transition can be checked without a neural model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import fmean
import tempfile
import time

from .world import ACTIONS, TidePool, WORLD_VERSION, event_uniform, teacher_action

SCHEMA = "poseidon-atlas-experiment-v1"
ARMS = ("policy", "neural_mpc", "atlas", "atlas_no_memory", "heuristic", "random")
LIMITS = [
    "Synthetic TidePool comparisons; no real-world survival or general intelligence claim.",
    "Paired bootstrap intervals are descriptive; small or repeatedly inspected suites are development evidence.",
    "Atlas error radii describe held-out one-step vital errors, not calibrated failure probabilities.",
    "Controllers have different compute budgets; measured latency and prediction quality are separate from survival.",
    "Receipt checksums detect accidental changes, not authenticated authorship. Replay verifies execution, not model provenance or optimality.",
    "No candidate is promoted or activated by this experiment.",
]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity() -> dict:
    from .provenance import assert_source_current
    return assert_source_current()


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int = 93000001
    episodes: int = 4
    max_steps: int = 64
    scarcity: float = 2.5

    def __post_init__(self):
        for name, low, high in (("seed", 0, 2**63 - 33), ("episodes", 1, 32), ("max_steps", 8, 512)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if isinstance(self.scarcity, bool) or not isinstance(self.scarcity, (int, float)) or not math.isfinite(self.scarcity) or not .5 <= self.scarcity <= 4:
            raise ValueError("scarcity must be finite and in [0.5, 4]")
        object.__setattr__(self, "scarcity", float(self.scarcity))


def paired_summary(rows: list[dict]) -> tuple[dict, dict]:
    summary, paired = {}, {}
    base = {row["seed"]: row for row in rows if row["arm"] == "policy"}
    for arm in ARMS:
        group = sorted((row for row in rows if row["arm"] == arm), key=lambda row: row["seed"])
        total_steps = sum(row["steps"] for row in group)
        summary[arm] = {
            "episodes": len(group), "survival_rate": fmean(row["survived"] for row in group),
            "mean_steps": fmean(row["steps"] for row in group),
            "mean_reward": fmean(row["reward"] for row in group),
            "mean_latency_ms": sum(row["decision_ms"] for row in group) / total_steps,
            "fallback_rate": sum(row["fallback_steps"] for row in group) / total_steps,
            "catastrophic_deaths": sum(bool(set((row["death_reason"] or "").split("+")) & {"starvation", "dehydration", "hazard"}) for row in group),
        }
        if arm == "policy":
            continue
        deltas = [row["reward"] - base[row["seed"]]["reward"] for row in group]
        rng = random.Random(782341)
        bootstrap = sorted(fmean(rng.choices(deltas, k=len(deltas))) for _ in range(1000))
        paired[arm] = {
            "baseline": "policy", "mean_reward_delta": fmean(deltas),
            "ci95": [bootstrap[24], bootstrap[974]],
            "wins": sum(x > 1e-10 for x in deltas), "ties": sum(abs(x) <= 1e-10 for x in deltas),
            "losses": sum(x < -1e-10 for x in deltas),
            "per_seed": [{"seed": row["seed"], "reward_delta": delta,
                          "survival_delta": int(row["survived"]) - int(base[row["seed"]]["survived"])}
                         for row, delta in zip(group, deltas)],
        }
    return summary, paired


def _episode(core, atlas, mpc, arm: str, seed: int, spec: ExperimentSpec) -> tuple[dict, dict]:
    env = TidePool(seed=seed, scarcity=spec.scarcity, max_steps=spec.max_steps)
    initial = env.snapshot()
    trace, errors = [], []
    decision_ms, fallbacks = 0.0, 0
    counts = {name: 0 for name in ACTIONS}
    while not env.done:
        obs = env.observe()
        start = time.perf_counter()
        if arm in ("atlas", "atlas_no_memory"):
            decision = atlas.plan(obs, use_memory=arm == "atlas")
        elif arm == "neural_mpc":
            decision = mpc.plan(obs)
        else:
            action = (core.act(obs) if arm == "policy" else teacher_action(obs) if arm == "heuristic"
                      else min(5, int(event_uniform(seed, env.tick, "random-control", "action") * 6)))
            decision = {"action": action, "action_name": ACTIONS[action]}
        elapsed = (time.perf_counter() - start) * 1000
        action = decision["action"]
        if type(action) is not int or not 0 <= action < 6:
            raise ValueError("Controller returned an invalid action")
        nxt, reward, _, info = env.step(action)
        decision_ms += elapsed
        counts[ACTIONS[action]] += 1
        fallbacks += int(bool(decision.get("fallback_reason")))
        prediction = None
        if arm in ("atlas", "atlas_no_memory"):
            prediction = decision["candidates"][action]["predicted_observation"]
        elif arm == "neural_mpc":
            prediction = decision["predicted_futures"][ACTIONS[action]]["step_1_observation"]
        if prediction is not None:
            mse = fmean((float(a) - b)**2 for a, b in zip(prediction, nxt))
            errors.append(mse)
            decision["observed_prediction_mse"] = mse
        trace.append({"observation": obs, "action": action, "action_name": ACTIONS[action],
                      "reward": reward, "next_observation": nxt, "info": info,
                      "state": env.state(), "decision": decision})
    row = {"arm": arm, "seed": seed, "steps": env.tick, "survived": env.alive,
           "reward": env.total_reward, "death_reason": env.death_reason,
           "final_health": env.health, "decision_ms": decision_ms,
           "fallback_steps": fallbacks, "prediction_mse": fmean(errors) if errors else None}
    episode = {"schema": "poseidon-episode-v1", "controller": arm, "world_version": WORLD_VERSION,
               "seed": seed, "scarcity": spec.scarcity, "max_steps": spec.max_steps,
               "steps": env.tick, "survived": env.alive, "alive": env.alive,
               "death": not env.alive, "death_reason": env.death_reason,
               "truncated": info["truncated"], "terminal_reason": info["terminal_reason"],
               "reward": env.total_reward, "final_health": env.health,
               "action_counts": counts, "initial_snapshot": initial,
               "final_snapshot": env.snapshot(), "trajectory": trace}
    return row, episode


def run_experiment(core, atlas, spec: ExperimentSpec | None = None) -> dict:
    from .planning import ModelPredictivePlanner, PlanningConfig
    spec = spec or ExperimentSpec()
    artifact = atlas.artifact
    from .atlas import _model_digest
    if artifact["checkpoint_sha256"] != hashlib.sha256(core.path.read_bytes()).hexdigest() or artifact["model_sha256"] != _model_digest(core):
        raise ValueError("Experiment core does not match the atlas-bound checkpoint and model")
    # Atlas records all seed partitions; never call an overlapping suite held out.
    fit = {"partition": artifact["partition"], "calibration": artifact["calibration"],
           "training_samples": len(artifact["records"])}
    partition = artifact["partition"]
    used = set(partition["train_seeds"]) | set(partition["calibration_seeds"])
    seeds = list(range(spec.seed, spec.seed + spec.episodes))
    if used.intersection(seeds):
        raise ValueError("Experiment seeds overlap atlas training or calibration")
    sources = source_identity()
    checkpoint_sha = hashlib.sha256(core.path.read_bytes()).hexdigest()
    planning_config = PlanningConfig()
    protocol = {"schema": SCHEMA, "spec": asdict(spec), "arms": list(ARMS),
                "world_version": WORLD_VERSION, "source_sha256": sources,
                "checkpoint_sha256": checkpoint_sha, "atlas_sha256": digest(artifact),
                "neural_mpc_config": asdict(planning_config), "randomization": "blake2b-named-events-v1",
                "paired_interval": "1000 episode-pair bootstrap resamples; fixed seed 782341; percentile 95%",
                "runtime": {"device": "cpu", "torch_threads": __import__("torch").get_num_threads()}}
    identifier = digest(protocol)
    mpc = ModelPredictivePlanner(core, planning_config)
    rows, episodes = [], []
    for index, seed in enumerate(seeds):
        # Rotate execution order; timing still includes Python/evaluator overhead.
        order = ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)]
        for arm in order:
            row, episode = _episode(core, atlas, mpc, arm, seed, spec)
            rows.append(row)
            episodes.append(episode)
    if source_identity() != sources or hashlib.sha256(core.path.read_bytes()).hexdigest() != checkpoint_sha or digest(atlas.artifact) != protocol["atlas_sha256"] or _model_digest(core) != artifact["model_sha256"]:
        raise RuntimeError("Source, checkpoint or atlas changed during the experiment; no receipt published")
    summary, paired = paired_summary(rows)
    result = {"schema": SCHEMA, "experiment_id": identifier, "protocol": protocol,
              "summary": summary, "paired": paired, "rows": rows, "episodes": episodes,
              "atlas": {"ready": True, "fit_receipt": fit}, "limits": LIMITS,
              "status": "completed", "promoted": False}
    result["receipt_sha256"] = digest(result)
    return result


def write_receipt(receipt: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Content-addressing avoids overwriting prior timing/outcome evidence.
    path = directory / (receipt["experiment_id"][:16] + "-" + receipt["receipt_sha256"][:12] + ".json")
    fd, temporary = tempfile.mkstemp(prefix="receipt-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def verify_receipt(receipt: dict) -> dict:
    try:
        return _verify_receipt_data(receipt)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("Malformed experiment receipt: missing or invalid evidence fields") from error


def _verify_receipt_data(receipt: dict) -> dict:
    """Replay actual actions independently; requires same TidePool source version."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Unsupported experiment receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest(unsigned) != receipt.get("receipt_sha256"):
        raise ValueError("Receipt checksum mismatch")
    if receipt.get("status") != "completed" or receipt.get("promoted") is not False:
        raise ValueError("Receipt status or activation claim is invalid")
    protocol = receipt["protocol"]
    if digest(protocol) != receipt.get("experiment_id") or protocol.get("arms") != list(ARMS) or protocol.get("schema") != SCHEMA or protocol.get("randomization") != "blake2b-named-events-v1":
        raise ValueError("Experiment identity or arms mismatch")
    if protocol.get("world_version") != WORLD_VERSION or protocol.get("source_sha256", {}).get("world.py") != source_identity()["world.py"]:
        raise ValueError("Replay requires the exact recorded world implementation")
    spec = ExperimentSpec(**protocol["spec"])
    partition = receipt["atlas"]["fit_receipt"]["partition"]
    train, calibration = partition["train_seeds"], partition["calibration_seeds"]
    if any(not isinstance(values, list) or not values or len(values) > 256 or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in values) or len(set(values)) != len(values) for values in (train, calibration)):
        raise ValueError("Invalid recorded atlas partition")
    test_seeds = set(range(spec.seed, spec.seed + spec.episodes))
    if set(train) & set(calibration) or (set(train) | set(calibration)) & test_seeds:
        raise ValueError("Recorded atlas partitions overlap experiment seeds")
    expected = {(arm, seed) for arm in ARMS for seed in range(spec.seed, spec.seed + spec.episodes)}
    episodes, rows = receipt.get("episodes"), receipt.get("rows")
    if not isinstance(episodes, list) or not isinstance(rows, list) or len(episodes) != len(expected) or len(rows) != len(expected):
        raise ValueError("Receipt is missing paired episodes or rows")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not {"arm", "seed", "steps", "survived", "reward", "death_reason", "final_health", "decision_ms", "fallback_steps", "prediction_mse"} <= set(row) or type(row["seed"]) is not int or type(row["survived"]) is not bool or type(row["steps"]) is not int:
            raise ValueError("Invalid episode row fields")
        key = (row["arm"], row["seed"])
        if key not in expected or key in indexed:
            raise ValueError("Duplicate or unexpected episode row")
        if not isinstance(row["decision_ms"], (int, float)) or not math.isfinite(row["decision_ms"]) or row["decision_ms"] < 0:
            raise ValueError("Invalid timing")
        indexed[key] = row
    checked, transitions, seen = 0, 0, set()
    for episode in episodes:
        required = {"schema", "controller", "world_version", "seed", "scarcity", "max_steps", "steps", "survived", "alive", "death", "death_reason", "truncated", "terminal_reason", "reward", "final_health", "action_counts", "initial_snapshot", "final_snapshot", "trajectory"}
        if not isinstance(episode, dict) or not required <= set(episode):
            raise ValueError("Missing episode fields")
        if episode["schema"] != "poseidon-episode-v1" or episode["world_version"] != WORLD_VERSION or episode["scarcity"] != spec.scarcity or episode["max_steps"] != spec.max_steps or type(episode["seed"]) is not int or type(episode["steps"]) is not int or any(type(episode[key]) is not bool for key in ("survived", "alive", "death", "truncated")):
            raise ValueError("Episode contract mismatch")
        key = (episode["controller"], episode["seed"])
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected episode")
        seen.add(key)
        env = TidePool(episode["seed"], spec.scarcity, spec.max_steps)
        TidePool.from_snapshot(episode["initial_snapshot"])
        TidePool.from_snapshot(episode["final_snapshot"])
        if env.snapshot() != episode["initial_snapshot"]:
            raise ValueError("Initial state does not match frozen specification")
        trace = episode["trajectory"]
        if not isinstance(trace, list) or not 1 <= len(trace) <= spec.max_steps:
            raise ValueError("Invalid trajectory length")
        fallback_steps, errors = 0, []
        counts = {name: 0 for name in ACTIONS}
        for event in trace:
            if not isinstance(event, dict) or not {"observation", "action", "action_name", "reward", "next_observation", "info", "state", "decision"} <= set(event) or not isinstance(event["decision"], dict):
                raise ValueError("Missing or invalid transition evidence")
            if env.done or env.observe() != event["observation"]:
                raise ValueError("Replay observation mismatch")
            action = event["action"]
            nxt, reward, _, info = env.step(action)
            if nxt != event["next_observation"] or reward != event["reward"] or info != event["info"] or env.state() != event["state"] or event["action_name"] != ACTIONS[action]:
                raise ValueError("Replay transition mismatch")
            decision = event["decision"]
            if decision.get("action") != action:
                raise ValueError("Decision differs from executed action")
            fallback_steps += int(bool(decision.get("fallback_reason")))
            prediction = None
            if key[0] in ("atlas", "atlas_no_memory"):
                if not isinstance(decision.get("candidates"), list) or len(decision["candidates"]) != 6 or any(not isinstance(candidate, dict) or "predicted_observation" not in candidate for candidate in decision["candidates"]):
                    raise ValueError("Missing atlas prediction evidence")
                prediction = decision["candidates"][action]["predicted_observation"]
            elif key[0] == "neural_mpc":
                if not isinstance(decision.get("predicted_futures"), dict) or not isinstance(decision["predicted_futures"].get(ACTIONS[action]), dict) or "step_1_observation" not in decision["predicted_futures"][ACTIONS[action]]:
                    raise ValueError("Missing neural prediction evidence")
                prediction = decision["predicted_futures"][ACTIONS[action]]["step_1_observation"]
            if prediction is not None:
                if len(prediction) != 16 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in prediction):
                    raise ValueError("Invalid recorded prediction")
                mse = fmean((a - b)**2 for a, b in zip(prediction, nxt))
                if mse != decision.get("observed_prediction_mse"):
                    raise ValueError("Prediction error mismatch")
                errors.append(mse)
            counts[ACTIONS[action]] += 1
            transitions += 1
        row = indexed[key]
        values = {"steps": env.tick, "survived": env.alive, "reward": env.total_reward,
                  "death_reason": env.death_reason, "final_health": env.health}
        if not env.done or env.snapshot() != episode["final_snapshot"] or episode["alive"] != env.alive or episode["death"] != (not env.alive) or any(episode[k] != v or row[k] != v for k, v in values.items()):
            raise ValueError("Replay terminal outcome mismatch")
        if episode.get("action_counts") != counts or episode.get("truncated") != info["truncated"] or episode.get("terminal_reason") != info["terminal_reason"] or row["fallback_steps"] != fallback_steps or row["prediction_mse"] != (fmean(errors) if errors else None):
            raise ValueError("Episode evidence mismatch")
        checked += 1
    summary, paired = paired_summary(rows)
    if receipt.get("summary") != summary or receipt.get("paired") != paired:
        raise ValueError("Summary does not match paired raw outcomes")
    return {"verified": True, "experiment_id": receipt["experiment_id"], "episodes_replayed": checked,
            "transitions_replayed": transitions, "scope": "same-version simulator execution and aggregate arithmetic; not model authorship"}


def load_and_verify(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Receipt exceeds 128 MiB limit")
    return verify_receipt(json.loads(path.read_text(encoding="utf-8")))
