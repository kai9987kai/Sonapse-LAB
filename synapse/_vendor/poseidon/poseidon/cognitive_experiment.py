"""POSEIDON COGNITIVE EXPERIMENT: Rigorous Matched-Seed Lesion Assay & POMDP Benchmark.

Evaluates Supermix Poseidon v2.0 Cognitive Platform across 6 matched arms:
1. intact: Full CognitiveAgent v2.0 with Oracle, 3-tier Mnemosyne, Horizon-X, Neuro.
2. erased: Working memory cleared at each step, episodic store cleared.
3. shifted: Circularly permuted episodic keys (norm-preserving semantic lesion).
4. random_retrieval: Uncorrelated episodic memory recall.
5. memory_free: Ablated memory subsystems (pure reactive baseline).
6. frozen_tidal: Pure un-augmented frozen baseline policy.

Evaluations run across identical paired seeds under standard and POMDP environments.
Produces an auditable cryptographically-hashed receipt.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, stdev
from typing import Any, Callable

import torch

from .cognitive_agent import CognitiveAgent
from .core import CoreRuntime, active_core_path
from .provenance import assert_source_current
from .world import ACTIONS, N_ACTIONS, TidePool, WORLD_VERSION
from .worldlab import POMDPConfig, POMDPTidePool

COGNITIVE_EXPERIMENT_SCHEMA = "poseidon-cognitive-experiment-v3"
ARMS = ("intact", "erased", "shifted", "random_retrieval", "memory_free", "frozen_tidal")
CONTROLLER_RANDOMIZATION = "python-torch-per-episode-seed-v1"
_RESEARCH_DIRECTORY = Path(__file__).parent / "research"
_RESEARCH_SOURCES = {f"research/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(_RESEARCH_DIRECTORY.glob("*.py"))}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _source_identity() -> dict[str, str]:
    sources = assert_source_current()
    research = {f"research/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(_RESEARCH_DIRECTORY.glob("*.py"))}
    if research != _RESEARCH_SOURCES:
        raise ValueError("Research source changed since import; restart before collecting or replaying evidence")
    return {**sources, **research}


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(core: CoreRuntime) -> dict[str, Any]:
    digest = hashlib.sha256()
    for name, tensor in sorted(core.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return {"checkpoint_sha256": _file_digest(core.path), "state_dict_sha256": digest.hexdigest(),
            "config": asdict(core.model.config)}


def _load_bound_core(path: str | Path) -> tuple[CoreRuntime, dict[str, Any]]:
    before = _file_digest(path)
    core = CoreRuntime(path)
    identity = _model_identity(core)
    if before != identity["checkpoint_sha256"]:
        raise ValueError("Core checkpoint changed while loading")
    return core, identity


def _oracle_identity(path: str | Path | None) -> dict[str, str] | None:
    return None if path is None else {"path": str(Path(path).resolve()), "sha256": _file_digest(path)}


def _validate_protocol(seed_base: int, episodes: int, max_steps: int, scarcity: float, use_pomdp: bool) -> None:
    if type(seed_base) is not int or not 0 <= seed_base < 2**63:
        raise ValueError("Invalid cognitive seed_base")
    if type(episodes) is not int or not 1 <= episodes <= 256 or seed_base + episodes > 2**63:
        raise ValueError("Cognitive episodes must contain 1 to 256 unique seeds")
    if (type(max_steps) is not int or not 1 <= max_steps <= 10000
            or episodes * max_steps * len(ARMS) > 100000):
        raise ValueError("Invalid or excessive cognitive transition budget")
    if type(scarcity) not in (float, int) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4.0:
        raise ValueError("Invalid cognitive scarcity")
    if type(use_pomdp) is not bool:
        raise ValueError("use_pomdp must be boolean")


@contextmanager
def _episode_randomness(seed: int):
    # Existing dream/retrieval code uses Python's generator and neural constructors
    # use Torch's generator. Scope both so paired runs and replay are repeatable.
    python_state = random.getstate()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)


def _environment(seed: int, scarcity: float, max_steps: int, use_pomdp: bool):
    if use_pomdp:
        return POMDPTidePool(POMDPConfig(scarcity=scarcity, seed=seed), max_steps=max_steps)
    return TidePool(scarcity=scarcity, seed=seed, max_steps=max_steps)


def _run_single_seed(
    arm: str,
    seed: int,
    max_steps: int,
    scarcity: float,
    use_pomdp: bool = False,
    core: CoreRuntime | None = None,
    oracle_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("Unsupported cognitive arm")
    with _episode_randomness(seed):
        core = core or CoreRuntime(active_core_path())
        env = _environment(seed, scarcity, max_steps, use_pomdp)
        if arm != "frozen_tidal":
            kwargs = {"oracle_checkpoint": oracle_checkpoint} if oracle_checkpoint is not None else {}
            agent = CognitiveAgent(core=core, mode=arm, **kwargs)
            rollout = agent.rollout(env, max_steps=max_steps, mode=arm)
            return {"arm": arm, "seed": seed, **{key: rollout[key] for key in (
                "steps", "total_reward", "survived", "death", "death_reason", "truncated", "terminal_reason",
                "final_health", "overrides", "override_rate", "initial_snapshot", "trajectory")}}

        obs = env.reset(seed=seed)
        initial_snap = env.snapshot()
        total_reward = 0.0
        steps = 0
        trajectory = []
        for _ in range(max_steps):
            act = core.act(obs)
            if use_pomdp:
                res = env.step(act)
                next_obs = res.observation
                reward = res.reward
                done = res.done
                info = res.info
            else:
                next_obs, reward, done, info = env.step(act)

            total_reward += reward
            steps += 1
            dec = {"action": act, "action_name": ACTIONS[act], "override": False}
            trajectory.append({
                "observation": obs,
                "action": act,
                "action_name": ACTIONS[act],
                "reward": reward,
                "done": bool(done),
                "next_observation": next_obs,
                "state": env.state(),
                "info": info,
                "decision": dec,
            })

            obs = next_obs
            if done:
                break

        return {
            "arm": arm,
            "seed": seed,
            "steps": steps,
            "total_reward": float(total_reward),
            "survived": bool(env.alive),
            "death": not env.alive,
            "death_reason": env.death_reason,
            "truncated": bool(env.alive and env.done),
            "terminal_reason": "death" if not env.alive else "horizon",
            "final_health": float(env.health),
            "overrides": 0,
            "override_rate": 0.0,
            "initial_snapshot": initial_snap,
            "trajectory": trajectory,
        }


def run_cognitive_benchmark(
    core_path: str | Path | None = None,
    seed_base: int = 90000000,
    episodes: int = 4,
    max_steps: int = 32,
    scarcity: float = 2.5,
    use_pomdp: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
    oracle_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Runs complete 6-arm matched lesion and POMDP benchmark."""
    _validate_protocol(seed_base, episodes, max_steps, scarcity, use_pomdp)
    sources = _source_identity()
    core, model_identity = _load_bound_core(core_path or active_core_path())
    oracle_identity = _oracle_identity(oracle_checkpoint)
    seeds = [seed_base + i for i in range(episodes)]

    arm_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}

    total_tasks = len(ARMS) * episodes
    completed = 0

    for arm in ARMS:
        for seed in seeds:
            res = _run_single_seed(
                arm=arm,
                seed=seed,
                max_steps=max_steps,
                scarcity=scarcity,
                use_pomdp=use_pomdp,
                core=core,
                oracle_checkpoint=oracle_checkpoint,
            )
            arm_results[arm].append(res)
            completed += 1
            if progress:
                progress({
                    "phase": "running",
                    "arm": arm,
                    "seed": seed,
                    "completed": completed,
                    "total": total_tasks,
                })

    arm_summaries, paired_comparisons = _summarize(arm_results)
    if (_source_identity() != sources or _model_identity(core) != model_identity
            or _oracle_identity(oracle_checkpoint) != oracle_identity):
        raise ValueError("Source or model identity changed during benchmark")
    receipt_body = {
        "schema": COGNITIVE_EXPERIMENT_SCHEMA,
        "world_version": WORLD_VERSION,
        "controller_randomization": CONTROLLER_RANDOMIZATION,
        "model": model_identity,
        "oracle_artifact": oracle_identity,
        "source_sha256": sources,
        "seed_base": seed_base,
        "episodes": episodes,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "use_pomdp": use_pomdp,
        "arms": arm_summaries,
        "paired": paired_comparisons,
        "episodes_data": arm_results,
    }
    receipt_body = json.loads(json.dumps(receipt_body))
    receipt_body["sha256"] = _digest(receipt_body)
    return receipt_body


def _summarize(arm_results: dict[str, list[dict[str, Any]]]) -> tuple[dict, dict]:
    """Recompute all outcome claims from complete seed-aligned episode records."""
    arm_summaries = {}
    for arm, records in arm_results.items():
        rewards = [r["total_reward"] for r in records]
        survivals = [1.0 if r["survived"] else 0.0 for r in records]
        steps_list = [r["steps"] for r in records]
        overrides_list = [r["overrides"] for r in records]

        m_rew = fmean(rewards)
        s_rew = stdev(rewards) if len(rewards) > 1 else 0.0
        surv_rate = fmean(survivals)
        m_steps = fmean(steps_list)

        arm_summaries[arm] = {
            "episodes": len(records),
            "survival_rate": round(surv_rate, 4),
            "mean_reward": round(m_rew, 4),
            "std_reward": round(s_rew, 4),
            "mean_steps": round(m_steps, 2),
            "total_overrides": sum(overrides_list),
            "mean_override_rate": round(fmean([r["override_rate"] for r in records]), 4),
        }

    # Paired comparisons vs intact
    intact_rewards = [r["total_reward"] for r in arm_results["intact"]]
    paired_comparisons = {}
    for arm in ARMS:
        if arm == "intact":
            continue
        deltas = [
            intact_r - other_r
            for intact_r, other_r in zip(intact_rewards, [r["total_reward"] for r in arm_results[arm]])
        ]
        mean_delta = fmean(deltas)
        wins = sum(d > 1e-6 for d in deltas)
        ties = sum(abs(d) <= 1e-6 for d in deltas)
        losses = sum(d < -1e-6 for d in deltas)

        paired_comparisons[f"intact_vs_{arm}"] = {
            "mean_reward_advantage": round(mean_delta, 4),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "deltas": [round(d, 4) for d in deltas],
        }

    return arm_summaries, paired_comparisons


def verify_cognitive_receipt(receipt: dict[str, Any], core_path: str | Path | None = None) -> bool:
    """Validate identities, replay every transition/controller and recompute claims.

    Legacy v2 receipts have no model/source binding and may omit exact sensor
    state. They are unsupported rather than certified by a hash.
    Success retains the public boolean contract; invalid evidence raises ValueError.
    """
    try:
        return _verify_cognitive_receipt(receipt, core_path)
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError, OSError, RecursionError) as error:
        raise ValueError("Malformed or unavailable cognitive evidence") from error


def _same(actual: Any, expected: Any, label: str) -> None:
    # JSON equality handles sensor-state tuples persisted as lists, distinguishes
    # booleans from integer claims, and rejects nonfinite numbers.
    if _digest(actual) != _digest(expected):
        raise ValueError(f"Cognitive receipt {label} mismatch")


def _verify_cognitive_receipt(receipt: dict[str, Any], core_path: str | Path | None) -> bool:
    fields = {"schema", "world_version", "controller_randomization", "model", "oracle_artifact", "source_sha256",
              "seed_base", "episodes", "max_steps", "scarcity", "use_pomdp", "arms", "paired", "episodes_data", "sha256"}
    if not isinstance(receipt, dict) or set(receipt) != fields:
        raise ValueError("Invalid cognitive receipt fields (complete v3 evidence required)")
    if (receipt["schema"] != COGNITIVE_EXPERIMENT_SCHEMA or receipt["world_version"] != WORLD_VERSION
            or receipt["controller_randomization"] != CONTROLLER_RANDOMIZATION):
        raise ValueError("Unsupported cognitive receipt protocol; v2 hash-only evidence cannot be replay-certified")
    _validate_protocol(*(receipt[key] for key in ("seed_base", "episodes", "max_steps", "scarcity", "use_pomdp")))
    if receipt["sha256"] != _digest({key: value for key, value in receipt.items() if key != "sha256"}):
        raise ValueError("Cognitive receipt hash mismatch")
    sources = _source_identity()
    _same(receipt["source_sha256"], sources, "source identity")
    core, model_identity = _load_bound_core(core_path or active_core_path())
    _same(receipt["model"], model_identity, "model identity")
    oracle_identity = receipt["oracle_artifact"]
    oracle_path = None
    if oracle_identity is not None:
        if (not isinstance(oracle_identity, dict) or set(oracle_identity) != {"path", "sha256"}
                or not isinstance(oracle_identity["path"], str) or not oracle_identity["path"]):
            raise ValueError("Invalid ORACLE artifact identity")
        oracle_path = oracle_identity["path"]
        _same(oracle_identity, _oracle_identity(oracle_path), "ORACLE artifact identity")

    data = receipt["episodes_data"]
    if not isinstance(data, dict) or set(data) != set(ARMS):
        raise ValueError("Missing or unexpected cognitive arms")
    seeds = list(range(receipt["seed_base"], receipt["seed_base"] + receipt["episodes"]))
    # Validate complete unique pairs before expensive neural replay.
    for arm in ARMS:
        records = data[arm]
        if (not isinstance(records, list) or len(records) != len(seeds)
                or any(not isinstance(record, dict) for record in records)
                or any(type(record.get("seed")) is not int for record in records)
                or [record["seed"] for record in records] != seeds
                or any(record.get("arm") != arm for record in records)):
            raise ValueError("Missing, duplicate or unordered cognitive arm/seed pairs")

    verified = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for record in data[arm]:
            _replay_environment(record, receipt)
            # Controller replay binds selected actions, forecast telemetry,
            # guardrail decisions and override claims to the model and source.
            replayed = _run_single_seed(arm, record["seed"], receipt["max_steps"], receipt["scarcity"],
                                        receipt["use_pomdp"], core, oracle_checkpoint=oracle_path)
            _same(record, replayed, f"controller replay {arm}/{record['seed']}")
            verified[arm].append(replayed)
    summaries, paired = _summarize(verified)
    _same(receipt["arms"], summaries, "arm summaries")
    _same(receipt["paired"], paired, "paired summaries")
    _same(_source_identity(), sources, "source identity after replay")
    _same(_model_identity(core), model_identity, "model identity after replay")
    _same(_oracle_identity(oracle_path), oracle_identity, "ORACLE identity after replay")
    return True


def _replay_environment(record: dict[str, Any], receipt: dict[str, Any]) -> None:
    fields = {"arm", "seed", "steps", "total_reward", "survived", "death", "death_reason", "truncated",
              "terminal_reason", "final_health", "overrides", "override_rate", "initial_snapshot", "trajectory"}
    if set(record) != fields:
        raise ValueError("Invalid cognitive episode fields")
    trace = record["trajectory"]
    if (not isinstance(trace, list) or not 1 <= len(trace) <= receipt["max_steps"]
            or type(record["steps"]) is not int or record["steps"] != len(trace)):
        raise ValueError("Empty, incomplete or invalid cognitive trajectory")
    env = _environment(record["seed"], receipt["scarcity"], receipt["max_steps"], receipt["use_pomdp"])
    observation = env.reset(seed=record["seed"])
    _same(record["initial_snapshot"], env.snapshot(), "initial snapshot after reset")
    # Restore actual evidence, including the post-observation sensor RNG state.
    # Calling observe() here would consume an extra noise draw.
    env = (POMDPTidePool.from_snapshot(record["initial_snapshot"]) if receipt["use_pomdp"]
           else TidePool.from_snapshot(record["initial_snapshot"]))
    total_reward = 0.0
    overrides = 0
    event_fields = {"observation", "action", "action_name", "reward", "done", "next_observation", "state", "info", "decision"}
    for index, event in enumerate(trace):
        if not isinstance(event, dict) or set(event) != event_fields:
            raise ValueError("Invalid cognitive transition fields")
        action = event["action"]
        if type(action) is not int or not 0 <= action < N_ACTIONS:
            raise ValueError("Invalid cognitive action")
        if env.done:
            raise ValueError("Cognitive trajectory continues after terminal transition")
        _same(event["observation"], observation, f"observation at {index}")
        if receipt["use_pomdp"]:
            result = env.step(action)
            observation, reward, done, info = result.observation, result.reward, result.done, result.info
        else:
            observation, reward, done, info = env.step(action)
        actual = {"action_name": ACTIONS[action], "next_observation": observation, "reward": reward,
                  "done": done, "info": info, "state": env.state()}
        _same({key: event[key] for key in actual}, actual, f"transition at {index}")
        decision = event["decision"]
        if (not isinstance(decision, dict) or type(decision.get("action")) is not int
                or decision["action"] != action or decision.get("action_name") != ACTIONS[action]
                or type(decision.get("override")) is not bool):
            raise ValueError("Cognitive decision disagrees with executed action")
        overrides += int(decision["override"])
        total_reward += reward
    if not env.done:
        raise ValueError("Cognitive trajectory omits the rest of the episode")
    outcomes = {"total_reward": total_reward, "survived": bool(env.alive), "death": not env.alive,
                "death_reason": env.death_reason, "truncated": bool(env.alive and env.done),
                "terminal_reason": "death" if not env.alive else "horizon", "final_health": float(env.health),
                "overrides": overrides, "override_rate": float(overrides / len(trace))}
    _same({key: record[key] for key in outcomes}, outcomes, "terminal outcomes")
