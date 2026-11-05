"""Paired causal-identity evaluation and replay verifier for TRIDENT."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Optional

import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.trident import TRIDENT_MODES, TridentController
from poseidon.world import WORLD_VERSION
from poseidon.world_controls import rollout_with_observed_transitions


ARMS = ("core", "trident", "erased", "shifted", "no_topology", "ungated")
SCHEMA = "poseidon-trident-experiment-v1"
_MODE = {"trident": "intact", "erased": "erased", "shifted": "shifted",
         "no_topology": "no_topology", "ungated": "ungated"}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _controller_set(core: CoreRuntime) -> dict:
    controllers = {"core": core}
    for arm, mode in _MODE.items():
        controllers[arm] = TridentController(core=core, mode=mode)
    return controllers


def _reset(controller, arm: str, scarcity: float, seed: int, max_steps: int) -> None:
    if arm != "core":
        controller.reset(scarcity=scarcity, seed=seed, max_steps=max_steps)


def _summarize(episodes: list[dict]) -> dict:
    rewards = [float(episode["reward"]) for episode in episodes]
    return {"episodes": len(episodes),
            "survival_rate": fmean(bool(episode["survived"]) for episode in episodes),
            "mean_steps": fmean(int(episode["steps"]) for episode in episodes),
            "mean_reward": fmean(rewards), "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "rows": [{key: episode[key] for key in
                      ("seed", "steps", "survived", "death_reason", "truncated", "reward", "final_health")}
                     for episode in episodes],
            "episodes_data": episodes}


def _telemetry(episodes: list[dict]) -> dict:
    decisions = [event["decision"] for episode in episodes for event in episode["trajectory"]]
    if not decisions:
        raise ValueError("TRIDENT produced no decisions")
    overrides = sum(bool(decision.get("overridden")) for decision in decisions)
    identity = sum(bool(decision.get("identity_specific")) for decision in decisions)
    return {
        "decision_count": len(decisions),
        "override_count": overrides,
        "identity_specific_count": identity,
        "mean_residual_norm": round(fmean(float(decision["residual_norm"]) for decision in decisions), 6),
        "mean_intact_advantage": round(fmean(float(decision["intact_advantage"]) for decision in decisions), 6),
        "mean_erased_advantage": round(fmean(float(decision["erased_advantage"]) for decision in decisions), 6),
        "mean_shifted_advantage": round(fmean(float(decision["shifted_advantage"]) for decision in decisions), 6),
        "mean_patch_count": round(fmean(float(decision["patch_count"]) for decision in decisions), 6),
        "fallback_reasons": sorted({str(decision.get("fallback_reason")) for decision in decisions}),
    }


def _paired(episodes_by_arm: dict, seeds: list[int]) -> list[dict]:
    result = []
    for seed in seeds:
        episodes = {arm: next(item for item in episodes_by_arm[arm] if item["seed"] == seed)
                    for arm in ARMS}
        trident = episodes["trident"]
        row = {"seed": seed}
        for arm in ARMS:
            if arm == "trident":
                continue
            baseline = episodes[arm]
            row[arm + "_steps_delta"] = trident["steps"] - baseline["steps"]
            row[arm + "_reward_delta"] = round(trident["reward"] - baseline["reward"], 12)
            row[arm + "_survival_delta"] = int(trident["survived"]) - int(baseline["survived"])
        result.append(row)
    return result


def run_trident_benchmark(
    episodes: int = 4,
    max_steps: int = 64,
    scarcity: float = 2.5,
    seed_base: int = 88000000,
    core_path: Optional[Path | str] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    if type(episodes) is not int or not 1 <= episodes <= 32:
        raise ValueError("episodes must be an integer from 1 to 32")
    if type(max_steps) is not int or not 1 <= max_steps <= 512:
        raise ValueError("max_steps must be an integer from 1 to 512")
    if type(seed_base) is not int or not 0 <= seed_base <= 2**63 - episodes:
        raise ValueError("seed_base cannot form the requested nonnegative seed range")
    if isinstance(scarcity, bool) or not isinstance(scarcity, (int, float)) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4:
        raise ValueError("scarcity must be finite and between 0.5 and 4")
    torch.set_num_threads(2)
    path = Path(core_path) if core_path else active_core_path()
    core = CoreRuntime(path)
    controllers = _controller_set(core)
    seeds = list(range(seed_base, seed_base + episodes))
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for seed in seeds:
        for arm in ARMS:
            controller = controllers[arm]
            _reset(controller, arm, float(scarcity), seed, max_steps)

            def progress_tick(event, arm=arm, seed=seed):
                if progress:
                    progress({"phase": "transition", "arm": arm, **event})

            episode = rollout_with_observed_transitions(
                controller, seed=seed, max_steps=max_steps, scarcity=float(scarcity),
                on_transition=progress_tick if progress else None,
            )
            results[arm].append(episode)
            if progress:
                progress({"phase": "episode", "arm": arm, "seed": seed,
                          "steps": episode["steps"], "survived": episode["survived"]})
    world_path = Path(__file__).with_name("world.py")
    receipt = {
        "schema": SCHEMA,
        "status": "completed-observational-evaluation",
        "promotion": False,
        "fitted_artifact_created": False,
        "benchmark_profile": "trident-causal-identity-six-arm-v1",
        "world_version": WORLD_VERSION,
        "world_source_sha256": _sha256(world_path),
        "core_sha256": _sha256(path),
        "scarcity": float(scarcity),
        "max_steps": max_steps,
        "episodes_per_arm": episodes,
        "seeds": seeds,
        "modes": {arm: _MODE.get(arm, "policy") for arm in ARMS},
        "arms": {arm: _summarize(results[arm]) for arm in ARMS},
        "paired_vs_trident": _paired(results, seeds),
        "trident_telemetry": _telemetry(results["trident"]),
        "scope": "Paired evaluation of residual identity and observed directed topology in synthetic TidePool; results do not promote a model or establish real-world capability.",
        "limits": [
            "Intact, erased, shifted, topology-ablated and ungated arms share seeds; each arm visits its own states.",
            "Directed support and replenishment estimates use only executed observations.",
            "Identity specificity is a ranking comparison, not a causal identification theorem.",
        ],
    }
    receipt["sha256"] = hashlib.sha256(_canonical(receipt).encode("utf-8")).hexdigest()
    return receipt


def verify_trident_receipt(receipt: dict, core_path: Optional[Path | str] = None) -> dict:
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Invalid TRIDENT receipt schema")
    recorded_sha = receipt.get("sha256")
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("sha256", None)
    computed_sha = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if not isinstance(recorded_sha, str) or computed_sha != recorded_sha:
        raise ValueError("TRIDENT receipt checksum mismatch")
    if receipt.get("status") != "completed-observational-evaluation" or receipt.get("promotion") is not False or receipt.get("fitted_artifact_created") is not False:
        raise ValueError("TRIDENT receipt status or activation boundary is invalid")
    seeds = receipt.get("seeds")
    episodes = receipt.get("episodes_per_arm")
    max_steps = receipt.get("max_steps")
    scarcity = receipt.get("scarcity")
    if (not isinstance(seeds, list) or not seeds or len(seeds) != episodes or
            any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds) or
            type(max_steps) is not int or not 1 <= max_steps <= 512 or
            isinstance(scarcity, bool) or not isinstance(scarcity, (int, float)) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4):
        raise ValueError("Invalid TRIDENT benchmark configuration")
    path = Path(core_path) if core_path else active_core_path()
    if _sha256(path) != receipt.get("core_sha256"):
        raise ValueError("TRIDENT receipt is bound to different core checkpoint bytes")
    world_path = Path(__file__).with_name("world.py")
    if receipt.get("world_version") != WORLD_VERSION or receipt.get("world_source_sha256") != _sha256(world_path):
        raise ValueError("TRIDENT receipt is bound to different TidePool source")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("TRIDENT receipt must contain the complete paired arm set")
    if any(not isinstance(arms[arm], dict) or not isinstance(arms[arm].get("episodes_data"), list) or
           len(arms[arm]["episodes_data"]) != len(seeds) for arm in ARMS):
        raise ValueError("TRIDENT receipt episode records are incomplete")
    if set(receipt.get("modes", {})) != set(ARMS):
        raise ValueError("TRIDENT receipt modes are incomplete")
    for arm, mode in receipt["modes"].items():
        if arm == "core":
            if mode != "policy":
                raise ValueError("Core arm must remain the frozen policy")
        elif mode not in TRIDENT_MODES:
            raise ValueError("Unknown TRIDENT evaluation mode")

    torch.set_num_threads(2)
    core = CoreRuntime(path)
    controllers = _controller_set(core)
    replayed = {arm: [] for arm in ARMS}
    for arm in ARMS:
        recorded = arms[arm]["episodes_data"]
        if [episode.get("seed") for episode in recorded] != seeds:
            raise ValueError(f"{arm} arm seed sequence does not match the paired configuration")
        for seed, expected in zip(seeds, recorded):
            controller = controllers[arm]
            _reset(controller, arm, float(scarcity), seed, max_steps)
            actual = rollout_with_observed_transitions(controller, seed=seed,
                                                       max_steps=max_steps, scarcity=float(scarcity))
            if _canonical(actual) != _canonical(expected):
                raise ValueError(f"{arm} replay mismatch for seed {seed}")
            replayed[arm].append(actual)
        if _canonical(_summarize(replayed[arm])) != _canonical(arms[arm]):
            raise ValueError(f"{arm} summary does not match replayed episodes")

    paired = _paired(replayed, seeds)
    telemetry = _telemetry(replayed["trident"])
    if _canonical(paired) != _canonical(receipt.get("paired_vs_trident")):
        raise ValueError("Paired differences do not match replayed episodes")
    if _canonical(telemetry) != _canonical(receipt.get("trident_telemetry")):
        raise ValueError("TRIDENT telemetry does not match replayed decisions")
    summary = arms["trident"]
    return {"verified": True, "verification_scope": "receipt integrity, controller replay, and TidePool transition replay",
            "receipt_sha256": recorded_sha, "episodes_per_arm": episodes,
            "survival_rate": summary["survival_rate"], "mean_reward": summary["mean_reward"],
            "override_count": telemetry["override_count"],
            "identity_specific_count": telemetry["identity_specific_count"],
            "promotion": False, "core_sha256": receipt["core_sha256"]}
