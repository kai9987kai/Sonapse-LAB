"""Paired, action-by-action evaluation and replay verifier for TITAN."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Dict, List, Optional

import torch

from poseidon.aura import AuraController
from poseidon.chimera import ChimeraController
from poseidon.core import CoreRuntime, active_core_path
from poseidon.hyperion import HyperionController
from poseidon.metamorph import MetamorphController
from poseidon.titan import TitanController
from poseidon.world import WORLD_VERSION
from poseidon.world_controls import rollout_with_observed_transitions


ARMS = ("core", "aura", "metamorph", "chimera", "hyperion", "titan")
SCHEMA = "poseidon-titan-experiment-v2"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _controller_set(core: CoreRuntime) -> dict:
    return {"core": core, "aura": AuraController(core=core),
            "metamorph": MetamorphController(core=core),
            "chimera": ChimeraController(core=core),
            "hyperion": HyperionController(core=core), "titan": TitanController(core=core)}


def _reset(controller, arm: str, scarcity: float, seed: int) -> None:
    if arm == "titan":
        controller.reset(scarcity=scarcity, seed=seed)
    elif arm != "core":
        controller.reset(scarcity=scarcity)


def _summarize(episodes: List[dict]) -> dict:
    rewards = [float(episode["reward"]) for episode in episodes]
    return {"episodes": len(episodes),
            "survival_rate": fmean(bool(episode["survived"]) for episode in episodes),
            "mean_steps": fmean(int(episode["steps"]) for episode in episodes),
            "mean_reward": fmean(rewards), "std_reward": pstdev(rewards),
            "rows": [{key: episode[key] for key in
                      ("seed", "steps", "survived", "death_reason", "truncated", "reward", "final_health")}
                     for episode in episodes],
            "episodes_data": episodes}


def _telemetry(episodes: List[dict]) -> dict:
    decisions = [event["decision"] for episode in episodes for event in episode["trajectory"]]
    if not decisions:
        raise ValueError("TITAN produced no decisions")
    mean_keys = ("spectral_divergence", "quantum_entropy", "lattice_coherence", "titan_coherence",
                 "bell_fidelity", "buoyancy_force", "trophic_richness")
    means = {"mean_" + key: fmean(float(decision[key]) for decision in decisions) for key in mean_keys}
    return {key: round(value, 6) for key, value in means.items()} | {
        "decision_count": len(decisions),
        "total_teleportations": sum(max(0, int(episode["trajectory"][-1]["decision"].get("total_teleportations", 0)))
                                     for episode in episodes),
        "total_rem_replays": sum(int(decision.get("rem_replays", 0)) for decision in decisions),
        "total_modder_overrides": sum(int(decision.get("total_modder_overrides", 0))
                                       for decision in decisions),
        "latest_svideo_line": decisions[-1].get("svideo_line_number"),
        "lineage_digests": [episode["trajectory"][-1]["decision"].get("lineage_digest")
                             for episode in episodes],
    }


def _paired(episodes_by_arm: dict, seeds: List[int]) -> List[dict]:
    result = []
    for seed in seeds:
        episodes = {arm: next(item for item in episodes_by_arm[arm] if item["seed"] == seed)
                    for arm in ARMS}
        titan = episodes["titan"]
        row = {"seed": seed}
        for arm in ARMS[:-1]:
            baseline = episodes[arm]
            row[arm + "_steps_delta"] = titan["steps"] - baseline["steps"]
            row[arm + "_reward_delta"] = round(titan["reward"] - baseline["reward"], 12)
            row[arm + "_survival_delta"] = int(titan["survived"]) - int(baseline["survived"])
        result.append(row)
    return result


def run_titan_benchmark(
    episodes: int = 4,
    max_steps: int = 64,
    scarcity: float = 2.5,
    seed_base: int = 99100000,
    core_path: Optional[Path | str] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> Dict[str, object]:
    """Run six controllers once per shared seed and retain replayable transitions."""
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
    results: Dict[str, List[dict]] = {arm: [] for arm in ARMS}
    for seed in seeds:
        for arm in ARMS:
            controller = controllers[arm]
            _reset(controller, arm, float(scarcity), seed)

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
        "benchmark_profile": "titan-six-arm-paired-replay-v2",
        "world_version": WORLD_VERSION,
        "world_source_sha256": _sha256(world_path),
        "core_sha256": _sha256(path),
        "scarcity": float(scarcity),
        "max_steps": max_steps,
        "episodes_per_arm": episodes,
        "seeds": seeds,
        "arms": {arm: _summarize(results[arm]) for arm in ARMS},
        "paired_vs_titan": _paired(results, seeds),
        "titan_telemetry": _telemetry(results["titan"]),
        "scope": "Paired evaluation in the synthetic TidePool environment; telemetry is descriptive and does not establish real-world, biological, quantum-hardware, or general-agent performance.",
    }
    receipt["sha256"] = hashlib.sha256(_canonical(receipt).encode("utf-8")).hexdigest()
    return receipt


def verify_titan_receipt(receipt: Dict[str, object], core_path: Optional[Path | str] = None) -> Dict[str, object]:
    """Check the seal, replay recorded policy decisions, and recompute world outcomes."""
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Invalid TITAN receipt schema")
    recorded_sha = receipt.get("sha256")
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("sha256", None)
    computed_sha = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    if not isinstance(recorded_sha, str) or computed_sha != recorded_sha:
        raise ValueError("TITAN receipt checksum mismatch")
    if receipt.get("status") != "completed-observational-evaluation" or receipt.get("promotion") is not False or receipt.get("fitted_artifact_created") is not False:
        raise ValueError("TITAN receipt status or activation boundary is invalid")
    seeds = receipt.get("seeds")
    episodes = receipt.get("episodes_per_arm")
    max_steps = receipt.get("max_steps")
    scarcity = receipt.get("scarcity")
    if (not isinstance(seeds, list) or not seeds or len(seeds) != episodes or
            any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds) or
            type(max_steps) is not int or not 1 <= max_steps <= 512 or
            isinstance(scarcity, bool) or not isinstance(scarcity, (int, float)) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4):
        raise ValueError("Invalid TITAN benchmark configuration")
    path = Path(core_path) if core_path else active_core_path()
    if _sha256(path) != receipt.get("core_sha256"):
        raise ValueError("TITAN receipt is bound to different core checkpoint bytes")
    world_path = Path(__file__).with_name("world.py")
    if receipt.get("world_version") != WORLD_VERSION or receipt.get("world_source_sha256") != _sha256(world_path):
        raise ValueError("TITAN receipt is bound to different TidePool source")
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("TITAN receipt must contain the complete paired arm set")
    if any(not isinstance(arms[arm], dict) or not isinstance(arms[arm].get("episodes_data"), list) or
           len(arms[arm]["episodes_data"]) != len(seeds) for arm in ARMS):
        raise ValueError("TITAN receipt episode records are incomplete")

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
            _reset(controller, arm, float(scarcity), seed)
            actual = rollout_with_observed_transitions(controller, seed=seed,
                                                       max_steps=max_steps, scarcity=float(scarcity))
            if _canonical(actual) != _canonical(expected):
                raise ValueError(f"{arm} replay mismatch for seed {seed}")
            replayed[arm].append(actual)
        if _canonical(_summarize(replayed[arm])) != _canonical(arms[arm]):
            raise ValueError(f"{arm} summary does not match replayed episodes")

    paired = _paired(replayed, seeds)
    telemetry = _telemetry(replayed["titan"])
    if _canonical(paired) != _canonical(receipt.get("paired_vs_titan")):
        raise ValueError("Paired differences do not match replayed episodes")
    if _canonical(telemetry) != _canonical(receipt.get("titan_telemetry")):
        raise ValueError("TITAN telemetry does not match replayed decisions")
    titan_summary = arms["titan"]
    return {"verified": True, "verification_scope": "receipt integrity, controller replay, and TidePool transition replay",
            "receipt_sha256": recorded_sha, "episodes_per_arm": episodes,
            "survival_rate": titan_summary["survival_rate"], "mean_reward": titan_summary["mean_reward"],
            "mean_spectral_divergence": telemetry["mean_spectral_divergence"],
            "mean_titan_coherence": telemetry["mean_titan_coherence"],
            "promotion": False, "core_sha256": receipt["core_sha256"]}
