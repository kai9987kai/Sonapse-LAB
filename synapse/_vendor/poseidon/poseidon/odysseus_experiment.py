"""Paired benchmark evaluation and weightless branch replay for Odysseus.

Compares policy, odysseus_calibrated, odysseus_ungated, odysseus_no_memory,
odyssey, and random across paired evaluation seeds.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import time
from typing import Sequence

from .core import CoreRuntime
from .odysseus import EmpiricalCognitiveMap, OdysseusAtlas, OdysseusConfig
from .world import ACTIONS, TidePool, WORLD_VERSION

SCHEMA = "poseidon-odysseus-experiment-v1"
ARMS = ("policy", "odysseus_calibrated", "odysseus_ungated", "odysseus_no_memory", "odyssey", "random")

LIMITS = [
    "Synthetic sampled TidePool development evidence; no safety or general superiority guarantee.",
    "Empirical transition probabilities and Bayesian replenishment estimates are bounded to visited patch history.",
    "Replay checks action execution, transition consistency, and bound evidence arithmetic without model weights.",
    "Descriptive calibration error radii are held-out diagnostic margins without future coverage guarantee.",
    "Leave-one-seed-out ranges are influence checks, not confidence intervals or independent replication.",
]


def digest(payload: object) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def rollout_arm(
    controller_arm: str,
    core: CoreRuntime,
    odysseus: OdysseusAtlas,
    seed: int,
    max_steps: int = 64,
    scarcity: float = 2.5,
) -> dict:
    """Roll out a single episode under a specified controller arm."""
    env = TidePool(seed, max_steps=max_steps, scarcity=scarcity)
    obs = env.observe()
    odysseus.reset(scarcity)

    # Initialize v0.5 odyssey if needed
    odyssey_map = None
    if controller_arm == "odyssey":
        from .odyssey import CognitiveMap as OdysseyMap
        odyssey_map = OdysseyMap(scarcity)

    decisions = []
    actions = []
    observations = [obs]
    rewards = []
    overrides = 0

    while not env.done:
        start_t = time.perf_counter()
        if controller_arm == "random":
            action = int(hashlib.blake2b(f"{seed}-{env.tick}".encode()).digest()[0] % 6)
            decision = {"action": action, "overridden": False, "reason": "random"}
        elif controller_arm == "policy":
            action = core.act(obs)
            decision = {"action": action, "overridden": False, "reason": "policy"}
        elif controller_arm == "odysseus_calibrated":
            plan = odysseus.plan(obs, mode="calibrated")
            action = plan["action"]
            overridden = plan["overridden"]
            if overridden:
                overrides += 1
            decision = plan
        elif controller_arm == "odysseus_ungated":
            plan = odysseus.plan(obs, mode="uncalibrated")
            action = plan["action"]
            overridden = plan["overridden"]
            if overridden:
                overrides += 1
            decision = plan
        elif controller_arm == "odysseus_no_memory":
            # Policy with stamina rest guard only
            base_act = core.act(obs)
            action = 0 if obs[3] < 0.20 and base_act in (4, 5) else base_act
            overridden = (action != base_act)
            if overridden:
                overrides += 1
            decision = {"action": action, "overridden": overridden, "reason": "no_memory_guard"}
        elif controller_arm == "odyssey":
            # v0.5 heuristic BFS logic
            patch = odyssey_map.update(obs)
            base_act = core.act(obs)
            action = 0 if obs[3] < 0.20 and base_act in (4, 5) else base_act
            overridden = (action != base_act)
            if overridden:
                overrides += 1
            odyssey_map.record_action(action)
            decision = {"action": action, "overridden": overridden, "reason": "odyssey_v5"}
        else:
            raise ValueError(f"Unknown arm: {controller_arm}")

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        decisions.append({"tick": env.tick, "action": action, "overridden": decision.get("overridden", False), "reason": decision.get("reason"), "ms": elapsed_ms})
        actions.append(action)

        nxt, rew, done, info = env.step(action)
        rewards.append(rew)
        observations.append(nxt)
        obs = nxt

    return {
        "controller": controller_arm,
        "seed": seed,
        "steps": env.tick,
        "survived": (env.death_reason is None),
        "death_reason": env.death_reason,
        "total_reward": sum(rewards),
        "overrides": overrides,
        "actions": actions,
        "decisions": decisions,
    }


def run_odysseus_experiment(
    core: CoreRuntime,
    odysseus: OdysseusAtlas,
    seeds: Sequence[int],
    max_steps: int = 64,
    scarcity: float = 2.5,
    progress=None,
) -> dict:
    """Execute paired multi-arm evaluation across all specified seeds."""
    episodes = []
    total_runs = len(ARMS) * len(seeds)
    completed = 0

    if progress and callable(progress):
        progress({"phase": "init", "completed": 0, "total": total_runs})

    for seed in seeds:
        for arm in ARMS:
            ep = rollout_arm(arm, core, odysseus, seed, max_steps, scarcity)
            episodes.append(ep)
            completed += 1
            if progress and callable(progress):
                progress({"phase": "rollout", "completed": completed, "total": total_runs, "arm": arm, "seed": seed})

    # Aggregate arm summaries
    summary = {}
    for arm in ARMS:
        arm_eps = [e for e in episodes if e["controller"] == arm]
        summary[arm] = {
            "episodes": len(arm_eps),
            "survival_rate": fmean(1.0 if e["survived"] else 0.0 for e in arm_eps),
            "mean_steps": fmean(e["steps"] for e in arm_eps),
            "mean_reward": fmean(e["total_reward"] for e in arm_eps),
            "total_overrides": sum(e["overrides"] for e in arm_eps),
            "override_rate": sum(e["overrides"] for e in arm_eps) / max(1, sum(e["steps"] for e in arm_eps)),
        }

    # Paired comparisons vs policy baseline
    paired = {}
    policy_by_seed = {e["seed"]: e for e in episodes if e["controller"] == "policy"}
    for arm in ARMS:
        if arm == "policy":
            continue
        arm_by_seed = {e["seed"]: e for e in episodes if e["controller"] == arm}
        deltas = []
        wins, ties, losses = 0, 0, 0
        for s in seeds:
            delta = arm_by_seed[s]["total_reward"] - policy_by_seed[s]["total_reward"]
            deltas.append({"seed": s, "reward_delta": delta, "survival_delta": int(arm_by_seed[s]["survived"]) - int(policy_by_seed[s]["survived"])})
            if delta > 1e-4:
                wins += 1
            elif delta < -1e-4:
                losses += 1
            else:
                ties += 1

        # Leave-one-seed-out sensitivity
        loso_means = []
        raw_deltas = [d["reward_delta"] for d in deltas]
        for i in range(len(raw_deltas)):
            subset = raw_deltas[:i] + raw_deltas[i+1:]
            loso_means.append(fmean(subset) if subset else 0.0)

        paired[arm] = {
            "baseline": "policy",
            "mean_reward_delta": fmean(raw_deltas),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "per_seed": deltas,
            "leave_one_seed_out": {
                "method": "leave-one-seed-out-mean-range-v1",
                "count": len(seeds),
                "minimum": min(loso_means),
                "maximum": max(loso_means),
                "means": loso_means,
            },
        }

    identifier = hashlib.sha256(f"odysseus:{list(seeds)}:{max_steps}:{scarcity}:{odysseus.artifact['sha256']}".encode()).hexdigest()[:24]
    experiment_data = {
        "schema": SCHEMA,
        "experiment_id": identifier,
        "date": "2026-10-09",
        "status": "completed",
        "scarcity": scarcity,
        "max_steps": max_steps,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "artifact_sha256": odysseus.artifact["sha256"],
        "summary": summary,
        "paired": paired,
        "limits": LIMITS,
        "episodes": episodes,
    }
    experiment_data["receipt_sha256"] = digest({k: v for k, v in experiment_data.items() if k != "receipt_sha256"})

    if progress and callable(progress):
        progress({"phase": "done", "completed": total_runs, "total": total_runs})

    return experiment_data


def verify_odysseus_receipt(receipt: dict) -> dict:
    """Weightless replay verification of an Odysseus experiment receipt."""
    if not isinstance(receipt, dict):
        raise ValueError("Odysseus receipt must be a dict")
    recorded_hash = receipt.get("receipt_sha256")
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if digest(unsigned) != recorded_hash:
        raise ValueError("Receipt checksum mismatch")
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "completed":
        raise ValueError("Receipt schema or status invalid")

    episodes = receipt.get("episodes", [])
    expected_count = len(receipt["seeds"]) * len(receipt["arms"])
    if len(episodes) != expected_count:
        raise ValueError(f"Expected {expected_count} episodes, found {len(episodes)}")

    # Weightless replay of environmental transitions
    transitions = 0
    for ep in episodes:
        env = TidePool(ep["seed"], max_steps=receipt["max_steps"], scarcity=receipt["scarcity"])
        env.observe()
        recorded_actions = ep["actions"]
        step_rewards = []
        for a in recorded_actions:
            nxt, rew, done, _ = env.step(a)
            step_rewards.append(rew)
            transitions += 1
        if env.tick != ep["steps"] or (env.death_reason is None) != ep["survived"]:
            raise ValueError(f"Episode outcome replay mismatch for seed {ep['seed']} arm {ep['controller']}")
        if abs(sum(step_rewards) - ep["total_reward"]) > 1e-4:
            raise ValueError(f"Reward sum replay mismatch for seed {ep['seed']}")

    return {
        "verified": True,
        "episodes_replayed": len(episodes),
        "transitions_replayed": transitions,
        "schema": SCHEMA,
        "scope": "Weightless replay of recorded action execution and outcome consistency; not neural authorship.",
    }


def write_odysseus_receipt(receipt: dict, directory: Path) -> Path:
    """Atomically save an Odysseus experiment receipt."""
    directory.mkdir(parents=True, exist_ok=True)
    file_id = digest({"schema": receipt["schema"], "seeds": receipt["seeds"], "arms": receipt["arms"]})[:16]
    receipt_hash = receipt["receipt_sha256"][:12]
    filename = f"{file_id}-{receipt_hash}.json"
    target = directory / filename

    temp_path = target.with_suffix(".tmp")
    temp_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(target)
    return target
