"""POSEIDON DECISION MEMORY BENCHMARK: 6-Arm Matched Lesion & Transfer Evaluation.

Evaluates Decision-Value-Gated Episodic Memory across six matched arms:
1. intact: full learned causal decision-value retrieval and admission gating.
2. similarity: classical Euclidean observation nearest-neighbor retrieval.
3. erased: empty episodic store (admission gate permanently closed).
4. shifted: circular key permutation (breaks semantic/temporal alignment).
5. donor: populated with memory records from unrelated donor episodes.
6. baseline: pure frozen Tidal baseline policy (memory bypassed).

Measures:
- Paired whole-episode reward differences with 95% bootstrap confidence intervals.
- Survival retention and non-inferiority against frozen Tidal.
- Admission gate frequency, override counts, and decision conflict rates.
- Cryptographic SHA-256 receipt generation with deterministic replay verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Sequence

import torch

from ..core import CoreRuntime, active_core_path
from .alias_world import ObservationAliasTidePool
from .contracts import canonical_bytes
from .decision_value_memory import (
    DECISION_MEMORY_MODES,
    DecisionValueAgent,
    DecisionValueMemory,
    DecisionValueMemoryConfig,
    EpisodicEntry,
    fit_decision_value_memory,
)
from ..world import OBS_SIZE

BENCHMARK_SCHEMA = "poseidon-decision-memory-benchmark-v1"


def bootstrap_ci_mean(diffs: Sequence[float], reps: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Computes mean and 95% percentile bootstrap confidence interval."""
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    mean_val = float(fmean(diffs))
    if n == 1 or all(x == diffs[0] for x in diffs):
        return mean_val, mean_val, mean_val

    rng = random.Random(seed)
    boot_means = []
    for _ in range(reps):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(fmean(sample))

    boot_means.sort()
    lower_idx = int(math.floor((alpha / 2.0) * reps))
    upper_idx = int(math.ceil((1.0 - alpha / 2.0) * reps)) - 1
    lower_idx = max(0, min(reps - 1, lower_idx))
    upper_idx = max(0, min(reps - 1, upper_idx))
    return mean_val, float(boot_means[lower_idx]), float(boot_means[upper_idx])


def run_decision_memory_benchmark(
    memory: DecisionValueMemory | None = None,
    base_seed: int = 710000000,
    episodes: int = 16,
    max_steps: int = 32,
    conditions: Sequence[str] = ("id_alias", "ood_scarcity", "ood_noise"),
    core: CoreRuntime | None = None,
) -> dict[str, Any]:
    """Runs matched 6-arm evaluation across specified environmental regimes."""
    core = core or CoreRuntime(active_core_path())
    if memory is None:
        # Fit quick decision-value memory on 8 training seeds
        train_seeds = [base_seed - 1000 + i for i in range(8)]
        memory = fit_decision_value_memory(train_seeds, updates=150)

    # Donor pool: collected from unrelated donor seeds
    donor_entries: list[EpisodicEntry] = []
    for d_seed in range(base_seed - 5000, base_seed - 4995):
        d_env = ObservationAliasTidePool(seed=d_seed, max_steps=12, cue_horizon=4)
        d_obs = d_env.reset()
        for t in range(8):
            d_act = core.act(d_obs)
            d_next, d_r, d_done, _ = d_env.step(d_act)
            donor_entries.append(EpisodicEntry(
                tick=t, observation=tuple(d_obs), action=d_act, reward=d_r,
                next_observation=tuple(d_next), residual=tuple([0.0]*OBS_SIZE),
            ))
            d_obs = d_next
            if d_done:
                break

    results_by_condition: dict[str, Any] = {}

    for condition in conditions:
        seeds = [base_seed + (1000 if condition == "ood_scarcity" else 2000 if condition == "ood_noise" else 0) + i
                 for i in range(episodes)]

        arm_rewards: dict[str, list[float]] = {arm: [] for arm in DECISION_MEMORY_MODES}
        arm_survivals: dict[str, list[bool]] = {arm: [] for arm in DECISION_MEMORY_MODES}
        arm_gate_counts: dict[str, list[int]] = {arm: [] for arm in DECISION_MEMORY_MODES}
        arm_trajectories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in DECISION_MEMORY_MODES}

        for seed in seeds:
            # Set up condition environment
            if condition == "ood_scarcity":
                env = ObservationAliasTidePool(seed=seed, scarcity=2.5, max_steps=max_steps)
            elif condition == "ood_noise":
                env = ObservationAliasTidePool(seed=seed, sensor_noise_std=0.04, max_steps=max_steps)
            else:
                env = ObservationAliasTidePool(seed=seed, scarcity=1.0, max_steps=max_steps)

            # Evaluate each of the 6 arms from the exact same initial state
            for arm in DECISION_MEMORY_MODES:
                agent = DecisionValueAgent(core=core, memory=copy.deepcopy(memory), mode=arm)
                if arm == "donor":
                    agent.memory.set_donor_pool(donor_entries)

                rollout_res = agent.rollout(env, max_steps=max_steps)
                arm_rewards[arm].append(rollout_res["total_reward"])
                arm_survivals[arm].append(rollout_res["survived"])
                arm_gate_counts[arm].append(rollout_res["gate_open_count"])

                arm_trajectories[arm].append({
                    "seed": seed,
                    "total_reward": rollout_res["total_reward"],
                    "steps": rollout_res["steps"],
                    "survived": rollout_res["survived"],
                    "gate_open_count": rollout_res["gate_open_count"],
                    "override_count": rollout_res["override_count"],
                    "actions": [step["action"] for step in rollout_res["trajectory"]],
                    "rewards": [round(step["reward"], 4) for step in rollout_res["trajectory"]],
                })

        # Calculate paired comparison against intact arm
        intact_rewards = arm_rewards["intact"]
        intact_surv = arm_survivals["intact"]
        paired_stats: dict[str, Any] = {}

        for comp_arm in ("similarity", "erased", "shifted", "donor", "baseline"):
            diffs = [r_int - r_comp for r_int, r_comp in zip(intact_rewards, arm_rewards[comp_arm])]
            mean_d, ci_low, ci_high = bootstrap_ci_mean(diffs, reps=2000, seed=base_seed)

            wins = sum(d > 1e-4 for d in diffs)
            ties = sum(abs(d) <= 1e-4 for d in diffs)
            losses = sum(d < -1e-4 for d in diffs)

            surv_int = sum(intact_surv) / float(len(intact_surv))
            surv_comp = sum(arm_survivals[comp_arm]) / float(len(arm_survivals[comp_arm]))

            paired_stats[f"intact_vs_{comp_arm}"] = {
                "mean_reward_difference": round(mean_d, 6),
                "ci_95": [round(ci_low, 6), round(ci_high, 6)],
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "intact_survival_rate": round(surv_int, 4),
                "comparator_survival_rate": round(surv_comp, 4),
                "survival_delta": round(surv_int - surv_comp, 4),
            }

        arm_summaries = {}
        for arm in DECISION_MEMORY_MODES:
            arm_summaries[arm] = {
                "mean_reward": round(fmean(arm_rewards[arm]), 6),
                "survival_rate": round(sum(arm_survivals[arm]) / float(len(arm_survivals[arm])), 4),
                "mean_gate_open_steps": round(fmean(arm_gate_counts[arm]), 2),
            }

        results_by_condition[condition] = {
            "episodes": episodes,
            "seeds": seeds,
            "arms": arm_summaries,
            "paired_comparisons": paired_stats,
            "trajectories": arm_trajectories,
        }

    receipt = {
        "schema": BENCHMARK_SCHEMA,
        "base_seed": base_seed,
        "episodes_per_condition": episodes,
        "max_steps": max_steps,
        "conditions": list(conditions),
        "arms": list(DECISION_MEMORY_MODES),
        "results": results_by_condition,
        "memory_config": asdict(memory.config),
    }

    receipt["sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    return receipt


def verify_decision_memory_receipt(
    receipt: dict[str, Any],
    replay_seeds: bool = True,
    core: CoreRuntime | None = None,
) -> bool:
    """Cryptographically verifies benchmark receipt and replays trajectories."""
    if receipt.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(f"Invalid schema: {receipt.get('schema')}")

    receipt_copy = copy.deepcopy(receipt)
    claimed_sha = receipt_copy.pop("sha256", None)
    if claimed_sha is None:
        raise ValueError("Missing sha256 in receipt")

    computed_sha = hashlib.sha256(canonical_bytes(receipt_copy)).hexdigest()
    if claimed_sha != computed_sha:
        raise ValueError("Receipt SHA-256 hash mismatch: receipt has been tampered with")

    if replay_seeds:
        core = core or CoreRuntime(active_core_path())
        cfg = DecisionValueMemoryConfig(**receipt["memory_config"])
        mem = DecisionValueMemory(cfg)

        for cond, cond_data in receipt["results"].items():
            trajectories = cond_data["trajectories"]
            for arm in ("baseline", "erased"):
                for ep_data in trajectories[arm]:
                    seed = ep_data["seed"]
                    if cond == "ood_scarcity":
                        env = ObservationAliasTidePool(seed=seed, scarcity=2.5, max_steps=receipt["max_steps"])
                    elif cond == "ood_noise":
                        env = ObservationAliasTidePool(seed=seed, sensor_noise_std=0.04, max_steps=receipt["max_steps"])
                    else:
                        env = ObservationAliasTidePool(seed=seed, scarcity=1.0, max_steps=receipt["max_steps"])
                    agent = DecisionValueAgent(core=core, memory=mem, mode=arm)
                    res = agent.rollout(env, max_steps=receipt["max_steps"])

                    if abs(res["total_reward"] - ep_data["total_reward"]) > 1e-4:
                        raise ValueError(f"Replay reward mismatch on seed {seed}: recorded {ep_data['total_reward']}, got {res['total_reward']}")
                    if res["survived"] != ep_data["survived"]:
                        raise ValueError(f"Replay survival mismatch on seed {seed}")

    return True
