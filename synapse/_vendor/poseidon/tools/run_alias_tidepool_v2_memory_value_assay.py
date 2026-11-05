"""Measure whether retaining the early alias cue improves later decisions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.alias_world_v2 import ObservationAliasTidePoolV2
from poseidon.research.contracts import (
    ALIAS_WORLD_V2_VERSION,
    alias_tidepool_v2_contract,
    canonical_bytes,
)


DEV_SEED_START = 742000000
EVAL_SEED_START = 742001000
SEED_COUNT = 64
MAX_STEPS = 24
TIE_MARGIN = 0.40
BOOTSTRAP_REPLICATES = 5000
CONDITIONS = {
    "id": {"scarcity": 1.0, "sensor_noise_std": 0.0},
    "scarcity": {"scarcity": 2.5, "sensor_noise_std": 0.0},
    "noise": {"scarcity": 1.0, "sensor_noise_std": 0.04},
    "scarcity_noise": {"scarcity": 2.5, "sensor_noise_std": 0.04},
}
ARMS = (
    "frozen_tidal",
    "reactive_balancer",
    "cue_memory_balancer",
    "cue_erased_balancer",
    "cue_shuffled_balancer",
)
COMPARISONS = (
    "reactive_balancer",
    "cue_erased_balancer",
    "cue_shuffled_balancer",
    "frozen_tidal",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_for(arm: str, obs: list[float], tick: int, cue_regime: int | None, core: CoreRuntime) -> tuple[int, bool]:
    if arm == "frozen_tidal":
        return int(core.act(obs)), False

    energy, hydration = float(obs[1]), float(obs[2])
    resource_action = 1 if energy <= hydration else 2
    if arm == "reactive_balancer" or tick < 4:
        return resource_action, False

    if abs(energy - hydration) > TIE_MARGIN:
        return resource_action, False

    if arm == "cue_memory_balancer":
        return (1 if cue_regime == 0 else 2), True
    if arm == "cue_erased_balancer":
        return 1, True
    if arm == "cue_shuffled_balancer":
        return (2 if cue_regime == 0 else 1), True
    raise ValueError(f"unknown arm: {arm}")


def _episode(seed: int, regime: int, condition: str, arm: str, core: CoreRuntime) -> dict[str, Any]:
    env = ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **CONDITIONS[condition],
    )
    obs = env.reset()
    # Retain the early cue in a one-bit episodic state; controls erase or invert it.
    cue_regime = 0 if obs[13] > 0.5 else 1
    total_reward = 0.0
    cue_influence_steps = 0
    trajectory = []

    while not env.done:
        tick = env.tick
        action, cue_influenced = _action_for(arm, obs, tick, cue_regime, core)
        next_obs, reward, done, info = env.step(action)
        trajectory.append({
            "tick": tick,
            "observation": [float(value) for value in obs],
            "action": int(action),
            "cue_influenced": bool(cue_influenced),
            "reward": float(reward),
            "next_observation": [float(value) for value in next_obs],
            "info": info,
        })
        cue_influence_steps += int(cue_influenced)
        total_reward += float(reward)
        obs = next_obs

    terminal = trajectory[-1]["info"]
    return {
        "seed": int(seed),
        "regime": int(regime),
        "condition": condition,
        "arm": arm,
        "steps": len(trajectory),
        "survived": not bool(terminal.get("death", False)),
        "death_reason": terminal.get("death_reason"),
        "total_reward": float(total_reward),
        "terminal_energy": float(env.energy),
        "terminal_hydration": float(env.hydration),
        "minimum_energy": float(min(row["info"]["energy"] for row in trajectory)),
        "minimum_hydration": float(min(row["info"]["hydration"] for row in trajectory)),
        "cue_influence_steps": cue_influence_steps,
        "trajectory": trajectory,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
        "mean_cue_influence_steps": float(np.mean([row["cue_influence_steps"] for row in rows])),
    }


def _by_seed(rows: list[dict[str, Any]], field: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[row["seed"]].append(float(row[field]))
    return {seed: float(np.mean(seed_values)) for seed, seed_values in values.items()}


def _paired_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_comparison(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    comparator: str,
    condition_index: int,
    comparator_index: int,
) -> dict[str, Any]:
    memory_rows = rows_by_arm["cue_memory_balancer"]
    comparator_rows = rows_by_arm[comparator]
    rewards_memory = _by_seed(memory_rows, "total_reward")
    rewards_other = _by_seed(comparator_rows, "total_reward")
    survival_memory = _by_seed(memory_rows, "survived")
    survival_other = _by_seed(comparator_rows, "survived")
    seeds = sorted(set(rewards_memory) & set(rewards_other))
    reward_delta = np.asarray([rewards_memory[s] - rewards_other[s] for s in seeds], dtype=float)
    survival_delta = np.asarray([survival_memory[s] - survival_other[s] for s in seeds], dtype=float)
    bootstrap_seed = 742100000 + condition_index * 100 + comparator_index
    reward_ci = _paired_interval(reward_delta, bootstrap_seed)
    survival_ci = _paired_interval(survival_delta, bootstrap_seed + 1)
    return {
        "paired_environment_seeds": len(seeds),
        "reward_delta_mean": float(np.mean(reward_delta)),
        "reward_delta_ci95_environment_seed_bootstrap": [reward_ci[0], reward_ci[1]],
        "reward_win_rate_by_seed": float(np.mean(reward_delta > 0.0)),
        "survival_delta_mean": float(np.mean(survival_delta)),
        "survival_delta_ci95_environment_seed_bootstrap": [survival_ci[0], survival_ci[1]],
    }


def run(seed_start: int = EVAL_SEED_START, seed_count: int = SEED_COUNT) -> dict[str, Any]:
    if seed_start < 0 or seed_count < 2:
        raise ValueError("seed_start must be nonnegative and seed_count at least two")
    contract = alias_tidepool_v2_contract()
    contract.validate()
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)

    episodes = []
    for condition in CONDITIONS:
        for seed in range(seed_start, seed_start + seed_count):
            for regime in (0, 1):
                for arm in ARMS:
                    episodes.append(_episode(seed, regime, condition, arm, core))

    replay_checks = 0
    for row in episodes:
        replay = _episode(row["seed"], row["regime"], row["condition"], row["arm"], core)
        if replay != row:
            raise RuntimeError(
                f"exact replay failed: seed={row['seed']} regime={row['regime']} "
                f"condition={row['condition']} arm={row['arm']}"
            )
        replay_checks += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["condition"], row["arm"])].append(row)
    summaries = {
        condition: {arm: _summary(grouped[(condition, arm)]) for arm in ARMS}
        for condition in CONDITIONS
    }
    paired = {
        condition: {
            comparator: _paired_comparison(
                {arm: grouped[(condition, arm)] for arm in ARMS},
                comparator,
                condition_index,
                comparator_index,
            )
            for comparator_index, comparator in enumerate(COMPARISONS)
        }
        for condition_index, condition in enumerate(CONDITIONS)
    }
    all_memory_viable = all(
        summaries[condition]["cue_memory_balancer"]["survival_rate"] == 1.0
        for condition in CONDITIONS
    )
    memory_beats_reactive = all(
        paired[condition]["reactive_balancer"]["reward_delta_ci95_environment_seed_bootstrap"][0] > 0.0
        for condition in CONDITIONS
    )
    memory_beats_erased = all(
        paired[condition]["cue_erased_balancer"]["reward_delta_ci95_environment_seed_bootstrap"][0] > 0.0
        for condition in CONDITIONS
    )

    source_paths = {
        "assay": Path(__file__).resolve(),
        "alias_world_v1": ROOT / "poseidon/research/alias_world.py",
        "alias_world_v2": ROOT / "poseidon/research/alias_world_v2.py",
        "contracts": ROOT / "poseidon/research/contracts.py",
        "world": ROOT / "poseidon/world.py",
        "core_runtime": ROOT / "poseidon/core.py",
    }
    return {
        "schema": "poseidon-alias-world-v2-memory-value-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "development_seed_range": [DEV_SEED_START, DEV_SEED_START + 23],
            "evaluation_seed_range": [seed_start, seed_start + seed_count - 1],
            "regimes": [0, 1],
            "conditions": CONDITIONS,
            "arms": list(ARMS),
            "comparisons": list(COMPARISONS),
            "max_steps": MAX_STEPS,
            "cue_horizon": 4,
            "cue_tie_margin": TIE_MARGIN,
            "tie_margin_selection": "0.40 selected on the separate 24-seed development block from a fixed 0.00-0.40 grid; evaluation seeds are disjoint",
            "shared_prefix": "all cue-based arms use the same reactive resource-balancing action until the cue closes at tick 4",
            "paired_unit": "environment seed, averaging both forced regimes within each seed",
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
            "promotion": False,
        },
        "results": {
            "by_condition": summaries,
            "paired_cue_memory_vs_comparator": paired,
            "gates": {
                "cue_memory_survives_all_conditions": all_memory_viable,
                "cue_memory_reward_ci_lower_positive_vs_reactive_all_conditions": memory_beats_reactive,
                "cue_memory_reward_ci_lower_positive_vs_erased_all_conditions": memory_beats_erased,
            },
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
        },
        "episodes": episodes,
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "core_checkpoint_path": str(core_path),
            "core_checkpoint_sha256": _sha256(core_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "limits": [
            "This tests a transparent one-bit cue-retention rule, not the trained DV-GEM scorer.",
            "The development sweep selected a heuristic threshold; this is not a learned or calibrated controller.",
            "The environment and cue are synthetic, fully controlled, and do not establish general memory or real-world performance.",
            "Sixty-four environment seeds are a confirmatory local block, not independent replication across environments.",
            "Passing gates do not promote a controller or establish a safety guarantee.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/research/alias_tidepool_v2_memory_value_assay_20261014.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    receipt = run(args.seed_start, args.seed_count)
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": receipt["results"],
        "source_sha256": receipt["identity"]["source_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
