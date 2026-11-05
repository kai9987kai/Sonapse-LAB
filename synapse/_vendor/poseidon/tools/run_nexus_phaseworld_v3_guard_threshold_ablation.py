"""Paired ablation of NEXUS phase-guard shelter resource thresholds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.nexus_omega.agent import NexusOmegaAgent
from tools.run_nexus_phaseworld_v3_drift_assay import _run_episode


HORIZON = 128
SHIFT_STEP = 64
ENVIRONMENT_SEED_START = 101401
ENVIRONMENT_SEED_COUNT = 64
AGENT_SEEDS = (57001, 57002)
BOOTSTRAP_REPLICATES = 20_000
PROFILES = {
    "previous": (0.75, 0.98),
    "reserve_aware": (0.80, 0.40),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_digest(episode: dict) -> str:
    canonical = json.dumps(episode, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_post_shift_reward": float(np.mean([row["post_shift_reward"] for row in rows])),
        "mean_terminal_health": float(np.mean([row["terminal_health"] for row in rows])),
        "mean_terminal_energy": float(np.mean([row["terminal_energy"] for row in rows])),
        "mean_terminal_hydration": float(np.mean([row["terminal_hydration"] for row in rows])),
        "minimum_health": float(min(row["minimum_health"] for row in rows)),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
        "post_shift_storm_forages": int(sum(row["post_shift_storm_forages"] for row in rows)),
        "boundary_latches": int(sum(row["boundary_latches"] for row in rows)),
        "guard_overrides": int(sum(row["guard_overrides"] for row in rows)),
    }


def _run_profile(
    profile: str,
    thresholds: tuple[float, float],
    environment_seeds: tuple[int, ...],
    agent_seeds: tuple[int, ...],
) -> list[dict]:
    NexusOmegaAgent.V2_SHELTER_MIN_ENERGY = thresholds[0]
    NexusOmegaAgent.V2_SHELTER_MIN_HYDRATION = thresholds[1]
    rows = []
    for seed in environment_seeds:
        for agent_seed in agent_seeds:
            episode = _run_episode(
                seed,
                "agent_aligned_prior",
                agent_seed,
                horizon=HORIZON,
                shift_step=SHIFT_STEP,
                initial_interval=10,
                initial_duration=6,
                shifted_interval=8,
                shifted_duration=7,
            )
            transitions = episode["transitions"]
            post = [row for row in transitions if row["step"] > SHIFT_STEP]
            row = {
                "environment_seed": int(seed),
                "agent_seed": int(agent_seed),
                "survived": bool(episode["survived"]),
                "total_reward": float(episode["total_reward"]),
                "post_shift_reward": float(episode["post_shift_reward"]),
                "terminal_health": float(episode["terminal_health"]),
                "terminal_energy": float(episode["terminal_energy"]),
                "terminal_hydration": float(episode["terminal_hydration"]),
                "minimum_health": float(min(t["health"] for t in transitions)),
                "minimum_energy": float(min(t["energy"] for t in transitions)),
                "minimum_hydration": float(min(t["hydration"] for t in transitions)),
                "post_shift_storm_forages": int(sum(t["is_storm"] and t["action"] == 0 for t in post)),
                "boundary_latches": int(sum(t["phase_boundary_latch_active"] for t in transitions)),
                "guard_overrides": int(sum(t["hazard_guard_override"] for t in transitions)),
                "trajectory_sha256": _episode_digest(episode),
            }
            rows.append(row)
    return rows


def _paired_delta(
    previous: list[dict],
    reserve_aware: list[dict],
    agent_seeds: tuple[int, ...],
    key: str,
    seed: int,
) -> dict:
    old = {(row["environment_seed"], row["agent_seed"]): row[key] for row in previous}
    new = {(row["environment_seed"], row["agent_seed"]): row[key] for row in reserve_aware}
    seeds = sorted({row["environment_seed"] for row in previous})
    deltas = np.asarray([
        np.mean([new[(env_seed, agent_seed)] - old[(env_seed, agent_seed)] for agent_seed in agent_seeds])
        for env_seed in seeds
    ], dtype=np.float64)
    rng = np.random.RandomState(seed)
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        bootstrap[index] = float(np.mean(deltas[rng.randint(0, len(deltas), size=len(deltas))]))
    return {
        "mean_delta": float(np.mean(deltas)),
        "seed_bootstrap_95pct_ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        "bootstrap_unit": "environment seed; paired across both agent initializations",
    }


def run(
    environment_seeds: tuple[int, ...],
    agent_seeds: tuple[int, ...] = AGENT_SEEDS,
) -> dict:
    if len(environment_seeds) < 2:
        raise ValueError("at least two environment seeds are required")
    if not agent_seeds:
        raise ValueError("at least one agent initialization is required")

    original_thresholds = (
        NexusOmegaAgent.V2_SHELTER_MIN_ENERGY,
        NexusOmegaAgent.V2_SHELTER_MIN_HYDRATION,
    )
    results = {}
    try:
        for profile, thresholds in PROFILES.items():
            rows = _run_profile(profile, thresholds, environment_seeds, agent_seeds)
            replay_count = 0
            NexusOmegaAgent.V2_SHELTER_MIN_ENERGY = thresholds[0]
            NexusOmegaAgent.V2_SHELTER_MIN_HYDRATION = thresholds[1]
            for row in rows:
                replay = _run_episode(
                    row["environment_seed"],
                    "agent_aligned_prior",
                    row["agent_seed"],
                    horizon=HORIZON,
                    shift_step=SHIFT_STEP,
                    initial_interval=10,
                    initial_duration=6,
                    shifted_interval=8,
                    shifted_duration=7,
                )
                if _episode_digest(replay) != row["trajectory_sha256"]:
                    raise RuntimeError(
                        f"exact replay failed for {profile}: seed={row['environment_seed']} "
                        f"agent_seed={row['agent_seed']}"
                    )
                replay_count += 1
            results[profile] = {
                "shelter_thresholds": {
                    "minimum_energy": float(thresholds[0]),
                    "minimum_hydration": float(thresholds[1]),
                },
                "summary": _summarize(rows),
                "replay_checks": replay_count,
                "episodes": rows,
            }
    finally:
        NexusOmegaAgent.V2_SHELTER_MIN_ENERGY = original_thresholds[0]
        NexusOmegaAgent.V2_SHELTER_MIN_HYDRATION = original_thresholds[1]

    source_paths = {
        "assay": Path(__file__).resolve(),
        "episode_runner": ROOT / "tools/run_nexus_phaseworld_v3_drift_assay.py",
        "agent": ROOT / "poseidon/nexus_omega/agent.py",
        "benchmark_v1": ROOT / "poseidon/nexus_omega/benchmark.py",
        "benchmark_v2": ROOT / "poseidon/nexus_omega/benchmark_v2.py",
        "benchmark_v3": ROOT / "poseidon/nexus_omega/benchmark_v3.py",
        "causal_efe": ROOT / "poseidon/nexus_omega/causal_efe.py",
        "causal_world_model": ROOT / "poseidon/nexus_omega/causal_world_model.py",
        "causal_world_model_v2": ROOT / "poseidon/nexus_omega/causal_world_model_v2.py",
        "contracts": ROOT / "poseidon/nexus_omega/contracts.py",
        "memory": ROOT / "poseidon/nexus_omega/memory.py",
    }
    previous = results["previous"]["episodes"]
    reserve_aware = results["reserve_aware"]["episodes"]
    return {
        "schema": "poseidon-nexus-phaseworld-v3-guard-threshold-ablation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment_seeds": list(environment_seeds),
            "agent_seeds": list(agent_seeds),
            "horizon": HORIZON,
            "shift_step": SHIFT_STEP,
            "initial_schedule": {"interval": 10, "duration": 6},
            "shifted_schedule": {"interval": 8, "duration": 7},
            "comparison": "same hidden-schedule episodes and initializations; only shelter reserve thresholds differ",
            "replay_identity": "SHA-256 over complete episode outputs, including observations and all transitions",
            "promotion": False,
        },
        "results": {
            "by_profile": {
                name: {key: value for key, value in item.items() if key != "episodes"}
                for name, item in results.items()
            },
            "paired_reward_delta": _paired_delta(previous, reserve_aware, agent_seeds, "total_reward", 2_026_101),
            "paired_post_shift_reward_delta": _paired_delta(previous, reserve_aware, agent_seeds, "post_shift_reward", 2_026_102),
        },
        "episodes": {
            name: item["episodes"] for name, item in results.items()
        },
        "limits": [
            "Synthetic one-time joint interval and duration shift only.",
            "Two fixed agent initializations do not establish independent model-family replication.",
            "Thresholds were selected after a smaller pilot and are not calibrated safety bounds.",
            "The study does not establish real-world safety or promote the controller.",
        ],
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=ENVIRONMENT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=ENVIRONMENT_SEED_COUNT)
    parser.add_argument("--agent-seeds", type=int, nargs="+", default=AGENT_SEEDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/nexus_omega/phaseworld_v3_guard_threshold_ablation_20261014.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    receipt = run(
        tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        tuple(args.agent_seeds),
    )
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": receipt["results"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
