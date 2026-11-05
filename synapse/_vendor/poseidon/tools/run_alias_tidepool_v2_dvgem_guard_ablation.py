"""Test whether a vitality fallback can bound an opt-in DV-GEM policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.alias_world_v2 import ObservationAliasTidePoolV2
from poseidon.research.contracts import canonical_bytes
from poseidon.research.decision_value_memory import DecisionValueMemory, DecisionValueMemoryConfig
from tools.run_alias_tidepool_v2_dvgem_pilot import (
    BASE_SEED,
    CONDITIONS,
    MAX_STEPS,
    _run_episode,
)


SEED_START = BASE_SEED + 50000
SEED_COUNT = 16
SELECTED_SCALE = 8.0
GUARD_THRESHOLDS = (None, 0.20, 0.30, 0.40, 0.50, 0.60)
BOOTSTRAP_REPLICATES = 5000
PILOT_DIR = ROOT / "runs/research/decision_value_memory_v2_pilot_20261014"
CALIBRATION_DIR = ROOT / "runs/research/decision_value_memory_v2_scale_calibration_20261014"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label(threshold: float | None) -> str:
    return "unguarded_scale8" if threshold is None else f"guard_{threshold:.2f}"


def _episode_summary(seed: int, regime: int, condition: str, threshold: float | None, memory, core):
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("guard thresholds must lie in [0, 1]")
    row = _run_episode(
        seed,
        regime,
        condition,
        "intact",
        memory,
        [],
        core,
        influence_scale=SELECTED_SCALE,
        resource_guard_threshold=threshold,
    )
    trajectory = row.pop("trajectory")
    return {
        **row,
        "policy": _label(threshold),
        "resource_guard_threshold": threshold,
        "trajectory_sha256": hashlib.sha256(canonical_bytes(trajectory)).hexdigest(),
    }


def _reactive_episode(seed: int, regime: int, condition: str) -> dict[str, Any]:
    env = ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **CONDITIONS[condition],
    )
    obs = env.reset()
    reward_total = 0.0
    trajectory = []
    while not env.done:
        energy, hydration = float(obs[1]), float(obs[2])
        action = 1 if energy <= hydration else 2
        next_obs, reward, _, info = env.step(action)
        trajectory.append({
            "tick": env.tick - 1,
            "action": action,
            "reward": float(reward),
            "energy": float(info["energy"]),
            "hydration": float(info["hydration"]),
            "death": bool(info["death"]),
        })
        reward_total += float(reward)
        obs = next_obs
    terminal = trajectory[-1]
    return {
        "seed": seed,
        "regime": regime,
        "condition": condition,
        "policy": "reactive_balancer",
        "steps": len(trajectory),
        "survived": not terminal["death"],
        "death_reason": env.death_reason,
        "total_reward": reward_total,
        "minimum_energy": min(row["energy"] for row in trajectory),
        "minimum_hydration": min(row["hydration"] for row in trajectory),
        "memory_override_count": 0,
        "memory_override_executed_count": 0,
        "resource_guard_trigger_count": 0,
        "resource_guard_override_count": 0,
        "trajectory_sha256": hashlib.sha256(canonical_bytes(trajectory)).hexdigest(),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "episodes_with_memory_override": int(sum(row["memory_override_count"] > 0 for row in rows)),
        "mean_memory_overrides_executed": float(np.mean([row["memory_override_executed_count"] for row in rows])),
        "mean_guard_triggers": float(np.mean([row["resource_guard_trigger_count"] for row in rows])),
        "mean_guard_overrides": float(np.mean([row["resource_guard_override_count"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
    }


def _by_seed(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def run(output: Path, seed_count: int = SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving evidence: {output}")
    if seed_count < 4:
        raise ValueError("seed_count must be at least four")
    pilot_receipt = json.loads((PILOT_DIR / "benchmark_receipt.json").read_text(encoding="utf-8"))
    calibration_receipt = json.loads((CALIBRATION_DIR / "calibration_receipt.json").read_text(encoding="utf-8"))
    if calibration_receipt["results"]["selected_scale"] != SELECTED_SCALE:
        raise ValueError("selected scale changed; update the guarded protocol explicitly")
    model_path = PILOT_DIR / "memory_model.pt"
    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    config_values = dict(model_state["config"])
    config_values["horizons"] = tuple(config_values.get("horizons", (4, 8)))
    memory = DecisionValueMemory(DecisionValueMemoryConfig(**config_values))
    memory.load_state_dict(model_state)
    model_sha = _sha256(model_path)
    if model_sha != pilot_receipt["model_sha256"]:
        raise ValueError("model checkpoint does not match the completed training pilot")

    eval_seeds = list(range(SEED_START, SEED_START + seed_count))
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    output.mkdir(parents=True)
    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "tools/run_alias_tidepool_v2_dvgem_pilot.py",
        ROOT / "tools/calibrate_alias_tidepool_v2_dvgem_scale.py",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths}
    thresholds = list(GUARD_THRESHOLDS)
    protocol = {
        "schema": "poseidon-decision-memory-v2-resource-guard-ablation-v1",
        "seed_range": [SEED_START, SEED_START + seed_count - 1],
        "seed_block_role": "development-only resource-guard threshold selection; fresh confirmation required",
        "world_version": "alias-tidepool-v2-resource-maintenance",
        "conditions": CONDITIONS,
        "regimes": [0, 1],
        "max_steps": MAX_STEPS,
        "influence_scale": SELECTED_SCALE,
        "guard_thresholds": thresholds,
        "controls": ["unguarded_scale8", "reactive_balancer"],
        "selection_rule": "among thresholds with 100% survival in every condition, maximize unweighted mean reward across conditions",
        "paired_unit": "environment seed averaged over both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
        "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
        "pilot_receipt_sha256": pilot_receipt["receipt_sha256"],
        "model_sha256": model_sha,
        "promotion": False,
        "source_sha256": source_hashes,
        "core_checkpoint_path": str(core_path),
        "core_checkpoint_sha256": _sha256(core_path),
    }
    protocol["protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())

    def ensure_source() -> None:
        for relative, digest in source_hashes.items():
            if _sha256(ROOT / relative) != digest:
                raise RuntimeError(f"source changed during ablation; evidence retained: {relative}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        episodes = []
        for condition_index, condition in enumerate(CONDITIONS):
            for seed in eval_seeds:
                for regime in (0, 1):
                    episodes.append(_reactive_episode(seed, regime, condition))
                    for threshold in thresholds:
                        episodes.append(_episode_summary(seed, regime, condition, threshold, memory, core))
            print(json.dumps({"phase": "condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            grouped[(row["condition"], row["policy"])].append(row)
        policy_names = ["reactive_balancer"] + [_label(threshold) for threshold in thresholds]
        by_condition = {
            condition: {policy: _summary(grouped[(condition, policy)]) for policy in policy_names}
            for condition in CONDITIONS
        }
        paired = {}
        for condition_index, condition in enumerate(CONDITIONS):
            reactive_rewards = _by_seed(grouped[(condition, "reactive_balancer")], "total_reward")
            no_guard_rewards = _by_seed(grouped[(condition, "unguarded_scale8")], "total_reward")
            no_guard_survival = _by_seed(grouped[(condition, "unguarded_scale8")], "survived")
            paired[condition] = {}
            for threshold_index, threshold in enumerate(thresholds):
                policy = _label(threshold)
                rows = grouped[(condition, policy)]
                rewards = _by_seed(rows, "total_reward")
                survivals = _by_seed(rows, "survived")
                seeds = sorted(set(rewards) & set(reactive_rewards))
                delta_reactive = np.asarray([rewards[s] - reactive_rewards[s] for s in seeds])
                delta_no_guard = np.asarray([rewards[s] - no_guard_rewards[s] for s in seeds])
                delta_survival = np.asarray([survivals[s] - no_guard_survival[s] for s in seeds])
                paired[condition][policy] = {
                    "paired_environment_seeds": len(seeds),
                    "reward_delta_vs_reactive_mean": float(np.mean(delta_reactive)),
                    "reward_delta_vs_reactive_ci95": _paired_ci(delta_reactive, BASE_SEED + condition_index * 100 + threshold_index),
                    "reward_delta_vs_unguarded_mean": float(np.mean(delta_no_guard)),
                    "reward_delta_vs_unguarded_ci95": _paired_ci(delta_no_guard, BASE_SEED + condition_index * 100 + threshold_index + 20),
                    "survival_delta_vs_unguarded_mean": float(np.mean(delta_survival)),
                    "survival_delta_vs_unguarded_ci95": _paired_ci(delta_survival, BASE_SEED + condition_index * 100 + threshold_index + 40),
                }

        viable_thresholds = [
            threshold for threshold in thresholds
            if all(by_condition[condition][_label(threshold)]["survival_rate"] == 1.0 for condition in CONDITIONS)
        ]
        selected = None
        if viable_thresholds:
            selected = max(viable_thresholds, key=lambda threshold: float(np.mean([
                by_condition[condition][_label(threshold)]["mean_reward"] for condition in CONDITIONS
            ])))

        print(json.dumps({"phase": "exact_replay", "episodes": len(episodes)}), flush=True)
        replay_checks = 0
        for row in episodes:
            if row["policy"] == "reactive_balancer":
                replay = _reactive_episode(row["seed"], row["regime"], row["condition"])
            else:
                threshold = None if row["policy"] == "unguarded_scale8" else float(row["policy"].split("_")[1])
                replay = _episode_summary(row["seed"], row["regime"], row["condition"], threshold, memory, core)
            if replay != row:
                raise RuntimeError(f"replay mismatch: {row['policy']} {row['condition']} seed={row['seed']} regime={row['regime']}")
            replay_checks += 1
        ensure_source()

        receipt = {
            "schema": "poseidon-decision-memory-v2-resource-guard-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_sha,
            "results": {
                "by_condition": by_condition,
                "paired_comparisons": paired,
                "selected_threshold": selected,
                "selection_rule": protocol["selection_rule"],
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
                "promoted": False,
            },
            "episodes": episodes,
            "identity": {
                "source_sha256": source_hashes,
                "model_sha256": model_sha,
                "pilot_receipt_sha256": pilot_receipt["receipt_sha256"],
                "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
                "core_checkpoint_path": str(core_path),
                "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            },
            "limits": [
                "Threshold selection reuses the prior 16-seed development block and is not confirmatory.",
                "The fallback is an explicit heuristic on observed energy and hydration, not a learned safety guarantee.",
                "Fresh-seed paired confirmation is required before any promotion or general performance claim.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = output / "guard_ablation_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "output": str(output),
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_checks": replay_checks,
            "selected_threshold": selected,
            "by_condition": by_condition,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/research/decision_value_memory_v2_guard_ablation_20261014")
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed_count=args.seeds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
