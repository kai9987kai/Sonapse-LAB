"""Fit DV-GEM on the viable alias world and measure actual policy overrides."""
from __future__ import annotations

import argparse
import copy
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
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes
from poseidon.research.decision_value_memory import (
    DECISION_MEMORY_MODES,
    DecisionValueAgent,
    DecisionValueMemory,
    DecisionValueMemoryConfig,
    EpisodicEntry,
    ResourceGuardedDecisionValueAgent,
    fit_decision_value_memory,
)
from poseidon.world import OBS_SIZE


BASE_SEED = 743000000
TRAIN_SEED_COUNT = 12
EVAL_SEED_COUNT = 16
UPDATES = 300
MAX_STEPS = 24
BOOTSTRAP_REPLICATES = 5000
CONDITIONS = {
    "id": {"scarcity": 1.0, "sensor_noise_std": 0.0},
    "scarcity": {"scarcity": 2.5, "sensor_noise_std": 0.0},
    "noise": {"scarcity": 1.0, "sensor_noise_std": 0.04},
    "scarcity_noise": {"scarcity": 2.5, "sensor_noise_std": 0.04},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_env(seed: int, regime: int, condition: str) -> ObservationAliasTidePoolV2:
    return ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **CONDITIONS[condition],
    )


def _collect_donor_records(core: CoreRuntime, seed_start: int) -> list[EpisodicEntry]:
    records = []
    for offset, regime in enumerate((0, 1)):
        env = ObservationAliasTidePoolV2(
            seed=seed_start + offset,
            max_steps=12,
            forced_regime=regime,
        )
        obs = env.reset()
        while not env.done:
            action = core.act(obs)
            next_obs, reward, _, _ = env.step(action)
            records.append(EpisodicEntry(
                tick=env.tick,
                observation=tuple(obs),
                action=action,
                reward=reward,
                next_observation=tuple(next_obs),
                residual=tuple([0.0] * OBS_SIZE),
                tag="donor-v2",
            ))
            obs = next_obs
    return records


def _run_episode(
    seed: int,
    regime: int,
    condition: str,
    arm: str,
    fitted_memory: DecisionValueMemory,
    donor_records: list[EpisodicEntry],
    core: CoreRuntime,
    influence_scale: float = 1.0,
    resource_guard_threshold: float | None = None,
) -> dict[str, Any]:
    env = _make_env(seed, regime, condition)
    memory = copy.deepcopy(fitted_memory)
    if resource_guard_threshold is None:
        agent = DecisionValueAgent(core=core, memory=memory, mode=arm, influence_scale=influence_scale)
    else:
        agent = ResourceGuardedDecisionValueAgent(
            core=core,
            memory=memory,
            mode=arm,
            influence_scale=influence_scale,
            resource_guard_threshold=resource_guard_threshold,
        )
    if arm == "donor":
        agent.memory.set_donor_pool(donor_records)
    agent.reset(mode=arm)
    obs = env.reset()
    total_reward = 0.0
    gates = 0
    overrides = 0
    trajectory = []

    while not env.done:
        decision = agent.act(obs)
        action = int(decision["action"])
        gate_open = bool(decision["admission_gate_open"])
        override = bool(decision["override"])
        next_obs, reward, done, info = env.step(action)
        agent.observe(obs, action, reward, next_obs)

        memory_decision = decision.get("decision_memory")
        trajectory.append({
            "tick": env.tick - 1,
            "observation": [float(x) for x in obs],
            "base_action": int(decision["base_action"]),
            "action": action,
            "override": override,
            "memory_override": bool(decision.get("memory_override", False)),
            "memory_override_executed": bool(decision.get("memory_override_executed", decision.get("memory_override", False))),
            "resource_guard_triggered": bool(decision.get("resource_guard_triggered", False)),
            "resource_guard_override": bool(decision.get("resource_guard_override", False)),
            "margin": float(decision["margin"]),
            "gate_open": gate_open,
            "memory": memory_decision,
            "reward": float(reward),
            "next_observation": [float(x) for x in next_obs],
            "info": info,
        })
        total_reward += float(reward)
        gates += int(gate_open)
        overrides += int(override)
        obs = next_obs
        if done:
            break

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
        "minimum_energy": float(min(float(row["info"]["energy"]) for row in trajectory)),
        "minimum_hydration": float(min(float(row["info"]["hydration"]) for row in trajectory)),
        "gate_open_count": gates,
        "override_count": overrides,
        "memory_override_count": int(sum(row["memory_override"] for row in trajectory)),
        "memory_override_executed_count": int(sum(row["memory_override_executed"] for row in trajectory)),
        "resource_guard_trigger_count": int(sum(row["resource_guard_triggered"] for row in trajectory)),
        "resource_guard_override_count": int(sum(row["resource_guard_override"] for row in trajectory)),
        "trajectory": trajectory,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_gate_open_steps": float(np.mean([row["gate_open_count"] for row in rows])),
        "mean_overrides": float(np.mean([row["override_count"] for row in rows])),
        "episodes_with_override": int(sum(row["override_count"] > 0 for row in rows)),
        "episodes_with_memory_override": int(sum(row["memory_override_count"] > 0 for row in rows)),
        "mean_resource_guard_overrides": float(np.mean([row["resource_guard_override_count"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
    }


def _seed_means(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def _paired_comparisons(by_arm: dict[str, list[dict[str, Any]]], condition_index: int) -> dict[str, Any]:
    comparisons = {}
    memory_rows = by_arm["intact"]
    memory_reward = _seed_means(memory_rows, "total_reward")
    memory_survival = _seed_means(memory_rows, "survived")
    for comparator_index, comparator in enumerate(("similarity", "erased", "shifted", "donor", "baseline")):
        other_reward = _seed_means(by_arm[comparator], "total_reward")
        other_survival = _seed_means(by_arm[comparator], "survived")
        seeds = sorted(set(memory_reward) & set(other_reward))
        reward_delta = np.asarray([memory_reward[s] - other_reward[s] for s in seeds], dtype=float)
        survival_delta = np.asarray([memory_survival[s] - other_survival[s] for s in seeds], dtype=float)
        comparisons[f"intact_vs_{comparator}"] = {
            "paired_environment_seeds": len(seeds),
            "reward_delta_mean": float(np.mean(reward_delta)),
            "reward_delta_ci95_environment_seed_bootstrap": _paired_ci(reward_delta, BASE_SEED + condition_index * 10 + comparator_index),
            "reward_win_rate_by_seed": float(np.mean(reward_delta > 0.0)),
            "survival_delta_mean": float(np.mean(survival_delta)),
            "survival_delta_ci95_environment_seed_bootstrap": _paired_ci(survival_delta, BASE_SEED + condition_index * 10 + comparator_index + 100),
        }
    return comparisons


def run(output: Path, train_seed_count: int = TRAIN_SEED_COUNT, eval_seed_count: int = EVAL_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving evidence: {output}")
    if train_seed_count < 4 or train_seed_count % 2:
        raise ValueError("train_seed_count must be an even integer of at least four")
    if eval_seed_count < 4:
        raise ValueError("eval_seed_count must be at least four")

    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    train_seeds = [BASE_SEED + 10000 + index for index in range(train_seed_count)]
    train_regime_by_seed = {seed: index % 2 for index, seed in enumerate(train_seeds)}
    evaluation_seeds = [BASE_SEED + 50000 + index for index in range(eval_seed_count)]
    donor_seed_start = BASE_SEED + 30000
    cfg = DecisionValueMemoryConfig(
        capacity=128,
        top_k=4,
        admission_threshold=0.04,
        shrinkage_alpha=0.50,
        hidden_dim=48,
        belief_dim=24,
    )
    output.mkdir(parents=True)

    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve()]
    source_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths}
    protocol = {
        "schema": "poseidon-decision-memory-v2-pilot-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "base_seed": BASE_SEED,
        "train_seeds": train_seeds,
        "training_regime_by_seed": {str(seed): train_regime_by_seed[seed] for seed in train_seeds},
        "eval_seeds": evaluation_seeds,
        "donor_seeds": [donor_seed_start, donor_seed_start + 1],
        "eval_seed_count": eval_seed_count,
        "updates": UPDATES,
        "max_steps": MAX_STEPS,
        "conditions": CONDITIONS,
        "arms": list(DECISION_MEMORY_MODES),
        "memory_config": vars(cfg),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "resampling_unit": "environment seed averaged across both regimes"},
        "torch_seed": BASE_SEED + 1,
        "torch_threads": 2,
        "promotion": False,
        "limits": [
            "Synthetic research pilot only; the memory controller remains inactive by default.",
            "Twelve balanced training episodes and sixteen paired evaluation seeds are not a powered final study.",
            "A gate opening is not counted as a benefit unless it changes actions and improves paired outcomes.",
        ],
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
                raise RuntimeError(f"source changed during pilot; evidence left in place: {relative}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(BASE_SEED + 1)
    started = time.monotonic()
    try:
        ensure_source()
        print(json.dumps({"phase": "fit", "train_seeds": train_seeds}), flush=True)

        def environment_factory(seed: int, max_steps: int, cue_horizon: int):
            return ObservationAliasTidePoolV2(
                seed=seed,
                max_steps=max_steps,
                cue_horizon=cue_horizon,
                forced_regime=train_regime_by_seed[seed],
            )

        fitted_memory = fit_decision_value_memory(
            train_seeds=train_seeds,
            config=cfg,
            updates=UPDATES,
            environment_factory=environment_factory,
        )
        model_path = output / "memory_model.pt"
        torch.save(fitted_memory.state_dict(), model_path)
        model_sha = _sha256(model_path)
        donor_records = _collect_donor_records(core, donor_seed_start)
        print(json.dumps({"phase": "evaluate", "eval_seeds": evaluation_seeds}), flush=True)

        episodes = []
        for condition_index, condition in enumerate(CONDITIONS):
            for seed in evaluation_seeds:
                for regime in (0, 1):
                    for arm in DECISION_MEMORY_MODES:
                        episodes.append(_run_episode(seed, regime, condition, arm, fitted_memory, donor_records, core))
            print(json.dumps({"phase": "condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in episodes:
            grouped.setdefault((row["condition"], row["arm"]), []).append(row)
        summaries = {
            condition: {arm: _summary(grouped[(condition, arm)]) for arm in DECISION_MEMORY_MODES}
            for condition in CONDITIONS
        }
        paired = {
            condition: _paired_comparisons(
                {arm: grouped[(condition, arm)] for arm in DECISION_MEMORY_MODES}, condition_index
            )
            for condition_index, condition in enumerate(CONDITIONS)
        }

        print(json.dumps({"phase": "exact_replay", "episodes": len(episodes)}), flush=True)
        replay_checks = 0
        for row in episodes:
            replay = _run_episode(row["seed"], row["regime"], row["condition"], row["arm"], fitted_memory, donor_records, core)
            if replay != row:
                raise RuntimeError(
                    f"exact replay failed: {row['condition']} seed {row['seed']} regime {row['regime']} arm {row['arm']}"
                )
            replay_checks += 1

        ensure_source()
        memory_beats_baseline = all(
            paired[condition]["intact_vs_baseline"]["reward_delta_ci95_environment_seed_bootstrap"][0] > 0.0
            for condition in CONDITIONS
        )
        any_override = any(summaries[condition]["intact"]["episodes_with_override"] > 0 for condition in CONDITIONS)
        receipt = {
            "schema": "poseidon-decision-memory-v2-benchmark-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_sha,
            "results": {
                "by_condition": summaries,
                "paired_comparisons": paired,
                "gates": {
                    "intact_arm_has_actual_overrides": any_override,
                    "intact_reward_ci_lower_positive_vs_baseline_all_conditions": memory_beats_baseline,
                    "promoted": False,
                },
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
            },
            "episodes": episodes,
            "identity": {
                "source_sha256": source_hashes,
                "core_checkpoint_path": str(core_path),
                "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            },
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = output / "benchmark_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "output": str(output),
            "receipt_sha256": receipt["receipt_sha256"],
            "model_sha256": model_sha,
            "replay_checks": replay_checks,
            "gates": receipt["results"]["gates"],
            "by_condition": summaries,
            "paired_comparisons": paired,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/research/decision_value_memory_v2_pilot_20261014",
    )
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEED_COUNT)
    parser.add_argument("--eval-seeds", type=int, default=EVAL_SEED_COUNT)
    args = parser.parse_args()
    result = run(args.output, train_seed_count=args.train_seeds, eval_seed_count=args.eval_seeds)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
