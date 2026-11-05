"""Confirm guarded DV-GEM against matched synthetic controls on fresh seeds."""
from __future__ import annotations

import argparse
import copy
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
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes
from poseidon.research.decision_value_memory import (
    DecisionValueMemory,
    DecisionValueMemoryConfig,
    ResourceGuardedDecisionValueAgent,
)
from poseidon.world import OBS_SIZE


PILOT_DIR = ROOT / "runs/research/decision_value_memory_v2_pilot_20261014"
CALIBRATION_DIR = ROOT / "runs/research/decision_value_memory_v2_scale_calibration_20261014"
GUARD_DIR = ROOT / "runs/research/decision_value_memory_v2_guard_ablation_20261014"
REANALYSIS = ROOT / "runs/research/decision_value_memory_v2_guard_margin_reanalysis_20261014.json"
MEMORY_VALUE_RECEIPT = ROOT / "runs/research/alias_tidepool_v2_memory_value_assay_20261014.json"
TIE_SELECTION_RECEIPT = ROOT / "runs/research/alias_tidepool_v2_tie_margin_selection_20261014.json"
BASE_SEED = 743060000
SEED_COUNT = 32
MAX_STEPS = 24
INFLUENCE_SCALE = 8.0
RESOURCE_GUARD_THRESHOLD = 0.30
CUE_TIE_MARGIN = 0.40
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
    "dvgem_intact_guarded",
    "dvgem_erased_guarded",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if not claimed or hashlib.sha256(canonical_bytes(body)).hexdigest() != claimed:
        raise ValueError(f"receipt failed self-hash: {path}")
    return receipt


def _environment(seed: int, regime: int, condition: str) -> ObservationAliasTidePoolV2:
    return ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **CONDITIONS[condition],
    )


def _episode(
    seed: int,
    regime: int,
    condition: str,
    arm: str,
    fitted_memory: DecisionValueMemory,
    core: CoreRuntime,
) -> dict[str, Any]:
    env = _environment(seed, regime, condition)
    obs = env.reset()
    cue_regime = 0 if obs[13] > 0.5 else 1
    agent = None
    if arm == "dvgem_intact_guarded":
        agent = ResourceGuardedDecisionValueAgent(
            core=core,
            memory=copy.deepcopy(fitted_memory),
            mode="intact",
            influence_scale=INFLUENCE_SCALE,
            resource_guard_threshold=RESOURCE_GUARD_THRESHOLD,
        )
    elif arm == "dvgem_erased_guarded":
        agent = ResourceGuardedDecisionValueAgent(
            core=core,
            memory=copy.deepcopy(fitted_memory),
            mode="erased",
            influence_scale=INFLUENCE_SCALE,
            resource_guard_threshold=RESOURCE_GUARD_THRESHOLD,
        )
    if agent is not None:
        agent.reset(mode=agent.mode)

    total_reward = 0.0
    gate_count = 0
    memory_candidates = 0
    memory_executed = 0
    guard_triggers = 0
    guard_overrides = 0
    trajectory = []
    while not env.done:
        tick = env.tick
        if arm == "frozen_tidal":
            action = int(core.act(obs))
            base_action = action
            decision = None
        elif arm == "reactive_balancer":
            action = 1 if float(obs[1]) <= float(obs[2]) else 2
            base_action = action
            decision = None
        elif arm == "cue_memory_balancer":
            resource_action = 1 if float(obs[1]) <= float(obs[2]) else 2
            if tick < 4 or abs(float(obs[1]) - float(obs[2])) > CUE_TIE_MARGIN:
                action = resource_action
            else:
                action = 1 if cue_regime == 0 else 2
            base_action = resource_action
            decision = None
        else:
            assert agent is not None
            decision = agent.act(obs)
            action = int(decision["action"])
            base_action = int(decision["base_action"])

        next_obs, reward, done, info = env.step(action)
        if agent is not None:
            agent.observe(obs, action, reward, next_obs)
            gate_count += int(bool(decision["admission_gate_open"]))
            memory_candidates += int(bool(decision.get("memory_override", False)))
            memory_executed += int(bool(decision.get("memory_override_executed", False)))
            guard_triggers += int(bool(decision.get("resource_guard_triggered", False)))
            guard_overrides += int(bool(decision.get("resource_guard_override", False)))

        trajectory.append({
            "tick": tick,
            "observation": [float(x) for x in obs],
            "base_action": base_action,
            "action": action,
            "reward": float(reward),
            "next_observation": [float(x) for x in next_obs],
            "info": info,
            "memory": None if decision is None else {
                "gate_open": bool(decision["admission_gate_open"]),
                "memory_override": bool(decision.get("memory_override", False)),
                "memory_override_executed": bool(decision.get("memory_override_executed", False)),
                "guard_triggered": bool(decision.get("resource_guard_triggered", False)),
                "guard_override": bool(decision.get("resource_guard_override", False)),
                "margin": float(decision["margin"]),
                "decision": decision.get("decision_memory"),
            },
        })
        total_reward += float(reward)
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
        "minimum_energy": float(min(float(row["info"]["energy"]) for row in trajectory)),
        "minimum_hydration": float(min(float(row["info"]["hydration"]) for row in trajectory)),
        "gate_open_count": gate_count,
        "memory_override_candidate_count": memory_candidates,
        "memory_override_executed_count": memory_executed,
        "resource_guard_trigger_count": guard_triggers,
        "resource_guard_override_count": guard_overrides,
        "trajectory": trajectory,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "episodes_with_memory_override_candidate": int(sum(row["memory_override_candidate_count"] > 0 for row in rows)),
        "episodes_with_memory_override_executed": int(sum(row["memory_override_executed_count"] > 0 for row in rows)),
        "mean_gate_open_steps": float(np.mean([row["gate_open_count"] for row in rows])),
        "mean_memory_override_candidates": float(np.mean([row["memory_override_candidate_count"] for row in rows])),
        "mean_memory_overrides_executed": float(np.mean([row["memory_override_executed_count"] for row in rows])),
        "mean_guard_triggers": float(np.mean([row["resource_guard_trigger_count"] for row in rows])),
        "mean_guard_overrides": float(np.mean([row["resource_guard_override_count"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
    }


def _seed_mean(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def _paired(rows_by_arm: dict[str, list[dict[str, Any]]], condition_index: int) -> dict[str, Any]:
    candidate = rows_by_arm["dvgem_intact_guarded"]
    candidate_rewards = _seed_mean(candidate, "total_reward")
    candidate_survival = _seed_mean(candidate, "survived")
    comparisons = {}
    for comparator_index, comparator in enumerate(("dvgem_erased_guarded", "reactive_balancer", "cue_memory_balancer", "frozen_tidal")):
        other_rewards = _seed_mean(rows_by_arm[comparator], "total_reward")
        other_survival = _seed_mean(rows_by_arm[comparator], "survived")
        seeds = sorted(set(candidate_rewards) & set(other_rewards))
        reward_delta = np.asarray([candidate_rewards[s] - other_rewards[s] for s in seeds])
        survival_delta = np.asarray([candidate_survival[s] - other_survival[s] for s in seeds])
        comparisons[f"intact_vs_{comparator}"] = {
            "paired_environment_seeds": len(seeds),
            "reward_delta_mean": float(np.mean(reward_delta)),
            "reward_delta_ci95_environment_seed_bootstrap": _paired_ci(reward_delta, BASE_SEED + condition_index * 10 + comparator_index),
            "reward_win_rate_by_seed": float(np.mean(reward_delta > 0.0)),
            "survival_delta_mean": float(np.mean(survival_delta)),
            "survival_delta_ci95_environment_seed_bootstrap": _paired_ci(survival_delta, BASE_SEED + condition_index * 10 + comparator_index + 100),
        }
    return comparisons


def run(output: Path, seed_count: int = SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving prior evidence: {output}")
    if seed_count < 8:
        raise ValueError("seed_count must be at least eight")
    pilot_receipt = _load_receipt(PILOT_DIR / "benchmark_receipt.json")
    calibration_receipt = _load_receipt(CALIBRATION_DIR / "calibration_receipt.json")
    guard_receipt = _load_receipt(GUARD_DIR / "guard_ablation_receipt.json")
    reanalysis = _load_receipt(REANALYSIS)
    memory_value_receipt = _load_receipt(MEMORY_VALUE_RECEIPT)
    tie_selection_receipt = _load_receipt(TIE_SELECTION_RECEIPT)
    if calibration_receipt["results"]["selected_scale"] != INFLUENCE_SCALE:
        raise ValueError("development scale changed; update confirmation protocol explicitly")
    if reanalysis["results"]["selected_threshold"] != RESOURCE_GUARD_THRESHOLD:
        raise ValueError("development guard threshold changed; update confirmation protocol explicitly")
    if tie_selection_receipt["results"]["selected_margin"] != CUE_TIE_MARGIN:
        raise ValueError("cue-memory development margin changed; update confirmation protocol explicitly")

    model_path = PILOT_DIR / "memory_model.pt"
    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    config_values = dict(model_state["config"])
    config_values["horizons"] = tuple(config_values.get("horizons", (4, 8)))
    fitted_memory = DecisionValueMemory(DecisionValueMemoryConfig(**config_values))
    fitted_memory.load_state_dict(model_state)
    model_sha = _sha256(model_path)
    if model_sha != pilot_receipt["model_sha256"]:
        raise ValueError("model checkpoint does not match the training pilot receipt")

    seeds = list(range(BASE_SEED, BASE_SEED + seed_count))
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    output.mkdir(parents=True)
    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "tools/run_alias_tidepool_v2_dvgem_pilot.py",
        ROOT / "tools/calibrate_alias_tidepool_v2_dvgem_scale.py",
        ROOT / "tools/run_alias_tidepool_v2_dvgem_guard_ablation.py",
        ROOT / "tools/reanalyze_alias_tidepool_v2_guard_margin.py",
        ROOT / "tools/run_alias_tidepool_v2_tie_margin_selection.py",
        ROOT / "tools/run_alias_tidepool_v2_memory_value_assay.py",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths}
    protocol = {
        "schema": "poseidon-decision-memory-v2-guard-confirmation-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "fresh_evaluation_seed_range": [BASE_SEED, BASE_SEED + seed_count - 1],
        "development_seed_ranges_excluded": {
            "training": [743010000, 743010011],
            "pilot_scale_selection": [743050000, 743050015],
            "guard_threshold_selection": [743050000, 743050015],
        },
        "conditions": CONDITIONS,
        "regimes": [0, 1],
        "max_steps": MAX_STEPS,
        "influence_scale": INFLUENCE_SCALE,
        "resource_guard_threshold": RESOURCE_GUARD_THRESHOLD,
        "cue_memory_tie_margin": CUE_TIE_MARGIN,
        "arms": list(ARMS),
        "paired_unit": "environment seed averaged over both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
        "selection_basis": {
            "cue_memory_value_receipt_sha256": memory_value_receipt["receipt_sha256"],
            "cue_tie_margin_selection_receipt_sha256": tie_selection_receipt["receipt_sha256"],
            "scale_calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
            "guard_ablation_receipt_sha256": guard_receipt["receipt_sha256"],
            "resource_margin_reanalysis_receipt_sha256": reanalysis["receipt_sha256"],
        },
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
                raise RuntimeError(f"source changed during confirmation; evidence retained: {relative}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        episodes = []
        for condition_index, condition in enumerate(CONDITIONS):
            for seed in seeds:
                for regime in (0, 1):
                    for arm in ARMS:
                        episodes.append(_episode(seed, regime, condition, arm, fitted_memory, core))
            print(json.dumps({"phase": "condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            grouped[(row["condition"], row["arm"])].append(row)
        by_condition = {
            condition: {arm: _summary(grouped[(condition, arm)]) for arm in ARMS}
            for condition in CONDITIONS
        }
        paired = {
            condition: _paired({arm: grouped[(condition, arm)] for arm in ARMS}, condition_index)
            for condition_index, condition in enumerate(CONDITIONS)
        }
        print(json.dumps({"phase": "exact_replay", "episodes": len(episodes)}), flush=True)
        replay_checks = 0
        for row in episodes:
            replay = _episode(row["seed"], row["regime"], row["condition"], row["arm"], fitted_memory, core)
            if replay != row:
                raise RuntimeError(f"replay mismatch: {row['condition']} seed={row['seed']} regime={row['regime']} arm={row['arm']}")
            replay_checks += 1
        ensure_source()

        candidate = {
            condition: by_condition[condition]["dvgem_intact_guarded"]
            for condition in CONDITIONS
        }
        gates = {
            "intact_guarded_survival_all_conditions": all(candidate[c]["survival_rate"] == 1.0 for c in CONDITIONS),
            "intact_guarded_minimum_resource_at_least_0_15_all_conditions": all(
                candidate[c]["minimum_energy"] >= 0.15 and candidate[c]["minimum_hydration"] >= 0.15
                for c in CONDITIONS
            ),
            "intact_has_executed_memory_overrides": any(
                candidate[c]["episodes_with_memory_override_executed"] > 0 for c in CONDITIONS
            ),
            "intact_reward_ci_lower_positive_vs_erased_all_conditions": all(
                paired[c]["intact_vs_dvgem_erased_guarded"]["reward_delta_ci95_environment_seed_bootstrap"][0] > 0.0
                for c in CONDITIONS
            ),
            "promoted": False,
        }
        receipt = {
            "schema": "poseidon-decision-memory-v2-guard-confirmation-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_sha,
            "results": {
                "by_condition": by_condition,
                "paired_comparisons": paired,
                "gates": gates,
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
            },
            "episodes": episodes,
            "identity": {
                "source_sha256": source_hashes,
                "model_sha256": model_sha,
                "pilot_receipt_sha256": pilot_receipt["receipt_sha256"],
                "calibration_receipt_sha256": calibration_receipt["receipt_sha256"],
                "guard_receipt_sha256": guard_receipt["receipt_sha256"],
                "margin_reanalysis_receipt_sha256": reanalysis["receipt_sha256"],
                "cue_memory_value_receipt_sha256": memory_value_receipt["receipt_sha256"],
                "cue_tie_margin_selection_receipt_sha256": tie_selection_receipt["receipt_sha256"],
                "core_checkpoint_path": str(core_path),
                "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            },
            "limits": [
                "Fresh local synthetic confirmation only; it is not independent replication across environments.",
                "The influence scale and resource threshold were selected on a separate development block.",
                "The low-resource fallback is a heuristic and does not establish safety outside this simulator.",
                "Passing these gates does not promote the controller by default.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = output / "confirmation_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_checks": replay_checks,
            "gates": gates,
            "by_condition": by_condition,
            "paired_comparisons": paired,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/research/decision_value_memory_v2_guard_confirmation_20261014")
    parser.add_argument("--seeds", type=int, default=SEED_COUNT)
    args = parser.parse_args()
    print(json.dumps(run(args.output, seed_count=args.seeds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
