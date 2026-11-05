"""Select a noise-aware reserve guard, then confirm structured DV-GEM on fresh seeds."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
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
from poseidon.research.decision_value_memory import DecisionValueMemory, DecisionValueMemoryConfig, ResourceGuardedDecisionValueAgent


TRAINING_DIR = ROOT / "runs/research/decision_value_memory_v2_structured_pilot_20261014"
DEVELOPMENT_SEED_START = 743070000
DEVELOPMENT_SEED_COUNT = 16
CONFIRMATION_SEED_START = 743080000
CONFIRMATION_SEED_COUNT = 64
MAX_STEPS = 24
INFLUENCE_SCALE = 8.0
RESOURCE_THRESHOLD = 0.30
SENSOR_NOISE_BOUND = 0.04
CONFIDENCE_Z_VALUES = (0.0, 1.0, 1.645, 2.0)
CUE_TIE_MARGIN = 0.40
BOOTSTRAP_REPLICATES = 5000
CONDITIONS = {
    "id": {"scarcity": 1.0, "sensor_noise_std": 0.0},
    "scarcity": {"scarcity": 2.5, "sensor_noise_std": 0.0},
    "noise": {"scarcity": 1.0, "sensor_noise_std": 0.04},
    "scarcity_noise": {"scarcity": 2.5, "sensor_noise_std": 0.04},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    digest = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    if not digest or hashlib.sha256(canonical_bytes(body)).hexdigest() != digest:
        raise ValueError(f"receipt self-hash failed: {path}")
    return receipt


def _load_model(path: Path) -> tuple[DecisionValueMemory, str]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    values = dict(state["config"])
    values["horizons"] = tuple(values.get("horizons", (4, 8)))
    memory = DecisionValueMemory(DecisionValueMemoryConfig(**values))
    memory.load_state_dict(state)
    return memory, _sha256(path)


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
    policy: str,
    core: CoreRuntime,
    memory: DecisionValueMemory | None = None,
    threshold: float = RESOURCE_THRESHOLD,
    confidence_z: float = 0.0,
    predictive_resource_guard: bool = False,
    predictive_floor: float = 0.15,
    resource_balance_margin: float | None = None,
    resource_balance_activation: float = 0.35,
) -> dict[str, Any]:
    env = _environment(seed, regime, condition)
    obs = env.reset()
    cue_regime = 0 if obs[13] > 0.5 else 1
    agent = None
    if policy in ("structured_intact_guarded", "structured_erased_guarded", "matched_gru_guarded", "development_candidate"):
        if memory is None:
            raise ValueError(f"{policy} requires a fitted memory model")
        mode = "erased" if policy == "structured_erased_guarded" else "intact"
        agent = ResourceGuardedDecisionValueAgent(
            core=core,
            memory=copy.deepcopy(memory),
            mode=mode,
            influence_scale=INFLUENCE_SCALE,
            resource_guard_threshold=threshold,
            sensor_noise_std=SENSOR_NOISE_BOUND,
            confidence_z=confidence_z,
            predictive_resource_guard=predictive_resource_guard,
            resource_guard_floor=predictive_floor,
            resource_balance_margin=resource_balance_margin,
            resource_balance_activation=resource_balance_activation,
        )
        agent.reset(mode=mode)

    total_reward = 0.0
    min_energy = 1.0
    min_hydration = 1.0
    memory_override_count = 0
    gate_count = 0
    guard_trigger_count = 0
    guard_override_count = 0
    trajectory = []
    while not env.done:
        tick = env.tick
        if agent is not None:
            decision = agent.act(obs)
            action = int(decision["action"])
            base_action = int(decision["base_action"])
            memory_override_count += int(bool(decision.get("memory_override_executed", False)))
            gate_count += int(bool(decision.get("admission_gate_open", False)))
            guard_trigger_count += int(bool(decision.get("resource_guard_triggered", False)))
            guard_override_count += int(bool(decision.get("resource_guard_override", False)))
            decision_trace = {
                "base_action": base_action,
                "memory_override_executed": bool(decision.get("memory_override_executed", False)),
                "gate_open": bool(decision.get("admission_gate_open", False)),
                "guard_triggered": bool(decision.get("resource_guard_triggered", False)),
                "guard_override": bool(decision.get("resource_guard_override", False)),
                "guard_energy_lcb": round(float(decision.get("resource_guard_energy_lcb", 0.0)), 8),
                "guard_hydration_lcb": round(float(decision.get("resource_guard_hydration_lcb", 0.0)), 8),
                "predictive_guard": bool(decision.get("predictive_resource_guard", False)),
                "guard_candidate_predicted_minimum": None if not predictive_resource_guard else round(float(decision["resource_guard_candidate_predicted_minimum"]), 8),
                "guard_foraging_predicted_minimum": None if not predictive_resource_guard else round(float(decision["resource_guard_foraging_predicted_minimum"]), 8),
                "guard_drinking_predicted_minimum": None if not predictive_resource_guard else round(float(decision["resource_guard_drinking_predicted_minimum"]), 8),
                "guard_balance_margin": resource_balance_margin,
                "guard_balance_activation": resource_balance_activation,
            }
        elif policy == "reactive_balancer":
            action = 1 if float(obs[1]) <= float(obs[2]) else 2
            base_action = action
            decision_trace = {"base_action": base_action}
        elif policy == "cue_memory_balancer":
            energy, hydration = float(obs[1]), float(obs[2])
            if tick < 4:
                action = 1 if energy <= hydration else 2
            elif energy - hydration > CUE_TIE_MARGIN:
                action = 2
            elif hydration - energy > CUE_TIE_MARGIN:
                action = 1
            else:
                action = 1 if cue_regime == 0 else 2
            base_action = action
            decision_trace = {"base_action": base_action, "cue_regime": cue_regime}
        elif policy == "frozen_tidal":
            action = int(core.act(obs))
            base_action = action
            decision_trace = {"base_action": base_action}
        else:
            raise ValueError(f"unknown policy: {policy}")

        next_obs, reward, done, info = env.step(action)
        if agent is not None:
            agent.observe(obs, action, reward, next_obs)
        total_reward += float(reward)
        min_energy = min(min_energy, float(info["energy"]))
        min_hydration = min(min_hydration, float(info["hydration"]))
        trajectory.append({
            "tick": tick,
            "observation": [float(x) for x in obs],
            "action": action,
            "reward": float(reward),
            "next_observation": [float(x) for x in next_obs],
            "info": info,
            "decision": decision_trace,
        })
        obs = next_obs
        if done:
            break

    digest = hashlib.sha256(canonical_bytes(trajectory)).hexdigest()
    return {
        "seed": int(seed),
        "regime": int(regime),
        "condition": condition,
        "policy": policy,
        "total_reward": float(total_reward),
        "steps": len(trajectory),
        "survived": not bool(trajectory[-1]["info"].get("death", False)),
        "minimum_energy": float(min_energy),
        "minimum_hydration": float(min_hydration),
        "memory_override_count": memory_override_count,
        "gate_open_count": gate_count,
        "resource_guard_trigger_count": guard_trigger_count,
        "resource_guard_override_count": guard_override_count,
        "trajectory_sha256": digest,
    }


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
        "episodes_with_memory_override": int(sum(row["memory_override_count"] > 0 for row in rows)),
        "total_memory_overrides": int(sum(row["memory_override_count"] for row in rows)),
        "total_guard_overrides": int(sum(row["resource_guard_override_count"] for row in rows)),
    }


def _seed_means(rows: list[dict[str, Any]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["seed"]].append(float(row[metric]))
    return {seed: float(np.mean(values)) for seed, values in grouped.items()}


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def _comparisons(rows_by_policy: dict[str, list[dict[str, Any]]], candidate: str, seed_offset: int) -> dict[str, Any]:
    candidate_reward = _seed_means(rows_by_policy[candidate], "total_reward")
    candidate_survival = _seed_means(rows_by_policy[candidate], "survived")
    out = {}
    for i, comparator in enumerate(rows_by_policy):
        if comparator == candidate:
            continue
        other_reward = _seed_means(rows_by_policy[comparator], "total_reward")
        other_survival = _seed_means(rows_by_policy[comparator], "survived")
        seeds = sorted(set(candidate_reward) & set(other_reward))
        reward_delta = np.asarray([candidate_reward[s] - other_reward[s] for s in seeds], dtype=float)
        survival_delta = np.asarray([candidate_survival[s] - other_survival[s] for s in seeds], dtype=float)
        out[f"{candidate}_vs_{comparator}"] = {
            "paired_environment_seeds": len(seeds),
            "reward_delta_mean": float(np.mean(reward_delta)),
            "reward_delta_ci95_seed_bootstrap": _paired_ci(reward_delta, seed_offset + i),
            "reward_win_rate_by_seed": float(np.mean(reward_delta > 0.0)),
            "survival_delta_mean": float(np.mean(survival_delta)),
        }
    return out


def _source_manifest(extra: list[Path]) -> dict[str, str]:
    paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve(), *extra]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _write_protocol(output: Path, protocol: dict[str, Any], source_hashes: dict[str, str]) -> None:
    protocol["source_sha256"] = source_hashes
    protocol["core_checkpoint_sha256"] = _sha256(Path(protocol["core_checkpoint_path"]))
    protocol["protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in source_hashes:
            archive.write(ROOT / relative, relative)


def _ensure_source(source_hashes: dict[str, str]) -> None:
    for relative, digest in source_hashes.items():
        if _sha256(ROOT / relative) != digest:
            raise RuntimeError(f"source changed during assay; evidence retained: {relative}")


def develop(output: Path, seed_count: int = DEVELOPMENT_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving prior evidence: {output}")
    if seed_count < 8:
        raise ValueError("development seed count must be at least eight")
    training_receipt = _load_receipt(TRAINING_DIR / "training_receipt.json")
    structured, structured_sha = _load_model(TRAINING_DIR / "structured_memory_model.pt")
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + seed_count))
    variants = [{"threshold": RESOURCE_THRESHOLD, "confidence_z": z} for z in CONFIDENCE_Z_VALUES]
    output.mkdir(parents=True)
    source_hashes = _source_manifest([ROOT / "tools/train_alias_tidepool_v2_structured_dvgem.py"])
    protocol = {
        "schema": "poseidon-alias-v2-structured-dvgem-guard-development-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "development_seed_range": [seeds[0], seeds[-1]],
        "conditions": CONDITIONS,
        "regimes": [0, 1],
        "max_steps": MAX_STEPS,
        "guard_variants": variants,
        "sensor_noise_bound_assumed_for_all_conditions": SENSOR_NOISE_BOUND,
        "fixed_threshold": RESOURCE_THRESHOLD,
        "influence_scale": INFLUENCE_SCALE,
        "paired_unit": "environment seed averaged across both regimes",
        "selection_rule": {
            "first": "require 100% survival and minimum energy/hydration >= 0.15 in every condition",
            "then": "maximize unweighted mean reward across the four conditions",
            "fallback_if_none_pass": "maximize worst-condition minimum resource, then survival, then mean reward",
        },
        "training_receipt_sha256": training_receipt["receipt_sha256"],
        "structured_model_sha256": structured_sha,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }
    _write_protocol(output, protocol, source_hashes)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        episodes: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for variant in variants:
                        label = f"z{variant['confidence_z']:g}"
                        row = _episode(
                            seed, regime, condition, "development_candidate", core, structured,
                            threshold=variant["threshold"], confidence_z=variant["confidence_z"],
                        )
                        row["policy"] = label
                        row["threshold"] = variant["threshold"]
                        row["confidence_z"] = variant["confidence_z"]
                        episodes.append(row)
                    for policy in ("reactive_balancer", "cue_memory_balancer", "frozen_tidal"):
                        episodes.append(_episode(seed, regime, condition, policy, core))
            print(json.dumps({"phase": "development_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            grouped[(row["condition"], row["policy"])].append(row)
        policies = [f"z{z:g}" for z in CONFIDENCE_Z_VALUES]
        by_condition = {
            condition: {policy: _condition_summary(grouped[(condition, policy)]) for policy in policies + ["reactive_balancer", "cue_memory_balancer", "frozen_tidal"]}
            for condition in CONDITIONS
        }
        candidate_results = {}
        for policy in policies:
            condition_rows = {c: grouped[(c, policy)] for c in CONDITIONS}
            mean_reward = float(np.mean([by_condition[c][policy]["mean_reward"] for c in CONDITIONS]))
            minimum_resource = min(
                min(by_condition[c][policy]["minimum_energy"], by_condition[c][policy]["minimum_hydration"])
                for c in CONDITIONS
            )
            survival = min(by_condition[c][policy]["survival_rate"] for c in CONDITIONS)
            passed = all(
                by_condition[c][policy]["survival_rate"] == 1.0
                and by_condition[c][policy]["minimum_energy"] >= 0.15
                and by_condition[c][policy]["minimum_hydration"] >= 0.15
                for c in CONDITIONS
            )
            candidate_results[policy] = {
                "threshold": RESOURCE_THRESHOLD,
                "confidence_z": float(policy[1:]),
                "mean_reward_across_conditions": mean_reward,
                "worst_condition_minimum_resource": minimum_resource,
                "minimum_condition_survival_rate": survival,
                "passes_development_margin_gate": passed,
            }
        passing = [name for name, result in candidate_results.items() if result["passes_development_margin_gate"]]
        if passing:
            selected_name = max(passing, key=lambda name: candidate_results[name]["mean_reward_across_conditions"])
            selection_basis = "reward among candidates passing all development resource/survival gates"
        else:
            selected_name = max(
                candidate_results,
                key=lambda name: (
                    candidate_results[name]["worst_condition_minimum_resource"],
                    candidate_results[name]["minimum_condition_survival_rate"],
                    candidate_results[name]["mean_reward_across_conditions"],
                ),
            )
            selection_basis = "best development worst-resource fallback; no variant passed all gates"

        replay_checks = 0
        for row in episodes:
            if row["policy"].startswith("z"):
                variant = {"threshold": row["threshold"], "confidence_z": row["confidence_z"]}
                replay = _episode(row["seed"], row["regime"], row["condition"], "development_candidate", core, structured, **variant)
                replay["policy"] = row["policy"]
                replay["threshold"] = row["threshold"]
                replay["confidence_z"] = row["confidence_z"]
            else:
                replay = _episode(row["seed"], row["regime"], row["condition"], row["policy"], core)
            if replay != row:
                raise RuntimeError(f"development replay mismatch: {row['policy']} seed={row['seed']} condition={row['condition']}")
            replay_checks += 1
        _ensure_source(source_hashes)

        selected = candidate_results[selected_name]
        selection = {
            "schema": "poseidon-alias-v2-structured-dvgem-guard-selection-v1",
            "selected": {"threshold": RESOURCE_THRESHOLD, "confidence_z": selected["confidence_z"]},
            "selected_label": selected_name,
            "selection_basis": selection_basis,
            "development_receipt_sha256": None,
        }
        receipt = {
            "schema": "poseidon-alias-v2-structured-dvgem-guard-development-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": training_receipt["receipt_sha256"],
            "structured_model_sha256": structured_sha,
            "results": {
                "by_condition": by_condition,
                "candidate_selection": candidate_results,
                "selected_guard": selection["selected"],
                "selected_label": selected_name,
                "selection_basis": selection_basis,
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
            },
            "episodes": episodes,
            "source_sha256": source_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": [
                "Development-only threshold selection; confirmation uses a disjoint seed block.",
                "The sensor-noise bound is configured from this synthetic benchmark, not inferred online.",
                "The reserve controller is heuristic and makes no real-world safety claim.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        selection["development_receipt_sha256"] = receipt["receipt_sha256"]
        selection["selection_sha256"] = hashlib.sha256(canonical_bytes(selection)).hexdigest()
        (output / "development_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output / "guard_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "selected_guard": selection["selected"],
            "selection_basis": selection_basis,
            "candidate_selection": candidate_results,
            "replay_checks": replay_checks,
            "receipt_sha256": receipt["receipt_sha256"],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def confirm(output: Path, seed_count: int = CONFIRMATION_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving prior evidence: {output}")
    if seed_count < 16:
        raise ValueError("confirmation seed count must be at least sixteen")
    training_receipt = _load_receipt(TRAINING_DIR / "training_receipt.json")
    development_dir = ROOT / "runs/research/decision_value_memory_v2_structured_guard_development_20261014"
    development_receipt = _load_receipt(development_dir / "development_receipt.json")
    selection = json.loads((development_dir / "guard_selection.json").read_text(encoding="utf-8"))
    selection_body = dict(selection)
    selection_sha = selection_body.pop("selection_sha256", None)
    if selection.get("development_receipt_sha256") != development_receipt["receipt_sha256"]:
        raise ValueError("guard selection does not bind the current development receipt")
    if not selection_sha or hashlib.sha256(canonical_bytes(selection_body)).hexdigest() != selection_sha:
        raise ValueError("guard selection self-hash failed")
    threshold = float(selection["selected"]["threshold"])
    confidence_z = float(selection["selected"]["confidence_z"])
    structured, structured_sha = _load_model(TRAINING_DIR / "structured_memory_model.pt")
    gru, gru_sha = _load_model(TRAINING_DIR / "matched_gru_control_model.pt")
    if structured_sha != training_receipt["model_sha256"]["structured_memory_model.pt"]:
        raise ValueError("structured model does not match its training receipt")
    if gru_sha != training_receipt["model_sha256"]["matched_gru_control_model.pt"]:
        raise ValueError("matched GRU control does not match its training receipt")
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + seed_count))
    policies = (
        "frozen_tidal",
        "reactive_balancer",
        "cue_memory_balancer",
        "structured_intact_guarded",
        "structured_erased_guarded",
        "matched_gru_guarded",
    )
    output.mkdir(parents=True)
    source_hashes = _source_manifest([ROOT / "tools/train_alias_tidepool_v2_structured_dvgem.py"])
    protocol = {
        "schema": "poseidon-alias-v2-structured-dvgem-fresh-confirmation-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "fresh_evaluation_seed_range": [seeds[0], seeds[-1]],
        "excluded_seed_ranges": {
            "structured_gru_training": [743020000, 743020011],
            "previous_v2_training": [743010000, 743010011],
            "guard_development": [DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + DEVELOPMENT_SEED_COUNT - 1],
            "previous_scale_and_guard_selection": [743050000, 743050015],
            "previous_confirmation": [743060000, 743060031],
        },
        "conditions": CONDITIONS,
        "regimes": [0, 1],
        "max_steps": MAX_STEPS,
        "guard": {"threshold": threshold, "sensor_noise_bound": SENSOR_NOISE_BOUND, "confidence_z": confidence_z},
        "influence_scale": INFLUENCE_SCALE,
        "cue_memory_tie_margin": CUE_TIE_MARGIN,
        "arms": list(policies),
        "paired_unit": "environment seed averaged over both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
        "training_receipt_sha256": training_receipt["receipt_sha256"],
        "development_receipt_sha256": development_receipt["receipt_sha256"],
        "guard_selection_sha256": selection_sha,
        "structured_model_sha256": structured_sha,
        "matched_gru_model_sha256": gru_sha,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }
    _write_protocol(output, protocol, source_hashes)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        memory_by_policy = {
            "structured_intact_guarded": structured,
            "structured_erased_guarded": structured,
            "matched_gru_guarded": gru,
        }
        rows = []
        for condition in CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for policy in policies:
                        rows.append(_episode(
                            seed, regime, condition, policy, core,
                            memory=memory_by_policy.get(policy), threshold=threshold, confidence_z=confidence_z,
                        ))
            print(json.dumps({"phase": "confirmation_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        by_condition = {
            condition: {policy: _condition_summary(grouped[(condition, policy)]) for policy in policies}
            for condition in CONDITIONS
        }
        comparisons = {
            condition: _comparisons(
                {policy: grouped[(condition, policy)] for policy in policies},
                "structured_intact_guarded",
                CONFIRMATION_SEED_START + condition_index * 31,
            )
            for condition_index, condition in enumerate(CONDITIONS)
        }
        replay_checks = 0
        for row in rows:
            replay = _episode(
                row["seed"], row["regime"], row["condition"], row["policy"], core,
                memory=memory_by_policy.get(row["policy"]), threshold=threshold, confidence_z=confidence_z,
            )
            if replay != row:
                raise RuntimeError(f"confirmation replay mismatch: {row['policy']} seed={row['seed']} condition={row['condition']}")
            replay_checks += 1
        _ensure_source(source_hashes)

        candidate = {condition: by_condition[condition]["structured_intact_guarded"] for condition in CONDITIONS}
        gates = {
            "full_survival_all_conditions": all(candidate[c]["survival_rate"] == 1.0 for c in CONDITIONS),
            "minimum_resource_at_least_0_15_all_conditions": all(
                candidate[c]["minimum_energy"] >= 0.15 and candidate[c]["minimum_hydration"] >= 0.15
                for c in CONDITIONS
            ),
            "memory_overrides_executed": sum(candidate[c]["total_memory_overrides"] for c in CONDITIONS) > 0,
            "positive_reward_ci_vs_erased_all_conditions": all(
                comparisons[c]["structured_intact_guarded_vs_structured_erased_guarded"]["reward_delta_ci95_seed_bootstrap"][0] > 0.0
                for c in CONDITIONS
            ),
            "beats_matched_gru_in_all_conditions": all(
                comparisons[c]["structured_intact_guarded_vs_matched_gru_guarded"]["reward_delta_mean"] > 0.0
                for c in CONDITIONS
            ),
            "promoted": False,
        }
        receipt = {
            "schema": "poseidon-alias-v2-structured-dvgem-confirmation-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": training_receipt["receipt_sha256"],
            "development_receipt_sha256": development_receipt["receipt_sha256"],
            "guard_selection_sha256": selection_sha,
            "structured_model_sha256": structured_sha,
            "matched_gru_model_sha256": gru_sha,
            "results": {
                "by_condition": by_condition,
                "paired_comparisons": comparisons,
                "gates": gates,
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
            },
            "episodes": rows,
            "source_sha256": source_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": [
                "Fresh local evaluation in a single synthetic benchmark, not independent external replication.",
                "The cue latch and confidence-bound guard are hand-specified from the synthetic observation contract.",
                "Reward scale remains an uncalibrated probability-plus-value mixture; a passing gate would not prove general safety.",
                "The controller remains opt-in and is not promoted by this receipt.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        (output / "confirmation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": receipt["elapsed_seconds"],
            "selected_guard": protocol["guard"],
            "by_condition": by_condition,
            "paired_comparisons": comparisons,
            "gates": gates,
            "replay_checks": replay_checks,
            "receipt_sha256": receipt["receipt_sha256"],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "confirm"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int)
    args = parser.parse_args()
    if args.phase == "develop":
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_structured_guard_development_20261014"
        result = develop(output, args.seeds or DEVELOPMENT_SEED_COUNT)
    else:
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_structured_confirmation_20261014"
        result = confirm(output, args.seeds or CONFIRMATION_SEED_COUNT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
