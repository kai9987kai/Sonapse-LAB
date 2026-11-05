from __future__ import annotations

"""Calibrate a conservative resource reserve on a separate seed block."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.assay_alias_tidepool_v2_predictive_guard as guard
import tools.assay_alias_tidepool_v2_structured_dvgem as assay
from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes
from poseidon.research.decision_value_memory import DecisionValueMemory


TRAINING_DIR = ROOT / "runs/research/decision_value_memory_v2_contrastive_pilot_20261014"
SEED_START = 743200000
SEED_COUNT = 8
FLOORS = (0.27, 0.30, 0.33, 0.36)
RESOURCE_TARGET = 0.20
BALANCE_MARGIN = 0.20
BALANCE_ACTIVATION = 0.35
INFLUENCE_SCALE = 32.0
LEARNED_ARMS = ("contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute")
REFERENCE_ARMS = ("reactive_balancer", "cue_memory_balancer", "frozen_tidal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.get("receipt_sha256")
    body = dict(value)
    body.pop("receipt_sha256", None)
    if not digest or hashlib.sha256(canonical_bytes(body)).hexdigest() != digest:
        raise ValueError(f"receipt self-hash failed: {path}")
    return value


def _sources() -> dict[str, str]:
    dependencies = [
        Path(__file__).resolve(),
        ROOT / "tools/assay_alias_tidepool_v2_predictive_guard.py",
        ROOT / "tools/assay_alias_tidepool_v2_structured_dvgem.py",
        ROOT / "tools/train_alias_tidepool_v2_contrastive_dvgem.py",
    ]
    paths = sorted((ROOT / "poseidon").rglob("*.py")) + dependencies
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _protocol(output: Path, body: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    body["source_sha256"] = sources
    body["core_checkpoint_sha256"] = _sha256(Path(body["core_checkpoint_path"]))
    body["protocol_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    (output / "protocol.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    return body


def _ensure(sources: dict[str, str]) -> None:
    for relative, digest in sources.items():
        if _sha256(ROOT / relative) != digest:
            raise RuntimeError(f"source changed during assay: {relative}")


def _paired(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], metric: str, seed: int) -> dict[str, Any]:
    a = assay._seed_means(a_rows, metric)
    b = assay._seed_means(b_rows, metric)
    seeds = sorted(set(a) & set(b))
    delta = np.asarray([a[seed] - b[seed] for seed in seeds], dtype=float)
    return {
        "paired_environment_seeds": len(delta),
        "mean": float(np.mean(delta)),
        "ci95": assay._paired_ci(delta, seed),
        "win_rate": float(np.mean(delta > 0.0)),
    }


def _run(seed: int, regime: int, condition: str, arm: str, floor: float | None,
         core: CoreRuntime, structured: DecisionValueMemory, gru: DecisionValueMemory,
         absolute: DecisionValueMemory) -> dict[str, Any]:
    assay.INFLUENCE_SCALE = INFLUENCE_SCALE
    if arm in LEARNED_ARMS:
        row = guard._run_episode(
            seed, regime, condition, arm, float(floor), core, structured, gru, absolute,
            balance_margin=BALANCE_MARGIN, balance_activation=BALANCE_ACTIVATION,
        )
        row.update({"policy": f"{arm}_floor{floor:.2f}", "model_role": arm,
                    "reserve_floor": floor, "balance_margin": BALANCE_MARGIN,
                    "balance_activation": BALANCE_ACTIVATION})
        return row
    return assay._episode(seed, regime, condition, arm, core)


def develop(output: Path | None = None, seed_count: int = SEED_COUNT) -> dict[str, Any]:
    output = output or ROOT / "runs/research/decision_value_memory_v2_reserve_calibration_v2_development_20261014"
    if output.exists():
        raise FileExistsError(f"output exists; preserving prior evidence: {output}")
    if seed_count < 8:
        raise ValueError("development requires at least eight environment seeds")
    train = guard._load_receipt(TRAINING_DIR / "training_receipt.json")
    model_hashes = train["model_sha256"]
    structured = guard._load_model(TRAINING_DIR / "structured_contrastive_model.pt", model_hashes["structured_contrastive_model.pt"])
    gru = guard._load_model(TRAINING_DIR / "matched_gru_contrastive_control_model.pt", model_hashes["matched_gru_contrastive_control_model.pt"])
    absolute = guard._load_model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", model_hashes["matched_structured_absolute_control_model.pt"])
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(SEED_START, SEED_START + seed_count))
    output.mkdir(parents=True)
    sources = _sources()
    protocol = _protocol(output, {
        "schema": "poseidon-alias-v2-reserve-calibration-development-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "development_seed_range": [seeds[0], seeds[-1]],
        "conditions": assay.CONDITIONS,
        "regimes": [0, 1],
        "max_steps": assay.MAX_STEPS,
        "selection_rule": f"choose the lowest reserve floor with full survival, minimum energy and hydration at least {RESOURCE_TARGET:.2f}, and positive paired 95% reward interval versus erased memory in every condition; otherwise choose the setting with the strongest worst-condition resource minimum",
        "reserve_guard": {
            "strategy": "one-step action-conditioned lower-bound filter plus adaptive balance deadband",
            "floor_grid": list(FLOORS), "minimum_observed_resource_target": RESOURCE_TARGET,
            "balance_margin": BALANCE_MARGIN,
            "balance_activation": BALANCE_ACTIVATION,
            "maximum_resource_drain": guard.MAXIMUM_RESOURCE_DRAIN,
            "minimum_resource_yield": guard.MINIMUM_RESOURCE_YIELD,
            "sensor_noise_bound": assay.SENSOR_NOISE_BOUND,
            "confidence_z": guard.CONFIDENCE_Z,
        },
        "influence_scale": INFLUENCE_SCALE,
        "arms": list(LEARNED_ARMS) + list(REFERENCE_ARMS),
        "training_receipt_sha256": train["receipt_sha256"],
        "model_sha256": model_hashes,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }, sources)

    old_threads = torch.get_num_threads()
    old_scale = assay.INFLUENCE_SCALE
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(2)
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for floor in FLOORS:
                        for arm in LEARNED_ARMS:
                            rows.append(_run(seed, regime, condition, arm, floor, core, structured, gru, absolute))
                    for arm in REFERENCE_ARMS:
                        rows.append(_run(seed, regime, condition, arm, None, core, structured, gru, absolute))
            print(json.dumps({"phase": "reserve_calibration_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        results: dict[str, Any] = {}
        eligible: list[str] = []
        for floor in FLOORS:
            label = f"contrastive_intact_floor{floor:.2f}"
            floor_result: dict[str, Any] = {"floor": floor, "by_condition": {}}
            min_resource, reward_values = 1.0, []
            safety, value = True, True
            for ci, condition in enumerate(assay.CONDITIONS):
                candidate = grouped[(condition, label)]
                stats = assay._condition_summary(candidate)
                erased = grouped[(condition, f"contrastive_erased_floor{floor:.2f}")]
                vs_erased = _paired(candidate, erased, "total_reward", SEED_START + ci + 1)
                comparisons = {name: _paired(candidate, grouped[(condition, name)], "total_reward", SEED_START + ci + 11 + j)
                               for j, name in enumerate(("cue_memory_balancer", "matched_gru_floor" + f"{floor:.2f}", "matched_absolute_floor" + f"{floor:.2f}", "reactive_balancer"))}
                floor_result["by_condition"][condition] = {"candidate": stats, "intact_vs_erased": vs_erased,
                                                              "comparisons": comparisons}
                min_resource = min(min_resource, stats["minimum_energy"], stats["minimum_hydration"])
                reward_values.append(stats["mean_reward"])
                safety = safety and stats["survival_rate"] == 1.0 and stats["minimum_energy"] >= RESOURCE_TARGET and stats["minimum_hydration"] >= RESOURCE_TARGET
                value = value and vs_erased["ci95"][0] > 0.0
            floor_result["passes_safety_floor"] = safety
            floor_result["positive_memory_value_ci_all_conditions"] = value
            floor_result["worst_condition_minimum_resource"] = min_resource
            floor_result["mean_reward_across_conditions"] = float(np.mean(reward_values))
            results[f"floor_{floor:.2f}"] = floor_result
            if safety and value:
                eligible.append(f"floor_{floor:.2f}")

        if eligible:
            selected = min(eligible, key=lambda key: results[key]["floor"])
            basis = "lowest reserve floor passing the preregistered safety and memory-value gates"
        else:
            selected = max(results, key=lambda key: (results[key]["worst_condition_minimum_resource"], results[key]["mean_reward_across_conditions"]))
            basis = "best worst-resource development fallback; primary gates were not all met"

        replay_checks = 0
        for row in rows:
            arm = row.get("model_role", row["policy"])
            floor = row.get("reserve_floor")
            replay = _run(row["seed"], row["regime"], row["condition"], arm, floor, core, structured, gru, absolute)
            if replay != row:
                raise RuntimeError(f"reserve-calibration replay mismatch: {row['policy']} seed={row['seed']}")
            replay_checks += 1
        _ensure(sources)
        chosen_floor = float(results[selected]["floor"])
        receipt = {
            "schema": "poseidon-alias-v2-reserve-calibration-development-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": train["receipt_sha256"],
            "model_sha256": model_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "source_sha256": sources,
            "results": results,
            "selected_floor": chosen_floor,
            "minimum_observed_resource_target": RESOURCE_TARGET,
            "selected_key": selected,
            "selection_basis": basis,
            "episodes": rows,
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "promotion": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        selection = {"schema": "poseidon-alias-v2-reserve-calibration-selection-v1",
                     "development_receipt_sha256": receipt["receipt_sha256"], "selected_floor": chosen_floor,
                     "balance_margin": BALANCE_MARGIN, "balance_activation": BALANCE_ACTIVATION,
                     "selection_basis": basis}
        selection["selection_sha256"] = hashlib.sha256(canonical_bytes(selection)).hexdigest()
        (output / "development_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"complete": True, "selected_floor": chosen_floor, "selected_key": selected,
                   "selection_basis": basis, "results": results, "replay_checks": replay_checks,
                   "receipt_sha256": receipt["receipt_sha256"], "elapsed_seconds": receipt["elapsed_seconds"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(old_threads)


def confirm(output: Path | None = None, seed_count: int = 32) -> dict[str, Any]:
    output = output or ROOT / "runs/research/decision_value_memory_v2_reserve_calibration_v2_confirmation_20261014"
    if output.exists():
        raise FileExistsError(f"output exists; preserving prior evidence: {output}")
    if seed_count < 16:
        raise ValueError("confirmation requires at least sixteen environment seeds")
    development_dir = ROOT / "runs/research/decision_value_memory_v2_reserve_calibration_v2_development_20261014"
    development = _receipt(development_dir / "development_receipt.json")
    selection = json.loads((development_dir / "selection.json").read_text(encoding="utf-8"))
    selection_body = dict(selection)
    selection_sha = selection_body.pop("selection_sha256", None)
    if (selection.get("development_receipt_sha256") != development["receipt_sha256"]
            or hashlib.sha256(canonical_bytes(selection_body)).hexdigest() != selection_sha):
        raise ValueError("reserve selection is not bound to its development receipt")
    floor = float(selection["selected_floor"])
    train = guard._load_receipt(TRAINING_DIR / "training_receipt.json")
    model_hashes = train["model_sha256"]
    structured = guard._load_model(TRAINING_DIR / "structured_contrastive_model.pt", model_hashes["structured_contrastive_model.pt"])
    gru = guard._load_model(TRAINING_DIR / "matched_gru_contrastive_control_model.pt", model_hashes["matched_gru_contrastive_control_model.pt"])
    absolute = guard._load_model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", model_hashes["matched_structured_absolute_control_model.pt"])
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seed_start = 743210000
    seeds = list(range(seed_start, seed_start + seed_count))
    arms = LEARNED_ARMS + REFERENCE_ARMS
    output.mkdir(parents=True)
    sources = _sources()
    protocol = _protocol(output, {
        "schema": "poseidon-alias-v2-reserve-calibration-confirmation-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "fresh_seed_range": [seeds[0], seeds[-1]],
        "excluded_seed_ranges": {
            "training": [743030000, 743030011],
            "prior_predictive_guard_development": [743130000, 743130007],
            "prior_adaptive_balance_development": [743170000, 743170007],
            "prior_adaptive_balance_confirmation": [743180000, 743180031],
            "first_reserve_calibration_development": [743190000, 743190007],
            "reserve_v2_development": [SEED_START, SEED_START + SEED_COUNT - 1],
        },
        "conditions": assay.CONDITIONS,
        "regimes": [0, 1],
        "max_steps": assay.MAX_STEPS,
        "frozen_selection": {"floor": floor, "balance_margin": BALANCE_MARGIN,
                             "balance_activation": BALANCE_ACTIVATION,
                             "development_receipt_sha256": development["receipt_sha256"],
                             "selection_sha256": selection_sha},
        "minimum_observed_resource_target": RESOURCE_TARGET,
        "sensor_noise_bound": assay.SENSOR_NOISE_BOUND,
        "confidence_z": guard.CONFIDENCE_Z,
        "influence_scale": INFLUENCE_SCALE,
        "arms": list(arms),
        "training_receipt_sha256": train["receipt_sha256"],
        "model_sha256": model_hashes,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }, sources)

    old_threads = torch.get_num_threads()
    old_scale = assay.INFLUENCE_SCALE
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        torch.set_num_threads(2)
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for arm in LEARNED_ARMS:
                        rows.append(_run(seed, regime, condition, arm, floor, core, structured, gru, absolute))
                    for arm in REFERENCE_ARMS:
                        rows.append(_run(seed, regime, condition, arm, None, core, structured, gru, absolute))
            print(json.dumps({"phase": "reserve_confirmation_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        candidate_label = f"contrastive_intact_floor{floor:.2f}"
        by_condition: dict[str, Any] = {}
        for ci, condition in enumerate(assay.CONDITIONS):
            candidate = grouped[(condition, candidate_label)]
            stats = assay._condition_summary(candidate)
            comparisons: dict[str, Any] = {}
            for j, arm in enumerate((f"contrastive_erased_floor{floor:.2f}", f"matched_absolute_floor{floor:.2f}",
                                     f"matched_gru_floor{floor:.2f}", "reactive_balancer", "cue_memory_balancer")):
                comparisons[arm] = _paired(candidate, grouped[(condition, arm)], "total_reward", seed_start + ci * 20 + j)
            by_condition[condition] = {"policies": {arm: assay._condition_summary(grouped[(condition, (f"{arm}_floor{floor:.2f}" if arm in LEARNED_ARMS else arm))]) for arm in arms},
                                       "candidate": stats, "comparisons": comparisons}

        gates = {
            "full_survival_all_conditions": all(by_condition[c]["candidate"]["survival_rate"] == 1.0 for c in assay.CONDITIONS),
            "minimum_resource_at_least_0_20_all_conditions": all(
                by_condition[c]["candidate"]["minimum_energy"] >= RESOURCE_TARGET
                and by_condition[c]["candidate"]["minimum_hydration"] >= RESOURCE_TARGET
                for c in assay.CONDITIONS),
            "positive_memory_value_ci_vs_erased_all_conditions": all(
                by_condition[c]["comparisons"][f"contrastive_erased_floor{floor:.2f}"]["ci95"][0] > 0.0
                for c in assay.CONDITIONS),
            "positive_value_ci_vs_absolute_all_conditions": all(
                by_condition[c]["comparisons"][f"matched_absolute_floor{floor:.2f}"]["ci95"][0] > 0.0
                for c in assay.CONDITIONS),
            "beats_reactive_in_all_conditions": all(
                by_condition[c]["comparisons"]["reactive_balancer"]["ci95"][0] > 0.0
                for c in assay.CONDITIONS),
            "beats_matched_gru_in_all_conditions": all(
                by_condition[c]["comparisons"][f"matched_gru_floor{floor:.2f}"]["ci95"][0] > 0.0
                for c in assay.CONDITIONS),
            "beats_cue_memory_in_all_conditions": all(
                by_condition[c]["comparisons"]["cue_memory_balancer"]["ci95"][0] > 0.0
                for c in assay.CONDITIONS),
        }
        gates["promoted"] = False

        replay_checks = 0
        for row in rows:
            arm = row.get("model_role", row["policy"])
            replay = _run(row["seed"], row["regime"], row["condition"], arm,
                          row.get("reserve_floor"), core, structured, gru, absolute)
            if replay != row:
                raise RuntimeError(f"reserve-confirmation replay mismatch: {row['policy']} seed={row['seed']}")
            replay_checks += 1
        _ensure(sources)
        receipt = {
            "schema": "poseidon-alias-v2-reserve-calibration-confirmation-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "development_receipt_sha256": development["receipt_sha256"],
            "selection_sha256": selection_sha,
            "training_receipt_sha256": train["receipt_sha256"],
            "model_sha256": model_hashes,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "source_sha256": sources,
            "selected_floor": floor,
            "minimum_observed_resource_target": RESOURCE_TARGET,
            "by_condition": by_condition,
            "gates": gates,
            "episodes": rows,
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "promotion": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        (output / "confirmation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {"complete": True, "selected_floor": floor, "by_condition": by_condition,
                   "gates": gates, "replay_checks": replay_checks,
                   "receipt_sha256": receipt["receipt_sha256"], "elapsed_seconds": receipt["elapsed_seconds"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        result = confirm()
    else:
        result = develop()
    print(json.dumps({"complete": result["complete"], "selected_floor": result["selected_floor"],
                      "replay_checks": result["replay_checks"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
