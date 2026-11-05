"""Select and confirm a resource-balance deadband for contrastive DV-GEM."""
from __future__ import annotations

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
from poseidon.research.decision_value_memory import DecisionValueMemory, DecisionValueMemoryConfig


TRAINING_DIR = guard.TRAINING_DIR
DEVELOPMENT_SEED_START = 743170000
DEVELOPMENT_SEED_COUNT = 8
CONFIRMATION_SEED_START = 743180000
CONFIRMATION_SEED_COUNT = 32
BALANCE_MARGINS = (0.10, 0.20, 0.30)
FLOOR = 0.15
BALANCE_ACTIVATION = 0.35
INFLUENCE_SCALE = 32.0
CONFIDENCE_Z = 1.0
BOOTSTRAP_REPLICATES = 5000


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


def _model(path: Path, expected: str) -> DecisionValueMemory:
    if _sha256(path) != expected:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(state["config"])
    config["horizons"] = tuple(config.get("horizons", (4, 8)))
    result = DecisionValueMemory(DecisionValueMemoryConfig(**config))
    result.load_state_dict(state)
    return result


def _source(extra: list[Path]) -> dict[str, str]:
    paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve(), *extra]
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
    delta = np.asarray([a[s] - b[s] for s in sorted(set(a) & set(b))], dtype=float)
    return {"paired_environment_seeds": len(delta), "mean": float(np.mean(delta)),
            "ci95": assay._paired_ci(delta, seed), "win_rate": float(np.mean(delta > 0.0))}


def _label(policy: str, margin: float) -> str:
    return f"{policy}_margin{margin:.2f}"


def _episode(seed: int, regime: int, condition: str, policy: str, margin: float | None,
             core: CoreRuntime, structured: DecisionValueMemory, gru: DecisionValueMemory,
             absolute: DecisionValueMemory) -> dict[str, Any]:
    assay.INFLUENCE_SCALE = INFLUENCE_SCALE
    if policy in ("contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute"):
        return guard._run_episode(seed, regime, condition, policy, FLOOR, core, structured, gru, absolute,
                                  balance_margin=margin, balance_activation=BALANCE_ACTIVATION)
    return guard._run_episode(seed, regime, condition, policy, None, core, structured, gru, absolute)


def develop(output: Path, seed_count: int = DEVELOPMENT_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output exists; preserving prior evidence: {output}")
    if seed_count < 8:
        raise ValueError("development requires at least eight environment seeds")
    train = _receipt(TRAINING_DIR / "training_receipt.json")
    hashes = train["model_sha256"]
    structured = _model(TRAINING_DIR / "structured_contrastive_model.pt", hashes["structured_contrastive_model.pt"])
    gru = _model(TRAINING_DIR / "matched_gru_contrastive_control_model.pt", hashes["matched_gru_contrastive_control_model.pt"])
    absolute = _model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", hashes["matched_structured_absolute_control_model.pt"])
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + seed_count))
    output.mkdir(parents=True)
    sources = _source([
        ROOT / "tools/assay_alias_tidepool_v2_predictive_guard.py",
        ROOT / "tools/assay_alias_tidepool_v2_structured_dvgem.py",
        ROOT / "tools/train_alias_tidepool_v2_contrastive_dvgem.py",
    ])
    protocol = _protocol(output, {
        "schema": "poseidon-alias-v2-balance-deadband-development-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "development_seed_range": [seeds[0], seeds[-1]],
        "conditions": assay.CONDITIONS,
        "regimes": [0, 1],
        "max_steps": assay.MAX_STEPS,
        "guard": {"strategy": "one-step conservative resource forecast plus imbalance deadband",
                  "floor": FLOOR, "balance_margin_grid": list(BALANCE_MARGINS),
                  "balance_margin_activation": BALANCE_ACTIVATION,
                  "maximum_resource_drain": guard.MAXIMUM_RESOURCE_DRAIN,
                  "minimum_resource_yield": guard.MINIMUM_RESOURCE_YIELD,
                  "sensor_noise_bound": assay.SENSOR_NOISE_BOUND, "confidence_z": CONFIDENCE_Z},
        "influence_scale": INFLUENCE_SCALE,
        "selection_rule": "require full survival, minimum energy/hydration >= 0.15, and positive paired lower 95% interval versus erased memory in every condition; among passing settings maximize reward; otherwise maximize worst-condition minimum resource",
        "training_receipt_sha256": train["receipt_sha256"],
        "model_sha256": hashes,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }, sources)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    old_scale = assay.INFLUENCE_SCALE
    started = time.monotonic()
    try:
        policies = ("contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute")
        rows = []
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for margin in BALANCE_MARGINS:
                        for policy in policies:
                            row = _episode(seed, regime, condition, policy, margin, core, structured, gru, absolute)
                            row.update({"policy": _label(policy, margin), "model_role": policy, "balance_margin": margin,
                                        "balance_activation": BALANCE_ACTIVATION, "floor": FLOOR})
                            rows.append(row)
                    for policy in ("reactive_balancer", "cue_memory_balancer", "frozen_tidal"):
                        rows.append(_episode(seed, regime, condition, policy, None, core, structured, gru, absolute))
            print(json.dumps({"phase": "balance_margin_development_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        results = {}
        for margin in BALANCE_MARGINS:
            labels = {policy: _label(policy, margin) for policy in policies}
            by_condition = {}
            rewards = []
            minimum_resource = 1.0
            safe = True
            value = True
            for ci, condition in enumerate(assay.CONDITIONS):
                stats = {policy: assay._condition_summary(grouped[(condition, labels[policy])]) for policy in policies}
                vs_erased = _paired(grouped[(condition, labels["contrastive_intact"])], grouped[(condition, labels["contrastive_erased"])], "total_reward", DEVELOPMENT_SEED_START + ci)
                vs_absolute = _paired(grouped[(condition, labels["contrastive_intact"])], grouped[(condition, labels["matched_absolute"])], "total_reward", DEVELOPMENT_SEED_START + ci + 10)
                by_condition[condition] = {"policies": stats, "intact_vs_erased": vs_erased, "intact_vs_absolute": vs_absolute}
                intact = stats["contrastive_intact"]
                rewards.append(intact["mean_reward"])
                minimum_resource = min(minimum_resource, intact["minimum_energy"], intact["minimum_hydration"])
                safe = safe and intact["survival_rate"] == 1.0 and intact["minimum_energy"] >= 0.15 and intact["minimum_hydration"] >= 0.15
                value = value and vs_erased["ci95"][0] > 0.0
            results[f"margin_{margin:.2f}"] = {"balance_margin": margin, "passes_safety_floor": safe,
                "positive_memory_value_ci_all_conditions": value, "worst_condition_minimum_resource": minimum_resource,
                "mean_reward_across_conditions": float(np.mean(rewards)), "by_condition": by_condition}
        eligible = [k for k, v in results.items() if v["passes_safety_floor"] and v["positive_memory_value_ci_all_conditions"]]
        if eligible:
            selected = max(eligible, key=lambda k: results[k]["mean_reward_across_conditions"])
            basis = "highest mean reward among candidates passing development safety and memory-value gates"
        else:
            selected = max(results, key=lambda k: (results[k]["worst_condition_minimum_resource"], results[k]["mean_reward_across_conditions"]))
            basis = "best worst-resource development fallback; primary gates were not all met"

        replay_checks = 0
        for row in rows:
            if row.get("model_role") in policies:
                margin = float(row["balance_margin"])
                model_role = row["model_role"]
                replay = _episode(row["seed"], row["regime"], row["condition"], model_role, margin, core, structured, gru, absolute)
                replay.update({"policy": row["policy"], "model_role": model_role, "balance_margin": margin,
                               "balance_activation": BALANCE_ACTIVATION, "floor": FLOOR})
            else:
                replay = _episode(row["seed"], row["regime"], row["condition"], row["policy"], None, core, structured, gru, absolute)
            if replay != row:
                raise RuntimeError(f"balance-margin replay mismatch: {row['policy']} seed={row['seed']}")
            replay_checks += 1
        _ensure(sources)
        selected_margin = results[selected]["balance_margin"]
        selection = {"schema": "poseidon-balance-deadband-selection-v1", "selected": {"balance_margin": selected_margin,
                    "balance_activation": BALANCE_ACTIVATION, "floor": FLOOR},
                    "selected_label": selected, "selection_basis": basis, "development_receipt_sha256": None}
        receipt = {"schema": "poseidon-alias-v2-balance-deadband-development-receipt-v1",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol["protocol_sha256"],
                   "training_receipt_sha256": train["receipt_sha256"], "model_sha256": hashes, "results": results,
                   "selected": selection["selected"], "selected_label": selected, "selection_basis": basis,
                   "episodes": rows, "complete_episode_replay_checks": replay_checks, "replay_failures": 0,
                   "source_sha256": sources, "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
                   "elapsed_seconds": round(time.monotonic() - started, 3), "promotion": False}
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        selection["development_receipt_sha256"] = receipt["receipt_sha256"]
        selection["selection_sha256"] = hashlib.sha256(canonical_bytes(selection)).hexdigest()
        (output / "development_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "selected": selection["selected"],
                   "selected_label": selected, "selection_basis": basis, "results": results,
                   "replay_checks": replay_checks, "receipt_sha256": receipt["receipt_sha256"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(previous)


def confirm(output: Path, seed_count: int = CONFIRMATION_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output exists; preserving prior evidence: {output}")
    if seed_count < 16:
        raise ValueError("confirmation requires at least sixteen environment seeds")
    train = _receipt(TRAINING_DIR / "training_receipt.json")
    dev_dir = ROOT / "runs/research/decision_value_memory_v2_balance_margin_adaptive_development_20261014"
    development = _receipt(dev_dir / "development_receipt.json")
    selection = json.loads((dev_dir / "selection.json").read_text(encoding="utf-8"))
    selection_body = dict(selection)
    selection_sha = selection_body.pop("selection_sha256", None)
    if selection.get("development_receipt_sha256") != development["receipt_sha256"] or hashlib.sha256(canonical_bytes(selection_body)).hexdigest() != selection_sha:
        raise ValueError("balance-margin selection is not bound to its development receipt")
    margin = float(selection["selected"]["balance_margin"])
    balance_activation = float(selection["selected"]["balance_activation"])
    hashes = train["model_sha256"]
    structured = _model(TRAINING_DIR / "structured_contrastive_model.pt", hashes["structured_contrastive_model.pt"])
    gru = _model(TRAINING_DIR / "matched_gru_contrastive_control_model.pt", hashes["matched_gru_contrastive_control_model.pt"])
    absolute = _model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", hashes["matched_structured_absolute_control_model.pt"])
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + seed_count))
    policies = ("frozen_tidal", "reactive_balancer", "cue_memory_balancer", "contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute")
    output.mkdir(parents=True)
    sources = _source([
        ROOT / "tools/assay_alias_tidepool_v2_predictive_guard.py",
        ROOT / "tools/assay_alias_tidepool_v2_structured_dvgem.py",
        ROOT / "tools/train_alias_tidepool_v2_contrastive_dvgem.py",
    ])
    protocol = _protocol(output, {"schema": "poseidon-alias-v2-balance-deadband-confirmation-v1",
        "world_version": ALIAS_WORLD_V2_VERSION, "fresh_seed_range": [seeds[0], seeds[-1]],
        "excluded_seed_ranges": {"training": [743030000, 743030011],
            "balance_margin_development": [DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + DEVELOPMENT_SEED_COUNT - 1],
            "prior_nonadaptive_balance_development": [743150000, 743150007],
            "predictive_guard_development": [743130000, 743130007],
            "contrastive_development": [743110000, 743110007],
            "contrastive_confirmation": [743120000, 743120031]},
        "conditions": assay.CONDITIONS, "regimes": [0, 1], "max_steps": assay.MAX_STEPS,
        "guard": {"strategy": "one-step conservative forecast plus balance deadband", "floor": FLOOR,
            "balance_margin": margin, "balance_activation": balance_activation,
            "maximum_resource_drain": guard.MAXIMUM_RESOURCE_DRAIN,
            "minimum_resource_yield": guard.MINIMUM_RESOURCE_YIELD, "sensor_noise_bound": assay.SENSOR_NOISE_BOUND,
            "confidence_z": CONFIDENCE_Z}, "influence_scale": INFLUENCE_SCALE, "arms": list(policies),
        "paired_unit": "environment seed averaged across both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
        "training_receipt_sha256": train["receipt_sha256"], "development_receipt_sha256": development["receipt_sha256"],
        "selection_sha256": selection_sha, "model_sha256": hashes, "core_checkpoint_path": str(core_path), "promotion": False}, sources)

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    old_scale = assay.INFLUENCE_SCALE
    assay.INFLUENCE_SCALE = INFLUENCE_SCALE
    started = time.monotonic()
    try:
        rows = []
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for policy in policies:
                        if policy in ("contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute"):
                            row = _episode(seed, regime, condition, policy, margin, core, structured, gru, absolute)
                            row.update({"policy": policy, "model_role": policy, "balance_margin": margin,
                                        "balance_activation": balance_activation, "floor": FLOOR})
                        else:
                            row = _episode(seed, regime, condition, policy, None, core, structured, gru, absolute)
                        rows.append(row)
            print(json.dumps({"phase": "balance_margin_confirmation_condition_complete", "condition": condition}), flush=True)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        by_condition = {condition: {policy: assay._condition_summary(grouped[(condition, policy)]) for policy in policies}
                        for condition in assay.CONDITIONS}
        comparisons = {}
        for ci, condition in enumerate(assay.CONDITIONS):
            pairs = {}
            candidate = grouped[(condition, "contrastive_intact")]
            for j, other in enumerate(policies):
                if other != "contrastive_intact":
                    pairs[f"contrastive_intact_vs_{other}"] = _paired(candidate, grouped[(condition, other)], "total_reward", CONFIRMATION_SEED_START + ci * 17 + j)
            comparisons[condition] = pairs
        replay_checks = 0
        for row in rows:
            if row["policy"] in ("contrastive_intact", "contrastive_erased", "matched_gru", "matched_absolute"):
                replay = _episode(row["seed"], row["regime"], row["condition"], row["policy"], margin, core, structured, gru, absolute)
                replay.update({"policy": row["policy"], "model_role": row["policy"], "balance_margin": margin,
                               "balance_activation": balance_activation, "floor": FLOOR})
            else:
                replay = _episode(row["seed"], row["regime"], row["condition"], row["policy"], None, core, structured, gru, absolute)
            if replay != row:
                raise RuntimeError(f"fresh balance-margin replay mismatch: {row['policy']} seed={row['seed']}")
            replay_checks += 1
        _ensure(sources)
        candidate = {condition: by_condition[condition]["contrastive_intact"] for condition in assay.CONDITIONS}
        gates = {"full_survival_all_conditions": all(candidate[c]["survival_rate"] == 1.0 for c in assay.CONDITIONS),
            "minimum_resource_at_least_0_15_all_conditions": all(candidate[c]["minimum_energy"] >= 0.15 and candidate[c]["minimum_hydration"] >= 0.15 for c in assay.CONDITIONS),
            "positive_memory_value_ci_vs_erased_all_conditions": all(comparisons[c]["contrastive_intact_vs_contrastive_erased"]["ci95"][0] > 0.0 for c in assay.CONDITIONS),
            "positive_value_ci_vs_absolute_all_conditions": all(comparisons[c]["contrastive_intact_vs_matched_absolute"]["ci95"][0] > 0.0 for c in assay.CONDITIONS),
            "beats_reactive_in_all_conditions": all(comparisons[c]["contrastive_intact_vs_reactive_balancer"]["mean"] > 0.0 for c in assay.CONDITIONS),
            "beats_matched_gru_in_all_conditions": all(comparisons[c]["contrastive_intact_vs_matched_gru"]["mean"] > 0.0 for c in assay.CONDITIONS),
            "promoted": False}
        receipt = {"schema": "poseidon-alias-v2-balance-deadband-confirmation-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": train["receipt_sha256"], "development_receipt_sha256": development["receipt_sha256"],
            "selection_sha256": selection_sha, "model_sha256": hashes, "by_condition": by_condition,
            "paired_comparisons": comparisons, "gates": gates, "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0, "episodes": rows, "source_sha256": sources,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"], "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": ["Fresh confirmation on one synthetic benchmark; not external replication.",
                "The resource forecast uses hand-specified bounds from the versioned synthetic world.",
                "The controller remains opt-in and is not promoted by this receipt."]}
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        (output / "confirmation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "selected_guard": {"floor": FLOOR,
                   "balance_margin": margin, "balance_activation": balance_activation},
            "by_condition": by_condition, "paired_comparisons": comparisons, "gates": gates,
            "replay_checks": replay_checks, "receipt_sha256": receipt["receipt_sha256"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(previous)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "confirm"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int)
    args = parser.parse_args()
    if args.phase == "develop":
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_balance_margin_adaptive_development_20261014"
        result = develop(output, args.seeds or DEVELOPMENT_SEED_COUNT)
    else:
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_balance_margin_adaptive_confirmation_20261014"
        result = confirm(output, args.seeds or CONFIRMATION_SEED_COUNT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
