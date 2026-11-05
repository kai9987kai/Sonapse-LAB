"""Calibrate and freshly confirm paired-advantage DV-GEM against matched controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.assay_alias_tidepool_v2_structured_dvgem as assay
from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes
from poseidon.research.decision_value_memory import DecisionValueMemory, DecisionValueMemoryConfig
from tools.train_alias_tidepool_v2_contrastive_dvgem import SCORER_TARGET_TYPE


TRAINING_DIR = ROOT / "runs/research/decision_value_memory_v2_contrastive_pilot_20261014"
DEVELOPMENT_SEED_START = 743110000
DEVELOPMENT_SEED_COUNT = 8
CONFIRMATION_SEED_START = 743120000
CONFIRMATION_SEED_COUNT = 32
SCALES = (4.0, 8.0, 16.0, 32.0)
THRESHOLDS = (0.40, 0.50)
CONFIDENCE_Z = 1.0
BOOTSTRAP_REPLICATES = 5000


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


def _load_model(path: Path, expected_sha: str) -> DecisionValueMemory:
    if _sha256(path) != expected_sha:
        raise ValueError(f"model hash mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(state["config"])
    config["horizons"] = tuple(config.get("horizons", (4, 8)))
    memory = DecisionValueMemory(DecisionValueMemoryConfig(**config))
    memory.load_state_dict(state)
    return memory


def _source_manifest(extra: list[Path]) -> dict[str, str]:
    paths = sorted((ROOT / "poseidon").rglob("*.py")) + [Path(__file__).resolve(), *extra]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _write_protocol(output: Path, protocol: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    protocol["source_sha256"] = sources
    protocol["core_checkpoint_sha256"] = _sha256(Path(protocol["core_checkpoint_path"]))
    protocol["protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import zipfile
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sources:
            archive.write(ROOT / relative, relative)
    return protocol


def _ensure_sources(sources: dict[str, str]) -> None:
    for relative, digest in sources.items():
        if _sha256(ROOT / relative) != digest:
            raise RuntimeError(f"source changed during assay; evidence retained: {relative}")


def _paired_delta(candidate: list[dict[str, Any]], comparator: list[dict[str, Any]], metric: str, seed: int) -> dict[str, Any]:
    a = assay._seed_means(candidate, metric)
    b = assay._seed_means(comparator, metric)
    paired = np.asarray([a[s] - b[s] for s in sorted(set(a) & set(b))], dtype=float)
    return {
        "paired_environment_seeds": int(len(paired)),
        "mean": float(np.mean(paired)),
        "ci95": assay._paired_ci(paired, seed),
        "win_rate": float(np.mean(paired > 0.0)),
    }


def _label(scale: float, threshold: float, role: str = "contrastive") -> str:
    return f"{role}_s{scale:g}_t{threshold:g}"


def develop(output: Path, seed_count: int = DEVELOPMENT_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving evidence: {output}")
    if seed_count < 8:
        raise ValueError("development seed count must be at least eight")
    train = _load_receipt(TRAINING_DIR / "training_receipt.json")
    hashes = train["model_sha256"]
    structured = _load_model(TRAINING_DIR / "structured_contrastive_model.pt", hashes["structured_contrastive_model.pt"])
    absolute = _load_model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", hashes["matched_structured_absolute_control_model.pt"])
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    seeds = list(range(DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + seed_count))
    variants = [{"scale": s, "threshold": t, "confidence_z": CONFIDENCE_Z} for s in SCALES for t in THRESHOLDS]
    output.mkdir(parents=True)
    extras = [
        ROOT / "tools/train_alias_tidepool_v2_contrastive_dvgem.py",
        ROOT / "tools/assay_alias_tidepool_v2_structured_dvgem.py",
    ]
    sources = _source_manifest(extras)
    protocol = _write_protocol(output, {
        "schema": "poseidon-alias-v2-counterfactual-advantage-development-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "development_seed_range": [seeds[0], seeds[-1]],
        "conditions": assay.CONDITIONS,
        "regimes": [0, 1],
        "max_steps": assay.MAX_STEPS,
        "scorer_target_type": SCORER_TARGET_TYPE,
        "grid": variants,
        "sensor_noise_bound": assay.SENSOR_NOISE_BOUND,
        "selection_rule": {
            "eligible": "100% survival, minimum energy/hydration >= 0.15, and positive paired 95% interval versus erased memory in every condition",
            "select": "maximum unweighted mean reward among eligible settings",
            "fallback": "if none eligible, maximize scarcity_noise paired reward versus erased among settings meeting survival/resource floors; otherwise maximize worst resource",
        },
        "training_receipt_sha256": train["receipt_sha256"],
        "model_sha256": hashes,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }, sources)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    old_scale = assay.INFLUENCE_SCALE
    try:
        rows = []
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for variant in variants:
                        assay.INFLUENCE_SCALE = variant["scale"]
                        label = _label(variant["scale"], variant["threshold"])
                        row = assay._episode(
                            seed, regime, condition, "development_candidate", core, structured,
                            threshold=variant["threshold"], confidence_z=variant["confidence_z"],
                        )
                        row.update({"policy": label, **variant, "model_role": "contrastive"})
                        rows.append(row)
                        absolute_label = _label(variant["scale"], variant["threshold"], "absolute")
                        absolute_row = assay._episode(
                            seed, regime, condition, "development_candidate", core, absolute,
                            threshold=variant["threshold"], confidence_z=variant["confidence_z"],
                        )
                        absolute_row.update({"policy": absolute_label, **variant, "model_role": "absolute-control"})
                        rows.append(absolute_row)
                    for threshold in THRESHOLDS:
                        erased_label = f"erased_t{threshold:g}"
                        erased = assay._episode(
                            seed, regime, condition, "structured_erased_guarded", core, structured,
                            threshold=threshold, confidence_z=CONFIDENCE_Z,
                        )
                        erased["policy"] = erased_label
                        erased.update({"scale": 0.0, "threshold": threshold, "confidence_z": CONFIDENCE_Z, "model_role": "erased-control"})
                        rows.append(erased)
                    for policy in ("reactive_balancer", "cue_memory_balancer", "frozen_tidal"):
                        rows.append(assay._episode(seed, regime, condition, policy, core))
            print(json.dumps({"phase": "contrastive_development_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        candidate_results = {}
        labels = [_label(v["scale"], v["threshold"]) for v in variants]
        for variant, label in zip(variants, labels):
            by_condition = {}
            safe = True
            value_vs_erased = True
            overall_reward = []
            combined_delta = None
            for ci, condition in enumerate(assay.CONDITIONS):
                candidate = grouped[(condition, label)]
                erased = grouped[(condition, f"erased_t{variant['threshold']:g}")]
                stats = assay._condition_summary(candidate)
                delta = _paired_delta(candidate, erased, "total_reward", DEVELOPMENT_SEED_START + ci)
                absolute_rows = grouped[(condition, _label(variant["scale"], variant["threshold"], "absolute"))]
                abs_delta = _paired_delta(candidate, absolute_rows, "total_reward", DEVELOPMENT_SEED_START + ci + 40)
                by_condition[condition] = {"candidate": stats, "vs_erased": delta, "vs_absolute_control": abs_delta}
                overall_reward.append(stats["mean_reward"])
                safe = safe and stats["survival_rate"] == 1.0 and stats["minimum_energy"] >= 0.15 and stats["minimum_hydration"] >= 0.15
                value_vs_erased = value_vs_erased and delta["ci95"][0] > 0.0
                if condition == "scarcity_noise":
                    combined_delta = delta["mean"]
            candidate_results[label] = {
                **variant,
                "mean_reward_across_conditions": float(np.mean(overall_reward)),
                "passes_safety_floor": safe,
                "positive_memory_value_ci_all_conditions": value_vs_erased,
                "scarcity_noise_delta_vs_erased": float(combined_delta or 0.0),
                "by_condition": by_condition,
            }
        eligible = [name for name, result in candidate_results.items() if result["passes_safety_floor"] and result["positive_memory_value_ci_all_conditions"]]
        if eligible:
            selected = max(eligible, key=lambda name: candidate_results[name]["mean_reward_across_conditions"])
            basis = "highest mean reward among safety-passing settings with positive paired memory-value intervals in every condition"
        else:
            safe = [name for name, result in candidate_results.items() if result["passes_safety_floor"]]
            pool = safe or list(candidate_results)
            selected = max(
                pool,
                key=lambda name: (
                    candidate_results[name]["scarcity_noise_delta_vs_erased"],
                    candidate_results[name]["mean_reward_across_conditions"],
                    -candidate_results[name]["scale"],
                ),
            )
            basis = "best scarcity-noise paired-value fallback; no setting met every primary development gate"

        replay_checks = 0
        for row in rows:
            policy = row["policy"]
            if policy.startswith("contrastive_") or policy.startswith("absolute_"):
                assay.INFLUENCE_SCALE = float(row["scale"])
                memory = structured if row["model_role"] == "contrastive" else absolute
                replay = assay._episode(row["seed"], row["regime"], row["condition"], "development_candidate", core, memory,
                                        threshold=float(row["threshold"]), confidence_z=float(row["confidence_z"]))
                replay.update({"policy": policy, "scale": row["scale"], "threshold": row["threshold"],
                               "confidence_z": row["confidence_z"], "model_role": row["model_role"]})
            elif policy.startswith("erased_"):
                replay = assay._episode(row["seed"], row["regime"], row["condition"], "structured_erased_guarded", core, structured,
                                        threshold=float(row["threshold"]), confidence_z=float(row["confidence_z"]))
                replay.update({"policy": policy, "scale": row["scale"], "threshold": row["threshold"],
                               "confidence_z": row["confidence_z"], "model_role": row["model_role"]})
            else:
                replay = assay._episode(row["seed"], row["regime"], row["condition"], policy, core)
            if replay != row:
                raise RuntimeError(f"development replay mismatch: {policy} seed={row['seed']} condition={row['condition']}")
            replay_checks += 1
        _ensure_sources(sources)
        selected_result = candidate_results[selected]
        selection = {
            "schema": "poseidon-alias-v2-counterfactual-advantage-selection-v1",
            "selected": {"scale": selected_result["scale"], "threshold": selected_result["threshold"], "confidence_z": selected_result["confidence_z"]},
            "selected_label": selected,
            "selection_basis": basis,
            "development_receipt_sha256": None,
        }
        receipt = {
            "schema": "poseidon-alias-v2-counterfactual-advantage-development-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": train["receipt_sha256"],
            "model_sha256": hashes,
            "candidate_results": candidate_results,
            "selected": selection["selected"],
            "selected_label": selected,
            "selection_basis": basis,
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
            "episodes": rows,
            "source_sha256": sources,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "promotion": False,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        selection["development_receipt_sha256"] = receipt["receipt_sha256"]
        selection["selection_sha256"] = hashlib.sha256(canonical_bytes(selection)).hexdigest()
        (output / "development_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "selected": selection["selected"],
                   "selected_label": selected, "selection_basis": basis, "candidate_results": candidate_results,
                   "replay_checks": replay_checks, "receipt_sha256": receipt["receipt_sha256"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(previous_threads)


def confirm(output: Path, seed_count: int = CONFIRMATION_SEED_COUNT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving evidence: {output}")
    if seed_count < 16:
        raise ValueError("fresh confirmation requires at least sixteen environment seeds")
    train = _load_receipt(TRAINING_DIR / "training_receipt.json")
    dev_dir = ROOT / "runs/research/decision_value_memory_v2_contrastive_development_highscale_20261014"
    development = _load_receipt(dev_dir / "development_receipt.json")
    selection = json.loads((dev_dir / "selection.json").read_text(encoding="utf-8"))
    selection_body = dict(selection)
    selection_sha = selection_body.pop("selection_sha256", None)
    if selection.get("development_receipt_sha256") != development["receipt_sha256"] or hashlib.sha256(canonical_bytes(selection_body)).hexdigest() != selection_sha:
        raise ValueError("development selection failed its receipt binding or self-hash")
    scale = float(selection["selected"]["scale"])
    threshold = float(selection["selected"]["threshold"])
    confidence_z = float(selection["selected"]["confidence_z"])
    hashes = train["model_sha256"]
    structured = _load_model(TRAINING_DIR / "structured_contrastive_model.pt", hashes["structured_contrastive_model.pt"])
    gru = _load_model(TRAINING_DIR / "matched_gru_contrastive_control_model.pt", hashes["matched_gru_contrastive_control_model.pt"])
    absolute = _load_model(TRAINING_DIR / "matched_structured_absolute_control_model.pt", hashes["matched_structured_absolute_control_model.pt"])
    seeds = list(range(CONFIRMATION_SEED_START, CONFIRMATION_SEED_START + seed_count))
    policies = (
        "frozen_tidal", "reactive_balancer", "cue_memory_balancer",
        "contrastive_intact", "contrastive_erased", "matched_gru_contrastive", "matched_absolute_control",
    )
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    output.mkdir(parents=True)
    sources = _source_manifest([
        ROOT / "tools/train_alias_tidepool_v2_contrastive_dvgem.py",
        ROOT / "tools/assay_alias_tidepool_v2_structured_dvgem.py",
    ])
    protocol = _write_protocol(output, {
        "schema": "poseidon-alias-v2-counterfactual-advantage-confirmation-v1",
        "world_version": ALIAS_WORLD_V2_VERSION,
        "fresh_seed_range": [seeds[0], seeds[-1]],
        "excluded_seed_ranges": {
            "training": [743030000, 743030011],
            "scale_guard_development": [DEVELOPMENT_SEED_START, DEVELOPMENT_SEED_START + DEVELOPMENT_SEED_COUNT - 1],
            "prior_low_scale_development": [743090000, 743090007],
            "structured_absolute_pilot": [743020000, 743020011],
            "structured_confirmation": [743080000, 743080063],
        },
        "conditions": assay.CONDITIONS,
        "regimes": [0, 1],
        "max_steps": assay.MAX_STEPS,
        "selected_policy": {"scale": scale, "threshold": threshold, "sensor_noise_bound": assay.SENSOR_NOISE_BOUND, "confidence_z": confidence_z},
        "scorer_target_type": SCORER_TARGET_TYPE,
        "arms": list(policies),
        "paired_unit": "environment seed averaged over both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
        "training_receipt_sha256": train["receipt_sha256"],
        "development_receipt_sha256": development["receipt_sha256"],
        "selection_sha256": selection_sha,
        "model_sha256": hashes,
        "core_checkpoint_path": str(core_path),
        "promotion": False,
    }, sources)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    old_scale = assay.INFLUENCE_SCALE
    assay.INFLUENCE_SCALE = scale
    model_by_policy = {
        "contrastive_intact": (structured, "structured_intact_guarded"),
        "contrastive_erased": (structured, "structured_erased_guarded"),
        "matched_gru_contrastive": (gru, "matched_gru_guarded"),
        "matched_absolute_control": (absolute, "structured_intact_guarded"),
    }
    try:
        rows = []
        for condition in assay.CONDITIONS:
            for seed in seeds:
                for regime in (0, 1):
                    for policy in policies:
                        if policy in model_by_policy:
                            memory, assay_policy = model_by_policy[policy]
                            row = assay._episode(seed, regime, condition, assay_policy, core, memory,
                                                 threshold=threshold, confidence_z=confidence_z)
                        else:
                            row = assay._episode(seed, regime, condition, policy, core)
                        row["policy"] = policy
                        row.update({"scale": scale, "threshold": threshold, "confidence_z": confidence_z})
                        rows.append(row)
            print(json.dumps({"phase": "contrastive_confirmation_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["policy"])].append(row)
        by_condition = {condition: {policy: assay._condition_summary(grouped[(condition, policy)]) for policy in policies}
                        for condition in assay.CONDITIONS}
        comparisons = {}
        for ci, condition in enumerate(assay.CONDITIONS):
            pairs = {}
            candidate = grouped[(condition, "contrastive_intact")]
            for j, comparator in enumerate(policies):
                if comparator == "contrastive_intact":
                    continue
                pairs[f"contrastive_intact_vs_{comparator}"] = _paired_delta(
                    candidate, grouped[(condition, comparator)], "total_reward", CONFIRMATION_SEED_START + ci * 19 + j
                )
            comparisons[condition] = pairs

        replay_checks = 0
        for row in rows:
            policy = row["policy"]
            if policy in model_by_policy:
                memory, assay_policy = model_by_policy[policy]
                replay = assay._episode(row["seed"], row["regime"], row["condition"], assay_policy, core, memory,
                                        threshold=threshold, confidence_z=confidence_z)
            else:
                replay = assay._episode(row["seed"], row["regime"], row["condition"], policy, core)
            replay["policy"] = policy
            replay.update({"scale": scale, "threshold": threshold, "confidence_z": confidence_z})
            if replay != row:
                raise RuntimeError(f"fresh confirmation replay mismatch: {policy} seed={row['seed']} condition={row['condition']}")
            replay_checks += 1
        _ensure_sources(sources)

        candidate = {condition: by_condition[condition]["contrastive_intact"] for condition in assay.CONDITIONS}
        gates = {
            "full_survival_all_conditions": all(candidate[c]["survival_rate"] == 1.0 for c in assay.CONDITIONS),
            "minimum_resource_at_least_0_15_all_conditions": all(candidate[c]["minimum_energy"] >= 0.15 and candidate[c]["minimum_hydration"] >= 0.15 for c in assay.CONDITIONS),
            "positive_memory_value_ci_vs_erased_all_conditions": all(comparisons[c]["contrastive_intact_vs_contrastive_erased"]["ci95"][0] > 0.0 for c in assay.CONDITIONS),
            "positive_advantage_objective_ci_vs_absolute_all_conditions": all(comparisons[c]["contrastive_intact_vs_matched_absolute_control"]["ci95"][0] > 0.0 for c in assay.CONDITIONS),
            "beats_matched_gru_in_all_conditions": all(comparisons[c]["contrastive_intact_vs_matched_gru_contrastive"]["mean"] > 0.0 for c in assay.CONDITIONS),
            "promoted": False,
        }
        receipt = {
            "schema": "poseidon-alias-v2-counterfactual-advantage-confirmation-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "training_receipt_sha256": train["receipt_sha256"],
            "development_receipt_sha256": development["receipt_sha256"],
            "selection_sha256": selection_sha,
            "model_sha256": hashes,
            "by_condition": by_condition,
            "paired_comparisons": comparisons,
            "gates": gates,
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
            "episodes": rows,
            "source_sha256": sources,
            "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": [
                "Fresh local confirmation on one synthetic world; not an independent implementation or external replication.",
                "The belief map and reserve rule are hand-specified; only the scorer was trained.",
                "A successful synthetic gate would not establish real-world safety or general intelligence.",
                "The candidate remains opt-in; promotion is explicitly false.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        (output / "confirmation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "selected": protocol["selected_policy"],
                   "by_condition": by_condition, "paired_comparisons": comparisons, "gates": gates,
                   "replay_checks": replay_checks, "receipt_sha256": receipt["receipt_sha256"]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
    finally:
        assay.INFLUENCE_SCALE = old_scale
        torch.set_num_threads(previous_threads)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "confirm"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int)
    args = parser.parse_args()
    if args.phase == "develop":
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_contrastive_development_highscale_20261014"
        result = develop(output, args.seeds or DEVELOPMENT_SEED_COUNT)
    else:
        output = args.output or ROOT / "runs/research/decision_value_memory_v2_contrastive_confirmation_highscale_20261014"
        result = confirm(output, args.seeds or CONFIRMATION_SEED_COUNT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
