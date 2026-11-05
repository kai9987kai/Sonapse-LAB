"""Calibrate opt-in DV-GEM influence scale on a development-only seed block."""
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
SCALES = (0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0)
BOOTSTRAP_REPLICATES = 5000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_summary(
    seed: int,
    regime: int,
    condition: str,
    scale: float,
    memory: DecisionValueMemory,
    core: CoreRuntime,
) -> dict[str, Any]:
    row = _run_episode(seed, regime, condition, "intact", memory, [], core, influence_scale=scale)
    trajectory = row.pop("trajectory")
    row["scale"] = float(scale)
    row["trajectory_sha256"] = hashlib.sha256(canonical_bytes(trajectory)).hexdigest()
    return row


def _paired_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists; preserving evidence: {output}")
    pilot_dir = ROOT / "runs/research/decision_value_memory_v2_pilot_20261014"
    model_path = pilot_dir / "memory_model.pt"
    pilot_receipt_path = pilot_dir / "benchmark_receipt.json"
    if not model_path.is_file() or not pilot_receipt_path.is_file():
        raise FileNotFoundError("the completed v2 DV-GEM pilot checkpoint and receipt are required")

    pilot_receipt = json.loads(pilot_receipt_path.read_text(encoding="utf-8"))
    model_state = torch.load(model_path, map_location="cpu", weights_only=True)
    config_values = dict(model_state["config"])
    config_values["horizons"] = tuple(config_values.get("horizons", (4, 8)))
    memory = DecisionValueMemory(DecisionValueMemoryConfig(**config_values))
    memory.load_state_dict(model_state)
    model_sha = _sha256(model_path)
    if model_sha != pilot_receipt.get("model_sha256"):
        raise ValueError("pilot model checkpoint does not match the source-bound pilot receipt")

    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)
    eval_seeds = list(range(SEED_START, SEED_START + SEED_COUNT))
    output.mkdir(parents=True)
    source_paths = sorted((ROOT / "poseidon").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "tools/run_alias_tidepool_v2_dvgem_pilot.py",
    ]
    source_hashes = {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths}
    protocol = {
        "schema": "poseidon-decision-memory-v2-scale-calibration-v1",
        "development_seed_range": [SEED_START, SEED_START + SEED_COUNT - 1],
        "seed_block_role": "reuse the prior 16-seed v2 pilot evaluation block for influence-scale selection; this is development data, not confirmation",
        "scales": list(SCALES),
        "conditions": CONDITIONS,
        "regimes": [0, 1],
        "max_steps": MAX_STEPS,
        "selection_rule": "among scales whose survival is no worse than scale zero in every condition, maximize unweighted mean reward across conditions; ties select the smaller scale",
        "paired_unit": "environment seed averaged across both forced regimes",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "confidence": 0.95, "resampling_unit": "environment seed"},
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
                raise RuntimeError(f"source changed during calibration; evidence retained: {relative}")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()
    try:
        ensure_source()
        episodes = []
        for condition_index, condition in enumerate(CONDITIONS):
            for scale in SCALES:
                for seed in eval_seeds:
                    for regime in (0, 1):
                        episodes.append(_episode_summary(seed, regime, condition, scale, memory, core))
            print(json.dumps({"phase": "development_condition_complete", "condition": condition}), flush=True)

        grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in episodes:
            grouped[(row["condition"], row["scale"])].append(row)
        by_scale = {}
        for scale in SCALES:
            condition_metrics = {}
            for condition in CONDITIONS:
                rows = grouped[(condition, scale)]
                condition_metrics[condition] = {
                    "episodes": len(rows),
                    "survived": int(sum(row["survived"] for row in rows)),
                    "survival_rate": float(np.mean([row["survived"] for row in rows])),
                    "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
                    "mean_gate_open_steps": float(np.mean([row["gate_open_count"] for row in rows])),
                    "mean_overrides": float(np.mean([row["override_count"] for row in rows])),
                    "episodes_with_override": int(sum(row["override_count"] > 0 for row in rows)),
                    "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
                    "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
                }
            scale_zero = {
                condition: {
                    seed: float(np.mean([row["total_reward"] for row in grouped[(condition, 0.0)] if row["seed"] == seed]))
                    for seed in eval_seeds
                }
                for condition in CONDITIONS
            }
            paired_reward = {}
            for condition_index, condition in enumerate(CONDITIONS):
                scale_rows = grouped[(condition, scale)]
                by_seed: dict[int, list[float]] = defaultdict(list)
                for row in scale_rows:
                    by_seed[row["seed"]].append(float(row["total_reward"]))
                deltas = np.asarray([
                    float(np.mean(by_seed[seed])) - scale_zero[condition][seed]
                    for seed in eval_seeds
                ])
                paired_reward[condition] = {
                    "mean_delta_vs_scale_zero": float(np.mean(deltas)),
                    "ci95_environment_seed_bootstrap": _paired_ci(deltas, BASE_SEED + int(scale * 100) + condition_index),
                    "win_rate_by_seed": float(np.mean(deltas > 0.0)),
                }
            viable = all(
                condition_metrics[condition]["survival_rate"] >= float(np.mean([
                    row["survived"] for row in grouped[(condition, 0.0)]
                ]))
                for condition in CONDITIONS
            )
            by_scale[str(scale)] = {
                "by_condition": condition_metrics,
                "paired_reward_vs_scale_zero": paired_reward,
                "survival_noninferior_to_scale_zero_all_conditions": viable,
                "unweighted_mean_reward_across_conditions": float(np.mean([
                    condition_metrics[condition]["mean_reward"] for condition in CONDITIONS
                ])),
            }

        eligible = [scale for scale in SCALES if by_scale[str(scale)]["survival_noninferior_to_scale_zero_all_conditions"]]
        selected = max(eligible, key=lambda scale: (by_scale[str(scale)]["unweighted_mean_reward_across_conditions"], -scale))

        print(json.dumps({"phase": "exact_replay", "episodes": len(episodes)}), flush=True)
        replay_checks = 0
        for row in episodes:
            replay = _episode_summary(row["seed"], row["regime"], row["condition"], row["scale"], memory, core)
            if replay != row:
                raise RuntimeError(
                    f"replay mismatch: scale={row['scale']} condition={row['condition']} "
                    f"seed={row['seed']} regime={row['regime']}"
                )
            replay_checks += 1
        ensure_source()

        receipt = {
            "schema": "poseidon-decision-memory-v2-scale-calibration-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": protocol["protocol_sha256"],
            "model_sha256": model_sha,
            "results": {
                "by_scale": by_scale,
                "selected_scale": float(selected),
                "selection_reason": "highest development reward among scales with survival noninferior to scale zero in every condition",
                "complete_episode_replay_checks": replay_checks,
                "replay_failures": 0,
                "promoted": False,
            },
            "episodes": episodes,
            "identity": {
                "source_sha256": source_hashes,
                "pilot_receipt_sha256": pilot_receipt["receipt_sha256"],
                "model_sha256": model_sha,
                "core_checkpoint_path": str(core_path),
                "core_checkpoint_sha256": protocol["core_checkpoint_sha256"],
            },
            "limits": [
                "Scale selection reuses the earlier pilot seed block and is therefore development evidence only.",
                "The scale mixes normalized policy probabilities with predicted reward differences and remains empirical, not calibrated to a safety guarantee.",
                "Fresh-seed paired confirmation is required before making a benefit claim or promoting any setting.",
            ],
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
        receipt_path = output / "calibration_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        summary = {
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "selected_scale": selected,
            "receipt_sha256": receipt["receipt_sha256"],
            "replay_checks": replay_checks,
            "by_scale": {
                scale: {
                    "viable": result["survival_noninferior_to_scale_zero_all_conditions"],
                    "mean_reward": result["unweighted_mean_reward_across_conditions"],
                    "overrides": {condition: result["by_condition"][condition]["episodes_with_override"] for condition in CONDITIONS},
                }
                for scale, result in by_scale.items()
            },
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
        default=ROOT / "runs/research/decision_value_memory_v2_scale_calibration_20261014",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
