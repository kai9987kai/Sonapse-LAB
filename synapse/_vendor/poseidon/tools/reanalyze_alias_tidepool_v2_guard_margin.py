"""Apply a stricter resource-margin gate to the recorded guard ablation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.research.contracts import canonical_bytes


INPUT = ROOT / "runs/research/decision_value_memory_v2_guard_ablation_20261014/guard_ablation_receipt.json"
PROTOCOL = INPUT.parent / "protocol.json"
OUTPUT = ROOT / "runs/research/decision_value_memory_v2_guard_margin_reanalysis_20261014.json"
RESOURCE_FLOOR = 0.15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, Any]:
    receipt = json.loads(INPUT.read_text(encoding="utf-8"))
    original_digest = receipt.get("receipt_sha256")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_bytes(receipt_body)).hexdigest() != original_digest:
        raise ValueError("source guard-ablation receipt failed its self-hash")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_digest = protocol.get("protocol_sha256")
    protocol_body = dict(protocol)
    protocol_body.pop("protocol_sha256", None)
    if hashlib.sha256(canonical_bytes(protocol_body)).hexdigest() != protocol_digest:
        raise ValueError("source guard-ablation protocol failed its self-hash")
    if receipt.get("protocol_sha256") != protocol_digest:
        raise ValueError("guard-ablation receipt and protocol digest do not match")
    conditions = list(receipt["results"]["by_condition"])
    thresholds = protocol["guard_thresholds"]
    results = {}
    eligible = []
    for threshold in thresholds:
        if threshold is None:
            continue
        label = f"guard_{float(threshold):.2f}"
        by_condition = {}
        for condition in conditions:
            rows = [
                row for row in receipt["episodes"]
                if row["condition"] == condition and row["policy"] == label
            ]
            if not rows:
                raise ValueError(f"missing episodes for {condition}/{label}")
            survived = all(bool(row["survived"]) for row in rows)
            energy_floor = min(float(row["minimum_energy"]) for row in rows)
            hydration_floor = min(float(row["minimum_hydration"]) for row in rows)
            by_condition[condition] = {
                "episodes": len(rows),
                "survived": int(sum(bool(row["survived"]) for row in rows)),
                "mean_reward": float(sum(float(row["total_reward"]) for row in rows) / len(rows)),
                "minimum_energy": energy_floor,
                "minimum_hydration": hydration_floor,
                "passes_survival": survived,
                "passes_resource_floor": energy_floor >= RESOURCE_FLOOR and hydration_floor >= RESOURCE_FLOOR,
            }
        passes = all(
            by_condition[c]["passes_survival"] and by_condition[c]["passes_resource_floor"]
            for c in conditions
        )
        results[str(threshold)] = {
            "by_condition": by_condition,
            "passes_all_survival_and_resource_gates": passes,
            "unweighted_mean_reward_across_conditions": float(
                sum(by_condition[c]["mean_reward"] for c in conditions) / len(conditions)
            ),
        }
        if passes:
            eligible.append(float(threshold))
    selected = max(eligible, key=lambda t: results[str(t)]["unweighted_mean_reward_across_conditions"]) if eligible else None
    return {
        "schema": "poseidon-alias-world-v2-guard-margin-reanalysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "input_receipt_sha256": original_digest,
            "resource_floor": RESOURCE_FLOOR,
            "selection_rule": "require 100% survival and minimum energy and hydration at or above 0.15 in every condition; then maximize unweighted mean reward",
            "reanalysis_only": True,
            "promotion": False,
        },
        "results": {
            "by_threshold": results,
            "selected_threshold": selected,
            "replay_checks_in_source_receipt": receipt["results"]["complete_episode_replay_checks"],
            "source_receipt_replay_failures": receipt["results"]["replay_failures"],
        },
        "identity": {
            "source_receipt_sha256": original_digest,
            "reanalysis_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "limits": [
            "This applies an additional descriptive gate to an existing development block; it is not independent confirmation.",
            "The 0.15 resource floor is a heuristic threshold and not a calibrated safety guarantee.",
        ],
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing reanalysis receipt: {OUTPUT}")
    result = run()
    result["receipt_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "receipt_sha256": result["receipt_sha256"],
        "selected_threshold": result["results"]["selected_threshold"],
        "by_threshold": {
            key: {
                "passes": value["passes_all_survival_and_resource_gates"],
                "mean_reward": value["unweighted_mean_reward_across_conditions"],
            }
            for key, value in result["results"]["by_threshold"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
