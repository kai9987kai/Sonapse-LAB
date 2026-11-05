"""Repeat the adapter control comparison after removing the false exposure-index assumption."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_nexus_adapter_control_assay as paired_assay


def main() -> None:
    paired_assay.SEEDS = tuple(range(74501, 74565))
    paired_assay.AGENT_SEED = 44002
    result = paired_assay.run()
    result["schema"] = "poseidon-nexus-adapter-schema-control-assay-v1"
    result["protocol"]["observation_semantics"] = (
        "exposure_index=None; canonical POMDP observation index 4 is episode progress, "
        "so the safety planner does not interpret it as exposure"
    )
    result["protocol"]["environment_seeds"] = list(paired_assay.SEEDS)
    result["identity"]["schema_assay_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["identity"]["causal_efe_source_sha256"] = hashlib.sha256(
        (ROOT / "poseidon/nexus_omega/causal_efe.py").read_bytes()
    ).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/adapter_schema_control_assay.json")
    args = parser.parse_args()
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": result["receipt_sha256"],
        "protocol": result["protocol"],
        "results": {key: value for key, value in result["results"].items() if key != "episodes"},
        "decision": result["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
