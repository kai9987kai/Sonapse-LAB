"""Run a fixed, matched CPU prediction experiment; never activate a controller.

Both architectures are fitted and frozen before either test set is collected.
An existing output directory is never reused. Test namespaces are consumed once
examined; repeating this command with the same seeds is a reproduction only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import torch

from poseidon.research.candidate_evaluation import evaluate_oracle
from poseidon.research.contracts import canonical_bytes
from poseidon.research.oracle_training import fit_oracle, verify_fit_receipt
from poseidon.research.prediction_gate import assess_prediction_gate


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(output: Path, *, seed: int = 652000000) -> dict:
    if type(seed) is not int or not 0 <= seed <= 2**63 - 5000000:
        raise ValueError("seed must leave room for five disjoint million-seed namespaces")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("pilot output must not exist; preserve previous evidence")
    root = Path(__file__).resolve().parents[1]
    settings = dict(base_seed=seed, train_episodes=128, selection_episodes=32,
                    calibration_episodes=64, max_steps=96, training_updates=1024,
                    batch_size=128, hidden_size=128, ensemble_size=5, coverage=0.90,
                    data_protocol="episode-mixture-v2")
    architectures = ("mlp-v1", "linear-residual-v1")
    protocol = {
        "schema": "poseidon-matched-residual-pilot-v1", "fit_settings": settings,
        "architectures": list(architectures), "test_episodes_per_condition": 64,
        "bootstrap_reps": 2000, "torch_threads": 2,
        "seed_sets": {name: list(range(seed + offset, seed + offset + count))
                      for name, offset, count in (("train", 0, 128), ("selection", 1000000, 32),
                                                 ("calibration", 2000000, 64),
                                                 ("id_test", 3000000, 64),
                                                 ("ood_scarcity_test", 4000000, 64))},
        "residual_scale_grid": [0.0, 0.25, 0.5, 1.0],
        "order": "freeze both fits before collecting either test condition",
        "acceptance": "both conditions: state gain over linear lower 95% bound > 0; reward gain lower bound >= 0",
        "control_run": False, "promoted": False,
        "limits": ["One training lineage per architecture; no seed-lineage uncertainty estimate.",
                   "Matched data/update counts, not equal FLOPs or elapsed runtime.",
                   "Scarcity shift is not structural transfer.",
                   "HorizonX scores reward/width; the composite cognitive controller also uses predicted state in homeostatic scoring and a health guard.",
                   "Prediction screen is a research filter, not a policy promotion decision.",
                   "A matched single network and frozen-core dynamics comparison remain future work."],
    }
    paths = sorted((root / "poseidon").rglob("*.py")) + [Path(__file__).resolve()]
    source_bytes = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    protocol["source_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()}
    protocol["protocol_sha256"] = hashlib.sha256(canonical_bytes(protocol)).hexdigest()
    output.mkdir(parents=True)
    _write(output / "protocol.json", protocol)
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in source_bytes.items():
            archive.writestr(name, data)

    def ensure_source():
        for name, digest in protocol["source_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"pilot source changed: {name}; evidence preserved, no continuation")

    summary = {"schema": protocol["schema"], "protocol_sha256": protocol["protocol_sha256"],
               "promoted": False, "control_run": False, "fits": {}, "evaluations": {}}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for architecture in architectures:
            ensure_source()
            started = time.monotonic()
            fit_dir = output / architecture / "fit"
            receipt = fit_oracle(fit_dir, architecture=architecture, **settings,
                progress=lambda event, name=architecture: print(json.dumps({"architecture": name, **event}), flush=True))
            verified = verify_fit_receipt(fit_dir / "fit_receipt.json", replay_data=True)
            summary["fits"][architecture] = {"artifact_path": receipt["artifact_path"],
                "receipt_sha256": receipt["receipt_sha256"], "verified": verified,
                "elapsed_seconds": time.monotonic() - started,
                "residual_selection": receipt["training"].get("residual_selection")}
            _write(output / "progress.json", summary)
        for architecture in architectures:
            ensure_source()
            print(json.dumps({"phase": "evaluate", "architecture": architecture}), flush=True)
            eval_dir = output / architecture / "prediction"
            receipt = evaluate_oracle(summary["fits"][architecture]["artifact_path"], eval_dir,
                                      base_seed=seed, episodes_per_condition=64, bootstrap_reps=2000)
            gate = assess_prediction_gate(eval_dir / "evaluation_receipt.json")
            _write(eval_dir / "prediction_gate.json", gate)
            if (gate.get("verification", {}).get("valid") is not True
                    or gate.get("verification", {}).get("replayed") is not True):
                raise ValueError(f"{architecture} evidence failed replay; pilot incomplete, evidence preserved")
            summary["evaluations"][architecture] = {
                "receipt_sha256": receipt["receipt_sha256"], "artifact_sha256": receipt["artifact_sha256"],
                "model_sha256": receipt["model_sha256"], "gate": gate,
                "conditions": {name: {"episodes": result["episodes"], "metrics": result["metrics"],
                                      "by_behavior": result.get("by_behavior", {})}
                               for name, result in receipt["results"].items()}}
            _write(output / "progress.json", summary)
        ensure_source()
        summary["complete"] = True
        _write(output / "summary.json", summary)
        print(json.dumps({"phase": "complete", "output": str(output), "promoted": False}), flush=True)
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=652000000)
    args = parser.parse_args()
    run(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
