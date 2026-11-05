"""Bounded paired control evaluation of an explicitly selected ORACLE candidate.

Each condition preserves all six arms and full replay receipts. Nothing is
promoted, and these seeds must not subsequently be used for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from poseidon.cognitive_experiment import run_cognitive_benchmark, verify_cognitive_receipt
from poseidon.research.candidate_evaluation import _bootstrap_interval
from poseidon.research.oracle_training import load_oracle_artifact


def evaluate(artifact: Path, output: Path, *, seed: int = 637000000,
             episodes: int = 16, max_steps: int = 64) -> dict:
    if type(episodes) is not int or not 2 <= episodes <= 32:
        raise ValueError("episodes must be in [2, 32]")
    if type(max_steps) is not int or not 2 <= max_steps <= 128:
        raise ValueError("max_steps must be in [2, 128]")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 3000000:
        raise ValueError("invalid control seed namespace")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("control output must be empty; retain previous evidence")
    payload = load_oracle_artifact(artifact)
    fit_seeds = {s for values in payload["training_manifest"]["seed_sets"].values() for s in values}
    conditions = [("clean_1p25", 1.25, False), ("scarcity_2p5", 2.5, False),
                  ("noisy_scarcity_2p5", 2.5, True)]
    all_seeds = {seed + index * 1000000 + offset for index in range(3) for offset in range(episodes)}
    if fit_seeds & all_seeds:
        raise ValueError("control seeds overlap training, selection, or calibration episodes")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "poseidon-learning-control-pilot-v1", "promoted": False,
               "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "episodes_per_condition": episodes, "max_steps": max_steps,
               "conditions": {}, "limits": [
                   "Synthetic TidePool pilot; no structural-world or general intelligence claim.",
                   "Controller-chosen actions differ from the calibration data policy; coverage is unverified.",
                   "Death probabilities are uncalibrated and do not contribute a failure-probability penalty.",
                   "Intervals are descriptive paired episode bootstraps; no multiple-comparison correction.",
                   "Frozen predictor and core; no online weight adaptation or model promotion."]}
    for index, (name, scarcity, noisy) in enumerate(conditions):
        print(json.dumps({"phase": "control", "condition": name}), flush=True)
        receipt = run_cognitive_benchmark(seed_base=seed + index * 1000000, episodes=episodes,
            max_steps=max_steps, scarcity=scarcity, use_pomdp=noisy, oracle_checkpoint=artifact,
            progress=lambda event: print(json.dumps({"condition": name, **event}), flush=True))
        verify_cognitive_receipt(receipt)
        path = output / f"{name}.json"
        path.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False), encoding="utf-8")
        intact = receipt["episodes_data"]["intact"]
        baseline = receipt["episodes_data"]["frozen_tidal"]
        deltas = [a["total_reward"] - b["total_reward"] for a, b in zip(intact, baseline)]
        rows = [step["decision"] for episode in intact for step in episode["trajectory"]]
        summary["conditions"][name] = {
            "receipt": path.name, "receipt_sha256": receipt["sha256"], "replay_verified": True,
            "seeds": [episode["seed"] for episode in intact], "arms": receipt["arms"],
            "mean_reward_difference": sum(deltas) / len(deltas),
            "paired_reward_95pct_bootstrap_interval": _bootstrap_interval(deltas, seed=seed + index, reps=2000),
            "reward_differences": deltas,
            "paired_survival_difference": sum(int(a["survived"]) - int(b["survived"])
                                               for a, b in zip(intact, baseline)) / episodes,
            "planning_enabled_fraction": sum(row["planning_enabled"] for row in rows) / len(rows),
            "member_forward_calls": sum(ep["trajectory"][-1]["decision"]["oracle"]["member_forward_calls"]
                                        for ep in intact),
            "memory_comparisons": {k: v for k, v in receipt["paired"].items() if k != "intact_vs_frozen_tidal"},
        }
        print(json.dumps({"condition_complete": name, "difference": summary["conditions"][name]["mean_reward_difference"]}), flush=True)
    summary["sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=637000000)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.artifact.resolve(), args.output, seed=args.seed, episodes=args.episodes, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
