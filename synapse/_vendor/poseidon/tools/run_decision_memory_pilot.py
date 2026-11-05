"""Run a fixed, matched CPU decision-value memory experiment.

Trains a DecisionValueMemory module on synthetic counterfactual branches,
freezes the weights, and evaluates across all 6 matched arms on held-out seeds.
Outputs cryptographically hashed receipts, paired bootstrap intervals, and replay verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch

from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.contracts import canonical_bytes
from poseidon.research.decision_memory_benchmark import (
    run_decision_memory_benchmark,
    verify_decision_memory_receipt,
)
from poseidon.research.decision_value_memory import (
    DecisionValueMemoryConfig,
    fit_decision_value_memory,
)


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(
    output: Path,
    *,
    seed: int = 710000000,
    train_seeds_count: int = 8,
    eval_episodes_per_condition: int = 16,
    max_steps: int = 24,
    updates: int = 200,
) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed <= 2**63 - 10000000:
        raise ValueError("seed must leave room for disjoint namespaces")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("pilot output must not exist; preserve previous evidence")

    root = Path(__file__).resolve().parents[1]
    protocol = {
        "schema": "poseidon-decision-memory-pilot-v1",
        "base_seed": seed,
        "train_seeds_count": train_seeds_count,
        "eval_episodes_per_condition": eval_episodes_per_condition,
        "max_steps": max_steps,
        "updates": updates,
        "arms": ["intact", "similarity", "erased", "shifted", "donor", "baseline"],
        "conditions": ["id_alias", "ood_scarcity", "ood_noise"],
        "bootstrap_reps": 2000,
        "torch_threads": 2,
        "promoted": False,
        "limits": [
            "Synthetic ObservationAliasTidePool testbed; no claims of real-world cognition.",
            "CPU-local matched evaluation across identical initial seeds.",
            "Immutable frozen Tidal core baseline preserved without retraining.",
            "Admission gate is fail-closed; default memory influence is zero.",
        ],
    }

    # Capture source state
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

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.monotonic()

    try:
        ensure_source()
        core = CoreRuntime(active_core_path())

        # 1. Train DecisionValueMemory on training partition
        train_seeds = [seed + 10000 + i for i in range(train_seeds_count)]
        cfg = DecisionValueMemoryConfig(
            capacity=128,
            top_k=4,
            admission_threshold=0.04,
            shrinkage_alpha=0.50,
            hidden_dim=48,
            belief_dim=24,
        )
        print(json.dumps({"phase": "fit_memory", "train_seeds": train_seeds}), flush=True)
        memory = fit_decision_value_memory(train_seeds=train_seeds, config=cfg, updates=updates)

        # Save memory model state
        memory_dir = output / "memory_model"
        memory_dir.mkdir(parents=True)
        torch.save(memory.state_dict(), memory_dir / "model.pt")

        # 2. Run matched 6-arm benchmark on untouched test seeds
        ensure_source()
        print(json.dumps({"phase": "benchmark_evaluation"}), flush=True)
        benchmark_receipt = run_decision_memory_benchmark(
            memory=memory,
            base_seed=seed + 50000,
            episodes=eval_episodes_per_condition,
            max_steps=max_steps,
            conditions=("id_alias", "ood_scarcity", "ood_noise"),
            core=core,
        )

        _write(output / "benchmark_receipt.json", benchmark_receipt)

        # 3. Verify receipt replay
        print(json.dumps({"phase": "verify_receipt"}), flush=True)
        verified = verify_decision_memory_receipt(benchmark_receipt, replay_seeds=True, core=core)

        # Assess gating criteria
        # Check if intact arm beats similarity and baseline with positive CI lower bound on ID
        id_comp = benchmark_receipt["results"]["id_alias"]["paired_comparisons"]
        intact_vs_sim = id_comp["intact_vs_similarity"]
        intact_vs_base = id_comp["intact_vs_baseline"]
        intact_vs_erased = id_comp["intact_vs_erased"]

        summary = {
            "schema": protocol["schema"],
            "protocol_sha256": protocol["protocol_sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "complete": True,
            "verified": verified,
            "promoted": False,  # Strict research rule: do not promote default policy
            "id_alias_summary": {
                "intact_mean_reward": benchmark_receipt["results"]["id_alias"]["arms"]["intact"]["mean_reward"],
                "baseline_mean_reward": benchmark_receipt["results"]["id_alias"]["arms"]["baseline"]["mean_reward"],
                "similarity_mean_reward": benchmark_receipt["results"]["id_alias"]["arms"]["similarity"]["mean_reward"],
                "erased_mean_reward": benchmark_receipt["results"]["id_alias"]["arms"]["erased"]["mean_reward"],
                "intact_vs_similarity_ci95": intact_vs_sim["ci_95"],
                "intact_vs_baseline_ci95": intact_vs_base["ci_95"],
                "intact_vs_erased_ci95": intact_vs_erased["ci_95"],
                "wins_vs_similarity": intact_vs_sim["wins"],
                "wins_vs_baseline": intact_vs_base["wins"],
            },
            "ood_scarcity_summary": {
                "intact_mean_reward": benchmark_receipt["results"]["ood_scarcity"]["arms"]["intact"]["mean_reward"],
                "baseline_mean_reward": benchmark_receipt["results"]["ood_scarcity"]["arms"]["baseline"]["mean_reward"],
                "similarity_mean_reward": benchmark_receipt["results"]["ood_scarcity"]["arms"]["similarity"]["mean_reward"],
            },
            "ood_noise_summary": {
                "intact_mean_reward": benchmark_receipt["results"]["ood_noise"]["arms"]["intact"]["mean_reward"],
                "baseline_mean_reward": benchmark_receipt["results"]["ood_noise"]["arms"]["baseline"]["mean_reward"],
                "similarity_mean_reward": benchmark_receipt["results"]["ood_noise"]["arms"]["similarity"]["mean_reward"],
            },
        }

        _write(output / "summary.json", summary)
        return summary
    finally:
        torch.set_num_threads(previous_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Decision-Value Memory Pilot Benchmark")
    parser.add_argument("output", type=Path, help="Target output directory")
    parser.add_argument("--seed", type=int, default=710000000, help="Base seed")
    parser.add_argument("--episodes", type=int, default=16, help="Evaluation episodes per condition")
    parser.add_argument("--steps", type=int, default=24, help="Maximum steps per episode")
    parser.add_argument("--updates", type=int, default=200, help="Training updates")
    args = parser.parse_args()

    res = run(
        args.output,
        seed=args.seed,
        eval_episodes_per_condition=args.episodes,
        max_steps=args.steps,
        updates=args.updates,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
