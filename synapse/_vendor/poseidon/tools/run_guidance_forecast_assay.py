"""Matched held-out forecast assay for NEXUS-OMEGA homeostatic guidance.

This measures one-step observation error against realized transitions in the
synthetic POMDP. It does not establish policy benefit or real-world validity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.trajectory_diffusion import TrajectoryDiffusionImaginer


ENV_SEEDS = tuple(range(71001, 71033))
MODEL_SEEDS = tuple(range(81001, 81009))
GUIDANCE_SCALES = (-1.25, 0.0, 0.625, 1.25, 2.5)
HORIZON = 32
PRIMARY_SCALE = 1.25
BOOTSTRAP_REPLICATES = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_test_two_sided(values: np.ndarray) -> dict:
    nonzero = [float(x) for x in values if abs(float(x)) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "p_value": 1.0}
    positive = sum(x > 0 for x in nonzero)
    tail = min(positive, n - positive)
    p_value = min(1.0, 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    return {"n_nonzero": n, "positive": positive, "p_value": p_value}


def _cluster_bootstrap_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        means[i] = np.mean(values[rng.randint(0, n, size=n)])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run_assay() -> dict:
    group_slices = {"vitals_0_2": slice(0, 3), "cue_and_context_3_7": slice(3, 8), "all_0_7": slice(0, 8)}
    raw_cells: list[dict] = []
    seed_scale_values: dict[int, dict[float, list[float]]] = {
        env_seed: {scale: [] for scale in GUIDANCE_SCALES} for env_seed in ENV_SEEDS
    }

    for model_seed in MODEL_SEEDS:
        for env_seed in ENV_SEEDS:
            env = POMDPPhaseWorld(seed=env_seed, max_steps=HORIZON)
            obs = env.reset()
            model = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=model_seed)
            imaginer = TrajectoryDiffusionImaginer(model, guidance_scale=1.0)
            memory = np.zeros(18, dtype=np.float32)
            errors = {scale: {name: [] for name in group_slices} for scale in GUIDANCE_SCALES}

            for tick in range(HORIZON):
                action = (tick + env_seed % 4) % 4
                prediction = model.predict_factored(np.asarray(obs, dtype=np.float32), memory, action)
                raw_next = prediction.predicted_obs.copy()
                direction = imaginer.compute_homeostatic_guidance(raw_next)
                next_obs, _reward, done, _info = env.step(action)
                actual = np.asarray(next_obs, dtype=np.float32)

                for scale in GUIDANCE_SCALES:
                    candidate = raw_next.copy() + (0.05 * scale) * direction
                    candidate[:3] = np.clip(candidate[:3], 0.0, 1.0)
                    step_error = np.abs(candidate - actual)
                    for group, channels in group_slices.items():
                        errors[scale][group].append(float(np.mean(step_error[channels])))

                obs = next_obs
                if done:
                    break

            for scale in GUIDANCE_SCALES:
                row = {"model_seed": model_seed, "environment_seed": env_seed, "guidance_scale": scale}
                for group in group_slices:
                    value = float(np.mean(errors[scale][group]))
                    row[f"mae_{group}"] = value
                raw_cells.append(row)
                seed_scale_values[env_seed][scale].append(row["mae_vitals_0_2"])

    per_seed = []
    for env_seed in ENV_SEEDS:
        means = {scale: float(np.mean(seed_scale_values[env_seed][scale])) for scale in GUIDANCE_SCALES}
        per_seed.append({
            "environment_seed": env_seed,
            "mae_vitals_by_scale": {str(scale): value for scale, value in means.items()},
            "improvement_by_scale": {str(scale): means[0.0] - means[scale] for scale in GUIDANCE_SCALES},
        })

    summaries = {}
    improvement_rows = {}
    for scale in GUIDANCE_SCALES:
        model_means = [row for row in raw_cells if row["guidance_scale"] == scale]
        summaries[str(scale)] = {
            f"mean_mae_{group}": float(np.mean([row[f"mae_{group}"] for row in model_means]))
            for group in group_slices
        }
        if scale != 0.0:
            diffs = np.asarray([
                entry["improvement_by_scale"][str(scale)] for entry in per_seed
            ], dtype=np.float64)
            improvement_rows[str(scale)] = {
                "mean_vital_mae_reduction": float(np.mean(diffs)),
                "environment_seed_cluster_bootstrap_95pct_ci": _cluster_bootstrap_interval(diffs, 99173 + int((scale + 10) * 100)),
                "two_sided_exact_sign_test": _sign_test_two_sided(diffs),
            }

    primary = improvement_rows[str(PRIMARY_SCALE)]
    return {
        "schema": "poseidon-guidance-forecast-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "question": "Does homeostatic state guidance reduce one-step vital-state forecast error on held-out synthetic POMDP transitions?",
            "primary_scale": PRIMARY_SCALE,
            "guidance_scales": list(GUIDANCE_SCALES),
            "environment_seeds": list(ENV_SEEDS),
            "model_initialization_seeds": list(MODEL_SEEDS),
            "steps_per_environment": HORIZON,
            "action_schedule": "open-loop action=(tick + environment_seed mod 4) mod 4, shared by all guidance arms",
            "primary_metric": "per-environment-seed mean absolute error on observation channels 0..2, averaged over 8 fixed model initializations",
            "controls": ["zero-guidance sham", "opposite-direction guidance", "two neighboring positive guidance scales"],
            "inference": "cluster bootstrap and exact sign test across 32 environment seeds; model initializations are averaged within environment seed",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "results": {
            "mean_mae_by_scale": summaries,
            "paired_vital_mae_reduction_vs_zero_guidance": improvement_rows,
            "primary_scale_result": primary,
            "primary_interpretation": "descriptive synthetic-model forecast comparison only; no policy or real-world claim",
        },
        "per_environment_seed": per_seed,
        "cell_receipts": raw_cells,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            "assay": _sha256(Path(__file__)),
            "trajectory_diffusion": _sha256(ROOT / "poseidon/nexus_omega/trajectory_diffusion.py"),
            "causal_world_model": _sha256(ROOT / "poseidon/nexus_omega/causal_world_model.py"),
            "benchmark": _sha256(ROOT / "poseidon/nexus_omega/benchmark.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/guidance_forecast_assay.json")
    args = parser.parse_args()
    receipt = run_assay()
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "primary_scale_result": receipt["results"]["primary_scale_result"],
        "mean_mae_by_scale": receipt["results"]["mean_mae_by_scale"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
