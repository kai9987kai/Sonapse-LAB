"""Three-way, model-bound residual-calibration gate for the NEXUS-OMEGA ensemble.

The candidate is action-conditioned mean residual bias correction. It is fitted
on calibration seeds, shrunk on selection seeds, and admitted only by a fresh
seed holdout. This does not update model weights or activate runtime control.
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


CALIBRATION_SEEDS = tuple(range(72001, 72025))
SELECTION_SEEDS = tuple(range(72101, 72117))
HOLDOUT_SEEDS = tuple(range(72201, 72265))
MODEL_SEEDS = tuple(range(81001, 81009))
SHRINKAGE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
HORIZON = 32
MIN_VITAL_MAE_GAIN = 0.001
MAX_ALL_CHANNEL_MAE_REGRESSION = 0.002
BOOTSTRAP_REPLICATES = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weights_digest(model: FactoredCausalWorldModel) -> str:
    digest = hashlib.sha256()
    for member_index, member in enumerate(model.members):
        for network_name in ("net_endo", "net_exo", "net_rew"):
            network = getattr(member, network_name)
            for parameter_name in ("W1", "b1", "W_out", "b_out"):
                value = np.ascontiguousarray(getattr(network, parameter_name))
                digest.update(f"{member_index}:{network_name}:{parameter_name}:{value.shape}".encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def _collect(seed_partition: tuple[int, ...]) -> tuple[dict, dict]:
    data = {}
    model_hashes = {}
    for model_seed in MODEL_SEEDS:
        model = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=model_seed)
        model_hashes[str(model_seed)] = _weights_digest(model)
        for env_seed in seed_partition:
            env = POMDPPhaseWorld(seed=env_seed, max_steps=HORIZON)
            obs = env.reset()
            memory = np.zeros(18, dtype=np.float32)
            rows = []
            for tick in range(HORIZON):
                action = (tick + env_seed % 4) % 4
                prediction = model.predict_factored(np.asarray(obs, dtype=np.float32), memory, action)
                next_obs, _reward, done, _info = env.step(action)
                rows.append({
                    "action": action,
                    "predicted": prediction.predicted_obs.astype(np.float64),
                    "actual": np.asarray(next_obs, dtype=np.float64),
                })
                obs = next_obs
                if done:
                    break
            data[(model_seed, env_seed)] = rows
    return data, model_hashes


def _fit_bias(calibration: dict) -> tuple[dict, dict]:
    sums = {seed: np.zeros((4, 8), dtype=np.float64) for seed in MODEL_SEEDS}
    counts = {seed: np.zeros(4, dtype=np.int64) for seed in MODEL_SEEDS}
    for (model_seed, _env_seed), rows in calibration.items():
        for row in rows:
            action = row["action"]
            sums[model_seed][action] += row["actual"] - row["predicted"]
            counts[model_seed][action] += 1
    biases, serialized_counts = {}, {}
    for model_seed in MODEL_SEEDS:
        table = np.zeros((4, 8), dtype=np.float64)
        for action in range(4):
            if counts[model_seed][action] > 0:
                table[action] = sums[model_seed][action] / counts[model_seed][action]
        biases[model_seed] = table
        serialized_counts[str(model_seed)] = [int(x) for x in counts[model_seed]]
    return biases, serialized_counts


def _episode_metrics(data: dict, biases: dict, scale: float) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, list[float]]] = {}
    for (model_seed, env_seed), rows in data.items():
        vital, all_channels = [], []
        for row in rows:
            candidate = row["predicted"] + scale * biases[model_seed][row["action"]]
            error = np.abs(candidate - row["actual"])
            vital.append(float(np.mean(error[:3])))
            all_channels.append(float(np.mean(error)))
        output.setdefault(env_seed, {"vital": [], "all": []})
        output[env_seed]["vital"].append(float(np.mean(vital)))
        output[env_seed]["all"].append(float(np.mean(all_channels)))
    return {
        seed: {name: float(np.mean(values)) for name, values in metrics.items()}
        for seed, metrics in output.items()
    }


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        replicates[index] = np.mean(values[rng.randint(0, len(values), size=len(values))])
    return [float(x) for x in np.quantile(replicates, [0.025, 0.975])]


def _sign_test(values: np.ndarray) -> dict:
    nonzero = [float(x) for x in values if abs(float(x)) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return {"n_nonzero": 0, "positive": 0, "p_value": 1.0}
    positive = sum(x > 0 for x in nonzero)
    tail = min(positive, n - positive)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    return {"n_nonzero": n, "positive": positive, "p_value": p}


def run() -> dict:
    calibration, model_hashes = _collect(CALIBRATION_SEEDS)
    biases, action_counts = _fit_bias(calibration)
    selection, selection_model_hashes = _collect(SELECTION_SEEDS)
    if selection_model_hashes != model_hashes:
        raise RuntimeError("model initialization identity changed between partitions")
    selection_rows = {
        scale: _episode_metrics(selection, biases, scale)
        for scale in SHRINKAGE_CANDIDATES
    }
    selected_scale = min(
        SHRINKAGE_CANDIDATES,
        key=lambda scale: (np.mean([row["vital"] for row in selection_rows[scale].values()]), scale),
    )

    holdout, holdout_model_hashes = _collect(HOLDOUT_SEEDS)
    if holdout_model_hashes != model_hashes:
        raise RuntimeError("model initialization identity changed before untouched holdout")
    raw = _episode_metrics(holdout, biases, 0.0)
    corrected = _episode_metrics(holdout, biases, selected_scale)
    vital_gain = np.asarray([raw[seed]["vital"] - corrected[seed]["vital"] for seed in HOLDOUT_SEEDS])
    all_gain = np.asarray([raw[seed]["all"] - corrected[seed]["all"] for seed in HOLDOUT_SEEDS])
    vital_ci = _bootstrap_ci(vital_gain, 552001)
    all_ci = _bootstrap_ci(all_gain, 552002)
    sign = _sign_test(vital_gain)
    admitted = bool(
        selected_scale > 0.0
        and vital_ci[0] > MIN_VITAL_MAE_GAIN
        and sign["p_value"] < 0.05
        and all_ci[0] > -MAX_ALL_CHANNEL_MAE_REGRESSION
    )

    seed_rows = [{
        "environment_seed": seed,
        "raw_vital_mae": raw[seed]["vital"],
        "candidate_vital_mae": corrected[seed]["vital"],
        "vital_mae_reduction": vital_gain[index],
        "raw_all_channel_mae": raw[seed]["all"],
        "candidate_all_channel_mae": corrected[seed]["all"],
        "all_channel_mae_reduction": all_gain[index],
    } for index, seed in enumerate(HOLDOUT_SEEDS)]

    return {
        "schema": "poseidon-nexus-residual-calibration-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "calibration_environment_seeds": list(CALIBRATION_SEEDS),
            "selection_environment_seeds": list(SELECTION_SEEDS),
            "untouched_holdout_environment_seeds": list(HOLDOUT_SEEDS),
            "fixed_model_initialization_seeds": list(MODEL_SEEDS),
            "episode_horizon": HORIZON,
            "action_schedule": "open-loop action=(tick + environment_seed mod 4) mod 4",
            "correction": "per-model-initialization, per-action mean next-observation residual across calibration partition",
            "selection": "one global shrinkage chosen by minimum mean vital MAE on disjoint selection partition; lower shrinkage wins ties",
            "shrinkage_candidates": list(SHRINKAGE_CANDIDATES),
            "primary_gate": f"95% environment-seed cluster-bootstrap lower bound on vital MAE reduction > {MIN_VITAL_MAE_GAIN}; exact two-sided sign test p < 0.05",
            "guardrail": f"95% environment-seed cluster-bootstrap lower bound on all-channel MAE reduction > {-MAX_ALL_CHANNEL_MAE_REGRESSION}",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "holdout_opened_once_after_selection": True,
        },
        "model_binding": {
            "model_weight_sha256_by_initialization_seed": model_hashes,
            "calibration_action_counts_by_model_seed": action_counts,
            "correction_offsets_by_model_seed_action_observation": {
                str(seed): biases[seed].tolist() for seed in MODEL_SEEDS
            },
        },
        "selection": {
            "selected_shrinkage": selected_scale,
            "candidate_vital_mae": {
                str(scale): float(np.mean([row["vital"] for row in selection_rows[scale].values()]))
                for scale in SHRINKAGE_CANDIDATES
            },
        },
        "untouched_holdout": {
            "admitted": admitted,
            "runtime_activated": False,
            "raw_mean_vital_mae": float(np.mean([row["vital"] for row in raw.values()])),
            "candidate_mean_vital_mae": float(np.mean([row["vital"] for row in corrected.values()])),
            "vital_mae_reduction": float(np.mean(vital_gain)),
            "vital_mae_reduction_95pct_ci": vital_ci,
            "vital_exact_sign_test_two_sided": sign,
            "raw_mean_all_channel_mae": float(np.mean([row["all"] for row in raw.values()])),
            "candidate_mean_all_channel_mae": float(np.mean([row["all"] for row in corrected.values()])),
            "all_channel_mae_reduction": float(np.mean(all_gain)),
            "all_channel_mae_reduction_95pct_ci": all_ci,
            "environment_seed_rows": seed_rows,
        },
        "limits": [
            "The ensemble is randomly initialized and not trained or artifact-bound to a released checkpoint.",
            "Inference is conditional on eight fixed model initializations and the named synthetic POMDP.",
            "Admission is a forecast metric gate only; it does not establish policy benefit or real-world calibration.",
            "The candidate is never activated by this script, even if admitted; runtime integration requires a separate review and fresh evaluation.",
            "Receipt hashes bind the recorded bytes but do not prove the assay executed as recorded.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {
            "assay": _sha256(Path(__file__)),
            "causal_world_model": _sha256(ROOT / "poseidon/nexus_omega/causal_world_model.py"),
            "benchmark": _sha256(ROOT / "poseidon/nexus_omega/benchmark.py"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/residual_calibration_gate.json")
    args = parser.parse_args()
    receipt = run()
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "selection": receipt["selection"],
        "untouched_holdout": {key: value for key, value in receipt["untouched_holdout"].items() if key != "environment_seed_rows"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
