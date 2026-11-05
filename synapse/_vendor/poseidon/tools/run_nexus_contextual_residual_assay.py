"""Cross-initialization assay of context-conditioned NEXUS residual adapters.

Fits a ridge residual map on one predictor-initialization family, selects its
regularization/shrinkage on a second family, and gates on unseen model and
environment seeds. This is a synthetic forecast experiment, not runtime policy.
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


TRAIN_ENV_SEEDS = tuple(range(74001, 74025))
SELECT_ENV_SEEDS = tuple(range(74101, 74117))
HOLDOUT_ENV_SEEDS = tuple(range(74201, 74265))
TRAIN_MODEL_SEEDS = tuple(range(81001, 81009))
SELECT_MODEL_SEEDS = tuple(range(82001, 82009))
HOLDOUT_MODEL_SEEDS = tuple(range(83001, 83009))
RIDGE_CANDIDATES = (0.1, 1.0, 10.0, 100.0)
SHRINKAGE_CANDIDATES = (0.25, 0.5, 0.75, 1.0)
HORIZON = 32
MIN_VITAL_MAE_GAIN = 0.001
MAX_ALL_CHANNEL_MAE_REGRESSION = 0.002
BOOTSTRAP_REPLICATES = 20_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _weight_hash(model: FactoredCausalWorldModel) -> str:
    digest = hashlib.sha256()
    for member_index, member in enumerate(model.members):
        for network_name in ("net_endo", "net_exo", "net_rew"):
            network = getattr(member, network_name)
            for parameter_name in ("W1", "b1", "W_out", "b_out"):
                value = np.ascontiguousarray(getattr(network, parameter_name))
                digest.update(f"{member_index}:{network_name}:{parameter_name}:{value.shape}".encode())
                digest.update(value.tobytes())
    return digest.hexdigest()


def _collect(env_seeds: tuple[int, ...], model_seeds: tuple[int, ...]) -> tuple[list[dict], dict]:
    rows = []
    model_hashes = {}
    for model_seed in model_seeds:
        model = FactoredCausalWorldModel(d_obs=8, d_mem=18, n_actions=4, n_members=3, base_seed=model_seed)
        model_hashes[str(model_seed)] = _weight_hash(model)
        for env_seed in env_seeds:
            env = POMDPPhaseWorld(seed=env_seed, max_steps=HORIZON)
            obs = env.reset()
            memory = np.zeros(18, dtype=np.float32)
            for tick in range(HORIZON):
                action = (tick + env_seed % 4) % 4
                observation = np.asarray(obs, dtype=np.float32)
                prediction = model.predict_factored(observation, memory, action)
                next_obs, _reward, done, _info = env.step(action)
                rows.append({
                    "environment_seed": env_seed,
                    "model_seed": model_seed,
                    "action": action,
                    "observation": observation.astype(np.float64),
                    "prediction": prediction.predicted_obs.astype(np.float64),
                    "actual": np.asarray(next_obs, dtype=np.float64),
                })
                obs = next_obs
                if done:
                    break
    return rows, model_hashes


def _features(observation: np.ndarray, action: int, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = np.clip((observation - center) / scale, -5.0, 5.0)
    features = np.zeros(36, dtype=np.float64)
    start = action * 9
    features[start] = 1.0
    features[start + 1:start + 9] = normalized
    return features


def _fit_contextual(rows: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, list[int]]:
    observations = np.stack([row["observation"] for row in rows])
    center = observations.mean(axis=0)
    scale = observations.std(axis=0)
    scale[scale < 1e-6] = 1.0
    design = np.stack([_features(row["observation"], row["action"], center, scale) for row in rows])
    target = np.stack([row["actual"] - row["prediction"] for row in rows])
    action_counts = [sum(row["action"] == action for row in rows) for action in range(4)]
    if min(action_counts) < 128:
        raise RuntimeError(f"Insufficient train support by action: {action_counts}")
    gram = design.T @ design
    cross = design.T @ target
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[np.arange(0, 36, 9), np.arange(0, 36, 9)] = 0.0
    models = {}
    for ridge in RIDGE_CANDIDATES:
        models[ridge] = np.linalg.solve(gram + ridge * penalty, cross)
    return models, center, scale, action_counts


def _fit_action_bias(rows: list[dict]) -> np.ndarray:
    sums = np.zeros((4, 8), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    for row in rows:
        sums[row["action"]] += row["actual"] - row["prediction"]
        counts[row["action"]] += 1
    return sums / counts[:, None]


def _error_rows(rows: list[dict], *, model: np.ndarray | None, center: np.ndarray | None,
                scale: np.ndarray | None, bias: np.ndarray | None, shrinkage: float) -> dict:
    by_environment: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for row in rows:
        correction = np.zeros(8, dtype=np.float64)
        if model is not None:
            correction = _features(row["observation"], row["action"], center, scale) @ model
        elif bias is not None:
            correction = bias[row["action"]]
        error = np.abs(row["prediction"] + shrinkage * correction - row["actual"])
        by_environment.setdefault(row["environment_seed"], {}).setdefault(row["model_seed"], []).append(
            (float(np.mean(error[:3])), float(np.mean(error)))
        )
    result = {}
    for env_seed, by_model in by_environment.items():
        per_model = [np.mean(values, axis=0) for values in by_model.values()]
        mean = np.mean(per_model, axis=0)
        result[env_seed] = {"vital": float(mean[0]), "all": float(mean[1])}
    return result


def _bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        means[index] = values[rng.randint(0, len(values), size=len(values))].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_test(values: np.ndarray) -> dict:
    nonzero = [float(value) for value in values if abs(float(value)) > 1e-12]
    n = len(nonzero)
    if not n:
        return {"n_nonzero": 0, "positive": 0, "p_value": 1.0}
    positive = sum(value > 0 for value in nonzero)
    tail = min(positive, n - positive)
    p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    return {"n_nonzero": n, "positive": positive, "p_value": p_value}


def run() -> dict:
    train, train_hashes = _collect(TRAIN_ENV_SEEDS, TRAIN_MODEL_SEEDS)
    contextual, center, obs_scale, action_counts = _fit_contextual(train)
    bias = _fit_action_bias(train)

    selection, selection_hashes = _collect(SELECT_ENV_SEEDS, SELECT_MODEL_SEEDS)
    if set(train_hashes.values()) & set(selection_hashes.values()):
        raise RuntimeError("selection reused a training predictor initialization")
    candidates = [{"kind": "none", "shrinkage": 0.0, "ridge": None}]
    candidates.extend({"kind": "action_bias", "shrinkage": value, "ridge": None}
                      for value in SHRINKAGE_CANDIDATES)
    candidates.extend({"kind": "contextual_ridge", "shrinkage": value, "ridge": ridge}
                      for ridge in RIDGE_CANDIDATES for value in SHRINKAGE_CANDIDATES)
    selection_results = {}
    for candidate in candidates:
        candidate_model = contextual[candidate["ridge"]] if candidate["kind"] == "contextual_ridge" else None
        metrics = _error_rows(selection, model=candidate_model, center=center, scale=obs_scale,
                              bias=bias if candidate["kind"] == "action_bias" else None,
                              shrinkage=candidate["shrinkage"])
        selection_results[str(candidate)] = float(np.mean([row["vital"] for row in metrics.values()]))
    selected = min(candidates, key=lambda item: (selection_results[str(item)], item["shrinkage"], item["kind"]))

    holdout, holdout_hashes = _collect(HOLDOUT_ENV_SEEDS, HOLDOUT_MODEL_SEEDS)
    if set(train_hashes.values()) & set(holdout_hashes.values()) or set(selection_hashes.values()) & set(holdout_hashes.values()):
        raise RuntimeError("holdout reused a training or selection predictor initialization")
    candidate_model = contextual[selected["ridge"]] if selected["kind"] == "contextual_ridge" else None
    candidate_bias = bias if selected["kind"] == "action_bias" else None
    raw = _error_rows(holdout, model=None, center=None, scale=None, bias=None, shrinkage=0.0)
    corrected = _error_rows(holdout, model=candidate_model, center=center, scale=obs_scale,
                            bias=candidate_bias, shrinkage=selected["shrinkage"])
    vital_gain = np.asarray([raw[seed]["vital"] - corrected[seed]["vital"] for seed in HOLDOUT_ENV_SEEDS])
    all_gain = np.asarray([raw[seed]["all"] - corrected[seed]["all"] for seed in HOLDOUT_ENV_SEEDS])
    vital_ci, all_ci = _bootstrap(vital_gain, 883001), _bootstrap(all_gain, 883002)
    sign = _sign_test(vital_gain)
    admitted = bool(
        selected["kind"] != "none"
        and selected["shrinkage"] > 0.0
        and vital_ci[0] > MIN_VITAL_MAE_GAIN
        and sign["p_value"] < 0.05
        and all_ci[0] > -MAX_ALL_CHANNEL_MAE_REGRESSION
    )
    per_seed = [{
        "environment_seed": seed,
        "raw_vital_mae": raw[seed]["vital"],
        "candidate_vital_mae": corrected[seed]["vital"],
        "vital_mae_reduction": vital_gain[index],
        "raw_all_channel_mae": raw[seed]["all"],
        "candidate_all_channel_mae": corrected[seed]["all"],
        "all_channel_mae_reduction": all_gain[index],
    } for index, seed in enumerate(HOLDOUT_ENV_SEEDS)]

    return {
        "schema": "poseidon-nexus-contextual-residual-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train_environment_seeds": list(TRAIN_ENV_SEEDS),
            "selection_environment_seeds": list(SELECT_ENV_SEEDS),
            "holdout_environment_seeds": list(HOLDOUT_ENV_SEEDS),
            "train_model_initializations": list(TRAIN_MODEL_SEEDS),
            "selection_model_initializations": list(SELECT_MODEL_SEEDS),
            "holdout_model_initializations": list(HOLDOUT_MODEL_SEEDS),
            "horizon": HORIZON,
            "action_schedule": "open-loop action=(tick + environment_seed mod 4) mod 4 shared across model arms",
            "context": "current eight-channel observation with per-action intercept and standardized observation channels",
            "candidate_selection": "minimum selection-partition vital MAE among zero, action-bias controls, and contextual ridge/shrinkage grid",
            "ridge_grid": list(RIDGE_CANDIDATES),
            "shrinkage_grid": list(SHRINKAGE_CANDIDATES),
            "admission_rule": f"vital MAE reduction 95% seed-cluster bootstrap lower bound > {MIN_VITAL_MAE_GAIN}; exact paired sign-test p < 0.05; all-channel reduction lower bound > {-MAX_ALL_CHANNEL_MAE_REGRESSION}",
            "holdout_opened_once_after_candidate_selection": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "fit": {
            "training_transitions": len(train),
            "training_action_counts": action_counts,
            "observation_center": center.tolist(),
            "observation_scale": obs_scale.tolist(),
            "contextual_coefficients_by_ridge": {str(ridge): contextual[ridge].tolist() for ridge in RIDGE_CANDIDATES},
            "action_bias_offsets": bias.tolist(),
            "training_model_weight_sha256": train_hashes,
        },
        "selection": {
            "selection_transitions": len(selection),
            "selection_model_weight_sha256": selection_hashes,
            "selected_candidate": selected,
            "mean_vital_mae_by_candidate": selection_results,
        },
        "holdout": {
            "admitted": admitted,
            "runtime_activated": False,
            "holdout_transitions": len(holdout),
            "holdout_model_weight_sha256": holdout_hashes,
            "raw_mean_vital_mae": float(np.mean([row["vital"] for row in raw.values()])),
            "candidate_mean_vital_mae": float(np.mean([row["vital"] for row in corrected.values()])),
            "vital_mae_reduction": float(vital_gain.mean()),
            "vital_mae_reduction_95pct_ci": vital_ci,
            "vital_exact_sign_test": sign,
            "raw_mean_all_channel_mae": float(np.mean([row["all"] for row in raw.values()])),
            "candidate_mean_all_channel_mae": float(np.mean([row["all"] for row in corrected.values()])),
            "all_channel_mae_reduction": float(all_gain.mean()),
            "all_channel_mae_reduction_95pct_ci": all_ci,
            "environment_seed_rows": per_seed,
        },
        "limits": [
            "The NEXUS predictors are randomly initialized; this does not validate a trained predictor.",
            "Evaluation is a synthetic open-loop one-step forecast assay, not policy or survival benefit.",
            "Inference averages eight unseen fixed predictor initializations within each of 64 environment seeds.",
            "Even if the gate passes, this script never activates the adapter in the runtime.",
            "The receipt and hashes support audit but cannot prove the experiment executed as recorded.",
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
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/contextual_residual_assay.json")
    args = parser.parse_args()
    receipt = run()
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "selected_candidate": receipt["selection"]["selected_candidate"],
        "holdout": {key: value for key, value in receipt["holdout"].items() if key not in {"environment_seed_rows", "holdout_model_weight_sha256"}},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
