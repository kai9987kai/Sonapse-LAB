"""Fresh-seed ID/OOD evaluation for trained ORACLE predictive candidates."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

from .contracts import EpisodeSplit, split_summary
from .oracle_model import OracleEnsemble
from .oracle_training import (_atomic_json, _batch, _file_sha256, _load_model,
                              load_oracle_artifact, source_hashes, collect_split,
                              LEGACY_DATA_PROTOCOL, MIXTURE_DATA_PROTOCOL, generation_protocol,
                              linear_delta_features, protocol_split_summary, episode_behavior)
from .contracts import canonical_bytes

EVALUATION_SCHEMA = "poseidon-oracle-evaluation-v1"


def _mean(rows: list[dict], name: str) -> float:
    value = sum(float(row[name]) for row in rows) / len(rows)
    if not math.isfinite(value):
        raise FloatingPointError(f"nonfinite evaluation metric: {name}")
    return value


def _bootstrap_interval(values: list[float], *, seed: int, reps: int) -> list[float]:
    if not values:
        raise ValueError("bootstrap interval needs episode values")
    if type(reps) is not int or not 100 <= reps <= 50000:
        raise ValueError("bootstrap_reps must be an integer in [100, 50000]")
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        samples.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
    samples.sort()
    return [samples[math.floor(0.025 * (reps - 1))], samples[math.ceil(0.975 * (reps - 1))]]


@torch.inference_mode()
def _evaluate_split(model: OracleEnsemble, split: EpisodeSplit, artifact: dict[str, Any],
                    bootstrap_reps: int) -> dict[str, Any]:
    train = artifact["training"]
    reward_baselines = train["reward_mean_by_action"]
    death_baseline = float(train["death_probability_prior"])
    state_quantile = float(artifact["calibration"]["state_standardized_error_quantile"])
    reward_quantile = float(artifact["calibration"]["reward_standardized_error_quantile"])
    data_protocol = artifact["training_manifest"].get("data_protocol", LEGACY_DATA_PROTOCOL)
    linear_baseline = train.get("linear_delta_baseline")
    linear_coefficients = (torch.tensor(linear_baseline["coefficients"], dtype=torch.float64)
                           if linear_baseline is not None else None)
    episode_metrics = []
    model.eval()
    for episode in split.episodes:
        rows = list(episode.transitions)
        obs, actions, deltas, rewards, deaths = _batch(rows)
        summary = model.summarize(model(obs, actions))
        delta_error = deltas - summary["delta_mean"]
        candidate_state_mse = float(delta_error.square().mean())
        baseline_state_mse = float(deltas.square().mean())
        candidate_reward_mae = float((rewards - summary["reward_mean"]).abs().mean())
        baseline_reward = torch.tensor([reward_baselines[row.action] for row in rows], dtype=torch.float32)
        baseline_reward_mae = float((rewards - baseline_reward).abs().mean())
        candidate_death_brier = float((deaths - summary["death_probability"]).square().mean())
        baseline_death_brier = float((deaths - death_baseline).square().mean())
        delta_scale = summary["delta_total_variance"].sqrt().clamp_min(1e-4)
        reward_scale = summary["reward_total_variance"].sqrt().clamp_min(1e-4)
        state_covered = bool(((delta_error.abs() <= state_quantile * delta_scale).all()).item())
        reward_covered = bool(((rewards - summary["reward_mean"]).abs() <= reward_quantile * reward_scale).all().item())
        if not all(math.isfinite(value) for value in (
                candidate_state_mse, baseline_state_mse, candidate_reward_mae,
                baseline_reward_mae, candidate_death_brier, baseline_death_brier)):
            raise FloatingPointError("nonfinite ORACLE evaluation metric")
        episode_metrics.append({
            "seed": episode.seed,
            "scarcity": episode.scarcity,
            "steps": len(rows),
            "deaths": sum(row.death for row in rows),
            "candidate_state_delta_mse": candidate_state_mse,
            "persistence_state_delta_mse": baseline_state_mse,
            "candidate_reward_mae": candidate_reward_mae,
            "action_mean_reward_mae": baseline_reward_mae,
            "candidate_death_brier": candidate_death_brier,
            "training_prior_death_brier": baseline_death_brier,
            "state_episode_interval_covered": state_covered,
            "reward_episode_interval_covered": reward_covered,
            "mean_state_interval_half_width": float((state_quantile * delta_scale).mean()),
            "mean_reward_interval_half_width": float((reward_quantile * reward_scale).mean()),
        })
        if linear_coefficients is not None:
            predicted_delta = linear_delta_features(obs.double(), actions) @ linear_coefficients
            episode_metrics[-1]["linear_action_delta_mse"] = float((deltas.double() - predicted_delta).square().mean())
            episode_metrics[-1]["behavior_policy"] = episode_behavior(episode.seed, data_protocol)
    state_gain = [row["persistence_state_delta_mse"] - row["candidate_state_delta_mse"] for row in episode_metrics]
    reward_gain = [row["action_mean_reward_mae"] - row["candidate_reward_mae"] for row in episode_metrics]
    death_gain = [row["training_prior_death_brier"] - row["candidate_death_brier"] for row in episode_metrics]
    result = {
        "episodes": len(episode_metrics),
        "transitions": sum(row["steps"] for row in episode_metrics),
        "death_transitions": sum(row["deaths"] for row in episode_metrics),
        "metrics": {
            "candidate_state_delta_mse": _mean(episode_metrics, "candidate_state_delta_mse"),
            "persistence_state_delta_mse": _mean(episode_metrics, "persistence_state_delta_mse"),
            "paired_state_mse_gain_over_persistence": _mean(episode_metrics, "persistence_state_delta_mse") - _mean(episode_metrics, "candidate_state_delta_mse"),
            "paired_state_mse_gain_95pct_bootstrap_interval": _bootstrap_interval(state_gain, seed=episode_metrics[0]["seed"] + 501, reps=bootstrap_reps),
            "candidate_reward_mae": _mean(episode_metrics, "candidate_reward_mae"),
            "action_mean_reward_mae": _mean(episode_metrics, "action_mean_reward_mae"),
            "paired_reward_mae_gain_over_action_mean": _mean(episode_metrics, "action_mean_reward_mae") - _mean(episode_metrics, "candidate_reward_mae"),
            "paired_reward_mae_gain_95pct_bootstrap_interval": _bootstrap_interval(reward_gain, seed=episode_metrics[0]["seed"] + 502, reps=bootstrap_reps),
            "candidate_death_brier": _mean(episode_metrics, "candidate_death_brier"),
            "training_prior_death_brier": _mean(episode_metrics, "training_prior_death_brier"),
            "paired_death_brier_gain_over_prior": _mean(episode_metrics, "training_prior_death_brier") - _mean(episode_metrics, "candidate_death_brier"),
            "paired_death_brier_gain_95pct_bootstrap_interval": _bootstrap_interval(death_gain, seed=episode_metrics[0]["seed"] + 503, reps=bootstrap_reps),
            "state_episode_interval_coverage": sum(row["state_episode_interval_covered"] for row in episode_metrics) / len(episode_metrics),
            "reward_episode_interval_coverage": sum(row["reward_episode_interval_covered"] for row in episode_metrics) / len(episode_metrics),
            "mean_state_interval_half_width": _mean(episode_metrics, "mean_state_interval_half_width"),
            "mean_reward_interval_half_width": _mean(episode_metrics, "mean_reward_interval_half_width"),
        },
        "per_episode": episode_metrics,
    }
    if linear_coefficients is not None:
        linear_gain = [row["linear_action_delta_mse"] - row["candidate_state_delta_mse"] for row in episode_metrics]
        result["metrics"].update({
            "linear_action_delta_mse": _mean(episode_metrics, "linear_action_delta_mse"),
            "paired_state_mse_gain_over_linear": sum(linear_gain) / len(linear_gain),
            "paired_state_mse_gain_over_linear_95pct_bootstrap_interval": _bootstrap_interval(linear_gain, seed=episode_metrics[0]["seed"] + 504, reps=bootstrap_reps),
        })
        result["death_target_status"] = ("uncalibrated_both_classes_present" if result["death_transitions"] else
                                          "unevaluated_no_positive_test_labels")
        result["by_behavior"] = {}
        for behavior in sorted({row["behavior_policy"] for row in episode_metrics}):
            group = [row for row in episode_metrics if row["behavior_policy"] == behavior]
            result["by_behavior"][behavior] = {
                "episodes": len(group), "death_transitions": sum(row["deaths"] for row in group),
                "candidate_state_delta_mse": _mean(group, "candidate_state_delta_mse"),
                "linear_action_delta_mse": _mean(group, "linear_action_delta_mse"),
                "state_episode_interval_coverage": _mean(group, "state_episode_interval_covered"),
                "reward_episode_interval_coverage": _mean(group, "reward_episode_interval_covered"),
                "mean_state_interval_half_width": _mean(group, "mean_state_interval_half_width"),
            }
    return result


def evaluate_oracle(
    artifact_path: str | Path,
    output_dir: str | Path,
    *,
    base_seed: int = 442_000_000,
    episodes_per_condition: int = 32,
    bootstrap_reps: int = 2000,
) -> dict[str, Any]:
    """Evaluate on new ID and scarcity-shift episodes; never fits or promotes."""
    if type(base_seed) is not int or not 0 <= base_seed <= 2**63 - 5_000_000:
        raise ValueError("base_seed must leave room for disjoint evaluation seed namespaces")
    if type(episodes_per_condition) is not int or not 1 <= episodes_per_condition <= 10000:
        raise ValueError("episodes_per_condition must be an integer in [1, 10000]")
    if type(bootstrap_reps) is not int or not 100 <= bootstrap_reps <= 50000:
        raise ValueError("bootstrap_reps must be an integer in [100, 50000]")
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError("evaluation output directory must be empty; previous receipts are preserved")
    artifact_file = Path(artifact_path).resolve()
    payload = load_oracle_artifact(artifact_file)
    model = _load_model(payload)
    fit_manifest = payload["training_manifest"]
    known_seeds = {seed for values in fit_manifest["seed_sets"].values() for seed in values}
    seed_sets = {
        "id_test": list(range(base_seed + 3_000_000, base_seed + 3_000_000 + episodes_per_condition)),
        "ood_scarcity_test": list(range(base_seed + 4_000_000, base_seed + 4_000_000 + episodes_per_condition)),
    }
    all_eval_seeds = [seed for values in seed_sets.values() for seed in values]
    if len(all_eval_seeds) != len(set(all_eval_seeds)) or known_seeds.intersection(all_eval_seeds):
        raise ValueError("evaluation episode seeds overlap fit partitions or each other")
    max_steps = int(fit_manifest["max_steps"])
    data_protocol = fit_manifest.get("data_protocol", LEGACY_DATA_PROTOCOL)
    test_splits = {
        "id_test": collect_split("id_test", seed_sets["id_test"], max_steps=max_steps,
                                  fixed_scarcity=1.25 if data_protocol == LEGACY_DATA_PROTOCOL else None,
                                  data_protocol=data_protocol),
        "ood_scarcity_test": collect_split("ood_scarcity_test", seed_sets["ood_scarcity_test"], max_steps=max_steps,
                                           fixed_scarcity=2.5, data_protocol=data_protocol),
    }
    summary_by_condition = {}
    for name, split in test_splits.items():
        summary_by_condition[name] = _evaluate_split(model, split, payload, bootstrap_reps)

    source = source_hashes()
    # Evaluation code is added separately to the training sources so the exact
    # analysis logic is visible even if the fitted artifact predates it.
    source["candidate_evaluation.py"] = _file_sha256(Path(__file__).resolve())
    training_summaries = {name: protocol_split_summary(split, data_protocol) for name, split in test_splits.items()}
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "status": "evaluated_candidate_not_promoted",
        "promoted": False,
        "artifact_path": str(artifact_file),
        "artifact_sha256": _file_sha256(artifact_file),
        "model_sha256": payload["model_sha256"],
        "base_seed": base_seed,
        "episodes_per_condition": episodes_per_condition,
        "max_steps": max_steps,
        "bootstrap_reps": bootstrap_reps,
        "test_seed_sets": seed_sets,
        "training_seed_sets": fit_manifest["seed_sets"],
        "seed_disjointness_verified": True,
        "test_splits": training_summaries,
        "results": summary_by_condition,
        "source_sha256": source,
        "protocol": {
            "id_condition": "TidePool, held-out episode seeds, scarcity 1.25",
            "ood_condition": "TidePool, held-out episode seeds, scarcity 2.5",
            "data_policy": fit_manifest["data_policy"],
            "state_baseline": "zero observation delta (persistence)",
            "reward_baseline": "training-split realized mean reward for the executed action",
            "death_baseline": "training-split smoothed death prevalence",
            "interval_coverage_unit": "all transition features within one complete episode",
            "calibrator": "fit split only; applied without adjustment to both ID and OOD",
        },
        "limits": [
            "This evaluates one-step prediction, not policy reward, planning benefit, or survival improvement.",
            "TidePool is a deterministic synthetic environment; OOD means scarcity shift within TidePool only.",
            "Bootstrap intervals quantify variation over this finite set of held-out episodes and are descriptive at small n.",
            "Episode-level conformal coverage is supported only under exchangeability with the calibration episode distribution; OOD coverage is diagnostic, not guaranteed.",
            "Death Brier scores are not a probability calibration claim; the binary death head was not calibrated.",
        ],
    }
    if data_protocol == MIXTURE_DATA_PROTOCOL:
        receipt["protocol"].update({
            "data_protocol": data_protocol,
            "id_condition": "held-out seeds from the identical frozen calibration environment/behavior mixture",
            "id_generation": generation_protocol(data_protocol),
            "state_linear_baseline": "training-only ridge affine observation delta conditioned on executed action",
            "calibrator": "disjoint calibration episodes only; applied unchanged to both ID and OOD",
        })
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    out_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_path / "evaluation_receipt.json", receipt)
    return receipt


def verify_evaluation_receipt(receipt_path: str | Path, *, replay: bool = True) -> dict[str, Any]:
    path = Path(receipt_path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("ORACLE evaluation receipt is missing or too large")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or stored.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("unsupported ORACLE evaluation receipt")
    expected = stored.pop("receipt_sha256", None)
    if not isinstance(expected, str) or hashlib.sha256(canonical_bytes(stored)).hexdigest() != expected:
        raise ValueError("ORACLE evaluation receipt digest mismatch")
    artifact_path = Path(stored["artifact_path"])
    if not artifact_path.is_file() or _file_sha256(artifact_path) != stored["artifact_sha256"]:
        raise ValueError("ORACLE evaluation artifact hash mismatch")
    current_sources = source_hashes() | {"candidate_evaluation.py": _file_sha256(Path(__file__).resolve())}
    if current_sources != stored["source_sha256"]:
        raise ValueError("ORACLE evaluation source fingerprints changed")
    if replay:
        with tempfile.TemporaryDirectory(prefix="poseidon-oracle-replay-") as temporary:
            replayed = evaluate_oracle(artifact_path, temporary, base_seed=stored["base_seed"],
                                       episodes_per_condition=stored["episodes_per_condition"],
                                       bootstrap_reps=stored["bootstrap_reps"])
        if replayed != {**stored, "receipt_sha256": expected}:
            raise ValueError("ORACLE evaluation metrics or generated test episodes do not replay")
    return {"valid": True, "replayed": replay, "artifact_sha256": stored["artifact_sha256"],
            "model_sha256": stored["model_sha256"], "conditions": list(stored["results"])}
