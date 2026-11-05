"""Train and calibrate ORACLE without changing Poseidon's active policy.

Each model member bootstraps complete training episodes. Checkpoint selection
uses disjoint episodes, uncertainty intervals use a second disjoint calibration
split, and final ID/OOD behavior belongs to candidate_evaluation.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from ..world import N_ACTIONS, TidePool, WORLD_VERSION, event_uniform, teacher_action
from .contracts import (Episode, EpisodeSplit, TransitionRecord, assert_disjoint_splits,
                        canonical_bytes, split_summary, tidepool_contract)
from .oracle_model import (ORACLE_SCHEMA, OracleConfig, OracleEnsemble, linear_delta_features,
                           MLP_ARCHITECTURE, LINEAR_RESIDUAL_ARCHITECTURE, RESIDUAL_SCALE_GRID)

FIT_RECEIPT_SCHEMA = "poseidon-oracle-fit-receipt-v1"
TRAIN_SCARCITY = (0.8, 1.0, 1.25, 1.5)
EXPLORATION_RATE = 0.18
LEGACY_DATA_PROTOCOL = "legacy-v1"
MIXTURE_DATA_PROTOCOL = "episode-mixture-v2"
BEHAVIOR_MIXTURE = (("teacher", 0.35), ("exploratory", 0.30),
                    ("uniform_random", 0.25), ("failure_rest", 0.10))
TARGET_SEMANTICS = {"state": "next_observation minus observation",
                    "reward": "TidePool realized one-step reward",
                    "death": "death on this transition; horizon truncation excluded"}


def generation_protocol(data_protocol: str) -> dict[str, Any]:
    """Registered immutable laws, shared by calibration and ID test collection."""
    if data_protocol == LEGACY_DATA_PROTOCOL:
        return {"version": LEGACY_DATA_PROTOCOL, "scarcity_cycle": list(TRAIN_SCARCITY),
                "policy": "teacher_action with deterministic event-keyed uniform-action exploration",
                "exploration_rate": EXPLORATION_RATE}
    if data_protocol != MIXTURE_DATA_PROTOCOL:
        raise ValueError("unsupported ORACLE data protocol")
    return {"version": MIXTURE_DATA_PROTOCOL, "scarcity_values": list(TRAIN_SCARCITY),
            "scarcity_sampling": "seed-keyed uniform per episode",
            "policy_sampling": "seed-keyed categorical once per complete episode",
            "behavior_mixture": [{"policy": policy, "probability": probability}
                                 for policy, probability in BEHAVIOR_MIXTURE],
            "exploratory_action_probability": EXPLORATION_RATE,
            "failure_rest_action": 0,
            "event_namespace": "oracle-episode-mixture-v2"}


def episode_behavior(seed: int, data_protocol: str) -> str:
    generation_protocol(data_protocol)
    if data_protocol == LEGACY_DATA_PROTOCOL:
        return "exploratory"
    draw = event_uniform(seed, 0, "oracle-episode-mixture-v2", "behavior")
    cumulative = 0.0
    for policy, probability in BEHAVIOR_MIXTURE:
        cumulative += probability
        if draw < cumulative:
            return policy
    return BEHAVIOR_MIXTURE[-1][0]


def protocol_split_summary(split: EpisodeSplit, data_protocol: str) -> dict:
    summary = split_summary(split, include_exposure=data_protocol == MIXTURE_DATA_PROTOCOL)
    if data_protocol == MIXTURE_DATA_PROTOCOL:
        summary["behavior_episode_counts"] = {
            policy: sum(episode_behavior(episode.seed, data_protocol) == policy for episode in split.episodes)
            for policy, _ in BEHAVIOR_MIXTURE}
    return summary


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    paths = {
        "world.py": package / "world.py",
        "contracts.py": package / "research" / "contracts.py",
        "oracle_model.py": package / "research" / "oracle_model.py",
        "oracle_training.py": package / "research" / "oracle_training.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_episode(seed: int, scarcity: float, max_steps: int = 96,
                    exploration_rate: float = EXPLORATION_RATE, *,
                    behavior_policy: str = "exploratory") -> Episode:
    """Collect one complete, observation-only teacher-plus-exploration episode."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if isinstance(scarcity, bool) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4:
        raise ValueError("scarcity must be finite and in [0.5, 4]")
    if type(max_steps) is not int or not 1 <= max_steps <= 10000:
        raise ValueError("max_steps must be an integer in [1, 10000]")
    if isinstance(exploration_rate, bool) or not math.isfinite(exploration_rate) or not 0 <= exploration_rate <= 1:
        raise ValueError("exploration_rate must be finite and in [0, 1]")
    if behavior_policy not in {policy for policy, _ in BEHAVIOR_MIXTURE}:
        raise ValueError("unsupported episode behavior policy")
    env = TidePool(seed=seed, scarcity=float(scarcity), max_steps=max_steps)
    observation = env.reset(seed)
    rows = []
    done = False
    while not done:
        if behavior_policy == "failure_rest":
            action = 0
        elif behavior_policy == "uniform_random" or (behavior_policy == "exploratory" and
                event_uniform(seed, env.tick, "oracle-data-policy", "explore") < exploration_rate):
            draw = event_uniform(seed, env.tick, "oracle-data-policy", "action")
            action = min(int(draw * N_ACTIONS), N_ACTIONS - 1)
        else:
            action = teacher_action(observation)
        next_observation, reward, done, info = env.step(action)
        row = TransitionRecord(
            episode_seed=seed,
            tick=env.tick - 1,
            scarcity=float(scarcity),
            observation=tuple(float(value) for value in observation),
            action=int(action),
            next_observation=tuple(float(value) for value in next_observation),
            reward=float(reward),
            death=bool(info["death"]),
            truncated=bool(info["truncated"]),
        )
        row.validate()
        rows.append(row)
        observation = next_observation
    episode = Episode(seed=seed, scarcity=float(scarcity), transitions=tuple(rows))
    episode.validate()
    return episode


def collect_split(name: str, seeds: Sequence[int], *, max_steps: int,
                  scarcity_values: Sequence[float] = TRAIN_SCARCITY,
                  fixed_scarcity: float | None = None,
                  exploration_rate: float = EXPLORATION_RATE,
                  data_protocol: str = LEGACY_DATA_PROTOCOL) -> EpisodeSplit:
    generation_protocol(data_protocol)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} seeds must be nonempty and unique")
    if not scarcity_values:
        raise ValueError("scarcity_values cannot be empty")
    episodes = []
    for index, seed in enumerate(seeds):
        if data_protocol == MIXTURE_DATA_PROTOCOL:
            index = min(int(event_uniform(seed, 0, "oracle-episode-mixture-v2", "scarcity") * len(scarcity_values)),
                        len(scarcity_values) - 1)
        scarcity = float(fixed_scarcity if fixed_scarcity is not None else scarcity_values[index % len(scarcity_values)])
        episodes.append(collect_episode(seed, scarcity, max_steps, exploration_rate,
                                        behavior_policy=episode_behavior(seed, data_protocol)))
    split = EpisodeSplit(name, tuple(episodes))
    split.validate()
    return split


def _flatten(split: EpisodeSplit) -> list[TransitionRecord]:
    return [row for episode in split.episodes for row in episode.transitions]


def _batch(rows: Sequence[TransitionRecord]) -> tuple[torch.Tensor, ...]:
    obs = torch.tensor([row.observation for row in rows], dtype=torch.float32)
    actions = torch.tensor([row.action for row in rows], dtype=torch.long)
    deltas = torch.tensor([row.next_observation for row in rows], dtype=torch.float32) - obs
    rewards = torch.tensor([row.reward for row in rows], dtype=torch.float32)
    deaths = torch.tensor([row.death for row in rows], dtype=torch.float32)
    return obs, actions, deltas, rewards, deaths


def _fit_linear_delta(rows: Sequence[TransitionRecord]) -> dict:
    obs, actions, deltas, _, _ = _batch(rows)
    features = linear_delta_features(obs.double(), actions)
    ridge = 1e-3
    gram = features.T @ features + ridge * torch.eye(features.shape[1], dtype=torch.float64)
    coefficients = torch.linalg.solve(gram, features.T @ deltas.double())
    return {"method": "action_conditioned_affine_delta_ridge_v1", "ridge": ridge,
            "fit_partition": "train", "coefficients": coefficients.tolist()}


def _bootstrap_exposure(split: EpisodeSplit, draws: list[list[int]]) -> list[dict]:
    episodes = {episode.seed: episode for episode in split.episodes}
    result = []
    for index, seeds in enumerate(draws):
        rows = [row for seed in seeds for row in episodes[seed].transitions]
        result.append({"member": index, "episode_draw_sha256": _sha256_bytes(canonical_bytes(seeds)),
                       "unique_episodes": len(set(seeds)),
                       "episode_multiplicities": [{"seed": seed, "draws": seeds.count(seed)} for seed in sorted(set(seeds))],
                       "pool_transitions": len(rows),
                       "pool_death_transitions": sum(row.death for row in rows),
                       "pool_action_counts": [sum(row.action == action for row in rows) for action in range(N_ACTIONS)]})
    return result


def _loss(output: dict[str, torch.Tensor], deltas: torch.Tensor, rewards: torch.Tensor,
          deaths: torch.Tensor, positive_weight: float) -> torch.Tensor:
    delta_logv = output["delta_log_variance"]
    delta_nll = 0.5 * (delta_logv + (deltas - output["delta_mean"]).square() * torch.exp(-delta_logv))
    reward_logv = output["reward_log_variance"]
    reward_nll = 0.5 * (reward_logv + (rewards - output["reward_mean"]).square() * torch.exp(-reward_logv))
    death_loss = F.binary_cross_entropy_with_logits(
        output["death_logit"], deaths, pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    return delta_nll.mean() + 0.35 * reward_nll.mean() + 0.20 * death_loss


@torch.inference_mode()
def _selection_loss(model: OracleEnsemble, split: EpisodeSplit, positive_weight: float, *,
                    residual_scale: float | None = None) -> float:
    rows = _flatten(split)
    obs, actions, deltas, rewards, deaths = _batch(rows)
    prediction = model(obs, actions, residual_scale=residual_scale)
    losses = [_loss({name: prediction[name][index] for name in prediction}, deltas, rewards, deaths, positive_weight)
              for index in range(model.config.ensemble_size)]
    result = float(torch.stack(losses).mean())
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite ORACLE selection loss")
    return result


@torch.inference_mode()
def _residual_selection(model: OracleEnsemble, split: EpisodeSplit) -> dict[str, Any]:
    """Frozen shrink grid, scored on equal-weight complete selection episodes.

    Read the raw neural residual independently of the saved scale. This makes
    both selection and later replay valid even when the selected scale is zero.
    """
    if model.config.architecture != LINEAR_RESIDUAL_ARCHITECTURE or split.name != "selection":
        raise ValueError("residual scaling requires the registered selection partition")
    split.validate()
    scores = [{"scale": scale, "state_mse": 0.0} for scale in RESIDUAL_SCALE_GRID]
    for episode in split.episodes:
        obs, actions, deltas, _, _ = _batch(episode.transitions)
        raw = model(obs, actions, residual_scale=1.0)["delta_residual_mean"]
        anchor = model.members[0].anchor_delta(obs, actions).unsqueeze(0)
        for score in scores:
            prediction = (anchor + score["scale"] * raw).mean(dim=0)
            mse = float((deltas.double() - prediction).square().mean())
            if not math.isfinite(mse):
                raise FloatingPointError("nonfinite residual selection score")
            score["state_mse"] += mse
    for score in scores:
        score["state_mse"] /= len(split.episodes)
    chosen = min(scores, key=lambda score: (score["state_mse"], score["scale"]))
    return {"method": "selection_state_mse_grid_v1", "partition": "selection",
            "aggregation": "equal_weight_episode_mean", "tie_break": "smaller_scale",
            "scores": scores, "selected_scale": chosen["scale"],
            "selection_episodes": len(split.episodes),
            "checkpoint_nll_residual_scale": 1.0}


@torch.inference_mode()
def episode_calibration_scores(model: OracleEnsemble, episode: Episode) -> tuple[float, float]:
    rows = episode.transitions
    obs, actions, deltas, rewards, _ = _batch(rows)
    summary = model.summarize(model(obs, actions))
    delta_scale = summary["delta_total_variance"].sqrt().clamp_min(1e-4)
    reward_scale = summary["reward_total_variance"].sqrt().clamp_min(1e-4)
    state_score = ((deltas - summary["delta_mean"]).abs() / delta_scale).max()
    reward_score = ((rewards - summary["reward_mean"]).abs() / reward_scale).max()
    result = float(state_score), float(reward_score)
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise FloatingPointError("invalid episode calibration score")
    return result


def conformal_order_statistic(scores: Sequence[float], coverage: float) -> float:
    if not scores or any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("calibration scores must be finite, nonnegative, and nonempty")
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0.5 <= coverage < 1:
        raise ValueError("coverage must be in [0.5, 1)")
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError("calibration split is too small for the requested finite-sample coverage")
    return float(sorted(scores)[rank - 1])


def _state_to_cpu(model: OracleEnsemble) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _derive_seeds(base_seed: int, train_episodes: int, selection_episodes: int,
                  calibration_episodes: int) -> dict[str, list[int]]:
    if type(base_seed) is not int or not 0 <= base_seed <= 2**63 - 3_000_000:
        raise ValueError("base_seed must leave room for four disjoint million-seed namespaces")
    result = {
        "train": list(range(base_seed, base_seed + train_episodes)),
        "selection": list(range(base_seed + 1_000_000, base_seed + 1_000_000 + selection_episodes)),
        "calibration": list(range(base_seed + 2_000_000, base_seed + 2_000_000 + calibration_episodes)),
    }
    if any(seed >= 2**63 for values in result.values() for seed in values):
        raise ValueError("derived episode seed exceeds the contract")
    if len({seed for values in result.values() for seed in values}) != sum(map(len, result.values())):
        raise ValueError("derived episode splits overlap")
    return result


def fit_oracle(
    output_dir: str | Path,
    *,
    base_seed: int = 442_000_000,
    train_episodes: int = 32,
    selection_episodes: int = 8,
    calibration_episodes: int = 12,
    max_steps: int = 96,
    training_updates: int = 256,
    batch_size: int = 128,
    hidden_size: int = 128,
    ensemble_size: int = 5,
    coverage: float = 0.90,
    data_protocol: str = MIXTURE_DATA_PROTOCOL,
    architecture: str = MLP_ARCHITECTURE,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit an independently initialized, episode-bootstrap predictive ensemble.

    Only train episodes optimize weights. Selection episodes choose the saved
    update, and calibration episodes set episode-level split-conformal widths.
    No test episodes are collected or accessed here.
    """
    for name, value, low, high in (
        ("train_episodes", train_episodes, 2, 10000),
        ("selection_episodes", selection_episodes, 2, 1000),
        ("calibration_episodes", calibration_episodes, 2, 1000),
        ("max_steps", max_steps, 2, 10000),
        ("training_updates", training_updates, 1, 100000),
        ("batch_size", batch_size, 8, 8192),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    generation = generation_protocol(data_protocol)
    config = OracleConfig(hidden_size=hidden_size, ensemble_size=ensemble_size, architecture=architecture)
    config.validate()
    if architecture == LINEAR_RESIDUAL_ARCHITECTURE and data_protocol != MIXTURE_DATA_PROTOCOL:
        raise ValueError("linear-residual-v1 requires episode-mixture-v2")
    if type(base_seed) is not int or not 0 <= base_seed <= 2**63 - 3_000_000:
        raise ValueError("base_seed must leave room for four disjoint million-seed namespaces")
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0.5 <= coverage < 1:
        raise ValueError("coverage must be in [0.5, 1)")
    minimum_calibration_episodes = math.ceil(coverage / (1 - coverage) - 1e-12)
    if calibration_episodes < minimum_calibration_episodes:
        raise ValueError(f"coverage {coverage} requires at least {minimum_calibration_episodes} calibration episodes")
    seed_groups = _derive_seeds(base_seed, train_episodes, selection_episodes, calibration_episodes)
    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        raise FileExistsError("ORACLE output directory must be empty; existing candidates are preserved")
    contract = tidepool_contract()
    if progress:
        progress({"phase": "collect", "split": "train", "completed": 0, "total": train_episodes})
    splits = {
        name: collect_split(name, seeds, max_steps=max_steps, data_protocol=data_protocol)
        for name, seeds in seed_groups.items()
    }
    split_manifest = assert_disjoint_splits(splits)
    train_rows = _flatten(splits["train"])
    positive_count = sum(row.death for row in train_rows)
    negative_count = len(train_rows) - positive_count
    positive_weight = min(30.0, negative_count / max(1, positive_count)) if positive_count else 1.0

    linear_baseline = _fit_linear_delta(train_rows) if data_protocol == MIXTURE_DATA_PROTOCOL else None
    linear_coefficients = (torch.tensor(linear_baseline["coefficients"], dtype=torch.float64)
                           if architecture == LINEAR_RESIDUAL_ARCHITECTURE else None)
    model = OracleEnsemble(config, seed=base_seed + 71_003, linear_coefficients=linear_coefficients)
    optimizers = [torch.optim.AdamW(member.parameters(), lr=3e-4, weight_decay=1e-5)
                  for member in model.members]
    training_groups = {episode.seed: list(episode.transitions) for episode in splits["train"].episodes}
    bootstrap_pools: list[list[TransitionRecord]] = []
    bootstrap_draws: list[list[int]] = []
    member_rngs: list[random.Random] = []
    for member_index in range(ensemble_size):
        rng = random.Random(base_seed + 83_117 + member_index * 1_000_003)
        sampled_episode_seeds = [rng.choice(seed_groups["train"]) for _ in seed_groups["train"]]
        bootstrap_draws.append(sampled_episode_seeds)
        bootstrap_pools.append([row for seed in sampled_episode_seeds for row in training_groups[seed]])
        member_rngs.append(rng)

    best_score = math.inf
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    selection_history = []
    eval_interval = max(1, min(32, training_updates // 8 or 1))
    model.train()
    for update in range(1, training_updates + 1):
        member_losses = []
        for index, member in enumerate(model.members):
            pool = bootstrap_pools[index]
            rng = member_rngs[index]
            rows = [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
            obs, actions, deltas, rewards, deaths = _batch(rows)
            prediction = member(obs, actions)
            loss = _loss(prediction, deltas, rewards, deaths, positive_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite ORACLE training loss; no candidate was written")
            optimizers[index].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), max_norm=2.0, error_if_nonfinite=True)
            optimizers[index].step()
            member_losses.append(float(loss.detach()))
        if update % eval_interval == 0 or update == training_updates:
            model.eval()
            score = _selection_loss(model, splits["selection"], positive_weight)
            selection_history.append({"update": update, "selection_nll": score,
                                     "mean_bootstrap_train_loss": sum(member_losses) / len(member_losses)})
            if score < best_score:
                best_score = score
                best_update = update
                best_state = _state_to_cpu(model)
            if progress:
                progress({"phase": "train", "completed": update, "total": training_updates,
                          "selection_nll": score, "best_update": best_update})
            model.train()
    if best_state is None:
        raise RuntimeError("ORACLE selection did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    residual_selection = None
    if architecture == LINEAR_RESIDUAL_ARCHITECTURE:
        residual_selection = _residual_selection(model, splits["selection"])
        model.set_residual_scale(residual_selection["selected_scale"])
        best_state = _state_to_cpu(model)

    state_scores, reward_scores = [], []
    calibration_episode_rows = []
    for episode in splits["calibration"].episodes:
        state_score, reward_score = episode_calibration_scores(model, episode)
        state_scores.append(state_score)
        reward_scores.append(reward_score)
        calibration_episode_rows.append({"seed": episode.seed, "state_max_score": state_score,
                                         "reward_max_score": reward_score})
    state_quantile = conformal_order_statistic(state_scores, coverage)
    reward_quantile = conformal_order_statistic(reward_scores, coverage)

    reward_sums = [0.0] * config.action_count
    reward_counts = [0] * config.action_count
    death_count_by_action = [0] * config.action_count
    for row in train_rows:
        reward_sums[row.action] += row.reward
        reward_counts[row.action] += 1
        death_count_by_action[row.action] += int(row.death)
    global_reward_mean = sum(row.reward for row in train_rows) / len(train_rows)
    reward_mean_by_action = [reward_sums[action] / reward_counts[action] if reward_counts[action] else global_reward_mean
                             for action in range(config.action_count)]
    death_prior = (positive_count + 1) / (len(train_rows) + 2)
    summaries = {name: protocol_split_summary(split, data_protocol) for name, split in splits.items()}
    source_digest = source_hashes()
    training_manifest = {
        "base_seed": base_seed,
        "seed_sets": split_manifest,
        "split_summaries": summaries,
        "max_steps": max_steps,
        "train_scarcity_cycle": list(TRAIN_SCARCITY),
        "exploration_rate": EXPLORATION_RATE,
        "data_policy": (generation["policy"] if data_protocol == LEGACY_DATA_PROTOCOL else
                        "frozen seed-keyed mixture of whole-episode teacher, exploratory, uniform_random and failure_rest policies"),
        "target_semantics": TARGET_SEMANTICS,
        "source_sha256": source_digest,
        "environment_contract": {"version": contract.version, "world_version": contract.world_version,
                                  "observation_names": list(contract.observation_names),
                                  "actions": list(contract.actions)},
    }
    if data_protocol == MIXTURE_DATA_PROTOCOL:
        training_manifest.update({"data_protocol": data_protocol, "generation": generation})
    model_sha = model_state_sha256(best_state)
    artifact_payload = {
        "schema": ORACLE_SCHEMA,
        "config": config.to_dict(),
        "model_seed": base_seed + 71_003,
        "model_state_dict": best_state,
        "model_sha256": model_sha,
        "training_manifest": training_manifest,
        "training": {"updates_requested": training_updates, "selected_update": best_update,
                     "selection_nll": best_score, "selection_history": selection_history,
                     "training_positive_death_labels": positive_count,
                     "death_positive_weight": positive_weight,
                     "bootstrap_episode_draws": bootstrap_draws,
                     "reward_mean_by_action": reward_mean_by_action,
                     "death_probability_prior": death_prior,
                     "death_count_by_action": death_count_by_action},
        "calibration": {"method": "episode_max_split_conformal_v1",
                        "coverage": coverage,
                        "exchangeability_unit": "complete episode seed",
                        "calibration_episode_count": calibration_episodes,
                        "state_standardized_error_quantile": state_quantile,
                        "reward_standardized_error_quantile": reward_quantile,
                        "state_episode_scores": state_scores,
                        "reward_episode_scores": reward_scores,
                        "episode_scores": calibration_episode_rows,
                        "minimum_calibration_episodes": minimum_calibration_episodes,
                        "scope": "ID calibration distribution only; no OOD coverage guarantee"},
        "activated": False,
    }
    if data_protocol == MIXTURE_DATA_PROTOCOL:
        artifact_payload["training"].update({
            "action_counts": reward_counts,
            "training_negative_death_labels": negative_count,
            "death_target_status": "uncalibrated_both_classes" if positive_count else "unevaluated_no_positive_training_labels",
            "batch_size": batch_size,
            "bootstrap_member_exposure": _bootstrap_exposure(splits["train"], bootstrap_draws),
            "linear_delta_baseline": linear_baseline,
        })
    if residual_selection is not None:
        artifact_payload["training"]["residual_selection"] = residual_selection
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_path = out_path / "oracle.pt"
    artifact_tmp = out_path / "oracle.pt.tmp"
    try:
        torch.save(artifact_payload, artifact_tmp)
        os.replace(artifact_tmp, artifact_path)
    finally:
        if artifact_tmp.exists():
            artifact_tmp.unlink()
    receipt = {
        "schema": FIT_RECEIPT_SCHEMA,
        "status": "trained_and_calibrated_candidate",
        "activated": False,
        "world_version": WORLD_VERSION,
        "model_sha256": model_sha,
        "artifact_sha256": _file_sha256(artifact_path),
        "artifact_path": str(artifact_path.resolve()),
        "config": config.to_dict(),
        "training_manifest": training_manifest,
        "training": artifact_payload["training"],
        "calibration": artifact_payload["calibration"],
        "limits": [
            "This artifact predicts TidePool transitions; it does not control or improve the active policy.",
            "Calibration is episode-level and applies to the registered ID distribution, not shifted scarcity.",
            "The synthetic TidePool benchmark is not evidence of biological, physical-world, or Unity transfer.",
            "Death scores are reported with Brier metrics later and are not probability-calibrated by this fit.",
        ],
    }
    receipt["receipt_sha256"] = _sha256_bytes(canonical_bytes(receipt))
    _atomic_json(out_path / "fit_receipt.json", receipt)
    if progress:
        progress({"phase": "completed", "completed": 1, "total": 1,
                  "artifact_sha256": receipt["artifact_sha256"], "model_sha256": model_sha})
    return receipt


def _bounded_read(path: Path, limit: int) -> bytes:
    if not path.is_file():
        raise ValueError("ORACLE evidence file is missing")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("ORACLE evidence file exceeds its size limit")
    return value


def _integer(value: Any, name: str, low: int = 0, high: int = 2**63 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid ORACLE {name}")
    return value


def _number(value: Any, name: str, low: float = -math.inf, high: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"invalid ORACLE {name}")
    return float(value)


def _digest(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("invalid ORACLE sha256")


def _counts(values: Any, total: int, name: str) -> None:
    if not isinstance(values, list) or len(values) != N_ACTIONS:
        raise ValueError(f"invalid ORACLE {name}")
    if sum(_integer(value, name) for value in values) != total:
        raise ValueError(f"ORACLE {name} total is inconsistent")


def _same(actual: Any, expected: Any, name: str) -> None:
    # Canonical comparison also distinguishes true from 1 and rejects NaNs.
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise ValueError(f"ORACLE {name} mismatch")


def _validate_manifest(manifest: dict) -> str:
    required = {"base_seed", "seed_sets", "split_summaries", "max_steps", "train_scarcity_cycle",
                "exploration_rate", "data_policy", "target_semantics", "source_sha256", "environment_contract"}
    if not isinstance(manifest, dict):
        raise ValueError("invalid ORACLE training manifest")
    protocol = manifest.get("data_protocol", LEGACY_DATA_PROTOCOL)
    generation = generation_protocol(protocol)
    if protocol == MIXTURE_DATA_PROTOCOL:
        required |= {"data_protocol", "generation"}
        _same(manifest.get("generation"), generation, "generation protocol")
    elif "data_protocol" in manifest:
        required.add("data_protocol")
    if set(manifest) != required:
        raise ValueError("invalid ORACLE manifest fields")
    _same(manifest["train_scarcity_cycle"], list(TRAIN_SCARCITY), "scarcity protocol")
    _same(manifest["exploration_rate"], EXPLORATION_RATE, "exploration protocol")
    policy = (generation["policy"] if protocol == LEGACY_DATA_PROTOCOL else
              "frozen seed-keyed mixture of whole-episode teacher, exploratory, uniform_random and failure_rest policies")
    _same(manifest["data_policy"], policy, "data policy")
    _same(manifest["target_semantics"], TARGET_SEMANTICS, "target semantics")
    contract = tidepool_contract()
    _same(manifest["environment_contract"], {"version": contract.version, "world_version": contract.world_version,
          "observation_names": list(contract.observation_names), "actions": list(contract.actions)}, "environment contract")
    if not isinstance(manifest["source_sha256"], dict) or set(manifest["source_sha256"]) != set(source_hashes()):
        raise ValueError("incomplete ORACLE source identity")
    for digest in manifest["source_sha256"].values():
        _digest(digest)
    max_steps = _integer(manifest["max_steps"], "max_steps", 2, 10000)
    seeds = manifest["seed_sets"]
    summaries = manifest["split_summaries"]
    if not isinstance(seeds, dict) or set(seeds) != {"train", "selection", "calibration"} or set(summaries) != set(seeds):
        raise ValueError("ORACLE requires complete train, selection and calibration partitions")
    all_seeds = []
    for name, values in seeds.items():
        cap = 10000 if name == "train" else 1000
        if not isinstance(values, list) or not 2 <= len(values) <= cap:
            raise ValueError("invalid ORACLE partition size")
        all_seeds.extend(_integer(seed, "episode seed") for seed in values)
        summary = summaries[name]
        _same(summary["episode_seeds"], values, "partition episode identity")
        if summary["name"] != name or type(summary["episodes"]) is not int or summary["episodes"] != len(values):
            raise ValueError("ORACLE partition summary identity mismatch")
        transitions = _integer(summary["transitions"], "partition transitions", len(values), len(values) * max_steps)
        deaths = _integer(summary["deaths"], "partition deaths", 0, len(values))
        _digest(summary["sha256"])
        if not isinstance(summary["scarcity_values"], list) or not summary["scarcity_values"]:
            raise ValueError("invalid ORACLE partition scarcity values")
        for scarcity in summary["scarcity_values"]:
            _number(scarcity, "scarcity", 0.5, 4)
        if protocol == MIXTURE_DATA_PROTOCOL:
            _counts(summary["action_counts"], transitions, "partition action counts")
            _counts(summary["death_count_by_action"], deaths, "partition death counts")
            if _integer(summary["truncations"], "truncations") + deaths != len(values):
                raise ValueError("ORACLE partition terminal counts disagree")
            episodes = summary["per_episode"]
            if not isinstance(episodes, list) or len(episodes) != len(values):
                raise ValueError("incomplete ORACLE episode exposure")
            for row, seed in zip(episodes, values):
                if row["seed"] != seed or type(row["death"]) is not bool:
                    raise ValueError("invalid ORACLE episode exposure identity")
                _counts(row["action_counts"], _integer(row["steps"], "episode steps", 1, max_steps), "episode action counts")
            _same(summary["action_counts"], [sum(row["action_counts"][action] for row in episodes) for action in range(N_ACTIONS)], "episode action totals")
            if sum(row["steps"] for row in episodes) != transitions or sum(row["death"] for row in episodes) != deaths:
                raise ValueError("ORACLE episode exposure totals disagree")
            _same(summary["behavior_episode_counts"], {policy: sum(episode_behavior(seed, protocol) == policy for seed in values)
                  for policy, _ in BEHAVIOR_MIXTURE}, "behavior exposure")
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("ORACLE episode partitions overlap")
    _same(seeds, _derive_seeds(manifest["base_seed"], len(seeds["train"]), len(seeds["selection"]), len(seeds["calibration"])), "seed namespace protocol")
    return protocol


def _validate_calibration(calibration: dict, seeds: list[int]) -> None:
    required = {"method", "coverage", "exchangeability_unit", "calibration_episode_count",
                "state_standardized_error_quantile", "reward_standardized_error_quantile",
                "state_episode_scores", "reward_episode_scores", "episode_scores",
                "minimum_calibration_episodes", "scope"}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError("invalid ORACLE calibration fields")
    if (calibration["method"] != "episode_max_split_conformal_v1" or
            calibration["exchangeability_unit"] != "complete episode seed" or
            calibration["scope"] != "ID calibration distribution only; no OOD coverage guarantee"):
        raise ValueError("unsupported ORACLE calibration semantics")
    coverage = _number(calibration["coverage"], "coverage", 0.5, 1)
    if coverage == 1 or _integer(calibration["calibration_episode_count"], "calibration size") != len(seeds):
        raise ValueError("ORACLE calibration size or coverage mismatch")
    minimum = math.ceil(coverage / (1 - coverage) - 1e-12)
    if _integer(calibration["minimum_calibration_episodes"], "minimum calibration size") != minimum:
        raise ValueError("ORACLE calibration finite-sample count mismatch")
    for target in ("state", "reward"):
        scores = calibration[f"{target}_episode_scores"]
        if not isinstance(scores, list) or len(scores) != len(seeds):
            raise ValueError("ORACLE calibration scores are incomplete")
        for score in scores:
            _number(score, "calibration score", 0)
        quantile = _number(calibration[f"{target}_standardized_error_quantile"], "calibration quantile", 0)
        if quantile != conformal_order_statistic(scores, coverage):
            raise ValueError("ORACLE calibration quantile contradicts scores")
    _same(calibration["episode_scores"], [{"seed": seed, "state_max_score": state, "reward_max_score": reward}
          for seed, state, reward in zip(seeds, calibration["state_episode_scores"], calibration["reward_episode_scores"])], "calibration episode scores")


def _validate_training(training: dict, manifest: dict, config: OracleConfig, protocol: str) -> None:
    required = {"updates_requested", "selected_update", "selection_nll", "selection_history",
                "training_positive_death_labels", "death_positive_weight", "bootstrap_episode_draws",
                "reward_mean_by_action", "death_probability_prior", "death_count_by_action"}
    if protocol == MIXTURE_DATA_PROTOCOL:
        required |= {"action_counts", "training_negative_death_labels", "death_target_status", "batch_size",
                     "bootstrap_member_exposure", "linear_delta_baseline"}
    if config.architecture == LINEAR_RESIDUAL_ARCHITECTURE:
        if protocol != MIXTURE_DATA_PROTOCOL:
            raise ValueError("linear-residual-v1 requires episode-mixture-v2")
        required.add("residual_selection")
    if not isinstance(training, dict) or set(training) != required:
        raise ValueError("invalid ORACLE training fields")
    updates = _integer(training["updates_requested"], "training updates", 1, 100000)
    selected = _integer(training["selected_update"], "selected update", 1, updates)
    history = training["selection_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("ORACLE selection history is empty")
    previous = 0
    for row in history:
        previous = _integer(row["update"], "selection update", previous + 1, updates)
        _number(row["selection_nll"], "selection loss")
        _number(row["mean_bootstrap_train_loss"], "train loss")
    best = min(history, key=lambda row: row["selection_nll"])
    if previous != updates or selected != best["update"] or training["selection_nll"] != best["selection_nll"]:
        raise ValueError("ORACLE selected checkpoint contradicts selection history")
    seeds = manifest["seed_sets"]["train"]
    expected_draws = []
    for member in range(config.ensemble_size):
        rng = random.Random(manifest["base_seed"] + 83_117 + member * 1_000_003)
        expected_draws.append([rng.choice(seeds) for _ in seeds])
    _same(training["bootstrap_episode_draws"], expected_draws, "bootstrap episode draws")
    summary = manifest["split_summaries"]["train"]
    positives = _integer(training["training_positive_death_labels"], "training deaths", 0, len(seeds))
    if positives != summary["deaths"]:
        raise ValueError("ORACLE training death count contradicts partition")
    negatives = summary["transitions"] - positives
    weight = min(30.0, negatives / max(1, positives)) if positives else 1.0
    if _number(training["death_positive_weight"], "death class weight") != weight:
        raise ValueError("ORACLE death class weight mismatch")
    if _number(training["death_probability_prior"], "death prior", 0, 1) != (positives + 1) / (summary["transitions"] + 2):
        raise ValueError("ORACLE death prior mismatch")
    _counts(training["death_count_by_action"], positives, "death action counts")
    means = training["reward_mean_by_action"]
    if not isinstance(means, list) or len(means) != N_ACTIONS:
        raise ValueError("invalid ORACLE reward action means")
    for mean in means:
        _number(mean, "reward action mean")
    if protocol == MIXTURE_DATA_PROTOCOL:
        _same(training["action_counts"], summary["action_counts"], "training action support")
        _same(training["death_count_by_action"], summary["death_count_by_action"], "training death support")
        _same(training["training_negative_death_labels"], negatives, "negative death labels")
        _same(training["death_target_status"], "uncalibrated_both_classes" if positives else "unevaluated_no_positive_training_labels", "death target status")
        _integer(training["batch_size"], "batch size", 8, 8192)
        exposure = training["bootstrap_member_exposure"]
        if not isinstance(exposure, list) or len(exposure) != config.ensemble_size:
            raise ValueError("incomplete ORACLE bootstrap member exposure")
        episodes = {row["seed"]: row for row in summary["per_episode"]}
        for index, (member, draws) in enumerate(zip(exposure, expected_draws)):
            expected = {"member": index, "episode_draw_sha256": _sha256_bytes(canonical_bytes(draws)),
                        "unique_episodes": len(set(draws)), "episode_multiplicities": [{"seed": seed, "draws": draws.count(seed)} for seed in sorted(set(draws))],
                        "pool_transitions": sum(episodes[seed]["steps"] for seed in draws),
                        "pool_death_transitions": sum(episodes[seed]["death"] for seed in draws),
                        "pool_action_counts": [sum(episodes[seed]["action_counts"][action] for seed in draws) for action in range(N_ACTIONS)]}
            _same(member, expected, "bootstrap member exposure")
        linear = training["linear_delta_baseline"]
        if (not isinstance(linear, dict) or set(linear) != {"method", "ridge", "fit_partition", "coefficients"}
                or linear["method"] != "action_conditioned_affine_delta_ridge_v1"
                or linear["fit_partition"] != "train" or linear["ridge"] != 1e-3):
            raise ValueError("invalid ORACLE linear delta baseline protocol")
        values = linear["coefficients"]
        if (not isinstance(values, list) or len(values) != N_ACTIONS * (config.observation_size + 1)
                or any(not isinstance(row, list) or len(row) != config.observation_size for row in values)):
            raise ValueError("invalid ORACLE linear baseline coefficient shape")
        for row in values:
            for value in row:
                _number(value, "linear baseline coefficient")
        coefficients = torch.tensor(linear["coefficients"], dtype=torch.float64)
        if coefficients.shape != (N_ACTIONS * (config.observation_size + 1), config.observation_size) or not bool(torch.isfinite(coefficients).all()):
            raise ValueError("invalid ORACLE linear baseline coefficients")
    if config.architecture == LINEAR_RESIDUAL_ARCHITECTURE:
        selection = training["residual_selection"]
        required_selection = {"method", "partition", "aggregation", "tie_break", "scores", "selected_scale",
                              "selection_episodes", "checkpoint_nll_residual_scale"}
        if not isinstance(selection, dict) or set(selection) != required_selection:
            raise ValueError("invalid ORACLE residual selection fields")
        for field, expected in {"method": "selection_state_mse_grid_v1", "partition": "selection",
                                "aggregation": "equal_weight_episode_mean", "tie_break": "smaller_scale",
                                "selection_episodes": len(manifest["seed_sets"]["selection"]),
                                "checkpoint_nll_residual_scale": 1.0}.items():
            _same(selection[field], expected, f"residual selection {field}")
        scores = selection["scores"]
        if not isinstance(scores, list) or len(scores) != len(RESIDUAL_SCALE_GRID):
            raise ValueError("ORACLE residual selection grid is incomplete")
        for row, scale in zip(scores, RESIDUAL_SCALE_GRID):
            if not isinstance(row, dict) or set(row) != {"scale", "state_mse"}:
                raise ValueError("invalid ORACLE residual selection score fields")
            _same(row["scale"], scale, "residual selection grid")
            _number(row["state_mse"], "residual selection MSE", 0)
        best_scale = min(scores, key=lambda row: (row["state_mse"], row["scale"]))["scale"]
        _same(selection["selected_scale"], best_scale, "residual selected scale")


def _validate_artifact(payload: dict[str, Any]) -> None:
    required = {"schema", "config", "model_seed", "model_state_dict", "model_sha256",
                "training_manifest", "training", "calibration", "activated"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != ORACLE_SCHEMA:
        raise ValueError("unsupported ORACLE artifact schema")
    if payload["activated"] is not False:
        raise ValueError("ORACLE candidate must remain opt-in")
    config_data = payload["config"]
    config_fields = {"hidden_size", "ensemble_size", "observation_size", "action_count"}
    if not isinstance(config_data, dict) or set(config_data) not in (config_fields, config_fields | {"architecture"}):
        raise ValueError("invalid ORACLE config")
    config = OracleConfig(**config_data)
    config.validate()
    protocol = _validate_manifest(payload["training_manifest"])
    _same(payload["model_seed"], payload["training_manifest"]["base_seed"] + 71_003, "model seed")
    _validate_training(payload["training"], payload["training_manifest"], config, protocol)
    _validate_calibration(payload["calibration"], payload["training_manifest"]["seed_sets"]["calibration"])
    # Reject non-JSON or nonfinite metadata before it can acquire receipt status.
    canonical_bytes({name: value for name, value in payload.items() if name != "model_state_dict"})
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("invalid ORACLE model tensors")
    anchor_keys = ({f"members.{index}.linear_coefficients" for index in range(config.ensemble_size)}
                   if config.architecture == LINEAR_RESIDUAL_ARCHITECTURE else set())
    for name, tensor in state.items():
        expected_dtype = torch.float64 if name in anchor_keys else torch.float32
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != expected_dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError("ORACLE model tensors must be finite values with registered dtypes")
    if model_state_sha256(payload["model_state_dict"]) != payload["model_sha256"]:
        raise ValueError("ORACLE model state hash mismatch")
    _load_model(payload)  # strict keys and tensor shapes
    if config.architecture == LINEAR_RESIDUAL_ARCHITECTURE:
        coefficients = torch.tensor(payload["training"]["linear_delta_baseline"]["coefficients"], dtype=torch.float64)
        scale = payload["training"]["residual_selection"]["selected_scale"]
        for index in range(config.ensemble_size):
            if not torch.equal(state[f"members.{index}.linear_coefficients"], coefficients):
                raise ValueError("ORACLE frozen linear anchor contradicts training baseline")
            if float(state[f"members.{index}.residual_scale"].item()) != scale:
                raise ValueError("ORACLE frozen residual scale contradicts selection")


def _decode_artifact(raw: bytes) -> dict[str, Any]:
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        _validate_artifact(payload)
        return payload
    except (KeyError, TypeError, RuntimeError, OverflowError, AttributeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"invalid ORACLE artifact: {error}") from error


def load_oracle_artifact(path: str | Path) -> dict[str, Any]:
    """Validate metadata and finite, shape-correct weights from one bounded read."""
    return _decode_artifact(_bounded_read(Path(path), 512 * 1024 * 1024))


def _load_model(payload: dict[str, Any]) -> OracleEnsemble:
    config = OracleConfig(**payload["config"])
    model = OracleEnsemble(config, seed=int(payload["model_seed"]))
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError("ORACLE model tensors do not match its config") from error
    return model.eval()


def verify_fit_receipt(receipt_path: str | Path, *, replay_data: bool = True) -> dict[str, Any]:
    path = Path(receipt_path)
    receipt = json.loads(_bounded_read(path, 32 * 1024 * 1024))
    required = {"schema", "status", "activated", "world_version", "model_sha256", "artifact_sha256",
                "artifact_path", "config", "training_manifest", "training", "calibration", "limits", "receipt_sha256"}
    if not isinstance(receipt, dict) or set(receipt) != required or receipt.get("schema") != FIT_RECEIPT_SCHEMA:
        raise ValueError("unsupported ORACLE fit receipt")
    expected_receipt_hash = receipt.pop("receipt_sha256", None)
    if not isinstance(expected_receipt_hash, str) or _sha256_bytes(canonical_bytes(receipt)) != expected_receipt_hash:
        raise ValueError("ORACLE fit receipt digest mismatch")
    if receipt.get("status") != "trained_and_calibrated_candidate":
        raise ValueError("invalid or unsupported ORACLE receipt status")
    if receipt.get("activated") is not False:
        raise ValueError("ORACLE candidate receipt must have activated=False")
    if receipt.get("world_version") != WORLD_VERSION:
        raise ValueError(f"ORACLE world version mismatch: {receipt.get('world_version')} != {WORLD_VERSION}")
    artifact_path = Path(receipt.get("artifact_path", ""))
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    artifact_bytes = _bounded_read(artifact_path, 512 * 1024 * 1024)
    if _sha256_bytes(artifact_bytes) != receipt.get("artifact_sha256"):
        raise ValueError("ORACLE artifact file hash mismatch")
    payload = _decode_artifact(artifact_bytes)
    if payload["model_sha256"] != receipt.get("model_sha256"):
        raise ValueError("ORACLE model identity mismatch")
    for key in ("config", "training", "calibration", "training_manifest"):
        if canonical_bytes(receipt.get(key)) != canonical_bytes(payload.get(key)):
            raise ValueError(f"ORACLE receipt {key} contradicts artifact payload")
    if not isinstance(receipt["limits"], list) or not receipt["limits"] or any(not isinstance(value, str) for value in receipt["limits"]):
        raise ValueError("invalid ORACLE fit limitations")
    if replay_data:
        protocol = receipt["training_manifest"]
        if source_hashes() != protocol["source_sha256"]:
            raise ValueError("ORACLE source fingerprints changed since fitting")
        seed_sets = protocol["seed_sets"]
        max_steps = int(protocol["max_steps"])
        data_protocol = protocol.get("data_protocol", LEGACY_DATA_PROTOCOL)
        replayed = {
            name: collect_split(name, seeds, max_steps=max_steps, data_protocol=data_protocol)
            for name, seeds in seed_sets.items()
        }
        assert_disjoint_splits(replayed)
        for name, split in replayed.items():
            actual = protocol_split_summary(split, data_protocol)
            expected = protocol["split_summaries"].get(name)
            if actual != expected:
                raise ValueError(f"ORACLE {name} transitions do not replay from the recorded seeds")
        model = _load_model(payload)
        training = payload["training"]
        train_rows = _flatten(replayed["train"])
        overall_reward = sum(row.reward for row in train_rows) / len(train_rows)
        action_rows = [[row for row in train_rows if row.action == action] for action in range(N_ACTIONS)]
        # Match the historical sequential accumulator exactly; Python 3.12's
        # compensated sum() differs in the last bits from repeated +=.
        reward_sums = [0.0] * N_ACTIONS
        for row in train_rows:
            reward_sums[row.action] += row.reward
        reward_means = [reward_sums[action] / len(rows) if rows else overall_reward
                        for action, rows in enumerate(action_rows)]
        _same(training["reward_mean_by_action"], reward_means, "replayed reward baseline")
        _same(training["death_count_by_action"], [sum(row.death for row in rows) for rows in action_rows], "replayed death exposure")
        if data_protocol == MIXTURE_DATA_PROTOCOL:
            _same(training["linear_delta_baseline"], _fit_linear_delta(train_rows), "replayed linear baseline")
            _same(training["bootstrap_member_exposure"], _bootstrap_exposure(replayed["train"], training["bootstrap_episode_draws"]), "replayed bootstrap exposure")
        if model.config.architecture == LINEAR_RESIDUAL_ARCHITECTURE:
            _same(training["residual_selection"], _residual_selection(model, replayed["selection"]),
                  "replayed residual selection scores")
        checkpoint_scale = 1.0 if model.config.architecture == LINEAR_RESIDUAL_ARCHITECTURE else None
        if _selection_loss(model, replayed["selection"], training["death_positive_weight"],
                           residual_scale=checkpoint_scale) != training["selection_nll"]:
            raise ValueError("ORACLE selected checkpoint loss does not replay")
        replayed_calibration = [episode_calibration_scores(model, episode) for episode in replayed["calibration"].episodes]
        state_scores = [score[0] for score in replayed_calibration]
        reward_scores = [score[1] for score in replayed_calibration]
        calibration = payload["calibration"]
        if state_scores != calibration["state_episode_scores"] or reward_scores != calibration["reward_episode_scores"]:
            raise ValueError("ORACLE calibration scores do not replay")
        if conformal_order_statistic(state_scores, calibration["coverage"]) != calibration["state_standardized_error_quantile"]:
            raise ValueError("ORACLE state conformal quantile does not replay")
        if conformal_order_statistic(reward_scores, calibration["coverage"]) != calibration["reward_standardized_error_quantile"]:
            raise ValueError("ORACLE reward conformal quantile does not replay")
    receipt["receipt_sha256"] = expected_receipt_hash
    return {"valid": True, "replayed_data": replay_data, "artifact_sha256": receipt["artifact_sha256"],
            "model_sha256": receipt["model_sha256"], "episode_splits": list(receipt["training_manifest"]["seed_sets"]),
            "data_protocol": receipt["training_manifest"].get("data_protocol", LEGACY_DATA_PROTOCOL),
            "verification_scope": "artifact_binding_environment_and_prediction_replay" if replay_data else "artifact_binding_and_structure"}
