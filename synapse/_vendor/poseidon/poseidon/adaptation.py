"""Bounded surprise replay and an evidence gate for experimental candidates.

An audit compares reactive policies, evaluates both dynamics ensembles on the
candidate's exact transitions, and checks held-out scene semantics. Passing
these descriptive thresholds grants eligibility for review, never activation.
Disagreement/error correlation is a diagnostic, not probability calibration.
"""
from __future__ import annotations

import json
import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from .core import SCENE_LABELS, hash_features
from .ensemble import WorldModelEnsemble
from .world import TidePool, OBS_SIZE, N_ACTIONS


@dataclass
class Transition:
    observation: list[float]
    action: int
    reward: float
    next_observation: list[float]
    disagreement: float
    prediction_error: float
    is_failure: bool
    priority: float


class SurprisePrioritizedBuffer:
    """Prioritizes surprising outcomes, failures, and ensemble disagreement."""

    def __init__(self, capacity: int = 2000, alpha: float = 0.6, seed: int = 42):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.capacity = capacity
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.storage: list[Transition] = []

    def __len__(self) -> int:
        return len(self.storage)

    def add(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        disagreement: float = 0.0,
        prediction_error: float = 0.0,
        is_failure: bool = False,
    ) -> None:
        if len(observation) != OBS_SIZE or len(next_observation) != OBS_SIZE:
            raise ValueError("observations must have 16 elements")
        if not all(math.isfinite(v) for v in [*observation, *next_observation, reward, disagreement, prediction_error]):
            raise ValueError("transition values must be finite")
        if type(action) is not int or not 0 <= action < N_ACTIONS:
            raise ValueError("action must be an integer in [0, 5]")
        if disagreement < 0 or prediction_error < 0:
            raise ValueError("disagreement and prediction_error must be nonnegative")
        # These are heuristic priorities, not calibrated probabilities.
        surprise_metric = disagreement + prediction_error + (2.0 if is_failure else 0.0) + 1e-4
        priority = (surprise_metric) ** self.alpha

        trans = Transition(
            observation=list(observation),
            action=action,
            reward=reward,
            next_observation=list(next_observation),
            disagreement=disagreement,
            prediction_error=prediction_error,
            is_failure=is_failure,
            priority=priority,
        )

        if len(self.storage) >= self.capacity:
            # Drop lowest priority or oldest item
            min_idx = min(range(len(self.storage)), key=lambda i: self.storage[i].priority)
            self.storage[min_idx] = trans
        else:
            self.storage.append(trans)

    def sample(self, batch_size: int = 64) -> list[Transition]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self.storage:
            return []
        k = min(batch_size, len(self.storage))
        priorities = [t.priority for t in self.storage]
        total_p = sum(priorities)
        probs = [p / total_p for p in priorities]
        # Weighted sampling with replacement using an isolated reproducible RNG.
        indices = self.rng.choices(range(len(self.storage)), weights=probs, k=k)
        return [self.storage[i] for i in indices]


@dataclass(frozen=True)
class PromotionCriteria:
    min_survival_rate: float = 0.95
    max_dynamics_mse: float = 0.040
    min_scene_accuracy: float = 0.98
    min_epistemic_correlation: float = 0.20
    max_dynamics_regression: float = 0.0
    max_scene_regression: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "min_epistemic_correlation":
                valid = -1 <= value <= 1
            elif name in ("min_survival_rate", "min_scene_accuracy", "max_scene_regression"):
                valid = 0 <= value <= 1
            else:
                valid = value >= 0
            if not valid:
                raise ValueError(f"{name} is outside its supported range")


def model_digest(model: Any) -> str:
    """Digest names, shapes, dtypes and tensor bytes without a pickle container."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")


def _validated_scenes(rows: Sequence[dict]) -> list[dict]:
    from .media import _group, _split, validate_scene
    if not rows:
        raise ValueError("scene_examples must be non-empty")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("scene examples need a non-empty prompt")
        scene = validate_scene(row.get("scene"))
        group = _group(scene)
        if row.get("group") != group or _split(group) != "test":
            raise ValueError("scene_examples must belong to the semantic test split")
        expected = {name: list(values).index(scene[name]) for name, values in SCENE_LABELS.items()}
        if row.get("labels") != expected:
            raise ValueError("scene labels must match the semantic scene")
        if row["prompt"] in seen:
            raise ValueError("scene_examples must contain unique prompts")
        seen.add(row["prompt"])
    return list(rows)


def _scene_scores(model: Any, rows: Sequence[dict]) -> tuple[float, list[dict]]:
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        features = hash_features([row["prompt"] for row in batch], model.config.hash_buckets).to(device)
        output = model(features)
        for name, values in SCENE_LABELS.items():
            if name not in output.get("scene", {}):
                raise ValueError(f"scene output is missing {name}")
            _finite_tensor(output["scene"][name], (len(batch), len(values)), f"scene.{name}")
        for i, row in enumerate(batch):
            predicted = {name: int(output["scene"][name][i].argmax()) for name in SCENE_LABELS}
            predictions.append({"prompt": row["prompt"], "group": row["group"], "expected": row["labels"],
                                "predicted": predicted, "correct": predicted == row["labels"]})
    return sum(row["correct"] for row in predictions) / len(rows), predictions


class PromotionAuditor:
    """Evaluate descriptive eligibility gates without changing active artifacts."""

    def __init__(self, criteria: PromotionCriteria | None = None):
        self.criteria = criteria or PromotionCriteria()

    @torch.inference_mode()
    def audit_candidate(
        self,
        candidate_model: Any,
        ensemble: Any,
        test_seeds: Sequence[int] = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110),
        scarcity: float = 2.0,
        incumbent_model: Any | None = None,
        incumbent_ensemble: Any | None = None,
        scene_examples: Sequence[dict] | None = None,
        max_steps: int = 128,
    ) -> dict[str, Any]:
        """Compare reactive policies and same-transition ensemble predictions.

        Legacy calls may omit incumbent/scene evidence. They still get measured
        candidate diagnostics, but cannot become eligible. Inputs are validated
        before measurement; nonfinite outputs abort rather than certify a run.
        """
        seeds = list(test_seeds)
        if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError("test_seeds must be non-empty valid integers")
        if len(seeds) != len(set(seeds)):
            raise ValueError("test_seeds must contain distinct seeds")
        if isinstance(scarcity, bool) or not math.isfinite(scarcity) or not 0.5 <= scarcity <= 4:
            raise ValueError("scarcity must be finite and in [0.5, 4]")
        if type(max_steps) is not int or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be an integer in [1, 10000]")
        scenes = None if scene_examples is None else _validated_scenes(scene_examples)
        modules = {id(m): m for m in (candidate_model, ensemble, incumbent_model, incumbent_ensemble) if m is not None}
        modes = {identity: module.training for identity, module in modules.items()}
        for module in modules.values():
            module.eval()
        try:
            return self._measure(candidate_model, ensemble, seeds, scarcity, incumbent_model,
                                 incumbent_ensemble, scenes, max_steps)
        finally:
            for identity, module in modules.items():
                module.train(modes[identity])

    def _measure(self, candidate, ensemble, seeds, scarcity, incumbent, incumbent_ensemble, scenes, max_steps):
        missing = [name for name, value in (("incumbent_model", incumbent),
                   ("incumbent_ensemble", incumbent_ensemble), ("scene_examples", scenes)) if value is None]
        errors, incumbent_errors, disagreements, samples, paired = [], [], [], [], []

        def model_output(model, observation):
            device = next(model.parameters()).device
            obs = torch.tensor([observation], dtype=torch.float32, device=device)
            _finite_tensor(obs, (1, OBS_SIZE), "observation")
            output = model(torch.zeros(1, model.config.hash_buckets, device=device), obs)
            _finite_tensor(output["action"], (1, N_ACTIONS), "action logits")
            _finite_tensor(output["memory"], (1, model.config.hidden_size), "latent state")
            return obs, output

        def predict(ensemble_model, output, obs, action):
            action_t = torch.tensor([action], dtype=torch.long, device=obs.device)
            prediction = ensemble_model(output["memory"], obs, action_t)
            _finite_tensor(prediction["mean_delta"], (1, OBS_SIZE), "dynamics prediction")
            _finite_tensor(prediction["disagreement"], (1,), "disagreement")
            if float(prediction["disagreement"][0]) < 0:
                raise ValueError("disagreement must be nonnegative")
            return prediction

        for seed in seeds:
            env = TidePool(seed=seed, scarcity=scarcity, max_steps=max_steps)
            obs = env.reset(seed)
            done, episode_steps, episode_reward = False, 0, 0.0
            episode_errors, episode_inc_errors = [], []
            while not done:
                obs_t, output = model_output(candidate, obs)
                action = int(output["action"].argmax(-1).item())
                pred = predict(ensemble, output, obs_t, action)
                baseline_pred = None
                if incumbent is not None and incumbent_ensemble is not None:
                    inc_obs_t, inc_output = model_output(incumbent, obs)
                    baseline_pred = predict(incumbent_ensemble, inc_output, inc_obs_t, action)
                next_obs, reward, done, info = env.step(action)
                next_t = torch.tensor([next_obs], dtype=torch.float32, device=obs_t.device)
                _finite_tensor(next_t, (1, OBS_SIZE), "next observation")
                if not math.isfinite(reward):
                    raise ValueError("reward must be finite")
                target = next_t - obs_t
                error = float((pred["mean_delta"] - target).double().square().mean())
                disagreement = float(pred["disagreement"][0])
                baseline_error = None
                if baseline_pred is not None:
                    baseline_error = float((baseline_pred["mean_delta"].cpu() - target.cpu()).double().square().mean())
                    incumbent_errors.append(baseline_error)
                    episode_inc_errors.append(baseline_error)
                if not math.isfinite(error) or (baseline_error is not None and not math.isfinite(baseline_error)):
                    raise ValueError("dynamics errors must be finite")
                errors.append(error)
                episode_errors.append(error)
                disagreements.append(disagreement)
                samples.append({"seed": seed, "step": episode_steps, "action": action,
                                "observation": list(obs), "next_observation": list(next_obs),
                                "disagreement": disagreement, "candidate_mse": error,
                                "incumbent_mse": baseline_error})
                episode_steps += 1
                episode_reward += reward
                obs = next_obs
                if episode_steps >= max_steps and not done:
                    raise ValueError("environment exceeded the declared audit horizon")
            candidate_alive = bool(env.alive)
            candidate_death = info.get("death")
            incumbent_alive, inc_steps, inc_reward = None, None, None
            if incumbent is not None:
                env_inc = TidePool(seed=seed, scarcity=scarcity, max_steps=max_steps)
                obs_inc = env_inc.reset(seed)
                done_inc, inc_steps, inc_reward = False, 0, 0.0
                while not done_inc:
                    _, inc_output = model_output(incumbent, obs_inc)
                    action_inc = int(inc_output["action"].argmax(-1).item())
                    obs_inc, reward_inc, done_inc, _ = env_inc.step(action_inc)
                    _finite_tensor(torch.tensor(obs_inc), (OBS_SIZE,), "incumbent next observation")
                    if not math.isfinite(reward_inc):
                        raise ValueError("incumbent reward must be finite")
                    inc_steps += 1
                    inc_reward += reward_inc
                    if inc_steps >= max_steps and not done_inc:
                        raise ValueError("environment exceeded the declared audit horizon")
                incumbent_alive = bool(env_inc.alive)
            paired.append({"seed": seed, "candidate_alive": candidate_alive, "incumbent_alive": incumbent_alive,
                           "survival_delta": None if incumbent_alive is None else int(candidate_alive) - int(incumbent_alive),
                           "candidate_steps": episode_steps, "incumbent_steps": inc_steps,
                           "candidate_reward": episode_reward, "incumbent_reward": inc_reward,
                           "candidate_death": candidate_death,
                           "candidate_dynamics_mse": sum(episode_errors) / len(episode_errors),
                           "incumbent_dynamics_mse": sum(episode_inc_errors) / len(episode_inc_errors) if episode_inc_errors else None})

        candidate_scene, incumbent_scene, scene_rows, incumbent_scene_rows = None, None, [], []
        if scenes is not None:
            candidate_scene, scene_rows = _scene_scores(candidate, scenes)
            if incumbent is not None:
                incumbent_scene, incumbent_scene_rows = _scene_scores(incumbent, scenes)
        d = torch.tensor(disagreements, dtype=torch.float64)
        e = torch.tensor(errors, dtype=torch.float64)
        d_centered, e_centered = d - d.mean(), e - e.mean()
        denominator = float(d_centered.norm() * e_centered.norm())
        calibration_valid = len(errors) >= 3 and denominator > 1e-15 and math.isfinite(denominator)
        correlation = max(-1.0, min(1.0, float(torch.dot(d_centered, e_centered)) / denominator)) if calibration_valid else None
        if not calibration_valid:
            missing.append("valid_disagreement_error_correlation")
        survival = sum(row["candidate_alive"] for row in paired) / len(seeds)
        baseline_survival = sum(row["incumbent_alive"] for row in paired) / len(seeds) if incumbent is not None else None
        mse = sum(errors) / len(errors)
        baseline_mse = sum(incumbent_errors) / len(incumbent_errors) if incumbent_errors else None
        passed = {
            "survival": survival >= self.criteria.min_survival_rate,
            "incumbent_survival": baseline_survival is not None and survival >= baseline_survival,
            "dynamics_mse": mse <= self.criteria.max_dynamics_mse,
            "incumbent_dynamics": baseline_mse is not None and mse <= baseline_mse + self.criteria.max_dynamics_regression,
            "scene_accuracy": candidate_scene is not None and candidate_scene >= self.criteria.min_scene_accuracy,
            "incumbent_scene": incumbent_scene is not None and candidate_scene >= incumbent_scene - self.criteria.max_scene_regression,
            "epistemic_calibration": calibration_valid and correlation >= self.criteria.min_epistemic_correlation,
        }
        complete = not missing
        eligible = bool(complete and all(passed.values()))
        contract = {"seeds": seeds, "scarcity": scarcity, "max_steps": max_steps,
                    "scene_examples": scenes, "criteria": asdict(self.criteria)}
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {
            "schema": "poseidon-candidate-audit-v2", "promoted": False, "activated": False,
            "eligible": eligible, "evidence_complete": complete,
            "status": "eligible_for_review" if eligible else "incomplete_evidence" if not complete else "criteria_failed",
            "missing_evidence": missing,
            "metrics": {"survival_rate": survival, "incumbent_survival_rate": baseline_survival,
                        "mean_dynamics_mse": mse, "incumbent_dynamics_mse": baseline_mse,
                        "epistemic_correlation": correlation, "scene_exact_accuracy": candidate_scene,
                        "incumbent_scene_exact_accuracy": incumbent_scene, "scene_examples": len(scenes) if scenes else 0,
                        "episodes_tested": len(seeds), "total_steps": len(errors)},
            "paired_episodes": paired, "calibration_samples": samples,
            "scene_predictions": scene_rows, "incumbent_scene_predictions": incumbent_scene_rows,
            "calibration": {"valid": calibration_valid, "method": "descriptive Pearson disagreement versus one-step squared error",
                            "samples": len(errors), "failure_probability_calibrated": False,
                            "reason": None if calibration_valid else "at least three samples and variation in both signals are required"},
            "criteria_passed": passed, "gate_thresholds": asdict(self.criteria),
            "identity": {"suite_sha256": digest, "candidate_core_sha256": model_digest(candidate),
                         "candidate_ensemble_sha256": model_digest(ensemble),
                         "incumbent_core_sha256": model_digest(incumbent) if incumbent is not None else None,
                         "incumbent_ensemble_sha256": model_digest(incumbent_ensemble) if incumbent_ensemble is not None else None},
            "scope": {"policy": "reactive core argmax; ensemble does not select actions in this audit",
                      "dynamics": "both ensembles predict the exact same candidate observations and executed actions",
                      "scene_split": "semantic test groups", "seeds": seeds, "scarcity": scarcity, "max_steps": max_steps},
            "limits": ["Eligibility is a descriptive threshold decision, not a significance test or activation.",
                       "Correlated within-episode samples do not establish independent calibration evidence.",
                       "Planner hazard scores are heuristics, not calibrated failure probabilities.",
                       "Reusing this test suite for candidate selection requires a fresh final audit."]}
