"""Explicit trained ORACLE interface for experimental runtime controllers.

One-step error intervals describe the recorded data policy. They are not
survival probabilities or guarantees for selected actions and imagined rollouts.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn

from .oracle_training import _bounded_read, _decode_artifact, _load_model, source_hashes
from ..world import N_ACTIONS, OBS_SIZE


class ResearchOracleAdapter(nn.Module):
    artifact_bound = True
    # Training/calibration currently supervises one-step predictions only.
    max_planning_horizon = 1
    failure_probability_calibrated = False

    def __init__(self, artifact_path: str | Path):
        super().__init__()
        self.artifact_path = Path(artifact_path).resolve()
        raw = _bounded_read(self.artifact_path, 512 * 1024 * 1024)
        self.payload = _decode_artifact(raw)
        self.artifact_sha256 = hashlib.sha256(raw).hexdigest()
        recorded = self.payload["training_manifest"]["source_sha256"]
        current = source_hashes()
        for name in ("world.py", "contracts.py", "oracle_model.py"):
            if recorded.get(name) != current[name]:
                raise ValueError(f"ORACLE inference source mismatch: {name}; refit for this contract")
        self.model = _load_model(self.payload).requires_grad_(False)
        self.config = SimpleNamespace(**self.payload["config"], latent_size=128, obs_size=OBS_SIZE)
        self.register_buffer("is_calibrated", torch.tensor(True))
        calibration = self.payload["calibration"]
        self.state_quantile = float(calibration["state_standardized_error_quantile"])
        self.reward_quantile = float(calibration["reward_standardized_error_quantile"])
        training = self.payload["training"]
        # Legacy artifacts record exposure under the reward-baseline counts.
        counts = training.get("action_counts", training.get("reward_count_by_action", []))
        if not counts:
            counts = training.get("baselines", {}).get("reward_count_by_action", [])
        self.action_counts = [int(count) for count in counts] if len(counts) == N_ACTIONS else [0] * N_ACTIONS
        self.member_forward_calls = 0
        self.prediction_rows = 0
        self.eval()

    def status(self) -> dict[str, Any]:
        return {
            "trained": True,
            "artifact_bound": True,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "model_sha256": self.payload["model_sha256"],
            "data_protocol": self.payload["training_manifest"].get("data_protocol", "legacy-v1"),
            "architecture": self.payload["config"].get("architecture", "mlp-v1"),
            "residual_selection": copy.deepcopy(self.payload["training"].get("residual_selection")),
            "calibration_available": True,
            "calibration_scope": "recorded data-policy episodes; selected-action and OOD coverage unverified",
            "failure_probability_calibrated": False,
            "max_planning_horizon": self.max_planning_horizon,
            "action_counts": self.action_counts.copy(),
            "support_scope": "training action exposure only; state-distribution support is unverified",
            "member_forward_calls": self.member_forward_calls,
            "prediction_rows": self.prediction_rows,
            "promoted": False,
        }

    def planning_support(self, observation: Sequence[float]) -> tuple[bool, str | None]:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        self._validate_observation(obs)
        if any(count < 8 for count in self.action_counts):
            return False, "insufficient_training_action_exposure"
        return True, None

    def reset_counters(self) -> None:
        self.member_forward_calls = 0
        self.prediction_rows = 0

    @staticmethod
    def _validate_observation(observation: torch.Tensor) -> None:
        if observation.ndim != 2 or observation.shape[1] != OBS_SIZE or observation.shape[0] == 0:
            raise ValueError(f"observation must have shape [batch, {OBS_SIZE}]")
        if not bool(torch.isfinite(observation).all()) or bool(((observation < 0) | (observation > 1)).any()):
            raise ValueError("observation values must be finite and in [0, 1]")

    @torch.inference_mode()
    def forward(self, latent_state: torch.Tensor | None, obs: torch.Tensor,
                actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # The trained research model is observation/action conditioned; the legacy
        # latent argument is deliberately unused, not represented as trained input.
        self._validate_observation(obs)
        output = self.model(obs, actions)
        summary = self.model.summarize(output)
        if any(not bool(torch.isfinite(value).all()) for value in summary.values()):
            raise FloatingPointError("ORACLE produced nonfinite predictions")
        state_half_width = self.state_quantile * summary["delta_total_variance"].sqrt()
        reward_half_width = self.reward_quantile * summary["reward_total_variance"].sqrt()
        result = {
            "mean_delta": summary["delta_mean"],
            "mean_reward": summary["reward_mean"],
            "mean_failure": summary["death_probability"],
            "raw_disagreement": summary["delta_epistemic_variance"].sum(-1),
            # Compatibility name: this is a one-step interval radius, not risk.
            "calibrated_uncertainty": state_half_width.mean(-1),
            "state_interval_half_width": state_half_width,
            "reward_interval_half_width": reward_half_width,
            "deltas": output["delta_mean"],
            "rewards": output["reward_mean"],
            "failures": output["death_logit"].sigmoid(),
        }
        if any(not bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("ORACLE produced nonfinite scaled predictions")
        self.member_forward_calls += len(self.model.members)
        self.prediction_rows += obs.shape[0]
        return result

    @torch.inference_mode()
    def imagine_trajectory(self, initial_latent: torch.Tensor | None,
                           initial_obs: torch.Tensor,
                           action_sequence: Sequence[int]) -> dict[str, Any]:
        if not 1 <= len(action_sequence) <= 32 or any(type(a) is not int or not 0 <= a < N_ACTIONS for a in action_sequence):
            raise ValueError("imagination requires 1-32 valid integer actions")
        observation = initial_obs.detach().clone().reshape(1, -1)
        self._validate_observation(observation)
        trajectory = []
        reward_sum = uncertainty_sum = 0.0
        for tick, action in enumerate(action_sequence):
            out = self(None, observation, torch.tensor([action], dtype=torch.long))
            # The ridge anchor deliberately retains float64 precision. Feed
            # recursive observations back in the neural encoder's input dtype.
            next_observation = (observation + out["mean_delta"]).clamp(0, 1).to(observation.dtype)
            reward = float(out["mean_reward"].item())
            radius = float(out["calibrated_uncertainty"].item())
            trajectory.append({"step": tick, "action": action,
                               "predicted_obs": next_observation[0].tolist(),
                               "step_reward": reward, "step_uncertainty": radius,
                               "step_failure_prob": float(out["mean_failure"].item())})
            reward_sum += reward
            uncertainty_sum += radius
            observation = next_observation
        return {"total_predicted_reward": reward_sum, "total_uncertainty": uncertainty_sum,
                "survival_probability": None, "failure_probability_calibrated": False,
                "calibration_applies_to_rollout": False, "horizon": len(action_sequence),
                "trajectory": trajectory, "final_predicted_obs": observation[0].tolist()}


def oracle_status(oracle: nn.Module) -> dict[str, Any]:
    if isinstance(oracle, ResearchOracleAdapter):
        return oracle.status()
    return {"trained": False, "artifact_bound": False, "calibration_available": False,
            "failure_probability_calibrated": False, "promoted": False,
            "reason": "No trained research ORACLE artifact selected; frozen Tidal fallback."}
