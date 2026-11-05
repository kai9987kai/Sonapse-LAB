"""Project FAULTLINE Oracle Adapter for Supermix-Poseidon.

Integrates FAULTLINE's condition-stratified joint-failure ensemble selection,
higher-order co-failure auditing, and continuous joint error covariance
minimization with Poseidon's ORACLE world model ensemble and TETHER reliability gate.

The default audit uses canonical synthetic predictions and simulated errors,
not forecasts collected from the selected ORACLE artifact. Panel slicing is
available for diagnostics, but selection and calibration evidence for the
changed predictor are absent. Planning therefore preserves the frozen policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..faultline_engine import (
    ContinuousJointErrorEngine,
    FaultlineDiscreteEngine,
    FaultlineEngine,
    POLICIES,
    POLICY_LABELS,
    analyze,
)
from ..world import N_ACTIONS, OBS_SIZE


class FaultlineOracleAdapter(nn.Module):
    """Adapter connecting FAULTLINE joint-failure selection to ORACLE and TETHER."""

    def __init__(
        self,
        oracle_adapter: Optional[Any] = None,
        artifact_path: Optional[Union[str, Path]] = None,
        panel_size: int = 3,
        primary_policy: str = "joint",
        seed: int = 42,
    ):
        super().__init__()
        self.panel_size = int(panel_size)
        if primary_policy not in POLICIES:
            raise ValueError(f"Unknown policy '{primary_policy}'; expected one of {POLICIES}")
        self.primary_policy = primary_policy
        self.seed = int(seed)

        # Connect or create underlying oracle model/adapter
        if oracle_adapter is not None:
            self.oracle = oracle_adapter
        elif artifact_path is not None:
            from .oracle_adapter import ResearchOracleAdapter
            self.oracle = ResearchOracleAdapter(artifact_path)
        else:
            self.oracle = None

        # Number of available ensemble members (default 5 for ORACLE ensemble)
        if self.oracle is not None and hasattr(self.oracle, "model") and hasattr(self.oracle.model, "members"):
            self.n_members = len(self.oracle.model.members)
        elif self.oracle is not None and hasattr(self.oracle, "heads"):
            self.n_members = len(self.oracle.heads)
        else:
            self.n_members = 5

        # Initialize audit state
        self.discrete_engine = FaultlineDiscreteEngine()
        self.continuous_engine = ContinuousJointErrorEngine()
        self.audit_receipt: Optional[Dict[str, Any]] = None
        self.active_panel: List[int] = list(range(min(self.panel_size, self.n_members)))
        self.active_panel_ids: List[str] = [f"m{i+1}" for i in self.active_panel]
        self.active_panel_source = "unvalidated_default_first_members"
        self.register_buffer("is_calibrated", torch.tensor(False))

        # Initial benchmark audit
        self.run_calibration_audit(seed=self.seed)

    def run_calibration_audit(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Audit an unbound dataset and synthetic continuous errors.

        The historical method name does not mean it calibrates the panel
        predictor. No audit here binds member IDs or forecasts to ORACLE.
        """
        self.panel_selection_scope = ("canonical_synthetic_fixture_not_oracle" if dataset is None
                                      else "caller_supplied_dataset_unbound_to_oracle")
        if dataset is None:
            dataset = FaultlineEngine._create_canonical_dataset(seed=seed)

        analysis = analyze(dataset, size=self.panel_size)

        # Continuous condition-stratified evaluation
        cond_errors = FaultlineEngine._simulate_conditioned_model_errors(
            N_per_cond=32, M=self.n_members, seed=seed
        )
        cont_eval = self.continuous_engine.select_minimax_continuous_panel(
            cond_errors, size=self.panel_size
        )
        # Compute covariance on pooled samples
        pooled_errors = []
        for c_mat in cond_errors.values():
            pooled_errors.extend(c_mat)
        cov_eval = self.continuous_engine.compute_joint_error_covariance(pooled_errors)

        # Find selected policy result
        selected_res = next(p for p in analysis["policies"] if p["id"] == self.primary_policy)
        accuracy_res = next(p for p in analysis["policies"] if p["id"] == "accuracy")

        if self.oracle is None:
            self.active_panel = list(selected_res["panel_indices"])
            self.active_panel_ids = list(selected_res["panel"])
            self.active_panel_source = "synthetic_fixture_selection"

        # Compute joint advantage: accuracy worst-case error minus joint worst-case error
        acc_worst = accuracy_res["test"]["worst_group_error"]
        joint_worst = selected_res["test"]["worst_group_error"]
        joint_advantage = acc_worst - joint_worst

        # SHA-256 audit receipt
        audit_body = {
            "dataset": analysis["dataset"],
            "primary_policy": self.primary_policy,
            "panel_size": self.panel_size,
            "active_panel": self.active_panel,
            "active_panel_ids": self.active_panel_ids,
            "joint_advantage": float(joint_advantage),
            "worst_condition_error": float(joint_worst),
            "minimax_continuous_panel": cont_eval["selected_panel"],
            "selection_scope": self.panel_selection_scope,
            "continuous_evaluation_scope": "synthetic_conditioned_errors_not_oracle",
            "applied_to_oracle": False,
        }
        receipt_str = json.dumps(audit_body, sort_keys=True)
        sha256 = hashlib.sha256(receipt_str.encode("utf-8")).hexdigest()

        self.audit_receipt = {
            "schema": "poseidon-faultline-oracle-audit-v1",
            "sha256": sha256,
            "analysis": analysis,
            "continuous_evaluation": cont_eval,
            "continuous_covariance": cov_eval,
            "active_panel": self.active_panel,
            "active_panel_ids": self.active_panel_ids,
            "joint_advantage": float(joint_advantage),
            "worst_condition_error": float(joint_worst),
            "accuracy_worst_condition_error": float(acc_worst),
            "selection_scope": self.panel_selection_scope,
            "continuous_evaluation_scope": "synthetic_conditioned_errors_not_oracle",
            "applied_to_oracle": False,
        }
        return self.audit_receipt

    def set_active_panel(self, panel_indices: Sequence[int]) -> None:
        """Manually override or lock the active specialist panel."""
        for idx in panel_indices:
            if idx < 0 or idx >= self.n_members:
                raise ValueError(f"Invalid member index {idx}; must be in 0..{self.n_members-1}")
        if len(panel_indices) != self.panel_size:
            raise ValueError(f"Panel size mismatch: expected {self.panel_size}, got {len(panel_indices)}")
        self.active_panel = list(panel_indices)
        self.active_panel_ids = [f"m{i+1}" for i in self.active_panel]
        self.active_panel_source = "manual_unvalidated_selection"

    def status(self) -> Dict[str, Any]:
        """Telemetry and status report for FAULTLINE audit & active panel."""
        base = {}
        if self.oracle is not None and hasattr(self.oracle, "status"):
            base = self.oracle.status()

        base.update({
            "trained": base.get("trained", False),
            "underlying_artifact_bound": base.get("artifact_bound", False),
            "artifact_bound": False,
            "calibration_available": False,
            "calibration_scope": "panel predictor uncalibrated; all-member intervals do not apply",
            "failure_probability_calibrated": False,
            "promoted": False,
            "faultline_enabled": True,
            "faultline_protocol": "faultline-oracle-panel-v1",
            "primary_policy": self.primary_policy,
            "panel_size": self.panel_size,
            "n_available_members": self.n_members,
            "active_panel": self.active_panel,
            "active_panel_ids": self.active_panel_ids,
            "active_panel_source": self.active_panel_source,
            "panel_selection_scope": self.panel_selection_scope,
            "joint_advantage": self.audit_receipt.get("joint_advantage", 0.0) if self.audit_receipt else 0.0,
            "worst_condition_error": self.audit_receipt.get("worst_condition_error", 0.0) if self.audit_receipt else 0.0,
            "sha256_receipt": self.audit_receipt.get("sha256") if self.audit_receipt else None,
        })
        return base

    def planning_support(self, observation: Sequence[float]) -> Tuple[bool, Optional[str]]:
        """Preserve the baseline until panel-specific calibration is supported."""
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        if obs.shape[-1] != OBS_SIZE:
            return False, f"invalid_obs_size_{obs.shape[-1]}_expected_{OBS_SIZE}"
        if not bool(torch.isfinite(obs).all()) or bool(((obs < 0) | (obs > 1)).any()):
            return False, "invalid_observation_values"
        if self.oracle is None:
            return False, "faultline_oracle_unavailable"
        return False, "faultline_panel_calibration_unavailable"

    def slice_panel_predictions(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Slice K-member tensor [M, batch, ...] down to active panel [K, batch, ...]."""
        if predictions.shape[0] < self.n_members:
            return predictions
        indices = torch.tensor(self.active_panel, dtype=torch.long, device=predictions.device)
        return torch.index_select(predictions, 0, indices)

    @torch.inference_mode()
    def forward(
        self,
        latent_state: Optional[torch.Tensor],
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run forward inference over the active panel selected by FAULTLINE."""
        if self.oracle is not None:
            # Check if oracle has underlying model or forward
            raw_out = self.oracle(latent_state, obs, actions)
            # If the raw output contains member deltas [M, batch, dim], slice to active panel
            if "deltas" in raw_out and raw_out["deltas"].shape[0] == self.n_members:
                panel_deltas = self.slice_panel_predictions(raw_out["deltas"])
                panel_rewards = self.slice_panel_predictions(raw_out["rewards"]) if "rewards" in raw_out else None
                panel_failures = self.slice_panel_predictions(raw_out["failures"]) if "failures" in raw_out else None

                # Compute panel-consensus
                mean_delta = panel_deltas.mean(dim=0)
                mean_reward = panel_rewards.mean(dim=0) if panel_rewards is not None else raw_out.get("mean_reward")
                mean_failure = panel_failures.mean(dim=0) if panel_failures is not None else raw_out.get("mean_failure")

                # Disagreement within the active panel
                panel_diff = panel_deltas - mean_delta.unsqueeze(0)
                panel_disagreement = (panel_diff ** 2).mean(dim=0).sum(dim=-1)

                out = dict(raw_out)
                # Equal-average calibration is tied to the original predictor.
                # Neither panel selection nor re-averaging transfers its widths.
                for interval in ("calibrated_uncertainty", "state_interval_half_width",
                                 "reward_interval_half_width"):
                    out.pop(interval, None)
                out["mean_delta"] = mean_delta
                if mean_reward is not None:
                    out["mean_reward"] = mean_reward
                if mean_failure is not None:
                    out["mean_failure"] = mean_failure
                out["raw_disagreement"] = panel_disagreement
                out["panel_deltas"] = panel_deltas
                out["faultline_active_panel"] = torch.tensor(self.active_panel, dtype=torch.long)
                return out
            return raw_out

        # Fallback simulation forward for standalone adapter testing
        batch_size = obs.shape[0]
        K = len(self.active_panel)
        sim_deltas = torch.zeros(K, batch_size, OBS_SIZE, dtype=obs.dtype, device=obs.device)
        for i, m_idx in enumerate(self.active_panel):
            sim_deltas[i] = obs * (0.01 * (m_idx + 1))
        mean_delta = sim_deltas.mean(dim=0)
        return {
            "mean_delta": mean_delta,
            "mean_reward": torch.zeros(batch_size, dtype=obs.dtype, device=obs.device),
            "mean_failure": torch.zeros(batch_size, dtype=obs.dtype, device=obs.device),
            "raw_disagreement": ((sim_deltas - mean_delta.unsqueeze(0)) ** 2).mean(0).sum(-1),
            "deltas": sim_deltas,
            "faultline_active_panel": torch.tensor(self.active_panel, dtype=torch.long),
        }
