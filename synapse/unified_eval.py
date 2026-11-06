"""Unified Evaluation Scorecard, Calibration Metrics & Joint Deployment Scoring.

Grounding:
Implements the multi-dimensional scorecard recommended for the kai9987kai stack:
    S = R - alpha * C - beta * U - gamma * F
Where:
    R = realized task return
    C = compute cost (normalized FLOPs or compute seconds)
    U = uncertainty miscalibration (Brier score + Expected Calibration Error ECE)
    F = safety / regression failures
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_brier_score(predictions: np.ndarray, binary_targets: np.ndarray) -> float:
    """Computes the mean squared error Brier score: (1/N) * sum((p_i - y_i)^2)."""
    p = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(binary_targets, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def compute_expected_calibration_error(
    confidences: np.ndarray,
    binary_accuracies: np.ndarray,
    num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across partitioned confidence bins."""
    confs = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    accs = np.asarray(binary_accuracies, dtype=np.float64)
    N = len(confs)
    if N == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs >= low) & (confs <= high) if i == num_bins - 1 else (confs >= low) & (confs < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = float(np.mean(accs[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / float(N)) * abs(bin_acc - bin_conf)

    return float(ece)


@dataclass
class EvaluationScorecard:
    task_name: str
    realized_return: float
    compute_seconds: float
    brier_score: float
    ece_score: float
    safety_failures: int
    alpha_compute: float = 0.05
    beta_calibration: float = 0.5
    gamma_safety: float = 2.0

    @property
    def joint_deployment_score(self) -> float:
        """S = R - alpha * C - beta * U - gamma * F"""
        calibration_gap = 0.5 * (self.brier_score + self.ece_score)
        s = (
            self.realized_return -
            self.alpha_compute * self.compute_seconds -
            self.beta_calibration * calibration_gap -
            self.gamma_safety * self.safety_failures
        )
        return float(s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task_name,
            'realized_return': round(self.realized_return, 4),
            'compute_seconds': round(self.compute_seconds, 4),
            'brier_score': round(self.brier_score, 4),
            'ece_score': round(self.ece_score, 4),
            'safety_failures': self.safety_failures,
            'joint_deployment_score': round(self.joint_deployment_score, 4)
        }


class UnifiedEvaluator:
    """Aggregates multi-dimensional performance across environments and tasks."""

    def __init__(
        self,
        alpha_compute: float = 0.05,
        beta_calibration: float = 0.5,
        gamma_safety: float = 2.0,
    ):
        self.alpha_compute = alpha_compute
        self.beta_calibration = beta_calibration
        self.gamma_safety = gamma_safety
        self.scorecards: List[EvaluationScorecard] = []

    def evaluate_task_run(
        self,
        task_name: str,
        returns: List[float],
        execution_times: List[float],
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        failures: int = 0
    ) -> EvaluationScorecard:
        """Evaluates an experimental run and produces an audited scorecard."""
        mean_r = float(np.mean(returns)) if returns else 0.0
        tot_time = float(np.sum(execution_times)) if execution_times else 0.1

        brier = compute_brier_score(predictions, ground_truth)
        ece = compute_expected_calibration_error(predictions, ground_truth)

        card = EvaluationScorecard(
            task_name=task_name,
            realized_return=mean_r,
            compute_seconds=tot_time,
            brier_score=brier,
            ece_score=ece,
            safety_failures=failures,
            alpha_compute=self.alpha_compute,
            beta_calibration=self.beta_calibration,
            gamma_safety=self.gamma_safety
        )
        self.scorecards.append(card)
        return card

    def get_summary_matrix(self) -> Dict[str, Any]:
        if not self.scorecards:
            return {'status': 'empty', 'mean_joint_score': 0.0}

        joint_scores = [c.joint_deployment_score for c in self.scorecards]
        return {
            'status': 'ok',
            'tasks_evaluated': len(self.scorecards),
            'mean_joint_deployment_score': round(float(np.mean(joint_scores)), 4),
            'cards': [c.to_dict() for c in self.scorecards]
        }
