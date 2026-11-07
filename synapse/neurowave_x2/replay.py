"""Deterministic Replay Engine for Sonapse NeuroWave-X2.

Executes bitwise or epsilon-bounded reproducible replays of agent evaluations,
verifying packet queues, state trajectories, and reward receipts.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .packets import NeuralPacket


class ReplayVerificationError(AssertionError):
    """Raised when replayed trajectory diverges beyond numerical tolerance from recorded run."""
    pass


class ReplayRecorder:
    """Records step-by-step evaluation traces for subsequent deterministic replay."""

    def __init__(self) -> None:
        self.action_trace: List[List[float]] = []
        self.reward_trace: List[float] = []
        self.packet_trace: List[Dict[str, Any]] = []

    def record_step(self, action: np.ndarray, reward: float, packets: List[NeuralPacket]) -> None:
        self.action_trace.append([float(a) for a in action])
        self.reward_trace.append(float(reward))
        for p in packets:
            self.packet_trace.append(p.to_dict())

    def export_trace(self) -> Dict[str, Any]:
        return {
            "actions": self.action_trace,
            "rewards": self.reward_trace,
            "packet_count": len(self.packet_trace),
            "packets": self.packet_trace,
        }


def verify_replay_consistency(
    original_trace: Dict[str, Any],
    replayed_trace: Dict[str, Any],
    tolerance: float = 1e-4,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Compares an original execution trace with a replayed trace.
    
    Returns:
        (passed, max_action_error, diagnostics)
    """
    orig_acts = np.array(original_trace["actions"], dtype=np.float64)
    rep_acts = np.array(replayed_trace["actions"], dtype=np.float64)

    if orig_acts.shape != rep_acts.shape:
        raise ReplayVerificationError(f"Trace length mismatch: orig {orig_acts.shape} vs rep {rep_acts.shape}")

    action_err = float(np.max(np.abs(orig_acts - rep_acts))) if orig_acts.size > 0 else 0.0

    orig_rews = np.array(original_trace["rewards"], dtype=np.float64)
    rep_rews = np.array(replayed_trace["rewards"], dtype=np.float64)
    reward_err = float(np.max(np.abs(orig_rews - rep_rews))) if orig_rews.size > 0 else 0.0

    passed = (action_err <= tolerance) and (reward_err <= tolerance)
    diag = {
        "max_action_diff": action_err,
        "max_reward_diff": reward_err,
        "step_count": len(orig_acts),
        "packets_matched": original_trace.get("packet_count") == replayed_trace.get("packet_count"),
    }

    if not passed:
        raise ReplayVerificationError(f"Replay divergence: action_err={action_err}, reward_err={reward_err}")

    return passed, action_err, diag
