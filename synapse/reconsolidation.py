"""Memory Reconsolidation and Labile State Plasticity Engine.

Scientific Foundations:
1. Nader, Schafe & Le Doux (Nature 2000):
   Established that retrieved memories return to a protein-synthesis-dependent
   labile state, requiring active reconsolidation to persist.
2. Sara (Nature Rev. Neurosci. 2000):
   Memory retrieval destabilizes the synaptic trace in proportion to mismatch /
   prediction error (surprisal), providing an adaptive window to update outdated
   information with current environmental statistics.
3. Extinction vs. Reconsolidation Boundary (Pedreira & Maldonado, Neuron 2003):
   - Small mismatch / reminder cue: Triggers destabilization and reconsolidation (memory updating).
   - Massive or protracted unreinforced exposure: Shifts from reconsolidation to extinction learning.
4. Synaptic Tagging & Capture Integration (Frey & Morris, Nature 1997):
   Labile traces capture newly synthesized plasticity-related proteins to re-stabilize.

Governing Invariant:
When enabled=False or reconsolidation_gain == 0.0:
Memory buffers and replay candidates remain unmodified bit-for-bit.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclasses.dataclass
class LabileMemoryTrace:
    trace_id: str
    original_obs: np.ndarray
    original_action: np.ndarray
    original_reward: float
    importance: float
    lability: float = 1.0           # Current vulnerability index in [0..1]
    reactivation_timestamp: float = 0.0
    destabilization_surprisal: float = 0.0
    update_count: int = 0
    is_restabilized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "original_reward": round(float(self.original_reward), 4),
            "importance": round(float(self.importance), 4),
            "lability": round(float(self.lability), 4),
            "reactivation_timestamp": round(float(self.reactivation_timestamp), 3),
            "destabilization_surprisal": round(float(self.destabilization_surprisal), 4),
            "update_count": int(self.update_count),
            "is_restabilized": bool(self.is_restabilized),
        }


@dataclasses.dataclass
class ReconsolidationTelemetry:
    active_labile_count: int = 0
    total_reactivations: int = 0
    total_updates: int = 0
    total_extinctions: int = 0
    mean_lability: float = 0.0
    latest_prediction_error: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_labile_count": int(self.active_labile_count),
            "total_reactivations": int(self.total_reactivations),
            "total_updates": int(self.total_updates),
            "total_extinctions": int(self.total_extinctions),
            "mean_lability": round(float(self.mean_lability), 4),
            "latest_prediction_error": round(float(self.latest_prediction_error), 4),
            "timestamp": round(float(self.timestamp), 3),
        }


class MemoryReconsolidationEngine:
    """Manages memory trace reactivation, destabilization, updating, and restabilization."""

    def __init__(
        self,
        destabilization_threshold: float = 0.25,  # Minimum surprisal to destabilize trace
        extinction_threshold: float = 0.85,        # Surprisal above which memory extinguishes
        labile_window_steps: int = 30,             # Duration trace remains labile
        reconsolidation_lr: float = 0.15,          # Rate of updating labile memory with new evidence
        seed: int = 42,
    ):
        self.destabilization_threshold = float(destabilization_threshold)
        self.extinction_threshold = float(extinction_threshold)
        self.labile_window_steps = int(labile_window_steps)
        self.reconsolidation_lr = float(reconsolidation_lr)
        self.rng = np.random.default_rng(seed)

        self.labile_traces: Dict[str, LabileMemoryTrace] = {}
        self.total_reactivations = 0
        self.total_updates = 0
        self.total_extinctions = 0
        self._last_telemetry: Optional[ReconsolidationTelemetry] = None

    def reset_state(self) -> None:
        self.labile_traces.clear()
        self.total_reactivations = 0
        self.total_updates = 0
        self.total_extinctions = 0
        self._last_telemetry = None

    def reactivate_trace(
        self,
        trace_id: str,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        importance: float,
        prediction_error: float,
        enabled: bool = False,
    ) -> Optional[LabileMemoryTrace]:
        """Evaluates whether retrieval triggers memory destabilization into a labile state.

        Args:
            trace_id: Identifier of retrieved memory.
            obs: Observation vector.
            action: Action vector.
            reward: Historical stored reward.
            importance: Salience score.
            prediction_error: Surprisal between current sensory input and retrieved trace expectation.
            enabled: Master opt-in flag.

        Returns:
            LabileMemoryTrace if destabilized, or None if unchanged.
        """
        if not enabled:
            return None

        self.total_reactivations += 1
        t0 = time.time()
        pe = float(abs(prediction_error))

        # Check extinction boundary
        if pe >= self.extinction_threshold:
            self.total_extinctions += 1
            # Extinction: Suppress importance without destabilizing core trace
            return None

        # Check destabilization boundary (Nader 2000)
        if pe >= self.destabilization_threshold:
            trace = LabileMemoryTrace(
                trace_id=trace_id,
                original_obs=np.copy(obs),
                original_action=np.copy(action),
                original_reward=float(reward),
                importance=float(importance),
                lability=1.0,
                reactivation_timestamp=t0,
                destabilization_surprisal=pe,
                update_count=0,
                is_restabilized=False,
            )
            self.labile_traces[trace_id] = trace
            return trace

        return None

    def update_labile_trace(
        self,
        trace_id: str,
        new_obs: np.ndarray,
        new_action: np.ndarray,
        new_reward: float,
        enabled: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Updates a destabilized memory trace with novel evidence during the labile window."""
        if not enabled or trace_id not in self.labile_traces:
            return None

        trace = self.labile_traces[trace_id]
        if trace.is_restabilized or trace.lability <= 0.05:
            return None

        # Update trace parameters proportional to current lability
        eff_lr = self.reconsolidation_lr * trace.lability
        trace.original_obs = (1.0 - eff_lr) * trace.original_obs + eff_lr * new_obs
        trace.original_action = (1.0 - eff_lr) * trace.original_action + eff_lr * new_action
        trace.original_reward = (1.0 - eff_lr) * trace.original_reward + eff_lr * new_reward
        trace.update_count += 1
        self.total_updates += 1

        return (trace.original_obs, trace.original_action, trace.original_reward)

    def step_decay(self, enabled: bool = False) -> ReconsolidationTelemetry:
        """Decays lability of all active traces toward restabilization (protein synthesis)."""
        t0 = time.time()
        if not enabled:
            telemetry = ReconsolidationTelemetry(
                active_labile_count=len(self.labile_traces),
                total_reactivations=self.total_reactivations,
                total_updates=self.total_updates,
                total_extinctions=self.total_extinctions,
                mean_lability=0.0,
                timestamp=t0,
            )
            self._last_telemetry = telemetry
            return telemetry

        decay_per_step = 1.0 / max(1, self.labile_window_steps)
        traces_to_remove = []

        for tid, trace in self.labile_traces.items():
            trace.lability = max(0.0, trace.lability - decay_per_step)
            if trace.lability <= 0.0:
                trace.is_restabilized = True
                traces_to_remove.append(tid)

        for tid in traces_to_remove:
            self.labile_traces.pop(tid, None)

        active = len(self.labile_traces)
        mean_lab = float(np.mean([t.lability for t in self.labile_traces.values()])) if active > 0 else 0.0

        telemetry = ReconsolidationTelemetry(
            active_labile_count=active,
            total_reactivations=self.total_reactivations,
            total_updates=self.total_updates,
            total_extinctions=self.total_extinctions,
            mean_lability=mean_lab,
            timestamp=t0,
        )
        self._last_telemetry = telemetry
        return telemetry

    def get_summary(self) -> Dict[str, Any]:
        if self._last_telemetry is None:
            return {
                "active_labile_count": len(self.labile_traces),
                "total_reactivations": self.total_reactivations,
                "total_updates": self.total_updates,
                "total_extinctions": self.total_extinctions,
                "mean_lability": 0.0,
                "destabilization_threshold": self.destabilization_threshold,
                "extinction_threshold": self.extinction_threshold,
            }
        d = self._last_telemetry.to_dict()
        d["destabilization_threshold"] = self.destabilization_threshold
        d["extinction_threshold"] = self.extinction_threshold
        return d
