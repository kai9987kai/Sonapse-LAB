"""Multi-Timescale Adaptation Schedules & Stability/Plasticity Telemetry.

Manages learning updates across three biological timescales:
1. Fast (Transitions, dt ~ 1 step):
   - Fast-weight associative memory, eligibility traces, neuromodulatory dynamics, local prediction errors.
2. Medium (Episodes, dt ~ 10-100 steps):
   - Replay updates, policy optimization, sleep-style memory consolidation, dormant expert ramp-up.
3. Slow (Developmental, dt ~ 100-1000 steps):
   - Structural plasticity, synaptogenesis, synaptic pruning, dormant slot retirement, topology consolidation.

Tracks plasticity health metrics: representation effective rank, stable rank, dead neuron fraction,
weight drift, predictive surprise, and catastrophic forgetting risk.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class AdaptationEvent:
    owning_subsystem: str
    timescale: str  # 'fast' | 'medium' | 'slow'
    trigger: str
    learning_rate: float
    expected_benefit: str
    computational_cost_ops: int
    state_modified: str
    verification_status: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiTimescaleScheduler:
    """Schedules and audits adaptation operations across fast, medium, and slow timescales."""

    def __init__(self, fast_interval: int = 1, medium_interval: int = 16, slow_interval: int = 64):
        self.fast_interval = max(1, int(fast_interval))
        self.medium_interval = max(1, int(medium_interval))
        self.slow_interval = max(1, int(slow_interval))

        self.step_counter: int = 0
        self.adaptation_log: List[AdaptationEvent] = []
        self.history_effective_rank: List[float] = []

    def should_trigger(self, timescale: str) -> bool:
        if timescale == "fast":
            return (self.step_counter % self.fast_interval) == 0
        if timescale == "medium":
            return (self.step_counter % self.medium_interval) == 0
        if timescale == "slow":
            return (self.step_counter % self.slow_interval) == 0
        return False

    def log_event(self, event: AdaptationEvent) -> None:
        self.adaptation_log.append(event)
        if len(self.adaptation_log) > 64:
            self.adaptation_log = self.adaptation_log[-64:]

    def evaluate_stability_metrics(self, representation_history: List[np.ndarray]) -> Dict[str, Any]:
        """Calculates representation effective rank, stable rank, and dead unit ratio."""
        if len(representation_history) < 4:
            return {
                "effective_rank": 1.0,
                "stable_rank": 1.0,
                "dead_units_fraction": 0.0,
                "plasticity_status": "insufficient_history",
            }

        X = np.stack(representation_history[-32:], axis=0)
        X_centered = X - np.mean(X, axis=0, keepdims=True)

        try:
            svals = np.linalg.svd(X_centered, compute_uv=False)
            s_sq = svals ** 2
            norm_sq = float(np.sum(s_sq)) + 1e-12
            p = s_sq / norm_sq
            ent = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
            eff_rank = float(math.exp(ent))
            max_s_sq = float(np.max(s_sq)) if len(s_sq) else 1e-6
            stable_rank = float(norm_sq / max_s_sq)
        except Exception:
            eff_rank = 1.0
            stable_rank = 1.0

        # Dead unit ratio: variance below threshold
        variances = np.var(X, axis=0)
        dead_units = float(np.mean(variances < 1e-5))

        self.history_effective_rank.append(eff_rank)
        if len(self.history_effective_rank) > 64:
            self.history_effective_rank = self.history_effective_rank[-64:]

        return {
            "effective_rank": round(eff_rank, 4),
            "stable_rank": round(stable_rank, 4),
            "dead_units_fraction": round(dead_units, 4),
            "plasticity_status": "plastic" if dead_units < 0.25 and eff_rank > 2.0 else "constrained",
        }
