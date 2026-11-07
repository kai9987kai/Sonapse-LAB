"""Adaptation Coordinator for Unified Multi-Timescale Plasticity Management.

Coordinates existing learning subsystems without duplicating optimizers or replacing owners:
- Continual backprop & fast-weight memory (fast)
- Neural graft active replay & DPO alignment (medium)
- Structural plasticity & dormant expert lifecycle (slow)
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np

from .multitimescale_learning import MultiTimescaleScheduler, AdaptationEvent


class AdaptationCoordinator:
    """Central coordinator for multi-timescale learning across agent subsystems."""

    def __init__(
        self,
        scheduler: Optional[MultiTimescaleScheduler] = None,
        num_neurons: int = 64,
        seed: int = 42,
    ):
        self.scheduler = scheduler or MultiTimescaleScheduler()
        self.num_neurons = int(num_neurons)
        self.rng = np.random.default_rng(seed)
        self.fast_weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        self.last_summary: Dict[str, Any] = {}

    def step(self, reward: float = 0.0, novelty: float = 0.0, **kwargs) -> Dict[str, Any]:
        self.scheduler.step_counter += 1
        self.fast_weights *= 0.95
        self.fast_weights += float(novelty) * 0.01 * self.rng.standard_normal(self.fast_weights.shape).astype(np.float32)
        norm = float(np.linalg.norm(self.fast_weights))

        regime = "balanced"
        if novelty > 0.5:
            regime = "exploratory_fast"
        elif reward > 1.0:
            regime = "consolidating_slow"

        return {
            "step": self.scheduler.step_counter,
            "fast_weight_norm": norm,
            "adaptation_regime": regime,
            "reward": float(reward),
            "novelty": float(novelty),
        }

    def export_state(self) -> Dict[str, Any]:
        return {
            "step_counter": int(self.scheduler.step_counter),
            "fast_weights": self.fast_weights.copy(),
            "last_summary": deepcopy(self.last_summary),
            "rng": deepcopy(self.rng.bit_generator.state),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        self.scheduler.step_counter = int(state.get("step_counter", self.scheduler.step_counter))
        if "fast_weights" in state:
            self.fast_weights = np.asarray(state["fast_weights"], dtype=np.float32).copy()
        if "last_summary" in state:
            self.last_summary = deepcopy(state["last_summary"])
        if "rng" in state:
            self.rng.bit_generator.state = deepcopy(state["rng"])

    def step_transition(
        self,
        agent,
        obs: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> Dict[str, Any]:
        self.scheduler.step_counter += 1
        actions_taken = []

        # 1. Fast timescale updates (every step)
        if self.scheduler.should_trigger("fast"):
            if getattr(agent, "fast_weight_memory", None) is not None:
                actions_taken.append("fast_weight_associative_update")
            self.scheduler.log_event(
                AdaptationEvent(
                    owning_subsystem="fast_weight_and_eligibility",
                    timescale="fast",
                    trigger="step_transition",
                    learning_rate=0.01,
                    expected_benefit="immediate associative memory retrieval",
                    computational_cost_ops=256,
                    state_modified="eligibility_and_fast_weights",
                    verification_status="active",
                )
            )

        # 2. Medium timescale updates (every ~16 steps or done)
        if self.scheduler.should_trigger("medium") or done:
            if getattr(agent, "neural_graft", None) is not None and getattr(agent.config, "active_training", False):
                actions_taken.append("neural_graft_active_replay")
            self.scheduler.log_event(
                AdaptationEvent(
                    owning_subsystem="active_training_and_dpo",
                    timescale="medium",
                    trigger="episode_boundary" if done else "medium_cadence",
                    learning_rate=0.001,
                    expected_benefit="policy refinement and value calibration",
                    computational_cost_ops=4096,
                    state_modified="graft_policy_heads",
                    verification_status="active",
                )
            )

        # 3. Slow timescale updates (every ~64 steps)
        if self.scheduler.should_trigger("slow"):
            if hasattr(agent, "dormant_lifecycle") and getattr(agent.config, "dormant_lifecycle", True):
                actions_taken.append("dormant_lifecycle_consolidation")
            self.scheduler.log_event(
                AdaptationEvent(
                    owning_subsystem="structural_plasticity_and_lifecycle",
                    timescale="slow",
                    trigger="slow_developmental_cadence",
                    learning_rate=0.0005,
                    expected_benefit="synaptogenesis and expert consolidation",
                    computational_cost_ops=16384,
                    state_modified="native_weights_and_dormant_slots",
                    verification_status="active",
                )
            )

        # Stability evaluation
        recent_feats = [agent.last_features] if hasattr(agent, "last_features") and agent.last_features is not None else []
        stability = self.scheduler.evaluate_stability_metrics(recent_feats)

        summary = {
            "step": self.scheduler.step_counter,
            "actions_triggered": actions_taken,
            "recent_events_logged": len(self.scheduler.adaptation_log),
            "stability_telemetry": stability,
        }
        self.last_summary = summary
        return summary
