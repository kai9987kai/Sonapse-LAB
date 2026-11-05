"""POSEIDON HORIZON-X: Multi-Horizon Risk-Sensitive Planner with Dynamic Reserve Modulation.

Key Innovations:
1. Multi-Horizon Evaluator:
   Evaluates candidate actions across multiple planning horizons H in {1, 4, 8, 16, 32}:
     V^{(H)}(a) = (1 / H) * R^{(H)}(a) - \\lambda_t * U^{(H)}(a) - \\beta * P_{fail}^{(H)}(a)
   aggregating near-term tactical safety and long-term strategic viability.

2. Dynamic Reserve-Dependent Risk Modulation:
   Prevents the "pessimism trap" where an agent in severe depletion refuses to act
   because all uncertain paths carry epistemic risk penalties:
     \\rho_t = health * min(energy, hydration)
     \\lambda_t = \\lambda_0 * max(\\mu_{min}, \\rho_t)
   When reserves are abundant, high risk aversion is maintained (avoiding unnecessary hazards);
   when reserves collapse, risk aversion dynamically relaxes to permit high-variance survival attempts.

3. Adverse Override Ledger:
   Audits every planner intervention against the frozen Tidal baseline:
   tracks margin, counterfactual score divergence, and realized post-intervention trajectory returns.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from .core import CoreRuntime
from .oracle import OracleEnsemble, OracleRuntime
from .research.oracle_adapter import oracle_status
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass(frozen=True)
class HorizonXConfig:
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    base_lambda: float = 0.75
    base_beta: float = 3.0
    min_risk_scale: float = 0.10
    override_threshold: float = 0.045
    horizon_decay: float = 0.85

    def __post_init__(self):
        if (not self.horizons or any(type(h) is not int or not 1 <= h <= 32 for h in self.horizons)
                or len(set(self.horizons)) != len(self.horizons)):
            raise ValueError("horizons must contain unique integers in [1, 32]")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in (self.base_lambda, self.base_beta, self.override_threshold)):
            raise ValueError("planner coefficients must be finite and nonnegative")
        if not 0 < self.horizon_decay <= 1 or not 0 <= self.min_risk_scale <= 1:
            raise ValueError("invalid planner risk scale or horizon decay")


@dataclass
class OverrideRecord:
    step: int
    base_action: int
    selected_action: int
    base_score: float
    selected_score: float
    margin: float
    reserve_index: float
    lambda_t: float
    horizon_scores: dict[int, float]


class HorizonXPlanner:
    """Multi-horizon risk-sensitive controller with dynamic reserve modulation."""

    def __init__(
        self,
        core: CoreRuntime,
        oracle: OracleEnsemble,
        config: HorizonXConfig | None = None,
    ):
        self.core = core
        self.oracle = oracle
        self.config = config or HorizonXConfig()
        self.override_history: list[OverrideRecord] = []
        self.step_counter: int = 0

    def compute_reserve_index(self, obs: Sequence[float]) -> float:
        """Computes metabolic reserve index rho_t = health * min(energy, hydration)."""
        health = max(0.0, min(1.0, float(obs[0])))
        energy = max(0.0, min(1.0, float(obs[1])))
        hydration = max(0.0, min(1.0, float(obs[2])))
        return float(health * min(energy, hydration))

    def compute_dynamic_lambda(self, obs: Sequence[float], metacognitive_state: Any | None = None) -> float:
        """Modulates risk aversion lambda_t based on reserves, preventing pessimism trap."""
        rho_t = self.compute_reserve_index(obs)
        scale = max(self.config.min_risk_scale, rho_t)
        base = self.config.base_lambda * scale
        if metacognitive_state is not None and getattr(metacognitive_state, "epistemic_caution_active", False):
            base *= 1.5
        return float(base)

    def evaluate_action_at_horizon(
        self,
        latent_state: torch.Tensor,
        obs_tensor: torch.Tensor,
        action: int,
        base_action: int,
        horizon: int,
        lambda_t: float,
    ) -> dict[str, float]:
        """Rolls out an action sequence of length H through the Oracle ensemble."""
        if horizon == 1:
            seq = [action]
        else:
            seq = [action] + [base_action] * (horizon - 1)

        imagination = self.oracle.imagine_trajectory(latent_state, obs_tensor, seq)
        r_exp = imagination["total_predicted_reward"]
        u_exp = imagination["total_uncertainty"]
        survival = imagination.get("survival_probability")
        calibrated_failure = imagination.get("failure_probability_calibrated", False)
        p_fail = 1.0 - survival if calibrated_failure and survival is not None else None

        # Normalized return per step - risk penalty - failure penalty
        score = (r_exp / horizon) - (lambda_t * u_exp) - (self.config.base_beta * (p_fail or 0.0))
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite ORACLE planning score")
        return {
            "score": float(score),
            "expected_reward": float(r_exp),
            "uncertainty": float(u_exp),
            "failure_prob": p_fail,
        }

    def plan(
        self,
        observation: Sequence[float],
        candidate_actions: Sequence[int] | None = None,
        memory_context: dict[str, Any] | None = None,
        metacognitive_state: Any | None = None,
    ) -> dict[str, Any]:
        """Selects action maximizing composite multi-horizon risk-adjusted score."""
        obs_tensor = torch.tensor(list(observation), dtype=torch.float32)
        if (obs_tensor.shape != (OBS_SIZE,) or not torch.isfinite(obs_tensor).all()
                or ((obs_tensor < 0) | (obs_tensor > 1)).any()):
            raise ValueError("observation must contain 16 finite values in [0, 1]")
        base_action = self.core.act(list(observation))
        lambda_t = self.compute_dynamic_lambda(observation, metacognitive_state=metacognitive_state)
        rho_t = self.compute_reserve_index(observation)

        status = oracle_status(self.oracle)
        supported, reason = (self.oracle.planning_support(observation) if status["trained"]
                             else (False, "no_trained_oracle_selected"))
        actions_to_eval = list(candidate_actions) if candidate_actions is not None else list(range(N_ACTIONS))
        if not actions_to_eval or any(type(a) is not int or not 0 <= a < N_ACTIONS for a in actions_to_eval):
            raise ValueError("candidate actions must be valid integers")
        actions_to_eval = sorted(set(actions_to_eval) | {base_action})
        horizons = tuple(h for h in self.config.horizons
                         if 1 <= h <= getattr(self.oracle, "max_planning_horizon", 0))
        if not supported or not horizons:
            self.step_counter += 1
            return {"selected_action": base_action, "selected_action_name": ACTIONS[base_action],
                    "base_action": base_action, "base_action_name": ACTIONS[base_action],
                    "override": False, "margin": 0.0, "chosen_score": 0.0,
                    "reserve_index": rho_t, "dynamic_lambda": lambda_t,
                    "planning_enabled": False, "fallback_reason": reason or "unsupported_horizons",
                    "oracle": status,
                    "candidate_evaluations": [{"action": a, "action_name": ACTIONS[a],
                                               "composite_score": 0.0, "horizon_scores": {},
                                               "evaluated": False} for a in actions_to_eval]}

        core_out = self.core.model(["Survive"], obs_tensor.reshape(1, -1))
        latent_state = core_out["memory"].squeeze(0)

        # Weights across horizons: w_H \propto decay^{idx}
        h_weights = {}
        total_w = 0.0
        for i, h in enumerate(horizons):
            w = math.pow(self.config.horizon_decay, i)
            h_weights[h] = w
            total_w += w
        for h in h_weights:
            h_weights[h] /= total_w

        action_evals = []
        for act in actions_to_eval:
            h_scores = {}
            composite_score = 0.0
            for h in horizons:
                h_res = self.evaluate_action_at_horizon(
                    latent_state=latent_state,
                    obs_tensor=obs_tensor,
                    action=act,
                    base_action=base_action,
                    horizon=h,
                    lambda_t=lambda_t,
                )
                h_scores[h] = h_res["score"]
                composite_score += h_weights[h] * h_res["score"]

            # Causal Memory Modulation from Mnemosyne / NEXUS-Omega
            if memory_context is not None and memory_context.get("mode") not in ("erased", "memory_free"):
                f_lcb = float(memory_context.get("estimated_food_lcb", observation[6]))
                w_lcb = float(memory_context.get("estimated_water_lcb", observation[7]))
                
                # Energy deficit modulation
                energy_deficit = max(0.0, 0.6 - float(observation[1]))
                if act == 1 and energy_deficit > 0.0 and f_lcb > float(observation[6]):
                    composite_score += 0.05 * energy_deficit * (f_lcb - float(observation[6]))

                # Hydration deficit modulation
                water_deficit = max(0.0, 0.6 - float(observation[2]))
                if act == 2 and water_deficit > 0.0 and w_lcb > float(observation[7]):
                    composite_score += 0.05 * water_deficit * (w_lcb - float(observation[7]))

                # Decision-Value-Gated Episodic Memory modulation
                dv_deltas = memory_context.get("decision_value_deltas")
                if dv_deltas is not None and len(dv_deltas) == N_ACTIONS:
                    composite_score += float(dv_deltas[act])

            if not math.isfinite(composite_score):
                raise FloatingPointError("nonfinite ORACLE composite score")
            action_evals.append({
                "action": act,
                "action_name": ACTIONS[act],
                "composite_score": composite_score,
                "horizon_scores": h_scores,
            })

        best_cand = max(action_evals, key=lambda x: x["composite_score"])
        base_eval = next((e for e in action_evals if e["action"] == base_action), action_evals[0])

        margin = best_cand["composite_score"] - base_eval["composite_score"]
        override = (best_cand["action"] != base_action) and (margin > self.config.override_threshold)

        chosen_action = best_cand["action"] if override else base_action
        chosen_score = best_cand["composite_score"] if override else base_eval["composite_score"]

        if override:
            self.override_history.append(
                OverrideRecord(
                    step=self.step_counter,
                    base_action=base_action,
                    selected_action=best_cand["action"],
                    base_score=base_eval["composite_score"],
                    selected_score=best_cand["composite_score"],
                    margin=margin,
                    reserve_index=rho_t,
                    lambda_t=lambda_t,
                    horizon_scores=best_cand["horizon_scores"],
                )
            )

        self.step_counter += 1

        return {
            "selected_action": chosen_action,
            "selected_action_name": ACTIONS[chosen_action],
            "base_action": base_action,
            "base_action_name": ACTIONS[base_action],
            "override": override,
            "margin": margin,
            "chosen_score": chosen_score,
            "reserve_index": rho_t,
            "dynamic_lambda": lambda_t,
            "candidate_evaluations": action_evals,
            "planning_enabled": True,
            "fallback_reason": None,
            "oracle": oracle_status(self.oracle),
        }

    def summary(self) -> dict[str, Any]:
        """Returns statistics on planner overrides and risk modulation history."""
        total_steps = self.step_counter
        overrides = len(self.override_history)
        override_rate = (overrides / total_steps) if total_steps > 0 else 0.0

        avg_margin = (
            sum(r.margin for r in self.override_history) / overrides if overrides > 0 else 0.0
        )
        return {
            "total_steps": total_steps,
            "override_count": overrides,
            "override_rate": float(override_rate),
            "average_override_margin": float(avg_margin),
            "recent_overrides": [asdict(r) for r in self.override_history[-10:]],
        }
