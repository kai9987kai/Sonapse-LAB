"""Model-Predictive Control (MPC) and lookahead planning using the Tidal dynamics head.

While behavior cloning and DAgger output reactive policies pi(a|o), the Tidal core
also learns an action-conditioned next-observation delta head:
    o_{t+1} = o_t + Delta(o_t, a)

This module implements model-based imagination:
1. Unrolls candidate action trajectories using the neural world model.
2. Evaluates physical survival viability (detecting predicted starvation,
   dehydration, hazard damage, and exhaustion before committing).
3. Hybridizes policy logits with rollout value estimates to filter out
   hallucinated or catastrophic actions under extreme environmental stress.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .core import CoreRuntime
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass(frozen=True)
class PlanningConfig:
    horizon: int = 2
    gamma: float = 0.90
    policy_weight: float = 0.60
    mpc_weight: float = 0.40
    def __post_init__(self):
        if type(self.horizon) is not int or self.horizon not in (1, 2):
            raise ValueError("The neural MPC supports horizons 1 and 2 only")
        for name in ("gamma", "policy_weight", "mpc_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.policy_weight + self.mpc_weight <= 0:
            raise ValueError("At least one planner weight must be positive")


def evaluate_predicted_observation(obs: Sequence[float], action: int) -> float:
    """Evaluate predicted survival value from a simulated 16-d observation vector."""
    if len(obs) != OBS_SIZE:
        raise ValueError(f"Observation must have {OBS_SIZE} elements")
    health, energy, hydration, stamina, exposure, threat, food, water, shelter, severity, temp, daylight, terrain, scent, last_action, progress = obs

    # Severe penalty for states near lethal physical thresholds
    penalty = 0.0
    if energy < 0.10:
        penalty += 4.0 * (0.10 - energy)
    if hydration < 0.10:
        penalty += 5.0 * (0.10 - hydration)
    if stamina < 0.05:
        penalty += 3.0 * (0.05 - stamina)
    if exposure > 0.80:
        penalty += 4.0 * (exposure - 0.80)
    if threat > 0.65 and action not in (3, 5):  # not sheltered or fleeing
        penalty += 6.0 * (threat - 0.65)

    # Reward for maintaining vital reserves
    vitality = 1.5 * min(energy, 1.0) + 1.8 * min(hydration, 1.0) + 0.8 * min(stamina, 1.0) + 2.0 * min(health, 1.0)
    exposure_penalty = 1.2 * max(0.0, exposure)

    return float(vitality - exposure_penalty - penalty)


class ModelPredictivePlanner:
    """Multi-step imagination planner powered by the learned Tidal dynamics head."""

    def __init__(self, core: CoreRuntime, config: PlanningConfig | None = None):
        self.core = core
        self.config = config or PlanningConfig()

    @torch.inference_mode()
    def plan(self, observation: Sequence[float]) -> dict:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        zeros_1 = torch.zeros(1, self.core.model.config.hash_buckets)

        # 1. Base policy prior logits
        result = self.core.model(zeros_1, obs_tensor)
        policy_logits = result["action"][0]
        policy_probs = F.softmax(policy_logits, dim=-1)

        # 2. Batched Horizon-1 imagination across all 6 actions
        zeros_6 = torch.zeros(N_ACTIONS, self.core.model.config.hash_buckets)
        obs_batch_6 = obs_tensor.repeat(N_ACTIONS, 1)
        actions_0 = torch.arange(N_ACTIONS)
        delta_1 = self.core.model(zeros_6, obs_batch_6, actions_0)["delta"]
        obs_1 = (obs_batch_6 + delta_1).clamp(0.0, 1.0)

        # Evaluate step 1 scores
        scores_1 = [evaluate_predicted_observation(obs_1[a].tolist(), a) for a in range(N_ACTIONS)]

        # 3. Batched Horizon-2 imagination (36 candidate branches in 1 model pass)
        if self.config.horizon >= 2:
            n_branches = N_ACTIONS * N_ACTIONS
            zeros_36 = torch.zeros(n_branches, self.core.model.config.hash_buckets)
            obs_batch_36 = obs_1.repeat_interleave(N_ACTIONS, dim=0)
            actions_1 = torch.arange(N_ACTIONS).repeat(N_ACTIONS)
            delta_2 = self.core.model(zeros_36, obs_batch_36, actions_1)["delta"]
            obs_2 = (obs_batch_36 + delta_2).clamp(0.0, 1.0)

            # Evaluate step 2 scores and reduce by max over sub-actions
            q_values = torch.zeros(N_ACTIONS)
            for a0 in range(N_ACTIONS):
                sub_scores = [evaluate_predicted_observation(obs_2[a0 * N_ACTIONS + a1].tolist(), a1) for a1 in range(N_ACTIONS)]
                q_values[a0] = scores_1[a0] + self.config.gamma * max(sub_scores)
        else:
            q_values = torch.tensor(scores_1, dtype=torch.float32)

        predicted_futures = {}
        for a in range(N_ACTIONS):
            predicted_futures[ACTIONS[a]] = {
                "step_1_observation": [round(x, 4) for x in obs_1[a].tolist()],
                "score": float(q_values[a]),
            }

        # Normalize Q-values to comparable scale
        q_norm = (q_values - q_values.mean()) / (q_values.std() + 1e-6)

        # 4. Hybrid decision: policy prior + imagination lookahead
        combined = self.config.policy_weight * policy_probs + self.config.mpc_weight * F.softmax(q_norm, dim=-1)

        chosen_action = int(combined.argmax())

        return {
            "action": chosen_action,
            "action_name": ACTIONS[chosen_action],
            "policy_action": int(policy_probs.argmax()),
            "mpc_action": int(q_values.argmax()),
            "combined_scores": [round(float(v), 4) for v in combined.tolist()],
            "q_values": [round(float(v), 4) for v in q_values.tolist()],
            "predicted_futures": predicted_futures,
        }

    def act(self, observation: Sequence[float]) -> int:
        return self.plan(observation)["action"]
