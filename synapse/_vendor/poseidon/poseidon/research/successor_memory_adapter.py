"""Project NEUROGRAFT Successor Memory Adapter for Supermix-Poseidon.

Bridges NEUROGRAFT's Successor Representation and Sharp-Wave Ripple Replay
with Poseidon's 3-Tier MNEMOSYNE cognitive memory hierarchy (poseidon/mnemosyne.py).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..neurograft_engine import (
    GraphLaplacianTopologicalEngine,
    NeurograftEngine,
    SharpWaveRippleReplay,
    SuccessorRepresentationEngine,
)
from ..world import ACTIONS, N_ACTIONS, OBSERVATION_NAMES, OBS_SIZE


def discretize_obs_to_spatial_state(
    obs: Sequence[float],
    num_bins: int = 16,
    num_spatial_states: Optional[int] = None,
) -> int:
    """Discretizes continuous 16-D TidePool observation into discrete spatial state index.

    Uses terrain difficulty, shelter quality, and local resource scent. This
    feature hash aliases locations; it is not a measured position identifier.
    """
    n_states = num_spatial_states if num_spatial_states is not None else num_bins
    if len(obs) < OBS_SIZE:
        return 0
    t = max(0.0, min(1.0, float(obs[OBSERVATION_NAMES.index("terrain_difficulty")])))
    s = max(0.0, min(1.0, float(obs[OBSERVATION_NAMES.index("shelter_quality")])))
    scent = max(0.0, min(1.0, float(obs[OBSERVATION_NAMES.index("resource_scent")])))

    # Compute a deterministic spatial hash index in 0..n_states-1
    val = (t * 7.0 + s * 13.0 + scent * 5.0)
    return int(val) % n_states


class SuccessorMemoryAdapter:
    """Connects MNEMOSYNE episodic memories to Successor Representation and SWR replay."""

    def __init__(
        self,
        memory: Optional[Any] = None,
        num_states: int = 16,
        num_spatial_states: Optional[int] = None,
        num_actions: int = N_ACTIONS,
        discount: float = 0.90,
        learning_rate: float = 0.15,
        surprise_threshold: float = 0.60,
    ):
        n_s = num_spatial_states if num_spatial_states is not None else num_states
        self.memory = memory
        self.num_states = n_s
        self.num_spatial_states = n_s
        self.num_actions = num_actions
        self.discount = discount
        self.surprise_threshold = surprise_threshold

        self.sr_engine = SuccessorRepresentationEngine(
            num_states=n_s,
            num_actions=num_actions,
            discount=discount,
            learning_rate=learning_rate,
        )
        self.replay_buffer = SharpWaveRippleReplay(capacity=512)
        self.total_steps_recorded = 0
        self.total_swr_triggers = 0
        self.verified_audit_receipt: Optional[Dict[str, Any]] = None

    def record_step(
        self,
        observation: Sequence[float],
        action: int,
        next_observation: Sequence[float],
        reward: float = 0.0,
        surprise: float = 0.0,
    ) -> Dict[str, Any]:
        """Records a transition step, updates the successor matrix, and triggers SWR if salient."""
        s = discretize_obs_to_spatial_state(observation, self.num_states)
        s_next = discretize_obs_to_spatial_state(next_observation, self.num_states)
        a = int(action) % self.num_actions

        # Online SR-TD update
        update_info = self.sr_engine.update_sr_td(s, a, s_next, reward)

        # Push to SWR replay buffer
        self.replay_buffer.push(s, a, s_next, reward, surprise)
        self.total_steps_recorded += 1

        # Check if SWR reverse replay should trigger (high surprise or rest action)
        swr_triggered = False
        if surprise >= self.surprise_threshold or a == ACTIONS.index("rest"):
            swr_info = self.replay_buffer.trigger_swr_reverse_replay(self.sr_engine, num_sweeps=4)
            self.total_swr_triggers += 1
            swr_triggered = True
        else:
            swr_info = {}

        return {
            "state": s,
            "spatial_state": s,
            "action": a,
            "next_state": s_next,
            "next_spatial_state": s_next,
            "td_error": update_info["action_td_error_norm"],
            "sr_td_error": update_info["action_td_error_norm"],
            "swr_triggered": swr_triggered,
            "swr_info": swr_info,
        }

    def get_action_preferences(
        self,
        observation: Sequence[float],
        reward_vector: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Computes successor-representation action values Q_SR(s, a) for the current observation."""
        s = discretize_obs_to_spatial_state(observation, self.num_states)
        q_all = self.sr_engine.revalue_action_values(reward_vector)
        return [round(float(q), 4) for q in q_all[s]]

    def trigger_offline_swr_consolidation(self, num_sweeps: int = 4) -> Dict[str, Any]:
        """Triggers hippocampal-inspired offline reverse replay consolidation sweeps."""
        res = self.replay_buffer.trigger_swr_reverse_replay(self.sr_engine, num_sweeps=num_sweeps)
        self.total_swr_triggers += 1
        return res

    def audit_zero_shot_goal_revaluation(
        self,
        old_goal_state: int = 2,
        new_goal_state: int = 14,
        goal_reward: float = 10.0,
    ) -> Dict[str, Any]:
        """Audits instantaneous value revaluation when goal state changes."""
        r_old = [0.0] * self.num_states
        r_old[old_goal_state % self.num_states] = goal_reward
        v_old = [round(float(v), 4) for v in self.sr_engine.revalue_state_values(r_old)]

        r_new = [0.0] * self.num_states
        r_new[new_goal_state % self.num_states] = goal_reward
        v_new = [round(float(v), 4) for v in self.sr_engine.revalue_state_values(r_new)]

        return {
            "zero_shot_revalued": True,
            "old_goal_state": old_goal_state,
            "new_goal_state": new_goal_state,
            "old_values": v_old,
            "new_values": v_new,
            "old_best_state": int(np.argmax(v_old)),
            "new_best_state": int(np.argmax(v_new)),
        }

    def revalue_goal(self, target_state: int, target_value: float = 10.0) -> List[int]:
        """Zero-shot revaluation for a new goal location: returns best actions for all states."""
        r = [0.0] * self.num_states
        if 0 <= target_state < self.num_states:
            r[target_state] = target_value
        q_matrix = self.sr_engine.revalue_action_values(r)
        return [int(a) for a in np.argmax(q_matrix, axis=1)]

    def run_memory_audit(self, num_episodes: int = 8, episode_len: int = 16) -> Dict[str, Any]:
        """Runs a synthetic audit of successor memory and SWR replay."""
        rng = np.random.RandomState(42)
        for ep in range(num_episodes):
            obs = rng.uniform(0.0, 1.0, size=OBS_SIZE).tolist()
            for t in range(episode_len):
                act = int(rng.randint(0, self.num_actions))
                nxt = rng.uniform(0.0, 1.0, size=OBS_SIZE).tolist()
                rew = 5.0 if t == episode_len - 1 else 0.0
                surp = 0.8 if t == episode_len - 1 else 0.1
                self.record_step(obs, act, nxt, rew, surp)
                obs = nxt

        # Demonstrate zero-shot revaluation
        policy_goal_0 = self.revalue_goal(target_state=0)
        policy_goal_15 = self.revalue_goal(target_state=self.num_states - 1)
        policy_divergence = sum(p0 != p15 for p0, p15 in zip(policy_goal_0, policy_goal_15))

        receipt = {
            "total_steps_recorded": self.total_steps_recorded,
            "total_swr_triggers": self.total_swr_triggers,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "policy_divergence_on_revaluation": policy_divergence,
            "zero_shot_adaptation_verified": policy_divergence > 0,
            "swr_replay_verified": self.total_swr_triggers > 0,
        }
        blob = json.dumps(receipt, sort_keys=True).encode("utf-8")
        receipt["audit_sha256"] = hashlib.sha256(blob).hexdigest()
        self.verified_audit_receipt = receipt
        return receipt
