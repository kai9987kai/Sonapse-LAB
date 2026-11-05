"""Dual Persistent Memory Architecture and Delayed-Recall Experiments.

Implements Experiment C of Supermix Beyond:
1. Cross-Step Recurrent Memory:
   - Maintains latent state across sequential environment steps h_t = Cell(o_t, a_{t-1}, h_{t-1}),
     allowing information to propagate through time beyond reactive single-step observations.
2. Episodic Experience Store:
   - Records structured events (tick, location, obs, action, reward, surprise, discoveries).
   - Content-addressable retrieval via spatial and semantic feature matching.
3. Explicit Episodic Store Assay (does not evaluate the neural module):
   - Tests whether the agent can utilize critical survival information after it is no longer visible.
   - Rigorously compared against:
     a) Active Persistent Memory
     b) No-Memory Control (reactive baseline)
     c) Shuffled-Memory Control (permuted historical records)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import OBS_SIZE, N_ACTIONS


@dataclass
class EpisodicRecord:
    tick: int
    x: int
    y: int
    observation: list[float]
    action: int
    reward: float
    uncertainty: float
    resource_found: str  # "food", "water", "shelter", or "none"


class CrossStepRecurrentMemory(nn.Module):
    """Maintains neural recurrent state persistently across environment steps."""

    def __init__(self, obs_dim: int = OBS_SIZE, action_dim: int = N_ACTIONS, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_emb = nn.Embedding(action_dim, 16)
        in_dim = obs_dim + 16 + hidden_dim

        # GRU-style gating mechanism for cross-step persistence
        self.update_gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.reset_gate = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.candidate = nn.Sequential(nn.Linear(obs_dim + 16 + hidden_dim, hidden_dim), nn.Tanh())

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_dim), dtype=torch.float32)

    def step(self, obs: torch.Tensor, last_action: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Propagate state across one environment transition."""
        act_vec = self.act_emb(last_action)
        joined = torch.cat((obs, act_vec, h_prev), dim=-1)
        z = self.update_gate(joined)
        r = self.reset_gate(joined)
        cand_in = torch.cat((obs, act_vec, r * h_prev), dim=-1)
        c = self.candidate(cand_in)
        h_next = (1.0 - z) * h_prev + z * c
        return h_next


class EpisodicMemoryStore:
    """Bounded episodic memory buffer with spatial and semantic retrieval."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.records: list[EpisodicRecord] = []

    def clear(self) -> None:
        self.records.clear()

    def append(self, record: EpisodicRecord) -> None:
        if len(self.records) >= self.capacity:
            self.records.pop(0)
        self.records.append(record)

    def retrieve_by_location(self, x: int, y: int, radius: int = 1) -> list[EpisodicRecord]:
        """Finds memories recorded at or near the given coordinate."""
        matches = [
            r for r in self.records
            if abs(r.x - x) <= radius and abs(r.y - y) <= radius
        ]
        return sorted(matches, key=lambda r: r.tick, reverse=True)

    def retrieve_by_resource(self, resource_type: str) -> list[EpisodicRecord]:
        """Finds memories where a key survival resource was located."""
        return [r for r in self.records if r.resource_found == resource_type]

    def retrieve_high_surprise(self, top_k: int = 5) -> list[EpisodicRecord]:
        """Finds the most surprising or uncertain past events for retrospective review."""
        return sorted(self.records, key=lambda r: r.uncertainty, reverse=True)[:top_k]


class DelayedRecallTask:
    """Controlled memory-isolation benchmark.
    
    Setup:
    - Step 0: Agent is shown a transient environmental cue:
      `target_action` in {1 (forage), 2 (drink), 3 (shelter)} encoded in the observation.
    - Steps 1 to K: The cue is completely blanked out (zeroed).
    - Step K: Agent must execute `target_action`.
    """

    def __init__(self, delay_steps: int = 4, seed: int = 42):
        self.delay_steps = delay_steps
        self.rng = random.Random(seed)
        self.current_step = 0
        self.target_action = 1
        self.reset()

    def reset(self) -> list[float]:
        self.current_step = 0
        self.target_action = self.rng.choice([1, 2, 3])
        obs = [0.0] * OBS_SIZE
        # In step 0, obs[6] indicates food cue, obs[7] indicates water cue, obs[8] indicates shelter cue
        if self.target_action == 1:
            obs[6] = 1.0
        elif self.target_action == 2:
            obs[7] = 1.0
        elif self.target_action == 3:
            obs[8] = 1.0
        obs[15] = 0.0  # progress
        return obs

    def step(self, action: int) -> tuple[list[float], float, bool]:
        self.current_step += 1
        done = self.current_step >= self.delay_steps
        # Blank observation during delay and at decision step
        obs = [0.0] * OBS_SIZE
        obs[15] = self.current_step / self.delay_steps

        reward = 0.0
        if done:
            # Check if chosen action matches the target cue from step 0
            if action == self.target_action:
                reward = 1.0
            else:
                reward = -0.5
        return obs, reward, done


def run_delayed_recall_benchmark(
    delays: Sequence[int] = (2, 4, 8),
    episodes_per_condition: int = 50,
    seed: int = 42,
    include_receipts: bool = False,
) -> dict[str, Any]:
    """Run a deterministic explicit-store assay with matched donor controls.

    The policy reads an observed cue into an EpisodicRecord and retrieves that
    record later. It is a software storage assay, not learned recurrent recall.
    The shuffled arm receives actual records from a permutation of the same
    episodes; the no-memory and missing-record fallback choose among the same
    three actions. The default return preserves the historical score mapping.
    """
    delays = list(delays)
    if not delays or any(type(delay) is not int or not 2 <= delay <= 10000 for delay in delays):
        raise ValueError("delays must be non-empty integers in [2, 10000] so the decision cue is hidden")
    if len(set(delays)) != len(delays):
        raise ValueError("delays must be unique")
    if type(episodes_per_condition) is not int or not 2 <= episodes_per_condition <= 10000:
        raise ValueError("episodes_per_condition must be an integer in [2, 10000] for donor controls")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    conditions = ("persistent_memory", "no_memory", "shuffled_memory")
    support = [1, 2, 3]
    results: dict[str, dict[int, float]] = {condition: {} for condition in conditions}
    receipts = []

    for delay in delays:
        records = []
        for ep in range(episodes_per_condition):
            ep_seed = seed + ep * 1337 + delay * 101
            env = DelayedRecallTask(delay_steps=delay, seed=ep_seed)
            initial = env.reset()
            # Decode the visible input only; policy never reads env.target_action.
            cue = next(action for action in support if initial[action + 5] > 0.5)
            records.append(EpisodicRecord(0, 0, 0, list(initial), cue, 0.0, 0.0, "cue"))
        order = list(range(episodes_per_condition))
        random.Random(f"episodic-donor:{seed}:{delay}").shuffle(order)
        donors = {episode: order[(i + 1) % len(order)] for i, episode in enumerate(order)}
        # Cyclic permutation of a shuffled order: no self-donors and cue counts preserved.
        for condition in conditions:
            correct = 0
            for ep in range(episodes_per_condition):
                ep_seed = seed + ep * 1337 + delay * 101
                env = DelayedRecallTask(delay_steps=delay, seed=ep_seed)
                obs = env.reset()
                store = EpisodicMemoryStore()
                source = ep if condition == "persistent_memory" else donors[ep] if condition == "shuffled_memory" else None
                if source is not None:
                    original = records[source]
                    store.append(EpisodicRecord(original.tick, original.x, original.y, list(original.observation),
                                               original.action, original.reward, original.uncertainty, original.resource_found))
                done = False
                retrieved_action = None
                while not done:
                    fallback = random.Random(f"episodic-fallback:{ep_seed}:{env.current_step}").choice(support)
                    retrieved = store.retrieve_by_resource("cue") if source is not None else []
                    retrieved_action = retrieved[0].action if retrieved and retrieved[0].action in support else None
                    chosen_action = retrieved_action if retrieved_action is not None else fallback
                    obs, reward, done = env.step(chosen_action)
                correct += reward > 0.5
                receipts.append({"condition": condition, "delay": delay, "episode": ep, "seed": ep_seed,
                                 "action": chosen_action, "retrieved_action": retrieved_action,
                                 "source_episode": source, "target": records[ep].action, "reward": reward,
                                 "correct": reward > 0.5})
            results[condition][delay] = round(correct / episodes_per_condition, 4)
    if not include_receipts:
        return results
    return {"schema": "poseidon-episodic-store-assay-v2", "assay": "explicit episodic cue storage and retrieval",
            "neural_memory_evaluated": False, "action_support": support, "delays": delays,
            "seed": seed, "episodes_per_condition": episodes_per_condition,
            "accuracy": results, "results": results, "episodes": receipts,
            "limits": ["No recurrent model is trained, called, or evaluated by this assay.",
                       "Perfect explicit-store recall does not establish learned memory or real-world survival skill.",
                       "Donor shuffling preserves this finite sample's cue frequencies; accuracy need not equal exactly one third."]}
