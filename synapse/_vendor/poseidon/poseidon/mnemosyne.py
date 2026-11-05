"""POSEIDON MNEMOSYNE: 3-Tier Cognitive Memory Architecture with Matched Lesion Assays.

Implements a biologically inspired tripartite memory hierarchy:
1. Tier 1 (Working Memory):
   Fast recurrent context integration (GRU cell) updating with current observation,
   prior action, and prediction error/surprise:
     e_{t-1} = || o_t - \\hat{o}_t ||_2
   maintaining dynamic temporal context h_t.

2. Tier 2 (Value-Gated Episodic Store):
   Longer-horizon experiential memory indexed by state representations.
   Retrieval is gated by value relevance:
     P(i) \\propto (\\alpha_r |r_i| + \\alpha_e e_i + \\alpha_{rec} \\gamma^{t - t_i}) * sim(o_t, o_i)

3. Tier 3 (Spatial & Semantic Cognitive Map):
   Topological patch graph with Bayesian online resource replenishment estimation:
     \\hat{R}_{food}^{LCB}(t) = \\min(C_{food}, o_{food} + \\Delta t * (\\mu_{food} - z * \\sigma_{food}))
   and empirical macro-transition distributions P(patch_dest | patch_src, action).

Includes 5 rigorous matched lesion assay modes:
- intact: Full 3-tier memory active.
- erased: Memory wiped at every tick (zero context retention).
- shifted: Circular permutation of episodic memory keys (breaks semantic alignment
  while preserving dimensionality, variance, and norm).
- random_retrieval: Episodic queries select randomly from history, ignoring value/similarity.
- memory_free: Memory completely bypassed (pure reactive baseline).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import math
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import ACTIONS, N_ACTIONS, OBS_SIZE

MNEMOSYNE_MODES = ("intact", "erased", "shifted", "random_retrieval", "memory_free")


def _fingerprint(observation: Sequence[float]) -> str:
    """Extracts invariant terrain and shelter hash from observation vector."""
    return f"{round(float(observation[12]), 6):.6f}_{round(float(observation[8]), 6):.6f}"


@dataclass
class EpisodicRecord:
    step: int
    observation: list[float]
    action: int
    reward: float
    next_observation: list[float]
    surprise: float
    context: list[float]


class WorkingMemory(nn.Module):
    """Tier 1: Recurrent working memory updating with surprise signals."""

    def __init__(self, obs_size: int = OBS_SIZE, action_count: int = N_ACTIONS, hidden_size: int = 64):
        super().__init__()
        self.obs_size = obs_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        # Input: obs + action one-hot + scalar surprise
        self.input_size = obs_size + action_count + 1
        self.cell = nn.GRUCell(self.input_size, hidden_size)
        self.register_buffer("initial_state", torch.zeros(1, hidden_size))

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size)

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        surprise: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Single-step recurrence."""
        batch_size = obs.shape[0]
        # One-hot encode previous action
        action_one_hot = F.one_hot(prev_action, num_classes=self.action_count).float()
        surprise_col = surprise.reshape(batch_size, 1)
        inputs = torch.cat([obs, action_one_hot, surprise_col], dim=-1)
        return self.cell(inputs, hidden)


@dataclass
class EpisodicStoreConfig:
    capacity: int = 256
    alpha_reward: float = 1.0
    alpha_surprise: float = 2.0
    alpha_recency: float = 0.5
    recency_gamma: float = 0.98
    similarity_temp: float = 0.2


class EpisodicStore:
    """Tier 2: Value-gated episodic retrieval buffer."""

    def __init__(self, config: EpisodicStoreConfig | None = None):
        self.config = config or EpisodicStoreConfig()
        self.records: list[EpisodicRecord] = []

    def clear(self) -> None:
        self.records.clear()

    def add(self, record: EpisodicRecord) -> None:
        if len(self.records) >= self.config.capacity:
            # Evict lowest value or oldest
            self.records.pop(0)
        self.records.append(record)

    def retrieve(
        self,
        query_obs: Sequence[float],
        current_step: int,
        top_k: int = 3,
        mode: str = "intact",
        rng: random.Random | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves top-k episodic matches weighted by value, surprise, and similarity."""
        if mode in ("erased", "memory_free") or not self.records:
            return []

        if mode == "random_retrieval":
            chosen = (rng or random).sample(self.records, min(top_k, len(self.records)))
            return [
                {
                    "step": r.step,
                    "action": r.action,
                    "reward": r.reward,
                    "surprise": r.surprise,
                    "similarity": 0.0,
                    "priority": 1.0,
                    "context": list(r.context),
                }
                for r in chosen
            ]

        q = list(query_obs)
        if mode == "shifted":
            # Circular shift by 4 elements: preserves norm/variance, breaks semantics
            q = q[-4:] + q[:-4]

        q_t = torch.tensor(q, dtype=torch.float32)
        q_norm = torch.norm(q_t).clamp_min(1e-6)

        results = []
        for r in self.records:
            r_t = torch.tensor(r.observation, dtype=torch.float32)
            r_norm = torch.norm(r_t).clamp_min(1e-6)
            sim = float(torch.dot(q_t, r_t) / (q_norm * r_norm))

            # Value & surprise priority weighting
            dt = max(0, current_step - r.step)
            recency = math.pow(self.config.recency_gamma, dt)
            priority = (
                self.config.alpha_reward * abs(r.reward)
                + self.config.alpha_surprise * r.surprise
                + self.config.alpha_recency * recency
                + 1e-4
            )
            score = sim * priority
            results.append((score, sim, priority, r))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        return [
            {
                "step": r.step,
                "action": r.action,
                "reward": r.reward,
                "surprise": r.surprise,
                "similarity": float(sim),
                "priority": float(priority),
                "score": float(score),
                "context": list(r.context),
            }
            for score, sim, priority, r in top
        ]


@dataclass
class PatchState:
    patch_id: str
    terrain: float
    shelter: float
    visit_count: int = 0
    revisit_count: int = 0
    last_seen_tick: int = 0
    observed_food: float = 0.0
    observed_water: float = 0.0
    food_cap: float = 0.0
    water_cap: float = 0.0
    food_rate_mean: float = 0.05
    food_rate_var: float = 0.01
    water_rate_mean: float = 0.05
    water_rate_var: float = 0.01

    def update_revisit(self, current_tick: int, food: float, water: float) -> None:
        dt = current_tick - self.last_seen_tick
        if dt > 0 and self.visit_count > 0:
            self.revisit_count += 1
            sample_food_rate = max(0.0, (food - self.observed_food) / dt)
            sample_water_rate = max(0.0, (water - self.observed_water) / dt)

            n = self.revisit_count
            delta_f = sample_food_rate - self.food_rate_mean
            self.food_rate_mean += delta_f / (n + 1)
            self.food_rate_var = ((n - 1) * self.food_rate_var + delta_f * (sample_food_rate - self.food_rate_mean)) / max(1, n)

            delta_w = sample_water_rate - self.water_rate_mean
            self.water_rate_mean += delta_w / (n + 1)
            self.water_rate_var = ((n - 1) * self.water_rate_var + delta_w * (sample_water_rate - self.water_rate_mean)) / max(1, n)

        self.last_seen_tick = current_tick
        self.visit_count += 1
        self.observed_food = food
        self.observed_water = water
        self.food_cap = max(self.food_cap, food)
        self.water_cap = max(self.water_cap, water)

    def food_lcb(self, current_tick: int, z: float = 1.5) -> float:
        dt = max(0, current_tick - self.last_seen_tick)
        std = math.sqrt(max(1e-8, self.food_rate_var))
        rate = max(0.0, self.food_rate_mean - z * std)
        return min(self.food_cap, self.observed_food + dt * rate)

    def water_lcb(self, current_tick: int, z: float = 1.5) -> float:
        dt = max(0, current_tick - self.last_seen_tick)
        std = math.sqrt(max(1e-8, self.water_rate_var))
        rate = max(0.0, self.water_rate_mean - z * std)
        return min(self.water_cap, self.observed_water + dt * rate)


class SpatialCognitiveMap:
    """Tier 3: Topological spatial cognitive map tracking resource replenishment cycles."""

    def __init__(self, lcb_z: float = 1.5):
        self.lcb_z = lcb_z
        self.patches: dict[str, PatchState] = {}
        self.transitions: dict[str, dict[int, dict[str, int]]] = {}
        self.action_counts: dict[str, dict[int, int]] = {}
        self.current_patch: str | None = None
        self.prev_patch: str | None = None
        self.prev_action: int | None = None

    def clear(self) -> None:
        self.patches.clear()
        self.transitions.clear()
        self.action_counts.clear()
        self.current_patch = None
        self.prev_patch = None
        self.prev_action = None

    def register(self, observation: Sequence[float], current_tick: int) -> PatchState:
        pid = _fingerprint(observation)
        if pid not in self.patches:
            self.patches[pid] = PatchState(
                patch_id=pid,
                terrain=float(observation[12]),
                shelter=float(observation[8]),
                visit_count=0,
                last_seen_tick=current_tick,
                observed_food=float(observation[6]),
                observed_water=float(observation[7]),
                food_cap=float(observation[6]),
                water_cap=float(observation[7]),
            )
        patch = self.patches[pid]
        patch.update_revisit(current_tick, float(observation[6]), float(observation[7]))

        # Record movement transition
        if self.prev_patch is not None and self.prev_action is not None:
            u, a, v = self.prev_patch, self.prev_action, pid
            self.transitions.setdefault(u, {}).setdefault(a, {})
            self.transitions[u][a][v] = self.transitions[u][a].get(v, 0) + 1
            self.action_counts.setdefault(u, {})
            self.action_counts[u][a] = self.action_counts[u].get(a, 0) + 1

        self.prev_patch = pid
        self.current_patch = pid
        return patch

    def transition_distribution(self, patch_id: str, action: int) -> dict[str, float]:
        attempts = self.action_counts.get(patch_id, {}).get(action, 0)
        if attempts == 0:
            return {}
        counts = self.transitions.get(patch_id, {}).get(action, {})
        return {dest: c / attempts for dest, c in counts.items()}

    def frontier_patches(self) -> list[str]:
        """Identifies discovered patches that have unattempted exploratory actions."""
        frontiers = []
        for pid in self.patches:
            counts = self.action_counts.get(pid, {})
            # Actions 1, 2, 4 (forage, drink, explore)
            unattempted = [a for a in (1, 2, 4) if counts.get(a, 0) == 0]
            if unattempted:
                frontiers.append(pid)
        return frontiers

    def topological_coverage(self, total_expected_patches: int = 36) -> float:
        """Computes ratio of mapped spatial patches to total world grid."""
        return min(1.0, len(self.patches) / max(1, total_expected_patches))

    def best_replenished_patch(self, current_tick: int, need: str = "food") -> str | None:
        """Finds patch with highest Lower Confidence Bound resource replenishment."""
        if not self.patches:
            return None
        if need == "food":
            return max(self.patches.keys(), key=lambda pid: self.patches[pid].food_lcb(current_tick, self.lcb_z))
        else:
            return max(self.patches.keys(), key=lambda pid: self.patches[pid].water_lcb(current_tick, self.lcb_z))


@dataclass
class MnemosyneConfig:
    hidden_size: int = 64
    episodic_capacity: int = 256
    retrieval_k: int = 3
    lcb_z: float = 1.5
    mode: str = "intact"
    seed: int = 42

    def __post_init__(self):
        if self.mode not in MNEMOSYNE_MODES:
            raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {self.mode}")


class MnemosyneMemory(nn.Module):
    """Integrated 3-Tier Cognitive Memory System with Matched Lesion Assays."""

    def __init__(self, config: MnemosyneConfig | None = None):
        super().__init__()
        self.config = config or MnemosyneConfig()
        self.rng = random.Random(self.config.seed)

        # Tier 1
        self.working = WorkingMemory(
            obs_size=OBS_SIZE,
            action_count=N_ACTIONS,
            hidden_size=self.config.hidden_size,
        )
        self.register_buffer("h_t", self.working.init_state(1))

        # Tier 2
        self.episodic = EpisodicStore(
            EpisodicStoreConfig(capacity=self.config.episodic_capacity)
        )

        # Tier 3
        self.spatial = SpatialCognitiveMap(lcb_z=self.config.lcb_z)

        # State tracking
        self.current_tick: int = 0
        self.last_action: int = 0
        self.last_surprise: float = 0.0
        self.last_prediction: list[float] | None = None

    def reset(self, mode: str | None = None) -> None:
        """Resets memory state and optionally adjusts assay mode."""
        if mode is not None:
            if mode not in MNEMOSYNE_MODES:
                raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {mode}")
            self.config.mode = mode

        self.h_t.zero_()
        self.episodic.clear()
        self.spatial.clear()
        self.current_tick = 0
        self.last_action = 0
        self.last_surprise = 0.0
        self.last_prediction = None

    def step_observation(
        self,
        observation: Sequence[float],
        predicted_observation: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Integrates new observation, updates working memory, queries episodic store."""
        obs_tensor = torch.tensor(list(observation), dtype=torch.float32).unsqueeze(0)
        mode = self.config.mode

        # Calculate surprise e_{t-1} = || o_t - \hat{o}_t ||_2
        if predicted_observation is not None:
            pred_tensor = torch.tensor(list(predicted_observation), dtype=torch.float32).unsqueeze(0)
            surprise_val = float(torch.norm(obs_tensor - pred_tensor, p=2).item())
        else:
            surprise_val = 0.0
        self.last_surprise = surprise_val

        # Tier 1 update: Working memory
        if mode in ("erased", "memory_free"):
            self.h_t.zero_()
        else:
            prev_act_tensor = torch.tensor([self.last_action], dtype=torch.long)
            surprise_tensor = torch.tensor([surprise_val], dtype=torch.float32)
            with torch.no_grad():
                self.h_t = self.working(obs_tensor, prev_act_tensor, surprise_tensor, self.h_t)

        # Tier 2 retrieval: Value-gated episodic retrieval
        retrieved = self.episodic.retrieve(
            query_obs=observation,
            current_step=self.current_tick,
            top_k=self.config.retrieval_k,
            mode=mode,
            rng=self.rng,
        )

        # Tier 3 spatial registration
        if mode == "memory_free":
            current_patch = None
            f_lcb = float(observation[6])
            w_lcb = float(observation[7])
        else:
            patch = self.spatial.register(observation, self.current_tick)
            current_patch = patch.patch_id
            f_lcb = patch.food_lcb(self.current_tick, z=self.config.lcb_z)
            w_lcb = patch.water_lcb(self.current_tick, z=self.config.lcb_z)

        # Aggregate context representation
        wm_vector = self.h_t.squeeze(0).tolist()
        return {
            "tick": self.current_tick,
            "mode": mode,
            "surprise": surprise_val,
            "working_memory": wm_vector,
            "retrieved_episodes": retrieved,
            "patch_id": current_patch,
            "estimated_food_lcb": f_lcb,
            "estimated_water_lcb": w_lcb,
        }

    def record_transition(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        surprise: float | None = None,
    ) -> None:
        """Stores transition in episodic memory and updates tracking."""
        mode = self.config.mode
        s_val = self.last_surprise if surprise is None else float(surprise)

        if mode not in ("erased", "memory_free"):
            rec = EpisodicRecord(
                step=self.current_tick,
                observation=list(observation),
                action=int(action),
                reward=float(reward),
                next_observation=list(next_observation),
                surprise=s_val,
                context=self.h_t.squeeze(0).tolist(),
            )
            self.episodic.add(rec)

        self.last_action = int(action)
        self.spatial.prev_action = int(action)
        self.current_tick += 1
