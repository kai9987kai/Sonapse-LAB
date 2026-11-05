"""POSEIDON DECISION-VALUE-GATED EPISODIC MEMORY (DV-GEM).

Flagship Causal Memory Architecture for Partially Observable & Aliased Regimes:
1. Compact Recurrent Belief Encoder (Working Context):
   Integrates history through o_t, completed action a_{t-1}, and realized residuals,
   producing a belief embedding h_t without leaking future returns or hidden simulator state.

2. Value-Gated Episodic Retrieval:
   Stores historical transitions e_i = (o_i, a_i, r_i, o'_i, delta_i).
   Query is formed from (o_t, a_cand, h_t).
   Evaluates whether retrieved past interventions provide positive causal decision value:
     \\Delta Q(s_t, a, m) = Q(s_t, a, m) - Q(s_t, a, \\emptyset)

3. Support & Conflict Screening:
   - Out-of-distribution or stale candidates are filtered (supported = False).
   - Conflicting evidence (retrieved records with diverging outcomes) raises conflict score.

4. Explicit Admission Gate:
   - Default influence is strictly 0.0 (baseline fallback).
   - Memory influence is admitted if and only if:
     supported and (conflict <= threshold) and (\\Delta Q - lambda * sigma_Q > admission_threshold)
   - When admitted: delta_m(a) = clip(alpha * \\Delta Q(a), -0.5, 0.5).

5. Six Rigorous Matched Lesion Arms:
   - intact: full learned retrieval and admission gating.
   - similarity: standard Euclidean observation similarity (no decision-value gating).
   - erased: empty episodic store / zero admission.
   - shifted: circular permutation of memory keys (breaks semantic alignment).
   - donor: populated from unrelated donor episodes.
   - baseline: pure frozen Tidal baseline (memory bypassed).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..world import ACTIONS, N_ACTIONS, OBS_SIZE

DECISION_MEMORY_MODES = ("intact", "similarity", "erased", "shifted", "donor", "baseline")
DECISION_MEMORY_SCHEMA = "poseidon-decision-value-memory-v1"
BELIEF_ENCODER_TYPES = ("gru-v1", "structured-cue-v1")
SCORER_TARGET_TYPES = ("absolute-q-v1", "counterfactual-advantage-v1")
STRUCTURED_CUE_BELIEF_FEATURES = (
    "health", "energy", "hydration", "stamina", "exposure", "threat",
    "food_visibility", "water_visibility", "shelter", "weather_severity",
    "temperature", "daylight", "terrain_difficulty", "resource_scent",
    "last_action_scaled", "progress",
    "previous_action_rest", "previous_action_forage", "previous_action_drink",
    "previous_action_shelter", "previous_action_explore", "previous_action_flee",
    "previous_reward", "latched_regime_cue_signed",
)


@dataclass(frozen=True)
class EpisodicEntry:
    tick: int
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    residual: tuple[float, ...]
    age: int = 0
    tag: str = "default"

    def validate(self) -> None:
        if type(self.tick) is not int or self.tick < 0:
            raise ValueError("tick must be a nonnegative integer")
        if type(self.action) is not int or not 0 <= self.action < N_ACTIONS:
            raise ValueError("action must be in [0, 5]")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        for name, vec in (("observation", self.observation), ("next_observation", self.next_observation)):
            if len(vec) != OBS_SIZE or not all(math.isfinite(x) for x in vec):
                raise ValueError(f"{name} must contain {OBS_SIZE} finite values")
        if len(self.residual) != OBS_SIZE or not all(math.isfinite(x) for x in vec):
            raise ValueError(f"residual must contain {OBS_SIZE} finite values")


@dataclass(frozen=True)
class DecisionValueMemoryConfig:
    capacity: int = 256
    top_k: int = 4
    admission_threshold: float = 0.05
    max_candidate_distance: float = 0.90
    conflict_threshold: float = 0.25
    uncertainty_discount: float = 0.50
    shrinkage_alpha: float = 0.50
    mode: str = "intact"
    belief_dim: int = 32
    hidden_dim: int = 64
    belief_encoder_type: str = "gru-v1"
    scorer_target_type: str = "absolute-q-v1"
    horizons: tuple[int, ...] = (4, 8)

    def __post_init__(self):
        if self.mode not in DECISION_MEMORY_MODES:
            raise ValueError(f"mode must be one of {DECISION_MEMORY_MODES}, got {self.mode}")
        if not 1 <= self.capacity <= 10000:
            raise ValueError("capacity must be in [1, 10000]")
        if not 1 <= self.top_k <= self.capacity:
            raise ValueError("top_k must be in [1, capacity]")
        if self.belief_encoder_type not in BELIEF_ENCODER_TYPES:
            raise ValueError(f"belief_encoder_type must be one of {BELIEF_ENCODER_TYPES}")
        if self.belief_encoder_type == "structured-cue-v1" and self.belief_dim != len(STRUCTURED_CUE_BELIEF_FEATURES):
            raise ValueError(
                f"structured-cue-v1 requires belief_dim={len(STRUCTURED_CUE_BELIEF_FEATURES)}"
            )
        if self.scorer_target_type not in SCORER_TARGET_TYPES:
            raise ValueError(f"scorer_target_type must be one of {SCORER_TARGET_TYPES}")
        if not math.isfinite(self.admission_threshold) or self.admission_threshold < 0.0:
            raise ValueError("admission_threshold must be nonnegative finite")
        if not 0.0 < self.shrinkage_alpha <= 1.0:
            raise ValueError("shrinkage_alpha must be in (0, 1]")


@dataclass
class MemoryDecision:
    action_values: list[float]
    decision_deltas: list[float]
    gated_deltas: list[float]
    admitted_actions: list[bool]
    admission_gate_open: bool
    retrieved_count: int
    retrieved_candidates: list[dict[str, Any]]
    conflict_score: float
    support_score: float
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_values": [round(x, 6) for x in self.action_values],
            "decision_deltas": [round(x, 6) for x in self.decision_deltas],
            "gated_deltas": [round(x, 6) for x in self.gated_deltas],
            "admitted_actions": list(self.admitted_actions),
            "admission_gate_open": self.admission_gate_open,
            "retrieved_count": self.retrieved_count,
            "conflict_score": round(self.conflict_score, 6),
            "support_score": round(self.support_score, 6),
            "mode": self.mode,
            "retrieved_candidates": self.retrieved_candidates,
        }


class RecurrentBeliefEncoder(nn.Module):
    """Compact GRU belief cell aggregating past observations and actions."""

    def __init__(self, obs_dim: int = OBS_SIZE, action_count: int = N_ACTIONS, belief_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_count = action_count
        self.belief_dim = belief_dim
        # Input: observation (16) + action one-hot (6) + scalar reward (1)
        self.input_dim = obs_dim + action_count + 1
        self.cell = nn.GRUCell(self.input_dim, belief_dim)

    def forward(self, obs: torch.Tensor, prev_action: int, prev_reward: float, hidden: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        act_one_hot = F.one_hot(torch.tensor([prev_action]), num_classes=self.action_count).float()
        r_tensor = torch.tensor([[prev_reward]], dtype=torch.float32)
        inp = torch.cat([obs, act_one_hot, r_tensor], dim=-1)
        return self.cell(inp, hidden)


class StructuredCueBeliefEncoder(nn.Module):
    """Parameter-free, inspectable state: current input, prior action/reward, and a cue latch.

    The signed latch is +1 for the early high-scent cue, -1 for the low-scent
    cue, and 0 before either cue is observed. During the aliased window, noisy
    neutral readings in [0.35, 0.65] preserve the last cue instead of changing it.
    This encoder is specific to the documented 16-channel TidePool contract.
    """

    def __init__(self, obs_dim: int = OBS_SIZE, action_count: int = N_ACTIONS):
        super().__init__()
        if obs_dim != OBS_SIZE or action_count != N_ACTIONS:
            raise ValueError("structured-cue-v1 is bound to the TidePool observation/action schema")
        self.obs_dim = obs_dim
        self.action_count = action_count
        self.belief_dim = len(STRUCTURED_CUE_BELIEF_FEATURES)

    def forward(self, obs: torch.Tensor, prev_action: int, prev_reward: float, hidden: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim or hidden.shape != (obs.shape[0], self.belief_dim):
            raise ValueError("structured-cue-v1 received an incompatible observation or hidden state")
        if not 0 <= int(prev_action) < self.action_count or not math.isfinite(float(prev_reward)):
            raise ValueError("previous action/reward is outside the structured belief contract")
        action = torch.full((obs.shape[0],), int(prev_action), dtype=torch.long, device=obs.device)
        action_one_hot = F.one_hot(action, num_classes=self.action_count).to(dtype=obs.dtype)
        cue_channel = obs[:, 13:14]
        prior_latch = hidden[:, -1:]
        high_cue = torch.where(cue_channel >= 0.65, torch.ones_like(cue_channel), prior_latch)
        cue_latch = torch.where(cue_channel <= 0.35, -torch.ones_like(cue_channel), high_cue)
        reward = torch.full((obs.shape[0], 1), float(prev_reward), dtype=obs.dtype, device=obs.device)
        return torch.cat([obs, action_one_hot, reward, cue_latch], dim=-1)


class DecisionValueScorer(nn.Module):
    """Neural Scorer predicting Q(s, a, m) vs Q(s, a, empty)."""

    def __init__(self, obs_dim: int = OBS_SIZE, belief_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        
        # Query: obs (16) + action one-hot (6) + belief (32) = 54
        query_dim = obs_dim + N_ACTIONS + belief_dim
        # Memory entry: obs (16) + action one-hot (6) + reward (1) + residual (16) = 39
        memory_dim = obs_dim + N_ACTIONS + 1 + obs_dim

        # Predictor with 2 ensemble heads for epistemic uncertainty estimation
        self.head1 = nn.Sequential(
            nn.Linear(query_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(query_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([query, memory], dim=-1)
        out1 = self.head1(inp)
        out2 = self.head2(inp)
        mean_q = (out1 + out2) / 2.0
        sigma_q = torch.abs(out1 - out2) / 2.0
        return mean_q, sigma_q


class DecisionValueMemory:
    """Decision-Value-Gated Episodic Memory Controller."""

    def __init__(self, config: DecisionValueMemoryConfig | None = None):
        self.config = config or DecisionValueMemoryConfig()
        self.mode = self.config.mode
        self.records: list[EpisodicEntry] = []
        self.donor_records: list[EpisodicEntry] = []

        if self.config.belief_encoder_type == "structured-cue-v1":
            self.belief_encoder = StructuredCueBeliefEncoder(obs_dim=OBS_SIZE, action_count=N_ACTIONS)
        else:
            self.belief_encoder = RecurrentBeliefEncoder(
                obs_dim=OBS_SIZE,
                action_count=N_ACTIONS,
                belief_dim=self.config.belief_dim,
            )
        self.scorer = DecisionValueScorer(
            obs_dim=OBS_SIZE,
            belief_dim=self.config.belief_dim,
            hidden_dim=self.config.hidden_dim,
        )

        self.current_belief = torch.zeros(1, self.config.belief_dim)
        self.last_action: int = 0
        self.last_reward: float = 0.0
        self.step_count: int = 0

    def reset(self, mode: str | None = None) -> None:
        if mode is not None:
            if mode not in DECISION_MEMORY_MODES:
                raise ValueError(f"mode must be one of {DECISION_MEMORY_MODES}, got {mode}")
            self.mode = mode
        self.records.clear()
        self.current_belief = torch.zeros(1, self.config.belief_dim)
        self.last_action = 0
        self.last_reward = 0.0
        self.step_count = 0

    def add_entry(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        residual: Sequence[float] | None = None,
        tag: str = "default",
    ) -> None:
        if len(self.records) >= self.config.capacity:
            self.records.pop(0)

        res = tuple(residual) if residual is not None else tuple([0.0] * OBS_SIZE)
        entry = EpisodicEntry(
            tick=self.step_count,
            observation=tuple(observation),
            action=action,
            reward=reward,
            next_observation=tuple(next_observation),
            residual=res,
            age=0,
            tag=tag,
        )
        entry.validate()
        self.records.append(entry)

    def set_donor_pool(self, entries: Sequence[EpisodicEntry]) -> None:
        """Injects donor transitions for the 'donor' lesion arm."""
        self.donor_records = list(entries)

    def update_belief(self, observation: Sequence[float], prev_action: int, prev_reward: float) -> torch.Tensor:
        obs_t = torch.tensor(list(observation), dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            self.current_belief = self.belief_encoder(obs_t, prev_action, prev_reward, self.current_belief)
        self.last_action = prev_action
        self.last_reward = prev_reward
        self.step_count += 1
        return self.current_belief

    def _get_active_records(self) -> list[EpisodicEntry]:
        if self.mode in ("erased", "baseline"):
            return []
        if self.mode == "donor":
            return self.donor_records
        if self.mode == "shifted":
            if not self.records:
                return []
            # Circular permutation of records: shift by 1
            return self.records[1:] + self.records[:1]
        return self.records

    def query(self, current_obs: Sequence[float]) -> MemoryDecision:
        """Evaluates causal decision value across all 6 candidate actions with admission gating."""
        active_records = self._get_active_records()
        obs_t = torch.tensor(list(current_obs), dtype=torch.float32).reshape(1, -1)

        action_values = [0.0] * N_ACTIONS
        decision_deltas = [0.0] * N_ACTIONS
        gated_deltas = [0.0] * N_ACTIONS
        admitted_actions = [False] * N_ACTIONS
        all_retrieved: list[dict[str, Any]] = []

        if not active_records or self.mode == "baseline":
            return MemoryDecision(
                action_values=action_values,
                decision_deltas=decision_deltas,
                gated_deltas=gated_deltas,
                admitted_actions=admitted_actions,
                admission_gate_open=False,
                retrieved_count=0,
                retrieved_candidates=[],
                conflict_score=0.0,
                support_score=0.0,
                mode=self.mode,
            )

        # 1. Candidate Pool Selection
        obs_np = list(current_obs)
        scored_records = []
        for r in active_records:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_np, r.observation))) / math.sqrt(OBS_SIZE)
            scored_records.append((dist, r))

        scored_records.sort(key=lambda x: x[0])

        if self.mode == "similarity":
            # Similarity mode strictly selects nearest neighbors by observation distance
            top_candidates = scored_records[: self.config.top_k]
            candidate_pool = [r for _, r in top_candidates]
        else:
            # Intact mode pools nearest neighbors plus distal anchor/cue transitions
            earliest_cues = active_records[:2]
            nearest_records = [r for _, r in scored_records[:2]]
            seen_ids = set()
            candidate_pool = []
            for r in earliest_cues + nearest_records:
                if id(r) not in seen_ids:
                    seen_ids.add(id(r))
                    candidate_pool.append(r)
            top_candidates = [(math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_np, r.observation))) / math.sqrt(OBS_SIZE), r)
                              for r in candidate_pool]

        # Check candidate support
        mean_dist = sum(d for d, _ in top_candidates) / max(1, len(top_candidates))
        support_score = max(0.0, 1.0 - mean_dist / self.config.max_candidate_distance)
        is_supported = mean_dist <= self.config.max_candidate_distance

        # Candidate Conflict Check: check variance of rewards for identical actions
        rewards_by_action: dict[int, list[float]] = {}
        for _, r in top_candidates:
            rewards_by_action.setdefault(r.action, []).append(r.reward)
        action_variances = [
            (sum((x - sum(vals)/len(vals))**2 for x in vals) / len(vals))
            for vals in rewards_by_action.values() if len(vals) > 1
        ]
        conflict_score = max(action_variances) if action_variances else 0.0
        has_conflict = conflict_score > self.config.conflict_threshold

        for dist, r in top_candidates:
            all_retrieved.append({
                "tick": r.tick,
                "action": r.action,
                "reward": round(r.reward, 4),
                "distance": round(dist, 4),
                "tag": r.tag,
            })

        # 2. Decision Value Evaluation across candidate actions
        gate_open_any = False
        with torch.no_grad():
            for cand_act in range(N_ACTIONS):
                act_one_hot = F.one_hot(torch.tensor([cand_act]), num_classes=N_ACTIONS).float()
                query_vec = torch.cat([obs_t, act_one_hot, self.current_belief], dim=-1)

                empty_mem_vec = torch.zeros(1, OBS_SIZE + N_ACTIONS + 1 + OBS_SIZE)
                if self.config.scorer_target_type == "counterfactual-advantage-v1":
                    q_empty = torch.zeros(1, 1, dtype=query_vec.dtype)
                    sigma_empty = torch.zeros_like(q_empty)
                else:
                    q_empty, sigma_empty = self.scorer(query_vec, empty_mem_vec)

                if self.mode == "similarity":
                    best_match = candidate_pool[0]
                    mem_act_oh = F.one_hot(torch.tensor([best_match.action]), num_classes=N_ACTIONS).float()
                    mem_vec = torch.cat([
                        torch.tensor(best_match.observation, dtype=torch.float32).reshape(1, -1),
                        mem_act_oh,
                        torch.tensor([[best_match.reward]], dtype=torch.float32),
                        torch.tensor(best_match.residual, dtype=torch.float32).reshape(1, -1),
                    ], dim=-1)
                    q_mem, _ = self.scorer(query_vec, mem_vec)
                    delta_q = float((q_mem - q_empty).item())
                    action_values[cand_act] = float(q_mem.item())
                    decision_deltas[cand_act] = delta_q
                    gated_deltas[cand_act] = max(-0.5, min(0.5, delta_q * self.config.shrinkage_alpha))
                    admitted_actions[cand_act] = True
                    gate_open_any = True
                else:
                    # Intact mode: find the candidate maximizing decision value LCB
                    best_cand_lcb = -float("inf")
                    best_delta_q = 0.0
                    best_q_mem = float(q_empty.item())

                    for cand_r in candidate_pool:
                        mem_act_oh = F.one_hot(torch.tensor([cand_r.action]), num_classes=N_ACTIONS).float()
                        mem_vec = torch.cat([
                            torch.tensor(cand_r.observation, dtype=torch.float32).reshape(1, -1),
                            mem_act_oh,
                            torch.tensor([[cand_r.reward]], dtype=torch.float32),
                            torch.tensor(cand_r.residual, dtype=torch.float32).reshape(1, -1),
                        ], dim=-1)

                        q_mem_cand, sigma_mem_cand = self.scorer(query_vec, mem_vec)
                        dq = float((q_mem_cand - q_empty).item())
                        s_net = float((sigma_mem_cand + sigma_empty).item())
                        cand_lcb = dq - self.config.uncertainty_discount * s_net
                        if cand_lcb > best_cand_lcb:
                            best_cand_lcb = cand_lcb
                            best_delta_q = dq
                            best_q_mem = float(q_mem_cand.item())

                    action_values[cand_act] = best_q_mem
                    decision_deltas[cand_act] = best_delta_q

                    admit = (
                        is_supported
                        and not has_conflict
                        and (best_cand_lcb > self.config.admission_threshold)
                    )

                    admitted_actions[cand_act] = admit
                    if admit:
                        gated_delta = max(-0.5, min(0.5, best_delta_q * self.config.shrinkage_alpha))
                        gated_deltas[cand_act] = gated_delta
                        gate_open_any = True
                    else:
                        gated_deltas[cand_act] = 0.0

        return MemoryDecision(
            action_values=action_values,
            decision_deltas=decision_deltas,
            gated_deltas=gated_deltas,
            admitted_actions=admitted_actions,
            admission_gate_open=gate_open_any,
            retrieved_count=len(top_candidates),
            retrieved_candidates=all_retrieved,
            conflict_score=conflict_score,
            support_score=support_score,
            mode=self.mode,
        )

    def get_influence_deltas(self, current_obs: Sequence[float]) -> list[float]:
        """Convenience method returning the admitted influence deltas across actions."""
        return list(self.query(current_obs).gated_deltas)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_MEMORY_SCHEMA,
            "config": asdict(self.config),
            "belief_encoder": self.belief_encoder.state_dict(),
            "scorer": self.scorer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != DECISION_MEMORY_SCHEMA:
            raise ValueError(f"Schema mismatch: expected {DECISION_MEMORY_SCHEMA}, got {state.get('schema')}")
        self.belief_encoder.load_state_dict(state["belief_encoder"])
        self.scorer.load_state_dict(state["scorer"])


class DecisionValueAgent:
    """Rigorous decision-value memory coordinator wrapping the frozen Tidal core."""

    def __init__(
        self,
        core: Any | None = None,
        memory: DecisionValueMemory | None = None,
        mode: str = "intact",
        override_margin: float = 0.04,
        influence_scale: float = 1.0,
    ):
        from ..core import CoreRuntime, active_core_path
        self.core = core or CoreRuntime(active_core_path())
        self.memory = memory or DecisionValueMemory(DecisionValueMemoryConfig(mode=mode))
        self.mode = mode
        if not math.isfinite(override_margin) or override_margin < 0.0:
            raise ValueError("override_margin must be a nonnegative finite value")
        if not math.isfinite(influence_scale) or influence_scale < 0.0:
            raise ValueError("influence_scale must be a nonnegative finite value")
        self.override_margin = override_margin
        self.influence_scale = influence_scale
        self.memory.mode = self.mode

    def reset(self, mode: str | None = None) -> None:
        if mode is not None:
            self.mode = mode
        self.memory.reset(mode=self.mode)

    def act(self, observation: Sequence[float]) -> dict[str, Any]:
        obs_list = list(observation)
        base_action = self.core.act(obs_list)

        # In baseline mode, completely bypass memory
        if self.mode == "baseline":
            return {
                "action": base_action,
                "base_action": base_action,
                "override": False,
                "memory_override": False,
                "margin": 0.0,
                "influence_scale": self.influence_scale,
                "admission_gate_open": False,
                "mode": self.mode,
                "decision_memory": None,
            }

        mem_decision = self.memory.query(obs_list)
        if not mem_decision.admission_gate_open:
            return {
                "action": base_action,
                "base_action": base_action,
                "override": False,
                "memory_override": False,
                "margin": 0.0,
                "influence_scale": self.influence_scale,
                "admission_gate_open": False,
                "mode": self.mode,
                "decision_memory": mem_decision.to_dict(),
            }

        # Retrieve core model logits
        obs_t = torch.as_tensor(obs_list, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.core.model([""], obs_t)
            logits = out["action"][0].cpu()

        # Score modulation: normalized policy probabilities + gated_deltas
        base_probs = F.softmax(logits, dim=-1)
        deltas = torch.tensor(mem_decision.gated_deltas, dtype=torch.float32)
        modulated_scores = base_probs + self.influence_scale * deltas

        best_action = int(modulated_scores.argmax().item())
        base_score = float(modulated_scores[base_action].item())
        best_score = float(modulated_scores[best_action].item())
        margin = best_score - base_score

        override = (best_action != base_action) and (margin > self.override_margin)
        chosen_action = best_action if override else base_action

        return {
            "action": chosen_action,
            "base_action": base_action,
            "override": override,
            "memory_override": override,
            "margin": margin,
            "influence_scale": self.influence_scale,
            "admission_gate_open": mem_decision.admission_gate_open,
            "mode": self.mode,
            "decision_memory": mem_decision.to_dict(),
        }

    def observe(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        residual: Sequence[float] | None = None,
        tag: str = "default",
    ) -> None:
        self.memory.add_entry(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            residual=residual,
            tag=tag,
        )
        self.memory.update_belief(next_observation, action, reward)

    def rollout(
        self,
        env: Any,
        max_steps: int = 64,
        mode: str | None = None,
    ) -> dict[str, Any]:
        self.reset(mode=mode)
        obs = env.reset()
        total_reward = 0.0
        trajectory = []
        gate_open_count = 0
        override_count = 0
        info: dict[str, Any] = {}

        for step in range(max_steps):
            dec = self.act(obs)
            act_idx = dec["action"]
            if dec["admission_gate_open"]:
                gate_open_count += 1
            if dec["override"]:
                override_count += 1

            next_obs, reward, done, info = env.step(act_idx)
            self.observe(obs, act_idx, reward, next_obs)
            total_reward += reward

            trajectory.append({
                "step": step,
                "observation": obs,
                "action": act_idx,
                "reward": reward,
                "next_observation": next_obs,
                "done": done,
                "override": dec["override"],
                "admission_gate_open": dec["admission_gate_open"],
                "info": info,
            })
            obs = next_obs
            if done:
                break

        return {
            "total_reward": total_reward,
            "steps": len(trajectory),
            "survived": not info.get("death", False),
            "gate_open_count": gate_open_count,
            "override_count": override_count,
            "mode": self.mode,
            "trajectory": trajectory,
        }


class ResourceGuardedDecisionValueAgent(DecisionValueAgent):
    """Opt-in hybrid with an optional conservative sensor-confidence reserve."""

    def __init__(
        self,
        *args: Any,
        resource_guard_threshold: float = 0.40,
        sensor_noise_std: float = 0.0,
        confidence_z: float = 0.0,
        predictive_resource_guard: bool = False,
        resource_guard_floor: float = 0.15,
        maximum_resource_drain: float = 0.10,
        minimum_resource_yield: float = 0.12,
        resource_balance_margin: float | None = None,
        resource_balance_activation: float = 0.35,
        **kwargs: Any,
    ):
        if not math.isfinite(resource_guard_threshold) or not 0.0 <= resource_guard_threshold <= 1.0:
            raise ValueError("resource_guard_threshold must be finite and in [0, 1]")
        if not math.isfinite(sensor_noise_std) or not 0.0 <= sensor_noise_std <= 0.5:
            raise ValueError("sensor_noise_std must be finite and in [0, 0.5]")
        if not math.isfinite(confidence_z) or confidence_z < 0.0:
            raise ValueError("confidence_z must be a nonnegative finite value")
        if not isinstance(predictive_resource_guard, bool):
            raise ValueError("predictive_resource_guard must be Boolean")
        if not math.isfinite(resource_guard_floor) or not 0.0 <= resource_guard_floor <= 1.0:
            raise ValueError("resource_guard_floor must be finite and in [0, 1]")
        if not math.isfinite(maximum_resource_drain) or not 0.0 <= maximum_resource_drain <= 1.0:
            raise ValueError("maximum_resource_drain must be finite and in [0, 1]")
        if not math.isfinite(minimum_resource_yield) or not 0.0 <= minimum_resource_yield <= 1.0:
            raise ValueError("minimum_resource_yield must be finite and in [0, 1]")
        if resource_balance_margin is not None and (
            not math.isfinite(resource_balance_margin) or not 0.0 <= resource_balance_margin <= 1.0
        ):
            raise ValueError("resource_balance_margin must be None or finite in [0, 1]")
        if not math.isfinite(resource_balance_activation) or not 0.0 <= resource_balance_activation <= 1.0:
            raise ValueError("resource_balance_activation must be finite and in [0, 1]")
        super().__init__(*args, **kwargs)
        self.resource_guard_threshold = resource_guard_threshold
        self.sensor_noise_std = sensor_noise_std
        self.confidence_z = confidence_z
        self.predictive_resource_guard = predictive_resource_guard
        self.resource_guard_floor = resource_guard_floor
        self.maximum_resource_drain = maximum_resource_drain
        self.minimum_resource_yield = minimum_resource_yield
        self.resource_balance_margin = resource_balance_margin
        self.resource_balance_activation = resource_balance_activation

    def act(self, observation: Sequence[float]) -> dict[str, Any]:
        decision = super().act(observation)
        candidate_action = int(decision["action"])
        energy, hydration = float(observation[1]), float(observation[2])
        uncertainty = self.confidence_z * self.sensor_noise_std
        energy_lcb = max(0.0, energy - uncertainty)
        hydration_lcb = max(0.0, hydration - uncertainty)
        predicted_minimum: dict[int, float] = {}

        def predicted_resources(action: int) -> tuple[float, float]:
            energy_next = energy_lcb - self.maximum_resource_drain
            hydration_next = hydration_lcb - self.maximum_resource_drain
            if action == 1:
                energy_next += self.minimum_resource_yield
            elif action == 2:
                hydration_next += self.minimum_resource_yield
            return max(0.0, energy_next), max(0.0, hydration_next)

        if self.predictive_resource_guard:
            candidate_next = predicted_resources(candidate_action)
            candidate_minimum = min(candidate_next)
            predicted_minimum = {action: min(predicted_resources(action)) for action in (1, 2)}
            lower_action = 1 if energy_lcb <= hydration_lcb else 2
            balance_guard_active = (
                self.resource_balance_margin is not None
                and min(energy_lcb, hydration_lcb) <= self.resource_balance_activation
            )
            balance_ok = (
                not balance_guard_active
                or abs(energy_lcb - hydration_lcb) <= float(self.resource_balance_margin)
                or candidate_action == lower_action
            )
            candidate_safe = (
                candidate_minimum >= self.resource_guard_floor
                and balance_ok
            )
            guard_triggered = self.mode != "baseline" and not candidate_safe
            if guard_triggered:
                best_value = max(predicted_minimum.values())
                best_actions = [action for action, value in predicted_minimum.items() if value >= best_value - 1e-9]
                if len(best_actions) == 1:
                    guard_action = best_actions[0]
                else:
                    guard_action = lower_action
            else:
                guard_action = candidate_action
            final_action = guard_action if guard_triggered else candidate_action
        else:
            candidate_minimum = float("nan")
            candidate_safe = False
            guard_triggered = (
                self.mode != "baseline"
                and min(energy_lcb, hydration_lcb) <= self.resource_guard_threshold
            )
            guard_action = 1 if energy_lcb <= hydration_lcb else 2
            final_action = guard_action if guard_triggered else candidate_action
        decision["memory_candidate_action"] = candidate_action
        decision["action"] = final_action
        decision["resource_guard_threshold"] = self.resource_guard_threshold
        decision["resource_guard_sensor_noise_std"] = self.sensor_noise_std
        decision["resource_guard_confidence_z"] = self.confidence_z
        decision["resource_guard_energy_lcb"] = energy_lcb
        decision["resource_guard_hydration_lcb"] = hydration_lcb
        decision["predictive_resource_guard"] = self.predictive_resource_guard
        decision["resource_guard_floor"] = self.resource_guard_floor
        decision["resource_guard_balance_margin"] = self.resource_balance_margin
        decision["resource_guard_balance_activation"] = self.resource_balance_activation
        decision["resource_guard_candidate_predicted_minimum"] = candidate_minimum
        decision["resource_guard_foraging_predicted_minimum"] = predicted_minimum.get(1)
        decision["resource_guard_drinking_predicted_minimum"] = predicted_minimum.get(2)
        decision["resource_guard_triggered"] = guard_triggered
        decision["resource_guard_override"] = guard_triggered and final_action != candidate_action
        decision["memory_override_executed"] = (
            bool(decision.get("memory_override", False)) and final_action == candidate_action
        )
        decision["override"] = final_action != int(decision["base_action"])
        return decision


def fit_decision_value_memory(
    train_seeds: Sequence[int],
    config: DecisionValueMemoryConfig | None = None,
    updates: int = 400,
    lr: float = 0.003,
    environment_factory: Callable[..., Any] | None = None,
) -> DecisionValueMemory:
    """Trains on cross-fitted branches from v1 or a caller-supplied world factory."""
    from .alias_world import ObservationAliasTidePool
    from ..core import CoreRuntime, active_core_path

    cfg = config or DecisionValueMemoryConfig()
    core = CoreRuntime(active_core_path())
    # Keep the belief encoder fixed and identical across every fitted episode
    # and at inference. The scorer is trained below; the encoder is not.
    fitted_mem = DecisionValueMemory(cfg)

    # Collect branch transition data across training seeds
    queries = []
    memories = []
    returns = []

    for seed in train_seeds:
        if environment_factory is None:
            env = ObservationAliasTidePool(seed=seed, max_steps=24, cue_horizon=4)
        else:
            env = environment_factory(seed=seed, max_steps=24, cue_horizon=4)
        obs = env.reset()
        mem = copy.deepcopy(fitted_mem)
        mem.reset()

        cue_entries: list[EpisodicEntry] = []
        for t in range(24):
            act = core.act(obs)
            next_obs, r, done, info = env.step(act)
            mem.add_entry(obs, act, r, next_obs)
            mem.update_belief(next_obs, act, r)
            if t < 4:
                cue_entries.append(EpisodicEntry(
                    tick=t, observation=tuple(obs), action=act, reward=r,
                    next_observation=tuple(next_obs), residual=tuple([0.0]*OBS_SIZE),
                ))

            # At aliased ticks t in [4, 8, 12], create branch anchors
            if t in (4, 8, 12) and not done and cue_entries:
                best_cue = cue_entries[0]
                snap = env.snapshot()
                branch_examples: list[tuple[torch.Tensor, torch.Tensor, float]] = []
                baseline_action = core.act(obs)

                # Measure actual synthetic branch returns for each action (horizon 4)
                for cand_act in range(N_ACTIONS):
                    env.restore(snap)
                    branch_r = 0.0
                    b_obs, r0, b_done, _ = env.step(cand_act)
                    branch_r += r0
                    curr = b_obs
                    for _ in range(3):
                        if b_done:
                            break
                        b_act = core.act(curr)
                        curr, r_step, b_done, _ = env.step(b_act)
                        branch_r += r_step

                    # Query vector: obs (16) + cand_act (6) + belief (32)
                    act_oh = F.one_hot(torch.tensor([cand_act]), num_classes=N_ACTIONS).float()
                    obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
                    q_vec = torch.cat([obs_t, act_oh, mem.current_belief], dim=-1)

                    # Memory vector: cue obs (16) + cue act (6) + cue reward (1) + residual (16)
                    cue_act_oh = F.one_hot(torch.tensor([best_cue.action]), num_classes=N_ACTIONS).float()
                    m_vec = torch.cat([
                        torch.tensor(best_cue.observation, dtype=torch.float32).reshape(1, -1),
                        cue_act_oh,
                        torch.tensor([[best_cue.reward]], dtype=torch.float32),
                        torch.tensor(best_cue.residual, dtype=torch.float32).reshape(1, -1),
                    ], dim=-1)

                    branch_examples.append((q_vec, m_vec, branch_r))

                env.restore(snap)
                baseline_return = next(value for action, (_, _, value) in enumerate(branch_examples) if action == baseline_action)
                for q_vec, m_vec, branch_return in branch_examples:
                    queries.append(q_vec)
                    memories.append(m_vec)
                    if cfg.scorer_target_type == "counterfactual-advantage-v1":
                        returns.append(branch_return - baseline_return)
                    else:
                        returns.append(branch_return)

            obs = next_obs
            if done:
                break

    if not queries:
        return fitted_mem

    # Train Scorer on collected counterfactual branches
    queries_tensor = torch.cat(queries, dim=0)
    memories_tensor = torch.cat(memories, dim=0)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(fitted_mem.scorer.parameters(), lr=lr)

    n_samples = queries_tensor.shape[0]
    batch_size = min(32, n_samples)

    for step in range(updates):
        idx = torch.randperm(n_samples)[:batch_size]
        q_b = queries_tensor[idx]
        m_b = memories_tensor[idx]
        y_b = returns_tensor[idx]

        if cfg.scorer_target_type == "counterfactual-advantage-v1":
            # The target is already paired against the frozen-core action.
            # Randomly removing the memory here would erase the contrast signal.
            m_input = m_b
        else:
            # Legacy objective: identical absolute-return targets with/without memory.
            drop_mask = (torch.rand(batch_size, 1) < 0.5).float()
            m_input = m_b * drop_mask

        optimizer.zero_grad()
        out1, out2 = fitted_mem.scorer.head1(torch.cat([q_b, m_input], dim=-1)), fitted_mem.scorer.head2(torch.cat([q_b, m_input], dim=-1))
        loss = F.mse_loss(out1, y_b) + F.mse_loss(out2, y_b)
        loss.backward()
        optimizer.step()

    return fitted_mem
