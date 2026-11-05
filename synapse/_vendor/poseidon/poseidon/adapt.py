"""POSEIDON ADAPT: Continual Policy Adaptation with Anti-Forgetting Anchor Gates.

Key Capabilities:
1. Replay Buffer with Prioritized Online Transitions:
   Collects environment transitions (s_t, a_t, r_t, s_{t+1}) during deployment.

2. Anti-Forgetting Anchor Dataset:
   Preserves canonical reference distributions from the frozen Tidal policy on
   standard deployment states, penalizing KL divergence:
     L_{anchor} = D_{KL}(p_{Tidal}(s) || p_{adapted}(s))

3. Residual Policy Adapter:
   Learns residual logit perturbations \\Delta \\pi_\\phi(s) around the frozen Tidal core:
     \\pi_{adapted}(a | s) = Softmax(Logits_{Tidal}(s) + \\alpha \\Delta \\pi_\\phi(s))
   Guarantees zero-initialization identity to the frozen core.

4. Formal Promotion Gate & Audit Ledger:
   Evaluates candidate adapters against the incumbent on held-out test seeds.
   Promotion requires:
     (1) Candidate return advantage >= \\delta_{min}
     (2) Anchor distribution regression <= 2% (preventing catastrophic forgetting)
   Produces structured audit receipts written to outputs/adaptation/ledger.json.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import datetime
import json
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import CoreRuntime
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool


@dataclass(frozen=True)
class AdaptConfig:
    replay_capacity: int = 500
    anchor_capacity: int = 100
    learning_rate: float = 1e-3
    anchor_weight: float = 1.0
    weight_decay: float = 1e-4
    max_training_steps: int = 40
    batch_size: int = 16
    promotion_min_delta: float = 0.04
    max_anchor_regression: float = 0.02
    eval_seeds: tuple[int, ...] = (101, 103, 107, 109)


class ResidualPolicyAdapter(nn.Module):
    """Residual logit perturbation network preserving the frozen base policy at origin."""

    def __init__(self, obs_size: int = OBS_SIZE, action_count: int = N_ACTIONS, hidden_size: int = 64, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_count),
        )
        # Initialize output layer to zero for strict identity initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        residual = self.net(obs)
        return base_logits + self.alpha * residual


@dataclass
class Transition:
    obs: list[float]
    action: int
    reward: float
    next_obs: list[float]
    priority: float = 1.0


class ReplayBuffer:
    """Transition replay buffer with capacity bounds."""

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.buffer: list[Transition] = []

    def add(self, obs: Sequence[float], action: int, reward: float, next_obs: Sequence[float], priority: float = 1.0) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(
            Transition(
                obs=list(obs),
                action=int(action),
                reward=float(reward),
                next_obs=list(next_obs),
                priority=max(1e-4, float(priority)),
            )
        )

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[Transition]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return (rng or random).sample(self.buffer, k)

    def __len__(self) -> int:
        return len(self.buffer)


class AnchorDataset:
    """Anti-forgetting dataset storing canonical observations and base policy distributions."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items: list[dict[str, Any]] = []

    def populate_from_core(self, core: CoreRuntime, count: int = 50, seed: int = 42) -> None:
        """Populates anchors across diverse seeds."""
        self.items.clear()
        rng = random.Random(seed)
        for i in range(count):
            env_seed = 1000 + i
            pool = TidePool(seed=env_seed, max_steps=10)
            obs = pool.observe()
            obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                out = core.model(["Survive"], obs_t)
                action_logits = out["action"][0]
                action_probs = F.softmax(action_logits, dim=-1)

            self.items.append({
                "obs": list(obs),
                "target_probs": action_probs.tolist(),
                "target_action": int(action_probs.argmax().item()),
            })

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[dict[str, Any]]:
        if not self.items:
            return []
        k = min(batch_size, len(self.items))
        return (rng or random).sample(self.items, k)

    def evaluate_retention(self, adapter: ResidualPolicyAdapter, core: CoreRuntime) -> float:
        """Computes fraction of anchor states where adapted policy agrees with base policy."""
        if not self.items:
            return 1.0
        correct = 0
        for item in self.items:
            obs_t = torch.tensor(item["obs"], dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                base_logits = core.model(["Survive"], obs_t)["action"]
                adapted_logits = adapter(obs_t, base_logits)
                pred_action = int(adapted_logits.argmax(dim=-1).item())
            if pred_action == item["target_action"]:
                correct += 1
        return correct / len(self.items)


class ContinualAdapter:
    """Manages offline policy adaptation and promotion auditing."""

    def __init__(self, core: CoreRuntime, config: AdaptConfig | None = None):
        self.core = core
        self.config = config or AdaptConfig()
        self.adapter = ResidualPolicyAdapter()
        self.replay = ReplayBuffer(capacity=self.config.replay_capacity)
        self.anchors = AnchorDataset(capacity=self.config.anchor_capacity)
        self.anchors.populate_from_core(self.core, count=min(50, self.config.anchor_capacity))

    def train_candidate(self) -> dict[str, float]:
        """Runs offline step-capped optimization on replay + anchor objectives."""
        if len(self.replay) < self.config.batch_size:
            return {"loss": 0.0, "steps": 0}

        optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        steps_run = 0
        total_loss = 0.0

        for step in range(self.config.max_training_steps):
            transitions = self.replay.sample(self.config.batch_size)
            anchor_batch = self.anchors.sample(self.config.batch_size)

            obs_t = torch.tensor([t.obs for t in transitions], dtype=torch.float32)
            act_t = torch.tensor([t.action for t in transitions], dtype=torch.long)
            rew_t = torch.tensor([t.reward for t in transitions], dtype=torch.float32)

            with torch.no_grad():
                base_logits = self.core.model(["Survive"] * len(transitions), obs_t)["action"]

            adapted_logits = self.adapter(obs_t, base_logits)

            # Policy gradient / behavioral cloning surrogate weighted by advantage/reward
            log_probs = F.log_softmax(adapted_logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
            # Maximize reward * log_prob -> minimize -rew * log_prob
            task_loss = -(chosen_log_probs * rew_t.clamp_min(0.0)).mean()

            # Anchor anti-forgetting regularization: KL divergence to base distribution
            anchor_loss = torch.tensor(0.0)
            if anchor_batch:
                a_obs_t = torch.tensor([a["obs"] for a in anchor_batch], dtype=torch.float32)
                a_tgt_t = torch.tensor([a["target_probs"] for a in anchor_batch], dtype=torch.float32)
                with torch.no_grad():
                    a_base_logits = self.core.model(["Survive"] * len(anchor_batch), a_obs_t)["action"]
                a_adapted_logits = self.adapter(a_obs_t, a_base_logits)
                a_adapted_log_probs = F.log_softmax(a_adapted_logits, dim=-1)
                anchor_loss = F.kl_div(a_adapted_log_probs, a_tgt_t, reduction="batchmean")

            loss = task_loss + self.config.anchor_weight * anchor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps_run += 1
            total_loss += float(loss.item())

        return {
            "mean_loss": total_loss / max(1, steps_run),
            "steps": steps_run,
        }

    def evaluate_policy_on_env(self, use_adapter: bool, seeds: Sequence[int], steps: int = 32) -> dict[str, float]:
        """Evaluates policy survival and return across evaluation seeds."""
        total_reward = 0.0
        survived = 0

        for seed in seeds:
            pool = TidePool(seed=seed, max_steps=steps)
            ep_reward = 0.0
            dead = False
            for _ in range(steps):
                obs = pool.observe()
                if obs[0] <= 0.0:
                    dead = True
                    break
                obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    base_logits = self.core.model(["Survive"], obs_t)["action"]
                    if use_adapter:
                        logits = self.adapter(obs_t, base_logits)
                    else:
                        logits = base_logits
                    action = int(logits.argmax(dim=-1).item())

                next_obs, reward, done, info = pool.step(action)
                ep_reward += float(reward)
                if done:
                    break

            total_reward += ep_reward
            if not dead:
                survived += 1

        n = len(seeds)
        return {
            "mean_reward": total_reward / n,
            "survival_rate": survived / n,
        }

    def evaluate_and_audit_promotion(self, ledger_path: Path | str | None = None) -> dict[str, Any]:
        """Evaluates candidate against incumbent with anchor retention gate."""
        incumbent_res = self.evaluate_policy_on_env(use_adapter=False, seeds=self.config.eval_seeds)
        candidate_res = self.evaluate_policy_on_env(use_adapter=True, seeds=self.config.eval_seeds)

        anchor_retention = self.anchors.evaluate_retention(self.adapter, self.core)
        anchor_regression = 1.0 - anchor_retention
        return_delta = candidate_res["mean_reward"] - incumbent_res["mean_reward"]

        promoted = False
        rejection_reason = None

        if anchor_regression > self.config.max_anchor_regression:
            rejection_reason = f"anchor_regression_exceeded: {anchor_regression:.4f} > {self.config.max_anchor_regression:.4f}"
        elif return_delta < self.config.promotion_min_delta:
            rejection_reason = f"insufficient_return_delta: {return_delta:.4f} < {self.config.promotion_min_delta:.4f}"
        else:
            promoted = True

        audit_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "PROMOTED" if promoted else "REJECTED",
            "reason": rejection_reason or "passed_all_gates",
            "anchor_retention": float(anchor_retention),
            "anchor_regression": float(anchor_regression),
            "return_delta": float(return_delta),
            "incumbent": incumbent_res,
            "candidate": candidate_res,
            "eval_seeds": list(self.config.eval_seeds),
        }

        if ledger_path is not None:
            p = Path(ledger_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ledger_data = []
            if p.is_file():
                try:
                    ledger_data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    ledger_data = []
            ledger_data.append(audit_entry)
            p.write_text(json.dumps(ledger_data, indent=2), encoding="utf-8")

        return audit_entry
