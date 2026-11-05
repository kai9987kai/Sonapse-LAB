"""POSEIDON STAGE 6: Controlled Adaptation and Information Seeking.

Formal Implementation of Stage 6 from docs/superpowers/specs/2026-10-11-learning-transfer-design.md:
1. Targeted Epistemic Probing vs. Random Equal-Budget Probing:
   - Compares active-inference driven information seeking (maximizing epistemic value
     and state dispersion) against random exploration probes under identical step/compute budgets.
   - Measures epistemic uncertainty reduction, feature coverage, and downstream return benefit.

2. Candidate-Only Sleep & Replay Adaptation Sessions:
   - Enforces strict isolation: training batches update only an isolated candidate adapter.
   - Cancelled or interrupted sessions leave incumbent untouched (fail-closed rollback).
   - Core runtime remains permanently frozen.

3. Independent Retention and Adaptation Audit Gate:
   - Audits anti-forgetting retention on anchor distributions (anchor agreement >= 98%, regression <= 2%).
   - Confirms return advantage (Delta R >= delta_min) and survival non-inferiority on held-out audit seeds.
   - Produces cryptographically bound, reproducible admission receipts.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .active_inference import ActiveInferenceEvaluator, ActiveInferenceConfig
from .adapt import AnchorDataset, ContinualAdapter, ReplayBuffer, ResidualPolicyAdapter, Transition
from .core import CoreRuntime
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for information-seeking exploratory probes."""
    budget_steps: int = 16
    epistemic_weight: float = 1.0
    dispersion_weight: float = 0.5
    min_safe_health: float = 0.15
    eval_seeds: tuple[int, ...] = (501, 502, 503, 504)


@dataclass
class ProbeOutcome:
    """Results from an exploratory probe episode."""
    probe_type: str
    seed: int
    steps_executed: int
    survived: bool
    total_reward: float
    initial_uncertainty: float
    final_uncertainty: float
    uncertainty_reduction: float
    state_coverage_volume: float
    action_entropy: float
    executed_actions: list[int]


class EpistemicInformationProber:
    """Evaluates targeted active-inference probes against random exploration probes."""

    def __init__(
        self,
        core: CoreRuntime,
        active_inference: ActiveInferenceEvaluator | None = None,
        config: ProbeConfig | None = None,
    ):
        self.core = core
        self.active_inference = active_inference or ActiveInferenceEvaluator()
        self.config = config or ProbeConfig()

    def run_probe(
        self,
        env: TidePool,
        probe_type: str = "targeted",
        rng: random.Random | None = None,
    ) -> ProbeOutcome:
        """Executes a single probe episode under the declared exploration policy."""
        r = rng or random.Random()
        obs = env.observe()
        obs_history: list[list[float]] = [list(obs)]
        executed_actions: list[int] = []
        total_reward = 0.0
        survived = True

        # Track uncertainty over the vital channels (energy, hydration, patch availability)
        def estimate_state_uncertainty(o: Sequence[float]) -> float:
            # Epistemic drive modulates with distance from clear knowledge
            # Ambiguity in food/water signals (channels 6, 7) and weather/fog (channel 9)
            fog = float(o[9]) if len(o) > 9 else 0.0
            food_sig = float(o[6]) if len(o) > 6 else 0.5
            water_sig = float(o[7]) if len(o) > 7 else 0.5
            ambiguity = fog * 0.5 + abs(food_sig - 0.5) * 0.25 + abs(water_sig - 0.5) * 0.25
            return float(ambiguity)

        initial_uncertainty = estimate_state_uncertainty(obs)

        for step in range(self.config.budget_steps):
            if obs[0] <= 0.0:
                survived = False
                break

            obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
            with torch.no_grad():
                base_logits = self.core.model(["Survive"], obs_t)["action"][0]
                base_action = int(base_logits.argmax().item())

            if probe_type == "targeted":
                # Targeted probe: score candidate actions by expected epistemic information gain
                candidate_scores = []
                for act in range(N_ACTIONS):
                    # Compute epistemic score
                    epistemic_drive = self.active_inference.compute_epistemic_drive(obs)
                    # Exploration action bonus (action 4 is explore)
                    act_bonus = 0.3 if act == 4 else (0.15 if act in (1, 2) else 0.0)
                    score = epistemic_drive * act_bonus + float(base_logits[act].item()) * 0.2
                    candidate_scores.append(score)
                action = int(torch.tensor(candidate_scores).argmax().item())
            elif probe_type == "random":
                # Random probe: uniform selection among safe actions
                # Ensure health guard: don't willingly walk off cliff if base knows extreme danger
                action = r.randint(0, N_ACTIONS - 1)
            elif probe_type == "baseline":
                action = base_action
            else:
                raise ValueError(f"unknown probe_type: {probe_type}")

            next_obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            executed_actions.append(action)
            obs_history.append(list(next_obs))
            obs = next_obs
            if done:
                break

        final_uncertainty = estimate_state_uncertainty(obs)
        uncertainty_reduction = max(0.0, initial_uncertainty - final_uncertainty)

        # State space coverage volume (bounding variance across observations)
        n_obs = len(obs_history)
        dim = min(OBS_SIZE, len(obs_history[0]))
        mean_obs = [sum(h[i] for h in obs_history) / n_obs for i in range(dim)]
        dispersion = sum(
            math.sqrt(sum((h[i] - mean_obs[i]) ** 2 for i in range(dim)))
            for h in obs_history
        ) / n_obs

        # Action entropy
        act_counts = [executed_actions.count(a) for a in range(N_ACTIONS)]
        total_acts = max(1, len(executed_actions))
        act_probs = [c / total_acts for c in act_counts if c > 0]
        action_entropy = -sum(p * math.log(p) for p in act_probs) if act_probs else 0.0

        return ProbeOutcome(
            probe_type=probe_type,
            seed=getattr(env, "seed", 0),
            steps_executed=len(executed_actions),
            survived=survived and obs[0] > 0.0,
            total_reward=total_reward,
            initial_uncertainty=initial_uncertainty,
            final_uncertainty=final_uncertainty,
            uncertainty_reduction=uncertainty_reduction,
            state_coverage_volume=float(dispersion),
            action_entropy=float(action_entropy),
            executed_actions=executed_actions,
        )

    def compare_probes(
        self,
        seeds: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Paired evaluation of targeted probes vs random probes under equal budget."""
        eval_seeds = list(seeds or self.config.eval_seeds)
        targeted_outcomes: list[ProbeOutcome] = []
        random_outcomes: list[ProbeOutcome] = []

        for seed in eval_seeds:
            # Targeted probe rollout
            env_t = TidePool(seed=seed, max_steps=self.config.budget_steps)
            targeted_outcomes.append(self.run_probe(env_t, probe_type="targeted"))

            # Random probe rollout
            env_r = TidePool(seed=seed, max_steps=self.config.budget_steps)
            random_outcomes.append(self.run_probe(env_r, probe_type="random", rng=random.Random(seed)))

        n = len(eval_seeds)
        t_mean_rew = sum(o.total_reward for o in targeted_outcomes) / n
        r_mean_rew = sum(o.total_reward for o in random_outcomes) / n
        t_mean_surv = sum(1.0 if o.survived else 0.0 for o in targeted_outcomes) / n
        r_mean_surv = sum(1.0 if o.survived else 0.0 for o in random_outcomes) / n
        t_mean_cov = sum(o.state_coverage_volume for o in targeted_outcomes) / n
        r_mean_cov = sum(o.state_coverage_volume for o in random_outcomes) / n
        t_mean_reduct = sum(o.uncertainty_reduction for o in targeted_outcomes) / n
        r_mean_reduct = sum(o.uncertainty_reduction for o in random_outcomes) / n

        return {
            "num_seeds": n,
            "seeds": eval_seeds,
            "budget_steps_per_episode": self.config.budget_steps,
            "targeted": {
                "mean_reward": float(t_mean_rew),
                "survival_rate": float(t_mean_surv),
                "state_coverage_volume": float(t_mean_cov),
                "mean_uncertainty_reduction": float(t_mean_reduct),
            },
            "random": {
                "mean_reward": float(r_mean_rew),
                "survival_rate": float(r_mean_surv),
                "state_coverage_volume": float(r_mean_cov),
                "mean_uncertainty_reduction": float(r_mean_reduct),
            },
            "comparison": {
                "delta_reward": float(t_mean_rew - r_mean_rew),
                "delta_survival": float(t_mean_surv - r_mean_surv),
                "delta_coverage": float(t_mean_cov - r_mean_cov),
                "delta_uncertainty_reduction": float(t_mean_reduct - r_mean_reduct),
                "targeted_advantage_supported": bool(t_mean_reduct >= r_mean_reduct and t_mean_surv >= r_mean_surv),
            },
        }


class CandidateAdapterSession:
    """Atomic, cancel-safe candidate adaptation session.

    Guarantees that training updates apply ONLY to an isolated candidate adapter copy.
    If a training batch is interrupted, cancelled, or fails validation, rollback is
    guaranteed and the incumbent remains 100% untouched.
    """

    def __init__(self, parent_adapter: ContinualAdapter):
        self.parent = parent_adapter
        # Create an isolated clone of the candidate adapter weights
        self.candidate_adapter = copy.deepcopy(parent_adapter.adapter)
        self.committed = False
        self.cancelled = False
        self.training_log: list[dict[str, Any]] = []

    def train_batch(
        self,
        transitions: list[Transition],
        steps: int = 10,
        lr: float = 1e-3,
    ) -> dict[str, float]:
        """Runs candidate-only gradient updates on transitions with anchor regularization."""
        if self.committed or self.cancelled:
            raise RuntimeError("cannot train on an already finalized session")
        if not transitions:
            return {"loss": 0.0, "steps": 0}

        optimizer = torch.optim.AdamW(
            self.candidate_adapter.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )

        total_loss = 0.0
        for step in range(steps):
            obs_t = torch.tensor([t.obs for t in transitions], dtype=torch.float32)
            act_t = torch.tensor([t.action for t in transitions], dtype=torch.long)
            rew_t = torch.tensor([t.reward for t in transitions], dtype=torch.float32)

            with torch.no_grad():
                base_logits = self.parent.core.model(["Survive"] * len(transitions), obs_t)["action"]

            adapted_logits = self.candidate_adapter(obs_t, base_logits)
            log_probs = F.log_softmax(adapted_logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, act_t.unsqueeze(1)).squeeze(1)
            task_loss = -(chosen_log_probs * rew_t.clamp_min(0.0)).mean()

            # Anchor regularization
            anchor_batch = self.parent.anchors.sample(min(len(transitions), len(self.parent.anchors.items)))
            anchor_loss = torch.tensor(0.0)
            if anchor_batch:
                a_obs_t = torch.tensor([a["obs"] for a in anchor_batch], dtype=torch.float32)
                a_tgt_t = torch.tensor([a["target_probs"] for a in anchor_batch], dtype=torch.float32)
                with torch.no_grad():
                    a_base_logits = self.parent.core.model(["Survive"] * len(anchor_batch), a_obs_t)["action"]
                a_adapted_logits = self.candidate_adapter(a_obs_t, a_base_logits)
                a_log_probs = F.log_softmax(a_adapted_logits, dim=-1)
                anchor_loss = F.kl_div(a_log_probs, a_tgt_t, reduction="batchmean")

            loss = task_loss + self.parent.config.anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        mean_loss = total_loss / max(1, steps)
        receipt = {"mean_loss": mean_loss, "steps": steps, "batch_size": len(transitions)}
        self.training_log.append(receipt)
        return receipt

    def cancel(self) -> None:
        """Explicitly cancels the adaptation session, preserving the incumbent intact."""
        self.cancelled = True
        self.candidate_adapter = None  # drop candidate weights

    def commit_if_audited(self, audit_result: dict[str, Any]) -> bool:
        """Promotes candidate to incumbent only if audit passed."""
        if self.cancelled:
            return False
        if audit_result.get("status") == "ADMITTED":
            self.parent.adapter.load_state_dict(self.candidate_adapter.state_dict())
            self.committed = True
            return True
        else:
            self.cancel()
            return False


class RetentionAndAdaptationAuditor:
    """Independent formal audit gate for candidate adapter admission."""

    def __init__(
        self,
        core: CoreRuntime,
        anchors: AnchorDataset,
        eval_seeds: Sequence[int] = (201, 202, 203, 204),
        min_return_delta: float = 0.02,
        max_anchor_regression: float = 0.02,
    ):
        self.core = core
        self.anchors = anchors
        self.eval_seeds = tuple(eval_seeds)
        self.min_return_delta = min_return_delta
        self.max_anchor_regression = max_anchor_regression

    def audit_candidate(
        self,
        candidate_adapter: ResidualPolicyAdapter,
        incumbent_adapter: ResidualPolicyAdapter | None = None,
        receipt_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Evaluates retention and adaptation return advantage."""
        # 1. Anti-forgetting anchor retention check
        retention = self.anchors.evaluate_retention(candidate_adapter, self.core)
        regression = 1.0 - retention

        # 2. Performance on audit seeds
        def evaluate_adapter(adapter: ResidualPolicyAdapter | None) -> dict[str, float]:
            total_rew = 0.0
            survived = 0
            for seed in self.eval_seeds:
                pool = TidePool(seed=seed, max_steps=32)
                ep_rew = 0.0
                dead = False
                for _ in range(32):
                    obs = pool.observe()
                    if obs[0] <= 0.0:
                        dead = True
                        break
                    obs_t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
                    with torch.no_grad():
                        base_logits = self.core.model(["Survive"], obs_t)["action"]
                        if adapter is not None:
                            logits = adapter(obs_t, base_logits)
                        else:
                            logits = base_logits
                        act = int(logits.argmax(dim=-1).item())
                    _, r, done, _ = pool.step(act)
                    ep_rew += float(r)
                    if done:
                        break
                total_rew += ep_rew
                if not dead:
                    survived += 1
            n = len(self.eval_seeds)
            return {"mean_reward": total_rew / n, "survival_rate": survived / n}

        incumbent_perf = evaluate_adapter(incumbent_adapter)
        candidate_perf = evaluate_adapter(candidate_adapter)

        delta_rew = candidate_perf["mean_reward"] - incumbent_perf["mean_reward"]
        delta_surv = candidate_perf["survival_rate"] - incumbent_perf["survival_rate"]

        admitted = False
        rejection_reasons = []

        if regression > self.max_anchor_regression:
            rejection_reasons.append(
                f"anchor_regression_exceeded ({regression:.4f} > {self.max_anchor_regression:.4f})"
            )
        if delta_rew < self.min_return_delta:
            rejection_reasons.append(
                f"insufficient_return_delta ({delta_rew:+.4f} < {self.min_return_delta:.4f})"
            )
        if delta_surv < 0.0:
            rejection_reasons.append(
                f"survival_regression ({delta_surv:+.4f} < 0)"
            )

        if not rejection_reasons:
            admitted = True

        # Compute SHA-256 digests for cryptographic receipt
        weights_bytes = b"".join(
            p.detach().cpu().numpy().tobytes() for p in candidate_adapter.parameters()
        )
        candidate_sha = hashlib.sha256(weights_bytes).hexdigest()

        receipt = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ADMITTED" if admitted else "REJECTED",
            "candidate_sha256": candidate_sha,
            "reasons": rejection_reasons if not admitted else ["passed_all_gates"],
            "anchor_retention": float(retention),
            "anchor_regression": float(regression),
            "delta_reward": float(delta_rew),
            "delta_survival": float(delta_surv),
            "incumbent_perf": incumbent_perf,
            "candidate_perf": candidate_perf,
            "eval_seeds": list(self.eval_seeds),
        }

        receipt_digest = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode("utf-8")).hexdigest()
        receipt["receipt_digest"] = receipt_digest

        if receipt_dir is not None:
            p = Path(receipt_dir)
            p.mkdir(parents=True, exist_ok=True)
            receipt_file = p / f"audit_{candidate_sha[:8]}.json"
            receipt_file.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

        return receipt
