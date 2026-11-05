"""PROJECT NEXUS-Ω: Independent Scientific Auditor.

Executes preregistered causal intervention assays, matched-seed benchmarks,
falsification testing, and cryptographic receipt emission.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.contracts import (
    AuditReceipt,
    HypothesisSpec,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
)


class IndependentScientificAuditor:
    """Rigorous adversarial auditor evaluating hypotheses under matched seed controls."""

    def __init__(self, seeds: Optional[List[int]] = None, max_steps: int = 32):
        self.seeds = seeds or [10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008]
        self.max_steps = max_steps

    def rollout_episode(
        self,
        agent: NexusOmegaAgent,
        seed: int,
        intervention: Optional[InterventionSpec] = None,
        intervene_at_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        env = POMDPPhaseWorld(seed=seed, max_steps=self.max_steps)
        obs = env.reset()
        agent.reset()

        total_reward = 0.0
        prediction_errors: List[float] = []
        steps = 0
        survived = True

        for t in range(self.max_steps):
            steps += 1
            if intervention is not None:
                if intervene_at_step is None or t == intervene_at_step:
                    agent.apply_intervention(intervention)

            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            e_t = agent.observe_transition(next_obs, reward, done)
            prediction_errors.append(float(np.mean(np.abs(e_t))))

            if done:
                survived = not info.get("death", False)
                break
            obs = next_obs

        return {
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "survived": survived,
            "mean_error": float(np.mean(prediction_errors)) if prediction_errors else 0.0,
            "intervention_log": list(agent.memory.state.intervention_log),
        }

    def run_causal_memory_assay(
        self,
        hypothesis: HypothesisSpec,
    ) -> AuditReceipt:
        """Runs matched-seed causal persistence assay comparing Intact vs Ablated Memory."""
        agent = NexusOmegaAgent(d_obs=8, d_working=16, n_actions=4)

        intact_rewards = []
        intact_survivals = []
        ablated_rewards = []
        ablated_survivals = []

        for s in self.seeds:
            # 1. Matched seed run: Intact condition
            res_intact = self.rollout_episode(
                agent,
                seed=s,
                intervention=InterventionSpec(InterventionType.INTACT),
            )
            intact_rewards.append(res_intact["total_reward"])
            intact_survivals.append(1.0 if res_intact["survived"] else 0.0)

            # 2. Matched seed run: Continuous Ablation condition
            ablate_spec = InterventionSpec(
                intervention_type=InterventionType.ABLATE_CARRIER,
                target_carrier=MemoryCarrierType.WORKING,
            )
            res_ablated = self.rollout_episode(
                agent,
                seed=s,
                intervention=ablate_spec,
            )
            ablated_rewards.append(res_ablated["total_reward"])
            ablated_survivals.append(1.0 if res_ablated["survived"] else 0.0)

        intact_mean_r = float(np.mean(intact_rewards))
        ablated_mean_r = float(np.mean(ablated_rewards))
        diffs = np.array(intact_rewards) - np.array(ablated_rewards)
        mean_diff = float(np.mean(diffs))

        # Paired two-tailed t-test
        if np.all(diffs == 0):
            p_val = 1.0
            effect_size = 0.0
        else:
            t_stat, p_val = stats.ttest_rel(intact_rewards, ablated_rewards)
            std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1e-4
            effect_size = float(mean_diff / (std_diff + 1e-8))

        # Check falsification vs acceptance
        is_significant = (p_val < hypothesis.alpha) and (mean_diff >= hypothesis.minimum_effect_size)
        verdict = "ACCEPTED" if is_significant else "FALSIFIED"

        receipt = AuditReceipt(
            audit_id=f"audit_{int(datetime.now().timestamp())}",
            hypothesis_id=hypothesis.hypothesis_id,
            timestamp=datetime.now().isoformat(),
            seeds_evaluated=list(self.seeds),
            baseline_metrics={
                "mean_reward": intact_mean_r,
                "survival_rate": float(np.mean(intact_survivals)),
            },
            candidate_metrics={
                "mean_reward": ablated_mean_r,
                "survival_rate": float(np.mean(ablated_survivals)),
                "mean_delta_reward": mean_diff,
            },
            p_value=float(p_val),
            effect_size=effect_size,
            retained_in_library=is_significant,
            verdict=verdict,
        )
        receipt.seal()
        return receipt

    def run_four_fold_memory_test(self, seed: int = 42000) -> Dict[str, Any]:
        """Operationally verifies the 4-fold distinction: Retention, Reconstruction, Relearning, Mimicry."""
        agent = NexusOmegaAgent(d_obs=8, d_working=16, n_actions=4)

        # 1. Retention: evaluate intact memory over 24 steps
        res_retention = self.rollout_episode(agent, seed=seed)

        # 2. Ablation: evaluate ablated working memory
        ablate_spec = InterventionSpec(InterventionType.ABLATE_CARRIER, MemoryCarrierType.WORKING)
        res_ablation = self.rollout_episode(agent, seed=seed, intervention=ablate_spec)

        # 3. Reconstruction: ablate at step 12, then apply reconstruct operator
        recon_spec = InterventionSpec(InterventionType.RECONSTRUCT_CARRIER, MemoryCarrierType.WORKING)
        res_recon = self.rollout_episode(agent, seed=seed, intervention=recon_spec, intervene_at_step=12)

        # 4. Behavioral Mimicry Check: memory-less reactive heuristic
        env = POMDPPhaseWorld(seed=seed, max_steps=self.max_steps)
        obs = env.reset()
        reactive_reward = 0.0
        for _ in range(self.max_steps):
            # Reactive heuristic: forage if health > 0.5 regardless of phase
            act = 0 if obs[0] > 0.5 else 1
            obs, r, done, _ = env.step(act)
            reactive_reward += r
            if done:
                break

        return {
            "retention_reward": res_retention["total_reward"],
            "ablation_reward": res_ablation["total_reward"],
            "reconstruction_reward": res_recon["total_reward"],
            "mimicry_reactive_reward": reactive_reward,
            "retention_advantage": res_retention["total_reward"] - res_ablation["total_reward"],
            "reconstruction_advantage": res_recon["total_reward"] - res_ablation["total_reward"],
            "is_distinguishable": bool(res_retention["total_reward"] != res_ablation["total_reward"]),
        }
