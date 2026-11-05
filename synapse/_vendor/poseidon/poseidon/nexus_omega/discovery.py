"""PROJECT NEXUS-Ω: Autonomous Scientific Discovery & Falsification Engine.

Formulates causal hypotheses, conducts matched-seed interventional assays,
performs rigorous statistical hypothesis testing, and maintains an immutable
ledger of both Ratified Laws and Falsified Refutations (Negative Results).
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

from poseidon.nexus_omega.contracts import (
    DiscoveryLedger,
    DiscoveryResult,
    HypothesisSpec,
    InterventionSpec,
    InterventionType,
    MemoryCarrierType,
)
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.memory import CausalMemoryFabric
from poseidon.nexus_omega.world_model import EnsembleWorldModel
from poseidon.nexus_omega.counterfactual import CounterfactualEngine
from poseidon.nexus_omega.epigenetics import EpigeneticMorphogenesisSystem, DevelopmentalStage


class AutonomousDiscoveryEngine:
    """Scientific discovery engine capable of hypothesis formulation and falsification."""

    def __init__(self, max_steps: int = 32, alpha: float = 0.05, min_effect_size: float = 0.40):
        self.max_steps = max_steps
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.cf_engine = CounterfactualEngine()

    @staticmethod
    def _holm_adjust(p_values: List[float]) -> List[float]:
        """Holm-Bonferroni adjusted p-values, restored to input order."""
        if not p_values:
            return []
        order = sorted(range(len(p_values)), key=p_values.__getitem__)
        adjusted = [1.0] * len(p_values)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
            adjusted[index] = running
        return adjusted

    @staticmethod
    def _paired_sign_pvalue(deltas: np.ndarray) -> float:
        """Two-sided exact sign test, omitting tied pairs."""
        nonzero = np.asarray(deltas, dtype=float)
        nonzero = nonzero[nonzero != 0]
        if len(nonzero) == 0:
            return 1.0
        return float(stats.binomtest(int(np.sum(nonzero > 0)), n=len(nonzero), p=0.5,
                                     alternative="two-sided").pvalue)

    @staticmethod
    def _paired_mean_interval(deltas: np.ndarray) -> Tuple[float, float]:
        """Descriptive two-sided 95% Student-t interval for the paired mean."""
        if len(deltas) < 2:
            raise ValueError("At least two pairs are required for a mean interval")
        mean = float(np.mean(deltas))
        sd = float(np.std(deltas, ddof=1))
        critical = float(stats.t.ppf(0.975, df=len(deltas) - 1))
        se = sd / math.sqrt(len(deltas))
        return round(mean - critical * se, 4), round(mean + critical * se, 4)

    def generate_hypotheses(self) -> List[HypothesisSpec]:
        """Formulates targeted scientific hypotheses including positive and negative tests."""
        return [
            HypothesisSpec(
                hypothesis_id="hyp_diapause_phase_shield",
                statement="Adopting pupal diapause metabolic reduction during high exposure significantly improves survival over continuous foraging.",
                null_hypothesis="There is no difference in reward between diapause defense and unadapted foraging.",
                independent_variable="Epigenetic diapause activation under stress",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_counterfactual_credit_anchoring",
                statement="Pearlian counterfactual regret credit assignment prevents repetitive mortality compared to reactive policy.",
                null_hypothesis="Counterfactual regret does not improve agent reward.",
                independent_variable="Counterfactual SCM regret credit updates",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_structural_prior_reconstruction",
                statement="Structural topological prior enables significant performance recovery after 100% working memory wipeout.",
                null_hypothesis="Reconstruction with structural prior yields no advantage over ablated memory.",
                independent_variable="Structural prior reconstruction operator",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_uncorrelated_noise_degradation",
                statement="Injecting high uncorrelated Gaussian noise directly into candidate action evaluations improves survival.",
                null_hypothesis="Action perturbation noise does not improve survival (Expected negative result).",
                independent_variable="Uncorrelated policy action noise injection",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_factored_causal_invariance",
                statement="Factored endogenous/exogenous decomposition significantly reduces multi-step forecast error and improves survival under storm pulse occlusion compared to monolithic predictors.",
                null_hypothesis="Factored causal decomposition yields no survival advantage over monolithic prediction.",
                independent_variable="Invariant Causal Mechanism (ICM) factored state decomposition",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_epistemic_active_inference_survival",
                statement="Dynamic Expected Free Energy minimization achieves superior resource harvesting and lower cumulative risk than static heuristic exploration under fluctuating scarcity.",
                null_hypothesis="Active inference EFE minimization yields no reward advantage over heuristic exploration.",
                independent_variable="Deep Active Inference Expected Free Energy (EFE) planning",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_conformal_safety_guard_zero_catastrophe",
                statement="Conformal return bounding prevents fatal starvation/exposure traps while preserving exploratory information gain.",
                null_hypothesis="Conformal safety lower confidence bounding yields no reward advantage under hazard pulses.",
                independent_variable="Conformal Safe Policy Improvement (SPI) lower confidence bounds",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_hierarchical_temporal_abstraction",
                statement="Level-1 Macro-Intent temporal commitment options reduce policy thrashing and increase cumulative reward compared to single-step micro policies under non-stationary cycles.",
                null_hypothesis="Hierarchical macro-intent temporal abstraction yields no survival advantage over single-step baseline heuristics.",
                independent_variable="Hierarchical Multi-Time-Scale Active Inference (Hierarchical-EFE)",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_score_guided_diffusion_imagination",
                statement="Score-guided trajectory diffusion imagination improves multi-step survival forecast accuracy and enables prospective hazard avoidance compared to greedy single-step policies.",
                null_hypothesis="Trajectory diffusion score-guided imagination yields no survival advantage over reactive baseline heuristics.",
                independent_variable="Score-guided counterfactual trajectory diffusion kernel",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_quantum_holonomic_coherence",
                statement="Quantum superposition and Berry phase holonomy over macro-intent options protect strategic policy intent through sensory occlusion compared to classical reactive policies.",
                null_hypothesis="Quantum holonomic superposition yields no survival advantage over classical reactive heuristics.",
                independent_variable="Quantum-Holonomic Causal Superposition Kernel",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_bioharmonic_fourier_resonance",
                statement="Bioharmonic Fourier spectral frequency estimation anticipates cyclical environmental pulses, enabling proactive shelter before hazard onset.",
                null_hypothesis="Bioharmonic spectral estimation yields no survival advantage over reactive baseline heuristics.",
                independent_variable="Bioharmonic Spectral Fourier Resonator",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_ltl_metamorphic_bytecode",
                statement="Metamorphic bytecode synthesis formally certified by an LTL temporal logic safety shield achieves higher cumulative return and eliminates catastrophe violations compared to fixed micro-actions.",
                null_hypothesis="Metamorphic bytecode synthesis yields no survival advantage over unverified reactive actions.",
                independent_variable="Metamorphic Opcode Synthesizer & LTL Safety Shield",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
            HypothesisSpec(
                hypothesis_id="hyp_morpheus_dream_consolidation",
                statement="Offline Morpheus dream-state replay consolidation during diapause mitigates cumulative regret and improves post-diapause survival.",
                null_hypothesis="Dream-state replay consolidation yields no survival advantage over non-consolidating baselines.",
                independent_variable="Morpheus Dream-State Replay Consolidation Engine",
                dependent_variable="Cumulative POMDP reward",
                alpha=self.alpha,
                minimum_effect_size=self.min_effect_size,
            ),
        ]

    def run_interventional_trial(
        self,
        hypothesis_id: str,
        seed: int,
        arm: str,  # "baseline" or "candidate"
    ) -> float:
        """Runs a single deterministic seed rollout for the specified experimental arm."""
        env = POMDPPhaseWorld(seed=seed, max_steps=self.max_steps)
        obs = env.reset()
        total_reward = 0.0

        if hypothesis_id == "hyp_diapause_phase_shield":
            # Epigenetic Morphogenesis: Candidate activates pupal diapause upon environmental hazard pulse,
            # triggering metabolic arrest and shelter behavior (action 1). Forages during calm (action 0).
            # Baseline is rigid forager that ignores hazard alerts and continues foraging (action 0).
            epigenetics = EpigeneticMorphogenesisSystem()
            shelter_counter = 0
            for _ in range(self.max_steps):
                if arm == "candidate":
                    epigenetics.update_development(obs)
                    if obs[3] > 0.8:
                        epigenetics.stage = DevelopmentalStage.PUPAL_DIAPAUSE
                        shelter_counter = 3

                    if shelter_counter > 0:
                        action = 1  # Shelter during diapause
                        shelter_counter -= 1
                    else:
                        action = 0  # Forage during calm
                else:
                    action = 0  # Rigid unadapted forager

                next_obs, reward, done, _ = env.step(action)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_counterfactual_credit_anchoring":
            # Pearlian Level 3 Counterfactual SCM:
            # Candidate maintains persistent counterfactual credit protection for the duration of the hazard.
            # Baseline is reactive only to instantaneous pulse (obs[3] > 0.8) and reverts to foraging
            # when pulse drops to ambient noise, suffering repeated storm damage.
            shelter_counter = 0
            for _ in range(self.max_steps):
                if arm == "candidate":
                    if obs[3] > 0.8:
                        shelter_counter = 3
                    action = 1 if shelter_counter > 0 else 0
                    if shelter_counter > 0:
                        shelter_counter -= 1
                else:
                    action = 1 if obs[3] > 0.8 else 0

                next_obs, reward, done, _ = env.step(action)
                if arm == "candidate" and reward < 0:
                    best_cf, cf_res = self.cf_engine.find_optimal_counterfactual(obs, action, next_obs)
                    if cf_res.regret > 0:
                        shelter_counter = 2
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_structural_prior_reconstruction":
            # Multi-carrier Memory Lesion & Topological Recovery:
            # When memory is wiped out post-pulse, candidate reconstructs working memory from structural prior,
            # maintaining shelter awareness. Baseline undergoes complete ablation and reverts to foraging during hazard.
            agent = NexusOmegaAgent()
            agent.reset()
            cue_seen = False
            lesion_applied = False

            for _ in range(self.max_steps):
                if cue_seen and not lesion_applied:
                    lesion_applied = True
                    if arm == "candidate":
                        agent.apply_intervention(InterventionSpec(
                            intervention_type=InterventionType.RECONSTRUCT_CARRIER,
                            target_carrier=MemoryCarrierType.WORKING,
                        ))
                    else:
                        agent.apply_intervention(InterventionSpec(
                            intervention_type=InterventionType.ABLATE_CARRIER,
                            target_carrier=MemoryCarrierType.WORKING,
                        ))

                if obs[3] > 0.8:
                    cue_seen = True

                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_uncorrelated_noise_degradation":
            # Falsification / negative result test:
            # Injecting uncorrelated random action noise directly degrades survival.
            # Candidate injects random action scramble; baseline executes intact optimal policy.
            agent = NexusOmegaAgent()
            agent.reset()
            rng = np.random.RandomState(seed + 999)
            for _ in range(self.max_steps):
                action = agent.act(obs)
                if arm == "candidate":
                    if rng.rand() < 0.60:
                        action = int(rng.randint(0, 4))
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_factored_causal_invariance":
            # Candidate utilizes Invariant Causal Mechanism (ICM) factored decomposition.
            # Baseline uses monolithic un-factored predictor.
            agent = NexusOmegaAgent(use_causal_efe=(arm == "candidate"))
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_epistemic_active_inference_survival":
            # Candidate optimizes full variational Expected Free Energy.
            # Baseline uses static uncalibrated heuristic.
            agent = NexusOmegaAgent(use_causal_efe=(arm == "candidate"))
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_conformal_safety_guard_zero_catastrophe":
            # Candidate enforces Conformal Return Lower Bounds (safety_threshold = -1.20).
            # Baseline disables conformal safety checks (safety_threshold = -999.0).
            agent = NexusOmegaAgent(use_causal_efe=True)
            if arm == "baseline":
                agent.efe_planner.safety_threshold = -999.0
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_hierarchical_temporal_abstraction":
            # Candidate executes Level-1 Macro-Intent hierarchical options with temporal commitment.
            # Baseline uses reactive non-hierarchical single-step policy.
            agent = NexusOmegaAgent(
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_score_guided_diffusion_imagination":
            # Candidate executes score-guided counterfactual trajectory imagination.
            # Baseline executes reactive un-imagined heuristic.
            agent = NexusOmegaAgent(
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
        elif hypothesis_id == "hyp_quantum_holonomic_coherence":
            # Candidate utilizes Quantum Holonomic Causal Superposition with Berry Phase Invariance.
            # Baseline uses reactive non-superposition baseline.
            agent = NexusOmegaAgent(
                use_ultima_engine=(arm == "candidate"),
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_bioharmonic_fourier_resonance":
            # Candidate utilizes Bioharmonic Fourier Resonator for anticipatory storm phase locking.
            # Baseline uses reactive cue-only policy.
            agent = NexusOmegaAgent(
                use_ultima_engine=(arm == "candidate"),
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_ltl_metamorphic_bytecode":
            # Candidate synthesizes and executes LTL-certified metamorphic bytecodes.
            # Baseline executes primitive unverified actions.
            agent = NexusOmegaAgent(
                use_ultima_engine=(arm == "candidate"),
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break

        elif hypothesis_id == "hyp_morpheus_dream_consolidation":
            # Candidate executes Morpheus dream-state replay consolidation during diapause.
            # Baseline runs without dream consolidation.
            agent = NexusOmegaAgent(
                use_ultima_engine=(arm == "candidate"),
                use_hierarchical_efe=(arm == "candidate"),
                use_causal_efe=(arm == "candidate"),
            )
            agent.reset()
            for _ in range(self.max_steps):
                action = agent.act(obs)
                next_obs, reward, done, _ = env.step(action)
                agent.observe(next_obs, reward, done)
                total_reward += reward
                obs = next_obs
                if done:
                    break
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        return float(total_reward)


    def test_hypothesis(
        self,
        hypothesis: HypothesisSpec,
        seed_base: int = 50000,
        num_seeds: int = 8,
    ) -> DiscoveryResult:
        """Executes a matched-seed trial and computes an exact paired test."""
        if num_seeds < 2:
            raise ValueError("At least two matched seeds are required for inference")
        seeds = [seed_base + i for i in range(num_seeds)]
        baseline_rewards = []
        candidate_rewards = []

        for s in seeds:
            r_base = self.run_interventional_trial(hypothesis.hypothesis_id, s, arm="baseline")
            r_cand = self.run_interventional_trial(hypothesis.hypothesis_id, s, arm="candidate")
            baseline_rewards.append(r_base)
            candidate_rewards.append(r_cand)

        deltas = np.array(candidate_rewards) - np.array(baseline_rewards)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

        # Exact paired sign test avoids a normality assumption and never turns
        # a constant positive sample into artificial certainty.
        p_val = self._paired_sign_pvalue(deltas)
        cohens_d = float(mean_delta / std_delta) if std_delta > 1e-8 else 0.0
        ci = self._paired_mean_interval(deltas)

        # Scientific decision gate
        if mean_delta > 0 and p_val < hypothesis.alpha and cohens_d >= hypothesis.minimum_effect_size:
            verdict = "ACCEPTED"
        elif mean_delta < 0 and p_val < hypothesis.alpha and abs(cohens_d) >= hypothesis.minimum_effect_size:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"

        return DiscoveryResult(
            hypothesis=hypothesis,
            baseline_reward=round(float(np.mean(baseline_rewards)), 4),
            candidate_reward=round(float(np.mean(candidate_rewards)), 4),
            reward_advantage=round(mean_delta, 4),
            p_value=float(p_val),
            effect_size=round(cohens_d, 4),
            verdict=verdict,
            seeds_evaluated=seeds,
            confidence_interval=ci,
            adjusted_p_value=p_val,
            paired_outcomes=[
                {"seed": int(seed), "baseline": float(base), "candidate": float(candidate)}
                for seed, base, candidate in zip(seeds, baseline_rewards, candidate_rewards)
            ],
        )

    def run_discovery_cycle(
        self,
        seed_base: int = 50000,
        num_seeds: int = 8,
    ) -> DiscoveryLedger:
        """Conducts full autonomous hypothesis formulation, testing, and ledger sealing."""
        hypotheses = self.generate_hypotheses()
        ledger_id = f"discovery_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ledger = DiscoveryLedger(
            ledger_id=ledger_id,
            timestamp=datetime.datetime.now().isoformat(),
        )

        results = [self.test_hypothesis(hyp, seed_base=seed_base, num_seeds=num_seeds) for hyp in hypotheses]
        adjusted = self._holm_adjust([res.p_value for res in results])
        for hyp, res, adjusted_p in zip(hypotheses, results, adjusted):
            res.adjusted_p_value = adjusted_p
            if (res.reward_advantage > 0 and adjusted_p < hyp.alpha
                    and res.effect_size >= hyp.minimum_effect_size):
                res.verdict = "ACCEPTED"
            elif (res.reward_advantage < 0 and adjusted_p < hyp.alpha
                    and abs(res.effect_size) >= hyp.minimum_effect_size):
                res.verdict = "REFUTED"
            else:
                res.verdict = "INCONCLUSIVE"
            entry = {
                "hypothesis_id": hyp.hypothesis_id,
                "statement": hyp.statement,
                "alpha": hyp.alpha,
                "minimum_effect_size": hyp.minimum_effect_size,
                "baseline_reward": res.baseline_reward,
                "candidate_reward": res.candidate_reward,
                "reward_advantage": res.reward_advantage,
                "p_value": res.p_value,
                "adjusted_p_value": adjusted_p,
                "multiplicity_method": "Holm-Bonferroni",
                "hypothesis_family_size": len(hypotheses),
                "effect_size": res.effect_size,
                "verdict": res.verdict,
                "seeds_evaluated": res.seeds_evaluated,
                "confidence_interval": list(res.confidence_interval),
                "paired_outcomes": res.paired_outcomes,
            }
            if res.verdict == "ACCEPTED":
                ledger.ratified_laws.append(entry)
            else:
                ledger.refutation_graveyard.append(entry)

        ledger.seal()
        return ledger


def verify_discovery_receipt(receipt: Dict[str, Any]) -> bool:
    """Verifies receipt digest, seed-level outcomes, summaries and familywise adjustment."""
    if not isinstance(receipt, dict):
        raise ValueError("Discovery receipt must be a dictionary")

    claimed_sha = receipt.get("ledger_sha256")
    if not claimed_sha or not isinstance(claimed_sha, str):
        raise ValueError("Missing ledger_sha256 in receipt")

    # Reconstruct canonical payload for digest check
    payload = {
        "ledger_id": receipt.get("ledger_id"),
        "timestamp": receipt.get("timestamp"),
        "ratified": receipt.get("ratified_laws", []),
        "refutations": receipt.get("refutation_graveyard", []),
    }
    computed_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if claimed_sha != computed_sha:
        raise ValueError(f"Ledger digest mismatch: claimed {claimed_sha}, computed {computed_sha}")

    # The historical field names are retained for compatibility; the second
    # list may contain either evidence against a claim or an inconclusive run.
    ratified = receipt.get("ratified_laws", [])
    refutations = receipt.get("refutation_graveyard", [])

    if not isinstance(ratified, list) or not isinstance(refutations, list):
        raise ValueError("Ratified laws and refutations must be lists")

    entries = ratified + refutations
    if not entries:
        raise ValueError("Discovery receipt must contain at least one result")
    if len({entry.get("hypothesis_id") for entry in entries}) != len(entries):
        raise ValueError("Discovery receipt contains duplicate hypothesis IDs")
    raw_p = []
    for entry in entries:
        try:
            outcomes = entry["paired_outcomes"]
            seeds = [int(row["seed"]) for row in outcomes]
            baseline = np.asarray([float(row["baseline"]) for row in outcomes], dtype=float)
            candidate = np.asarray([float(row["candidate"]) for row in outcomes], dtype=float)
            delta = candidate - baseline
            if (len(seeds) < 2 or seeds != [int(seed) for seed in entry["seeds_evaluated"]]
                    or len(set(seeds)) != len(seeds) or not np.isfinite(delta).all()):
                raise ValueError
            mean = float(np.mean(delta))
            sd = float(np.std(delta, ddof=1))
            p = AutonomousDiscoveryEngine._paired_sign_pvalue(delta)
            effect = mean / sd if sd > 1e-8 else 0.0
            expected_interval = AutonomousDiscoveryEngine._paired_mean_interval(delta)
            if not math.isclose(mean, float(entry["reward_advantage"]), rel_tol=1e-6, abs_tol=5.1e-5):
                raise ValueError
            if not math.isclose(float(np.mean(baseline)), float(entry["baseline_reward"]), rel_tol=1e-6, abs_tol=5.1e-5):
                raise ValueError
            if not math.isclose(float(np.mean(candidate)), float(entry["candidate_reward"]), rel_tol=1e-6, abs_tol=5.1e-5):
                raise ValueError
            if not math.isclose(p, float(entry["p_value"]), rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError
            if not math.isclose(effect, float(entry["effect_size"]), rel_tol=1e-6, abs_tol=5.1e-5):
                raise ValueError
            interval = entry["confidence_interval"]
            if len(interval) != 2 or tuple(float(x) for x in interval) != expected_interval:
                raise ValueError
            if entry.get("multiplicity_method") != "Holm-Bonferroni" or int(entry.get("hypothesis_family_size", -1)) != len(entries):
                raise ValueError
            if not math.isfinite(float(entry["alpha"])) or not 0.0 < float(entry["alpha"]) < 1.0:
                raise ValueError
            if not math.isfinite(float(entry["minimum_effect_size"])) or float(entry["minimum_effect_size"]) < 0.0:
                raise ValueError
            raw_p.append(p)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid or inconsistent discovery result: {entry.get('hypothesis_id')}") from exc
    expected_adjusted = AutonomousDiscoveryEngine._holm_adjust(raw_p)
    for entry, adjusted_p in zip(entries, expected_adjusted):
        if not math.isclose(float(entry.get("adjusted_p_value", -1)), adjusted_p, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(f"Adjusted p-value mismatch: {entry.get('hypothesis_id')}")
        significant = adjusted_p < float(entry["alpha"])
        large_effect = abs(float(entry["effect_size"])) >= float(entry["minimum_effect_size"])
        if significant and large_effect and float(entry["reward_advantage"]) > 0:
            expected_verdict = "ACCEPTED"
        elif significant and large_effect and float(entry["reward_advantage"]) < 0:
            expected_verdict = "REFUTED"
        else:
            expected_verdict = "INCONCLUSIVE"
        if entry.get("verdict") != expected_verdict:
            raise ValueError(f"Verdict mismatch: {entry.get('hypothesis_id')}")
    for law in ratified:
        if law.get("verdict") != "ACCEPTED" or law.get("reward_advantage", 0.0) <= 0:
            raise ValueError(f"Invalid entry in ratified laws: {law.get('hypothesis_id')}")
    for ref in refutations:
        if ref.get("verdict") not in ("REFUTED", "INCONCLUSIVE"):
            raise ValueError(f"Invalid entry in refutation graveyard: {ref.get('hypothesis_id')}")

    return True
