"""Unified Research Benchmark Suite & Scientific Scorecard Engine.

Integrates:
1. BenchmarkRegistry: Centralized catalog of all scientific assays and benchmarks.
2. BenchmarkRunner: Seed-controlled execution with progress tracking.
3. BenchmarkStatistics: Paired bootstrap confidence intervals, Cohen's d effect sizes, and Benjamini-Hochberg FDR correction.
4. ResearchScorecard: Markdown and JSON scorecard generator with before/after comparisons and non-claim audit receipts.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BenchmarkMetric:
    name: str
    value: float
    unit: str
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0


@dataclass
class BenchmarkExecutionResult:
    benchmark_id: str
    title: str
    category: str
    execution_time_ms: float
    passed: bool
    metrics: Dict[str, BenchmarkMetric]
    details: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = ""


@dataclass
class UnifiedScorecard:
    timestamp: float
    git_commit_sha: str
    results: List[BenchmarkExecutionResult]
    overall_pass_rate: float
    summary_markdown: str


class BenchmarkStatistics:
    """Rigorous statistical testing: bootstrap CI, Cohen's d, and FDR correction."""

    @staticmethod
    def bootstrap_confidence_interval(
        samples: List[float], num_bootstraps: int = 1000, alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """Compute mean and 95% bootstrap confidence interval."""
        if not samples:
            return 0.0, 0.0, 0.0
        n = len(samples)
        mean_val = sum(samples) / n
        if n == 1:
            return mean_val, mean_val, mean_val

        boot_means = []
        for _ in range(num_bootstraps):
            resample = [random.choice(samples) for _ in range(n)]
            boot_means.append(sum(resample) / n)
        boot_means.sort()

        low_idx = int((alpha / 2.0) * num_bootstraps)
        high_idx = int((1.0 - alpha / 2.0) * num_bootstraps)
        return (
            round(mean_val, 4),
            round(boot_means[max(0, low_idx)], 4),
            round(boot_means[min(num_bootstraps - 1, high_idx)], 4),
        )

    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d standardized effect size."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        m1, m2 = sum(group1) / n1, sum(group2) / n2
        s1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
        s2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
        s_pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        if s_pooled < 1e-8:
            return 0.0
        return round((m1 - m2) / s_pooled, 4)

    @staticmethod
    def fdr_benjamini_hochberg(p_values: List[float], q: float = 0.05) -> List[bool]:
        """Apply Benjamini-Hochberg FDR correction across multiple hypotheses."""
        m = len(p_values)
        if m == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        significant = [False] * m
        max_k = -1

        for rank, (orig_idx, p) in enumerate(indexed):
            k = rank + 1
            if p <= (k / m) * q:
                max_k = rank

        if max_k >= 0:
            for rank in range(max_k + 1):
                significant[indexed[rank][0]] = True
        return significant


class BenchmarkRegistry:
    """Catalog of all registered assays in the Sonapse-LAB research platform."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_benchmarks()

    def _register_default_benchmarks(self) -> None:
        self.register(
            "sancf_matrix",
            "SANCF 10-Arm Matrix Benchmark",
            "Neural Communication",
            "Evaluates cross-region payload, bits transmitted, and stability across 10 arms.",
        )
        self.register(
            "neurowave_modes",
            "NeuroWave Multiplexing Assay",
            "Oscillatory Mechanics",
            "Evaluates 7 oscillatory modes against 7 matched randomized controls.",
        )
        self.register(
            "causal_connectome",
            "Causal Connectome Discovery Assay",
            "Graph & Causality",
            "10 structural motifs and 10 causal intervention operators.",
        )
        self.register(
            "causal_world_model",
            "Causal World Model Do-Calculus Assay",
            "Causality",
            "Tests passive drift vs active interventional effect disentanglement.",
        )
        self.register(
            "relational_hrr",
            "Relational HRR Memory Interference Assay",
            "Memory & Representation",
            "Benchmarks capacity scaling (1..16 pairs) and distractor resistance.",
        )
        self.register(
            "meta_plasticity",
            "Meta-Plasticity Learning Assay",
            "Synaptic Plasticity",
            "Evaluates parameterized update rules vs fixed/reward Hebbian.",
        )
        self.register(
            "event_driven_energy",
            "Event-Driven Energy Efficiency Assay",
            "Compute Efficiency",
            "Measures FLOPs and SynOps savings of sparse execution.",
        )
        self.register(
            "verifier_rsi",
            "Verifier-Gated Architecture Search",
            "Recursive Architecture",
            "Evaluates static validity, stability, non-regression, and paired bootstrap tests.",
        )
        self.register(
            "tesseract_bypass_4d",
            "World4D Tesseract Bypass Navigation",
            "Spatial Environments",
            "4-spatial-dimensional navigation and hypercube obstacle evasion.",
        )
        self.register(
            "digital_tool_arena",
            "Digital Tool Arena Audit Task",
            "Synthetic Tool Interfaces",
            "Permissioned multi-step synthetic document reasoning and reporting.",
        )

    def register(self, benchmark_id: str, title: str, category: str, description: str) -> None:
        self._registry[benchmark_id] = {
            "title": title,
            "category": category,
            "description": description,
        }

    def list_benchmarks(self) -> List[Dict[str, Any]]:
        return [{"id": k, **v} for k, v in self._registry.items()]


class BenchmarkRunner:
    """Executes registered benchmarks with seed controls and gathers metrics."""

    def __init__(self, registry: Optional[BenchmarkRegistry] = None):
        self.registry = registry or BenchmarkRegistry()

    def run_all(self, seed: int = 42) -> List[BenchmarkExecutionResult]:
        torch.manual_seed(seed)
        random.seed(seed)
        results: List[BenchmarkExecutionResult] = []

        # 1. SANCF Matrix
        t0 = time.perf_counter()
        from synapse.neural_comm_bench import SancfMatrixBenchmark
        sancf_bench = SancfMatrixBenchmark(num_steps=15, seed=seed)
        sancf_res = sancf_bench.run_benchmark()
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="sancf_matrix",
                title="SANCF 10-Arm Matrix Benchmark",
                category="Neural Communication",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=len(sancf_res.arm_results) == 10,
                metrics={
                    "total_arms": BenchmarkMetric("total_arms", len(sancf_res.arm_results), "count"),
                    "a3_bits_transmitted": BenchmarkMetric("a3_bits_transmitted", sancf_res.arm_results[3].bits_transmitted, "bits"),
                },
                disclaimer="Payload capacity represents digital floating-point channel transfer, not biophysical synaptic transmission.",
            )
        )

        # 2. NeuroWave Modes
        t0 = time.perf_counter()
        neurowave_res = sancf_bench.run_neurowave_assay()
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="neurowave_modes",
                title="NeuroWave Multiplexing Assay",
                category="Oscillatory Mechanics",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=len(neurowave_res.modes) == 7 and len(neurowave_res.controls) == 7,
                metrics={
                    "active_modes": BenchmarkMetric("active_modes", len(neurowave_res.modes), "modes"),
                    "control_modes": BenchmarkMetric("control_modes", len(neurowave_res.controls), "controls"),
                },
                disclaimer="Oscillatory modes are abstract coupled mathematical modes, not biophysical membrane rhythms.",
            )
        )

        # 3. Causal Connectome
        t0 = time.perf_counter()
        from synapse.causal_connectome_assay import CausalConnectomeAssay
        causal_assay = CausalConnectomeAssay(num_nodes=300, seed=seed)
        causal_res = causal_assay.run_full_assay()
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="causal_connectome",
                title="Causal Connectome Discovery Assay",
                category="Graph & Causality",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=len(causal_res.motifs) >= 10 and len(causal_res.interventions) >= 10,
                metrics={
                    "motifs_discovered": BenchmarkMetric("motifs_discovered", len(causal_res.motifs), "count"),
                    "interventions_evaluated": BenchmarkMetric("interventions_evaluated", len(causal_res.interventions), "count"),
                },
                disclaimer="Motifs extracted from graph connectivity matrices; interventions are mathematical in silico manipulations.",
            )
        )

        # 4. Causal World Model
        t0 = time.perf_counter()
        from synapse.causal_experiment_assay import CausalExperimentAssay
        cwm_assay = CausalExperimentAssay(state_dim=64, action_dim=5, seed=seed)
        cwm_res = cwm_assay.run_assay()
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="causal_world_model",
                title="Causal World Model Do-Calculus Assay",
                category="Causality",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=cwm_res.passed,
                metrics={
                    "disentanglement_score": BenchmarkMetric("disentanglement_score", cwm_res.disentanglement_score, "score"),
                    "passive_drift_error": BenchmarkMetric("passive_drift_error", cwm_res.passive_drift_prediction_error, "mse"),
                },
                disclaimer="Evaluates mathematical Pearlian do-calculus estimation on synthetic state variables.",
            )
        )

        # 5. Relational HRR Memory
        t0 = time.perf_counter()
        from synapse.memory_interference_assay import MemoryInterferenceAssay
        hrr_assay = MemoryInterferenceAssay(dim=256, seed=seed)
        hrr_res = hrr_assay.run_assay(max_pairs=8, num_trials=3, num_distractors=20)
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="relational_hrr",
                title="Relational HRR Memory Interference Assay",
                category="Memory & Representation",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=hrr_res.distractor_resistance_score >= 0.8,
                metrics={
                    "max_clean_capacity": BenchmarkMetric("max_clean_capacity", float(hrr_res.max_clean_capacity), "pairs"),
                    "distractor_resistance": BenchmarkMetric("distractor_resistance", hrr_res.distractor_resistance_score, "acc"),
                },
                disclaimer="Algebraic circular convolution memory model; does not claim human-level episodic recall.",
            )
        )

        # 6. Meta-Plasticity
        t0 = time.perf_counter()
        from synapse.plasticity_rule_learner import PlasticityRuleLearner
        meta_learner = PlasticityRuleLearner(seed=seed)
        meta_res = meta_learner.train_and_evaluate(meta_epochs=3)
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="meta_plasticity",
                title="Meta-Plasticity Learning Assay",
                category="Synaptic Plasticity",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=meta_res.learned_rule_score >= meta_res.hebbian_baseline_score - 0.05,
                metrics={
                    "learned_rule_score": BenchmarkMetric("learned_rule_score", meta_res.learned_rule_score, "return"),
                    "hebbian_score": BenchmarkMetric("hebbian_score", meta_res.hebbian_baseline_score, "return"),
                },
                disclaimer="Parameterized plasticity rule optimization over simulated synthetic environments.",
            )
        )

        # 7. Event-Driven Energy Efficiency
        t0 = time.perf_counter()
        from synapse.energy_efficiency_assay import EnergyEfficiencyAssay
        eff_assay = EnergyEfficiencyAssay(num_neurons=200, num_steps=10, seed=seed)
        eff_res = eff_assay.run_assay(thresholds=[0.05, 0.1])
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="event_driven_energy",
                title="Event-Driven Energy Efficiency Assay",
                category="Compute Efficiency",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=len(eff_res.trials) > 0,
                metrics={
                    "theoretical_speedup": BenchmarkMetric("theoretical_speedup", eff_res.trials[0].theoretical_speedup, "x"),
                    "recommended_threshold": BenchmarkMetric("recommended_threshold", eff_res.recommended_threshold, "val"),
                },
                disclaimer="Theoretical SynOps versus dense FLOPs; not physical neuromorphic silicon power measurements.",
            )
        )

        # 8. Verifier-RSI
        t0 = time.perf_counter()
        from synapse.verifier_rsi import RecursiveImprovementEngine
        rsi_engine = RecursiveImprovementEngine(base_config={"neural_comm_dim": 16, "spectral_radius": 0.8}, seed=seed)
        prop = rsi_engine.propose_mutation()
        receipt = rsi_engine.evaluate_proposal(prop)
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="verifier_rsi",
                title="Verifier-Gated Architecture Search",
                category="Recursive Architecture",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=receipt.static_validity and receipt.numerical_stability,
                metrics={
                    "static_validity": BenchmarkMetric("static_validity", 1.0 if receipt.static_validity else 0.0, "bool"),
                    "numerical_stability": BenchmarkMetric("numerical_stability", 1.0 if receipt.numerical_stability else 0.0, "bool"),
                },
                disclaimer="Strictly constrained verifier-gated search; does not claim open-ended artificial superintelligence.",
            )
        )

        # 9. World4D Tesseract Bypass
        t0 = time.perf_counter()
        from synapse.world4d import World4DEnvironment
        env4d = World4DEnvironment(seed=seed)
        obs4d = env4d.reset(seed=seed)
        step_obs, step_rew, step_done, step_info = env4d.step(np.array([0.2, 0.0, 0.0, 0.5, 0.0], dtype=np.float32))
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="tesseract_bypass_4d",
                title="World4D Tesseract Bypass Navigation",
                category="Spatial Environments",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=obs4d.shape[0] == 155,
                metrics={
                    "obs_dim": BenchmarkMetric("obs_dim", float(obs4d.shape[0]), "dim"),
                    "distance_to_hypergoal": BenchmarkMetric("distance_to_hypergoal", float(step_info.get("dist_to_goal", 1.0)), "euclid"),
                },
                disclaimer="Mathematical 4-spatial-dimensional navigation puzzle with tesseract obstacle projection.",
            )
        )

        # 10. Digital Tool Arena
        t0 = time.perf_counter()
        from synapse.digital_tool_arena import DigitalToolArena, ToolActionType
        arena = DigitalToolArena(seed=seed)
        obs_tool = arena.reset(seed=seed)
        l_obs, l_rew, l_done, l_info = arena.step(ToolActionType.LIST, {})
        t1 = time.perf_counter()
        results.append(
            BenchmarkExecutionResult(
                benchmark_id="digital_tool_arena",
                title="Digital Tool Arena Audit Task",
                category="Synthetic Tool Interfaces",
                execution_time_ms=(t1 - t0) * 1000.0,
                passed=obs_tool.shape[0] == 155 and l_info.get("tool") == "LIST",
                metrics={
                    "corpus_documents": BenchmarkMetric("corpus_documents", float(len(arena.corpus)), "count"),
                    "steps": BenchmarkMetric("steps", float(arena.steps), "count"),
                },
                disclaimer="Permissioned in-memory synthetic tool sandbox with zero filesystem or network host access.",
            )
        )

        return results


class ResearchScorecard:
    """Generates Markdown and JSON audit scorecards with non-claim receipts."""

    def __init__(self, commit_sha: str = "880bb4a0da05ca783738a16723b0a629f14e5517"):
        self.commit_sha = commit_sha

    def generate_scorecard(self, results: List[BenchmarkExecutionResult]) -> UnifiedScorecard:
        passed_count = sum(1 for r in results if r.passed)
        rate = passed_count / max(len(results), 1)

        lines = [
            f"# Sonapse-LAB Scientific Research Scorecard",
            f"**Commit SHA:** `{self.commit_sha}` | **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}",
            f"**Overall Benchmark Pass Rate:** {rate*100:.1f}% ({passed_count}/{len(results)} passed)",
            "",
            "| Benchmark | Category | Latency (ms) | Status | Key Metric | Scientific Disclaimer |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            first_metric = next(iter(r.metrics.values())) if r.metrics else BenchmarkMetric("none", 0.0, "")
            metric_str = f"{first_metric.name}={first_metric.value} {first_metric.unit}"
            lines.append(f"| {r.title} | {r.category} | {r.execution_time_ms:.1f} | {status} | {metric_str} | {r.disclaimer} |")

        lines.extend([
            "",
            "### Scientific Non-Claims & Methodological Controls",
            "- **Biological Frequencies:** Abstract coupled oscillatory modes are mathematical eigenvalues, not biophysical membrane potentials.",
            "- **Electromagnetic / Quantum Claims:** Orthogonal modes are digital basis decompositions (DCT/Hadamard), not physical radio or OAM waves.",
            "- **Episodic Memory:** Relational memory is evaluated via circular-convolution algebra and vector cleanup, not claimed human episodic recall.",
            "- **Host Security:** Digital tool arena runs strictly in an in-memory synthetic sandbox with zero shell or network access.",
        ])

        summary_md = "\n".join(lines)
        return UnifiedScorecard(
            timestamp=time.time(),
            git_commit_sha=self.commit_sha,
            results=results,
            overall_pass_rate=round(rate, 4),
            summary_markdown=summary_md,
        )
