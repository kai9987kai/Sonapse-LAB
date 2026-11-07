"""Illustrative verifier-gated RSI scaffold with simulated outcomes.

Grounding:
Implements the AIDE² / Darwin Gödel Machine / AlphaEvolve pattern for safe, bounded RSI:
1. Bounded proposal generation (parameter adapters, configuration mutations, routing weights).
2. Immutable Verification Pipeline:
   - Schema & static constraints (AST bounds, parameter clamps)
   - Deterministic test suite (isolated execution)
   - Support-world evaluation
   - Fresh held-out query-world evaluation (preventing evaluation leakage)
   - Lower Confidence Bound dominance: LCB(Delta) = mean(Delta) - t_alpha * (std / sqrt(n)) > epsilon
   - Canary activation & regression pass (regression rate <= 1%)
3. Cryptographic Provenance Receipts (SHA-256 digests, seeds, commit hashes).
4. Dual Archive: Promoted Success Lineage vs. Failure Archive for diagnostic learning.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class VerificationStage(str, Enum):
    SCHEMA_CHECK = "schema_check"
    STATIC_ANALYSIS = "static_analysis"
    DETERMINISTIC_TESTS = "deterministic_tests"
    SUPPORT_EVAL = "support_eval"
    QUERY_EVAL = "query_eval"
    DOMINANCE_LCB = "dominance_lcb"
    CANARY_REGRESSION = "canary_regression"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class CandidatePatch:
    patch_id: str
    parent_hash: str
    patch_type: str  # 'config', 'adapter', 'routing', 'hyperparameter'
    diff: Dict[str, Any]
    mutation_scale: float = 1.0
    generator_id: str = "aide2_alphaevolve_scaffold"
    timestamp: float = field(default_factory=time.time)

    @property
    def digest(self) -> str:
        payload = json.dumps({
            'patch_id': self.patch_id,
            'parent_hash': self.parent_hash,
            'patch_type': self.patch_type,
            'diff': self.diff,
            'mutation_scale': self.mutation_scale,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class VerificationReceipt:
    patch_id: str
    parent_hash: str
    patch_digest: str
    passed_all: bool
    final_stage: VerificationStage
    stage_results: Dict[str, Dict[str, Any]]
    delta_mean: float
    lcb_score: float
    regression_rate: float
    receipt_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    simulated: bool = True
    measured: bool = False
    evidence_kind: str = 'illustrative'

    def finalize(self) -> VerificationReceipt:
        payload = json.dumps({
            'patch_id': self.patch_id,
            'parent_hash': self.parent_hash,
            'patch_digest': self.patch_digest,
            'passed_all': self.passed_all,
            'final_stage': self.final_stage.value,
            'delta_mean': self.delta_mean,
            'lcb_score': self.lcb_score,
            'regression_rate': self.regression_rate,
            'timestamp': self.timestamp,
            'stage_results': self.stage_results,
            'simulated': self.simulated,
            'measured': self.measured,
            'evidence_kind': self.evidence_kind,
        }, sort_keys=True)
        self.receipt_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self


class BoundedRSIEngine:
    """Orchestrates verifier-gated evolutionary RSI cycles under strict isolation."""

    def __init__(
        self,
        epsilon_threshold: float = 0.05,
        alpha_t: float = 1.96,  # 95% confidence interval multiplier
        max_regression_rate: float = 0.01,
        seed: int = 42,
    ):
        self.epsilon_threshold = epsilon_threshold
        self.alpha_t = alpha_t
        self.max_regression_rate = max_regression_rate
        self.rng = np.random.default_rng(seed)

        # Evolutionary Archives
        self.success_lineage: List[Dict[str, Any]] = []
        self.failure_archive: List[Dict[str, Any]] = []
        self.receipts: List[VerificationReceipt] = []

        # Root policy state
        self.current_parent_hash = hashlib.sha256(b"sonapse_root_policy_v1").hexdigest()
        self.current_config: Dict[str, Any] = {
            'learning_rate': 0.001,
            'neurogenesis_grow_threshold': 0.35,
            'neurogenesis_prune_threshold': 0.02,
            'mimomix_ponder_cost': 0.02,
            'mimomix_deliberate_threshold': 0.45,
            'mpc_horizon': 5,
            'uncertainty_penalty': 0.3,
            'fast_memory_inner_lr': 0.05,
        }

    def generate_candidate(
        self,
        patch_type: str = "config",
        mutation_scale: Optional[float] = None
    ) -> CandidatePatch:
        """Generates a bounded mutation candidate constrained by safety limits."""
        scale = mutation_scale if mutation_scale is not None else float(self.rng.uniform(0.5, 1.5))
        patch_id = f"patch_{len(self.receipts) + 1}_{int(time.time()*1000) % 100000}"

        diff: Dict[str, Any] = {}
        if patch_type == "config":
            param_key = str(self.rng.choice(list(self.current_config.keys())))
            cur_val = self.current_config[param_key]
            # Bounded multiplicative or additive perturbation
            mult = float(self.rng.choice([0.8, 0.9, 1.1, 1.25])) ** scale
            new_val = float(np.clip(cur_val * mult, 0.0001, 10.0))
            diff[param_key] = round(new_val, 6)
        elif patch_type == "routing":
            diff['expert_temperature'] = round(float(self.rng.uniform(0.5, 1.5)), 4)
            diff['dormant_novelty_gate'] = round(float(self.rng.uniform(0.2, 0.8)), 4)
        elif patch_type == "adapter":
            diff['adapter_rank'] = int(self.rng.choice([2, 4, 8]))
            diff['adapter_scale'] = round(float(self.rng.uniform(0.1, 0.5)), 3)
        elif patch_type == "connectome_graft":
            diff['inter_graft_gain'] = round(float(self.rng.uniform(0.8, 1.25)), 4)
            diff['cross_region_prune_threshold'] = round(float(self.rng.uniform(0.01, 0.05)), 4)
            diff['da_receptor_sensitivity'] = round(float(self.rng.uniform(0.7, 1.3)), 4)
            diff['octopamine_locomotion_gain'] = round(float(self.rng.uniform(0.8, 1.4)), 4)
        else:
            diff['delta_lr'] = round(float(self.rng.uniform(-0.0005, 0.0005)), 6)

        return CandidatePatch(
            patch_id=patch_id,
            parent_hash=self.current_parent_hash,
            patch_type=patch_type,
            diff=diff,
            mutation_scale=scale
        )

    def verify_and_eval(
        self,
        candidate: CandidatePatch,
        num_support_trials: int = 5,
        num_query_trials: int = 5,
        simulated_parent_perf: float = 1.0,
    ) -> VerificationReceipt:
        """Runs an explicitly simulated demo; no environment or unit tests execute."""
        for name, value in [('num_support_trials', num_support_trials), ('num_query_trials', num_query_trials)]:
            if type(value) is not int or not 1 <= value <= 1000:
                raise ValueError(f'{name} must be an integer in [1,1000]')
        stage_results: Dict[str, Dict[str, Any]] = {}

        # Stage 1: Schema Check
        if not candidate.diff or not isinstance(candidate.diff, dict):
            return self._reject(candidate, VerificationStage.SCHEMA_CHECK, stage_results, "Empty or invalid diff format")
        if candidate.parent_hash != self.current_parent_hash:
            return self._reject(candidate, VerificationStage.SCHEMA_CHECK, stage_results, 'stale or unknown parent hash')
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v)
               for v in candidate.diff.values()):
            return self._reject(candidate, VerificationStage.SCHEMA_CHECK, stage_results, 'diff values must be finite numeric values')
        stage_results['schema_check'] = {'status': 'pass', 'keys_modified': list(candidate.diff.keys())}

        # Stage 2: Static Analysis / Range Bounds
        forbidden_ranges = {
            'learning_rate': (1e-6, 0.1),
            'neurogenesis_grow_threshold': (0.05, 0.95),
            'neurogenesis_prune_threshold': (0.001, 0.2),
            'mimomix_ponder_cost': (0.001, 0.5),
            'mpc_horizon': (1, 20),
            'uncertainty_penalty': (0.01, 2.0),
            'inter_graft_gain': (0.1, 5.0),
            'cross_region_prune_threshold': (0.001, 0.1),
            'da_receptor_sensitivity': (0.1, 3.0),
            'octopamine_locomotion_gain': (0.1, 3.0),
        }
        for k, val in candidate.diff.items():
            if k in forbidden_ranges:
                low, high = forbidden_ranges[k]
                if not (low <= val <= high):
                    return self._reject(candidate, VerificationStage.STATIC_ANALYSIS, stage_results, f"Value {val} for {k} out of bounds [{low}, {high}]")
        stage_results['static_analysis'] = {'status': 'pass'}

        # Stage 3: Deterministic Tests (Simulated unit test suite)
        test_seed = int(candidate.digest[:8], 16) % 10000
        test_rng = np.random.default_rng(test_seed)
        unit_test_passed = bool(test_rng.uniform(0, 1) > 0.05)  # 95% pass base
        if not unit_test_passed:
            return self._reject(candidate, VerificationStage.DETERMINISTIC_TESTS, stage_results, "Deterministic invariant unit tests failed")
        stage_results['deterministic_tests'] = {'status': 'pass', 'invariants_checked': 12}

        # Stage 4: Support-World Evaluation
        # Simulates support world performance delta
        support_noise = test_rng.normal(0.05, 0.08, size=num_support_trials)
        support_deltas = [float(x) for x in support_noise]
        mean_support_delta = float(np.mean(support_deltas))
        stage_results['support_eval'] = {
            'status': 'pass',
            'support_trials': num_support_trials,
            'mean_delta': round(mean_support_delta, 4)
        }

        # Stage 5: Fresh Held-out Query Evaluation (disjoint seeds to prevent leakage)
        query_seed = (test_seed + 99991) % 100000
        query_rng = np.random.default_rng(query_seed)
        query_deltas = query_rng.normal(mean_support_delta * 0.9, 0.06, size=num_query_trials)
        mean_query_delta = float(np.mean(query_deltas))
        std_query_delta = float(np.std(query_deltas, ddof=1)) if num_query_trials > 1 else 0.05

        stage_results['query_eval'] = {
            'status': 'pass',
            'query_trials': num_query_trials,
            'mean_delta': round(mean_query_delta, 4),
            'std_delta': round(std_query_delta, 4)
        }

        # Stage 6: Dominance Check with Lower Confidence Bound (LCB)
        # LCB(Delta) = mean(Delta) - t_alpha * (std / sqrt(n))
        se = std_query_delta / math.sqrt(num_query_trials)
        lcb = mean_query_delta - self.alpha_t * se
        dominates = lcb > self.epsilon_threshold

        stage_results['dominance_lcb'] = {
            'lcb_score': round(lcb, 4),
            'threshold': self.epsilon_threshold,
            'dominates': dominates
        }

        if not dominates:
            return self._reject(
                candidate,
                VerificationStage.DOMINANCE_LCB,
                stage_results,
                f"LCB score {lcb:.4f} did not exceed epsilon threshold {self.epsilon_threshold:.4f}",
                delta_mean=mean_query_delta,
                lcb_score=lcb
            )

        # Stage 7: Canary Activation & Regression Pass (regression rate on baseline benchmarks <= 1%)
        canary_regressions = int(np.sum(query_deltas < -0.15))
        reg_rate = canary_regressions / float(num_query_trials)
        stage_results['canary_regression'] = {
            'regression_rate': reg_rate,
            'threshold': self.max_regression_rate,
            'pass': reg_rate <= self.max_regression_rate
        }

        if reg_rate > self.max_regression_rate:
            return self._reject(
                candidate,
                VerificationStage.CANARY_REGRESSION,
                stage_results,
                f"Regression rate {reg_rate:.2%} exceeded maximum allowed {self.max_regression_rate:.2%}",
                delta_mean=mean_query_delta,
                lcb_score=lcb,
                regression_rate=reg_rate
            )

        # Candidate passed ALL stages -> Promote to versioned child policy!
        # Apply diff to current config
        self.current_config.update(candidate.diff)
        new_parent_hash = candidate.digest
        self.current_parent_hash = new_parent_hash

        receipt = VerificationReceipt(
            patch_id=candidate.patch_id,
            parent_hash=candidate.parent_hash,
            patch_digest=candidate.digest,
            passed_all=True,
            final_stage=VerificationStage.PROMOTED,
            stage_results=stage_results,
            delta_mean=mean_query_delta,
            lcb_score=lcb,
            regression_rate=reg_rate
        ).finalize()

        self.success_lineage.append({
            'generation': len(self.success_lineage) + 1,
            'patch_id': candidate.patch_id,
            'parent_hash': candidate.parent_hash,
            'child_hash': new_parent_hash,
            'diff': candidate.diff,
            'lcb_score': lcb,
            'delta_mean': mean_query_delta,
            'receipt_hash': receipt.receipt_hash,
            'timestamp': receipt.timestamp
        })
        self.receipts.append(receipt)
        return receipt

    def _reject(
        self,
        candidate: CandidatePatch,
        failed_stage: VerificationStage,
        stage_results: Dict[str, Dict[str, Any]],
        reason: str,
        delta_mean: float = 0.0,
        lcb_score: float = 0.0,
        regression_rate: float = 0.0
    ) -> VerificationReceipt:
        """Archives rejected candidate in the failure archive."""
        stage_results[failed_stage.value] = {
            'status': 'failed',
            'rejection_reason': reason
        }
        receipt = VerificationReceipt(
            patch_id=candidate.patch_id,
            parent_hash=candidate.parent_hash,
            patch_digest=candidate.digest,
            passed_all=False,
            final_stage=VerificationStage.REJECTED,
            stage_results=stage_results,
            delta_mean=delta_mean,
            lcb_score=lcb_score,
            regression_rate=regression_rate
        ).finalize()

        self.failure_archive.append({
            'patch_id': candidate.patch_id,
            'patch_type': candidate.patch_type,
            'failed_stage': failed_stage.value,
            'reason': reason,
            'diff': candidate.diff,
            'receipt_hash': receipt.receipt_hash,
            'timestamp': receipt.timestamp
        })
        self.receipts.append(receipt)
        return receipt

    def get_summary(self) -> Dict[str, Any]:
        return {
            'status': 'ok',
            'simulated': True,
            'measured': False,
            'evidence_kind': 'illustrative',
            'scope': 'synthetic verification scaffold; tests and support/query scores are simulated; no measured RSI gain',
            'current_parent_hash': self.current_parent_hash,
            'promoted_count': len(self.success_lineage),
            'failure_count': len(self.failure_archive),
            'total_evaluated': len(self.receipts),
            'promotion_rate': round(len(self.success_lineage) / max(1, len(self.receipts)), 4),
            'active_config': copy.deepcopy(self.current_config),
            'recent_promoted': self.success_lineage[-3:] if self.success_lineage else [],
            'recent_failures': self.failure_archive[-3:] if self.failure_archive else [],
        }
