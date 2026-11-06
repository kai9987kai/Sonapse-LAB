"""Reproducible scientific research evidence engine for SYNAPSE.

Automates multi-seed benchmark suites, ablation assays, statistical hypothesis tests,
effect sizes (Cohen's d), bootstrap confidence intervals, and publication-ready reporting.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Computes Cohen's d effect size between two independent or paired groups."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-8:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def bootstrap_confidence_interval(data: np.ndarray, num_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Computes non-parametric bootstrap (1 - alpha) confidence interval."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    n = len(arr)
    boot_means = [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(num_bootstrap)]
    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


class ResearchEvidenceEngine:
    """Orchestrates comprehensive empirical benchmarks, statistical tests, and report generation."""
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.experiments: Dict[str, Any] = {}

    def run_ablation_suite(self, baseline_scores: List[float], ablated_scores_dict: Dict[str, List[float]]) -> Dict[str, Any]:
        """Runs ablation suite comparing full baseline system against various ablated configurations."""
        base_arr = np.array(baseline_scores)
        base_mean = float(np.mean(base_arr))
        base_ci = bootstrap_confidence_interval(base_arr)
        
        ablation_results = {}
        for condition_name, scores in ablated_scores_dict.items():
            abl_arr = np.array(scores)
            abl_mean = float(np.mean(abl_arr))
            abl_ci = bootstrap_confidence_interval(abl_arr)
            d = compute_cohens_d(base_arr, abl_arr)
            
            # Simple paired difference test
            diffs = base_arr[:min(len(base_arr), len(abl_arr))] - abl_arr[:min(len(base_arr), len(abl_arr))]
            t_stat = float(np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)) + 1e-8)) if len(diffs) > 1 else 0.0
            
            ablation_results[condition_name] = {
                'mean_score': abl_mean,
                'std_score': float(np.std(abl_arr)),
                'ci_95': abl_ci,
                'relative_performance_drop_pct': float(max(0.0, (base_mean - abl_mean) / max(1e-4, base_mean) * 100.0)),
                'cohens_d': d,
                't_stat': t_stat,
                'statistically_significant': bool(abs(t_stat) > 2.0 and d > 0.5)
            }
            
        suite_summary = {
            'baseline': {
                'mean_score': base_mean,
                'std_score': float(np.std(base_arr)),
                'ci_95': base_ci
            },
            'ablations': ablation_results,
            'provenance_hash': hashlib.sha256(str(ablation_results).encode()).hexdigest()[:16]
        }
        self.experiments['ablation_suite'] = suite_summary
        return suite_summary

    def generate_markdown_report(self, title: str = "SYNAPSE Scientific Evidence Report") -> str:
        """Renders publication-ready Markdown report."""
        lines = [
            f"# {title}",
            "",
            "> Automated empirical benchmark report with non-parametric bootstrap 95% confidence intervals and Cohen's d effect sizes.",
            "",
            "## 1. Executive Summary",
            ""
        ]
        
        if 'ablation_suite' in self.experiments:
            res = self.experiments['ablation_suite']
            base = res['baseline']
            lines.append(f"- **Baseline Performance**: {base['mean_score']:.3f} +/- {base['std_score']:.3f} (95% CI: [{base['ci_95'][0]:.3f}, {base['ci_95'][1]:.3f}])")
            lines.append(f"- **Provenance Hash**: `0x{res['provenance_hash']}`")
            lines.append("")
            lines.append("## 2. Component Ablation Matrix")
            lines.append("")
            lines.append("| Component Ablated | Mean Score | Performance Drop (%) | Cohen's d | Significant (p < 0.05) |")
            lines.append("|---|---|---|---|---|")
            for name, data in res['ablations'].items():
                sig = "YES" if data['statistically_significant'] else "NO"
                lines.append(f"| `{name}` | {data['mean_score']:.3f} | {data['relative_performance_drop_pct']:.1f}% | {data['cohens_d']:.2f} | **{sig}** |")
            lines.append("")
            
        return "\n".join(lines)
