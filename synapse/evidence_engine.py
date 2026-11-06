"""Paired statistics for supplied measurements; this module never invents scores."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t


def _scores(values, name='scores'):
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a nonempty finite numeric vector') from exc
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
        raise ValueError(f'{name} must be a nonempty finite numeric vector')
    return arr


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Pooled-SD standardized mean difference (not paired Cohen's dz)."""
    g1, g2 = _scores(group1), _scores(group2)
    n1, n2 = len(g1), len(g2)
    if min(n1, n2) < 2:
        return 0.0
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1)+(n2-1)*g2.var(ddof=1))/(n1+n2-2))
    return float((g1.mean()-g2.mean())/pooled) if pooled > 1e-12 else 0.0


def bootstrap_confidence_interval(data: np.ndarray, num_bootstrap: int = 2000, alpha: float = .05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap of measurement units, not individual time steps."""
    arr = _scores(data)
    if type(num_bootstrap) is not int or not 1 <= num_bootstrap <= 100000 or not 0 < alpha < 1:
        raise ValueError('invalid bootstrap configuration')
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(num_bootstrap)])
    return tuple(float(x) for x in np.quantile(means, [alpha/2, 1-alpha/2]))


class ResearchEvidenceEngine:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.experiments: Dict[str, Any] = {}

    def run_ablation_suite(self, baseline_scores: List[float], ablated_scores_dict: Dict[str, List[float]], source_label: str = 'caller-supplied measurements; provenance not independently verified') -> Dict[str, Any]:
        """Compare exactly paired units supplied in identical order by the caller.

        Positive differences mean baseline exceeds control. Bootstrap intervals are
        descriptive with small samples. Two-sided paired t tests assume independent
        units and approximately normal differences; p values are unadjusted.
        """
        base = _scores(baseline_scores, 'baseline')
        if not isinstance(ablated_scores_dict, dict) or not ablated_scores_dict:
            raise ValueError('at least one measured comparison is required')
        if not isinstance(source_label, str) or not source_label.strip():
            raise ValueError('source_label must be nonempty')
        controls = {name: _scores(values, str(name)) for name, values in ablated_scores_dict.items()}
        if any(not isinstance(name, str) or not name for name in controls):
            raise ValueError('comparison names must be nonempty strings')
        if any(len(arr) != len(base) for arr in controls.values()):
            raise ValueError('paired measurements must have identical lengths and unit order')
        rows = {}
        for name, arr in controls.items():
            diffs = base-arr
            sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.
            t_stat, p_value = None, None
            if len(diffs) > 1 and sd > 1e-12:
                t_stat = float(diffs.mean()/(sd/np.sqrt(len(diffs))))
                p_value = float(2*student_t.sf(abs(t_stat), df=len(diffs)-1))
            # Constant paired differences have no estimated variance. Do not turn
            # that degeneracy into a claim of infinite statistical evidence.
            rows[name] = {
                'mean_score': float(arr.mean()), 'std_score': float(arr.std()),
                'ci_95': list(bootstrap_confidence_interval(arr)),
                'paired_mean_difference': float(diffs.mean()),
                'paired_ci_95': list(bootstrap_confidence_interval(diffs)),
                'relative_performance_drop_pct': float(diffs.mean()/abs(base.mean())*100) if abs(base.mean()) > 1e-12 else None,
                'cohens_d': compute_cohens_d(base, arr),
                't_stat': t_stat, 'p_value': p_value,
                'statistically_significant': bool(p_value < .05) if p_value is not None else None,
            }
        raw = {'baseline': base.tolist(), 'comparisons': {k: v.tolist() for k, v in controls.items()}, 'source_label': source_label}
        result = {
            'baseline': {'mean_score': float(base.mean()), 'std_score': float(base.std()), 'ci_95': list(bootstrap_confidence_interval(base))},
            'ablations': rows, 'paired_units': len(base), 'source_label': source_label,
            'raw_measurements': raw,
            'statistical_method': 'paired-unit percentile bootstrap; two-sided paired Student t test when variance estimable; unadjusted exploratory p values',
            'provenance_hash': hashlib.sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        }
        self.experiments['ablation_suite'] = result
        return result

    def generate_markdown_report(self, title: str = 'SYNAPSE Scientific Evidence Report') -> str:
        lines = [f'# {title}', '', 'Paired supplied measurements. Component names and provenance are provided by the caller.', '']
        if 'ablation_suite' not in self.experiments:
            return '\n'.join(lines)
        result = self.experiments['ablation_suite']
        base = result['baseline']
        lines.extend([f"Source: {result['source_label']}", '', f"Baseline mean: {base['mean_score']:.4f}; paired units: {result['paired_units']}.", '', '## Component Ablation Matrix', '',
                      '| Comparison | Mean | Baseline minus comparison | Paired 95% interval | Exploratory p |', '|---|---:|---:|---|---:|'])
        for name, row in result['ablations'].items():
            low, high = row['paired_ci_95']
            p = f"{row['p_value']:.4g}" if row['p_value'] is not None else 'not estimable'
            lines.append(f"| `{name}` | {row['mean_score']:.4f} | {row['paired_mean_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] | {p} |")
        lines.extend(['', result['statistical_method']+'. Small samples limit inference; no multiplicity correction is applied.', '', f"Raw-measurement SHA-256: `{result['provenance_hash']}`", ''])
        return '\n'.join(lines)
