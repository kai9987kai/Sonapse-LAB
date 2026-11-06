import numpy as np
import pytest

from synapse.evidence_engine import ResearchEvidenceEngine, bootstrap_confidence_interval


@pytest.mark.parametrize('baseline,other', [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), ([[1, 2]], [1, 2])])
def test_ablation_rejects_invalid_or_unpaired_measurements(baseline, other):
    with pytest.raises(ValueError):
        ResearchEvidenceEngine().run_ablation_suite(baseline, {'control': other})


def test_paired_intervals_preserve_regressions_and_bind_raw_data():
    engine = ResearchEvidenceEngine()
    result = engine.run_ablation_suite([1., 2., 3., 4.], {'improved': [2., 3., 4., 5.]}, source_label='measured test fixture')
    row = result['ablations']['improved']
    assert row['paired_mean_difference'] == -1.
    assert row['paired_ci_95'] == [-1., -1.]
    assert row['relative_performance_drop_pct'] < 0
    assert result['source_label'] == 'measured test fixture'
    other = ResearchEvidenceEngine().run_ablation_suite([2., 3., 4., 5.], {'improved': [3., 4., 5., 6.]}, source_label='measured test fixture')
    assert other['provenance_hash'] != result['provenance_hash']
    report = engine.generate_markdown_report()
    assert 'paired' in report.lower()
    assert 'p < 0.05' not in report


def test_single_pair_has_no_inferential_success_claim():
    result = ResearchEvidenceEngine().run_ablation_suite([1.], {'control': [0.]})
    assert result['ablations']['control']['p_value'] is None
    assert result['ablations']['control']['statistically_significant'] is None
    with pytest.raises(ValueError):
        bootstrap_confidence_interval([])


def test_evidence_capability_requires_measured_inputs_or_runs_study(monkeypatch):
    import synapse.adaptation_study as study
    from synapse.capabilities import invoke
    monkeypatch.setattr(study, 'run_adaptation_study', lambda *a, **kw: {'status': 'complete', 'summary': {}, 'scope': 'test fixture'})
    result = invoke('evidence.research_benchmark', {})
    assert result['status'] == 'ok'
    assert result['evidence_kind'] == 'native_world_development_study'
    supplied = invoke('evidence.research_benchmark', {'baseline_scores': [1., 2.], 'ablated_scores': {'control': [0., 1.]}, 'source_label': 'user measured cohort'})
    assert supplied['status'] == 'ok'
    assert supplied['suite']['source_label'] == 'user measured cohort'

