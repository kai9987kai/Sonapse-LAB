"""Tests for Neuromorphic Hardware Efficiency & Pareto Frontiers."""
import pytest
from synapse.efficiency import NeuromorphicEfficiencyBenchmark


def test_neuromorphic_efficiency_benchmark():
    bench = NeuromorphicEfficiencyBenchmark(in_features=8, num_neurons=32, seed=42)
    res = bench.run_benchmark_suite(steps=20)

    assert res['num_neurons'] == 32
    assert 'dense_rate' in res['profiles']
    assert 'lif_spiking' in res['profiles']
    assert 'adex_spiking' in res['profiles']
    assert len(res['pareto_frontier']) > 0
    assert res['spiking_synops_efficiency_ratio'] > 1.0
    assert res['energy_savings_ratio'] > 1.0
