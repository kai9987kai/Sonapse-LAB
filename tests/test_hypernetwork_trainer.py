"""Tests for Synthetic Failure Generation & Repair Hypernetwork Benchmarks."""
import pytest
from synapse.hypernetwork_trainer import SyntheticFailureGenerator, HypernetworkRepairBenchmark


def test_synthetic_failure_generator():
    gen = SyntheticFailureGenerator(seed=42)
    cases = gen.generate_failure_cases(count=20)
    assert len(cases) == 20
    for c in cases:
        assert c.category in ['deleted_synapses', 'disrupted_motor', 'degraded_memory', 'corrupted_sensory']
        assert 0.0 <= c.damaged_perf <= c.pre_damage_perf


def test_hypernetwork_repair_benchmark():
    gen = SyntheticFailureGenerator(seed=42)
    cases = gen.generate_failure_cases(count=20)

    benchmark = HypernetworkRepairBenchmark(seed=42)
    train_res = benchmark.train_hypernetwork(cases[:15], epochs=1)
    assert 'initial_loss' in train_res
    assert 'final_loss' in train_res

    res = benchmark.run_comprehensive_benchmark(cases[15:])
    assert res['test_cases_evaluated'] == 5
    assert 'hypernetwork' in res['benchmark_comparison']
    assert 'random_search' in res['benchmark_comparison']
    assert 'gradient_finetuning' in res['benchmark_comparison']
    assert 'heuristic_repair' in res['benchmark_comparison']
    assert res['hypernetwork_speedup_vs_gradient'] > 1.0
