import numpy as np
import pytest

from synapse.neurowave_x2.stress_tasks import (
    StrictSeparationBenchmark,
    CompositionalInformationBenchmark,
    CommunicationStressHarness,
)


def test_strict_separation_benchmark():
    bench = StrictSeparationBenchmark(cue_dim=6, action_dim=3, seed=42)
    obs = bench.reset()
    assert obs.shape == (6,)

    # Correct action
    rew, corr, info = bench.evaluate_receiver_action(bench.expected_action)
    assert rew == 1.0
    assert corr is True

    # Incorrect action
    wrong_action = (bench.expected_action + 1) % 3
    rew_bad, corr_bad, _ = bench.evaluate_receiver_action(wrong_action)
    assert rew_bad == -0.5
    assert corr_bad is False


def test_compositional_information_benchmark():
    bench = CompositionalInformationBenchmark(vector_dim=8, seed=42)
    x, target = bench.generate_sample()
    assert x.shape == (8,)
    assert target.shape == (14,)  # 8 linear + 6 non-linear relations

    # Perfect reconstruction
    res_perfect = bench.evaluate_reconstruction(target, target)
    assert res_perfect["total_mse"] == 0.0
    assert res_perfect["compositional_accuracy"] == 1.0

    # Perturbed reconstruction
    decoded = target + 0.1
    res_perturbed = bench.evaluate_reconstruction(target, decoded)
    assert res_perturbed["total_mse"] > 0.0
    assert res_perturbed["compositional_accuracy"] < 1.0


def test_communication_stress_harness():
    harness = CommunicationStressHarness(seed=42)

    # 1. Packet drop rates
    drops = [harness.should_drop_packet(base_loss_rate=0.2, burst_loss_prob=0.0) for _ in range(500)]
    drop_fraction = sum(drops) / 500.0
    assert 0.10 <= drop_fraction <= 0.30

    # 2. Channel noise SNR
    payload = np.ones(100, dtype=np.float32)
    noisy = harness.apply_channel_noise(payload, snr_db=10.0)
    assert noisy.shape == payload.shape
    diff = np.abs(noisy - payload)
    assert np.mean(diff) > 0.0
