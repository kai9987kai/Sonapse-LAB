import math
import torch
import numpy as np
import pytest

from synapse.neurowave_x2.compute_profiler import ComputeProfiler
from synapse.neurowave_x2.resource_budget import ResourceBudgetController
from synapse.neurowave_x2.information_metrics import (
    compute_effective_rank,
    compute_message_entropy_estimate,
    compute_communication_sparsity,
    compute_spectral_energy_concentration,
    compute_channel_redundancy,
    compute_rate_distortion_proxy,
)


def test_compute_profiler():
    profiler = ComputeProfiler(history_size=10)
    linear = torch.nn.Linear(10, 5)
    mem = profiler.measure_model_memory(linear)
    assert mem["total_parameters"] == 55  # 10*5 + 5
    assert mem["trainable_parameters"] == 55
    assert mem["parameter_memory_kb"] > 0

    flops = profiler.estimate_step_flops(active_routes=4, message_dim=16, target_size=32, spectral_modes=4)
    assert flops["analytical_flops_per_step"] > 0
    assert flops["analytical_macs_per_step"] == flops["analytical_flops_per_step"] / 2.0

    # Latency recording
    profiler.record_latency(0.005)
    profiler.record_latency(0.007)
    lat = profiler.get_latency_percentiles()
    assert lat["latency_p50_ms"] > 0


def test_resource_budget_controller():
    controller = ResourceBudgetController(max_wire_bits_per_step=1000, max_flops_per_step=5000.0)
    assert controller.check_within_budget(500, 2500.0) is True
    assert controller.check_within_budget(1500, 2500.0) is False
    assert controller.check_within_budget(500, 6000.0) is False

    penalties = controller.compute_communication_loss_penalties(
        total_wire_bits=256,
        total_flops=1000.0,
        mean_latency=0.002,
        residual_norm=0.5,
    )
    assert "loss_comm_total_penalty" in penalties
    assert penalties["loss_comm_total_penalty"].item() > 0


def test_information_metrics():
    # 1. Effective rank
    mat = np.random.randn(8, 16)
    eff_rank = compute_effective_rank(mat)
    assert 1.0 <= eff_rank <= 8.5

    # 2. Message entropy
    msgs = np.random.uniform(-0.8, 0.8, size=(10, 8))
    ent = compute_message_entropy_estimate(msgs, num_bins=16)
    assert ent > 0.0

    # 3. Sparsity
    curr = np.array([0.0, 0.0001, 0.5, -0.2])
    sp = compute_communication_sparsity(curr, threshold=1e-3)
    assert sp == 0.5

    # 4. Spectral energy concentration
    modes = np.array([10.0, 5.0, 1.0, 0.5])
    conc = compute_spectral_energy_concentration(modes)
    assert 0.8 < conc <= 1.0

    # 5. Channel redundancy
    red = compute_channel_redundancy(mat)
    assert 0.0 <= red <= 1.0

    # 6. Rate distortion proxy
    orig = np.ones((4, 4))
    recon = np.ones((4, 4)) * 0.9
    rd = compute_rate_distortion_proxy(orig, recon, bits=128)
    assert rd["wire_bits"] == 128.0
    assert math.isclose(rd["distortion_mse"], 0.01, rel_tol=1e-4)
