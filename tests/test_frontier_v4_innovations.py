"""
Comprehensive Unit Tests for Frontier V4 Scientific Innovations:
- Spiking Neuromorphic Dynamics with Heterogeneous Axonal Delays
- Astrocytic Glymphatic Clearance & Interstitial Metabolic Homeostat
- Quantum-Inspired Kuramoto Phase-Coherent Swarm Synchrony
"""

import math
import numpy as np
import pytest
import torch

from synapse.spiking_neuromorphic import SpikingNeuromorphicEngine, SpikingTelemetry
from synapse.glymphatic_clearance import GlymphaticClearanceEngine, GlymphaticTelemetry
from synapse.quantum_kuramoto import QuantumKuramotoEngine, KuramotoTelemetry


def test_spiking_neuromorphic_dynamics_and_delays():
    """Verify LIF-SFA integration, heterogeneous axonal delays, and refractory resets."""
    engine = SpikingNeuromorphicEngine(num_neurons=32, max_delay_steps=8, dt_ms=1.0, seed=42)

    # Sub-threshold drive should not cause spikes
    sub_drive = torch.full((32,), 0.2, dtype=torch.float32)
    delayed, instant, telem = engine(sub_drive)
    assert instant.sum().item() == 0, "Sub-threshold drive should produce 0 spikes"
    assert telem.mean_membrane_potential < engine.theta_base

    # Strong suprathreshold drive triggers spikes
    strong_drive = torch.full((32,), 15.0, dtype=torch.float32)
    spike_occurred = False
    for step in range(20):
        delayed, instant, telem = engine(strong_drive)
        if instant.sum().item() > 0:
            spike_occurred = True
            break
    assert spike_occurred, "Suprathreshold drive must trigger spiking action potentials"

    # Verify that adaptive threshold increased on spiking neurons
    assert telem.mean_adaptive_threshold > engine.theta_base

    # Verify delay ring buffer: delayed spikes must be binary floats
    assert torch.all((delayed == 0.0) | (delayed == 1.0))

    # Test state checkpoint roundtrip
    saved = engine.export_state()
    engine.reset()
    assert engine.v.mean().item() == pytest.approx(-65.0, abs=1e-3)
    engine.import_state(saved)
    assert engine.ring_ptr == saved['ring_ptr']


def test_glymphatic_clearance_and_sleep_flush():
    """Verify interstitial waste production in wakefulness and convective flush during sleep."""
    engine = GlymphaticClearanceEngine(seed=42)

    # 1. Waking cognition produces metabolic waste
    initial_waste = engine.waste_concentration
    for _ in range(50):
        damping, telem = engine.step(neural_activity_norm=2.5, is_quiescent_sleep=False)

    assert engine.waste_concentration > initial_waste, "Waking activity must accumulate metabolic waste"
    assert damping < 1.0, "Accumulated waste must dampen synaptic efficiency via A1-receptor effect"
    assert telem.interstitial_space_fraction == pytest.approx(0.14, abs=1e-3)

    high_waste = engine.waste_concentration

    # 2. Quiescent sleep phase initiates accelerated AQP4 convective clearance
    for _ in range(100):
        damping, telem = engine.step(neural_activity_norm=0.1, is_quiescent_sleep=True)

    assert engine.waste_concentration < high_waste, "Quiescent sleep phase must flush metabolic waste"
    assert telem.interstitial_space_fraction == pytest.approx(0.23, abs=1e-3)
    assert telem.is_quiescent_flush_active is True
    assert telem.cumulative_waste_cleared > 0.0
    assert damping > 0.95, "Flushing waste restores synaptic transmission efficiency toward baseline"


def test_quantum_kuramoto_phase_synchrony():
    """Verify Kuramoto order parameter phase transition from uncoupled to coherent broadcast."""
    engine = QuantumKuramotoEngine(num_oscillators=32, coupling_k=0.0, dt=0.05, seed=42)

    # 1. Uncoupled (K=0) network should exhibit low order parameter R (desynchronized)
    for _ in range(30):
        gates, telem = engine.step(coupling_override=0.0)

    r_uncoupled = telem.order_parameter_r
    assert 0.0 <= r_uncoupled < 0.65, f"Uncoupled oscillators should have low R, got {r_uncoupled}"

    # 2. Strong coupling (K=4.0) should drive global phase synchronization (R -> 1.0)
    for _ in range(150):
        gates, telem = engine.step(coupling_override=4.0)

    r_coupled = telem.order_parameter_r
    assert r_coupled > r_uncoupled, "Strong coupling must increase phase synchrony order parameter R"
    assert r_coupled > 0.70, f"Expected strong synchronization R > 0.70, got {r_coupled}"

    # Quantum-inspired phase gating factors G_i must be bounded in [0, 1]
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)
    assert telem.mean_ensemble_phase >= -math.pi and telem.mean_ensemble_phase <= math.pi
