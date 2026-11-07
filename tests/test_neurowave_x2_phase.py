import math
import torch
import numpy as np
import pytest

from synapse.neurowave_x2.phase_channel import PhaseCoherenceAdapter, WaveChannel, BAND_FREQUENCIES


def test_phase_coherence_adapter_initialization():
    adapter = PhaseCoherenceAdapter(num_regions=5, bands=("theta", "beta", "gamma"))
    assert adapter.num_regions == 5
    assert len(adapter.bands) == 3


def test_phase_coherence_gain_formula():
    """Verify CTC gain formula: g_ij = (1 + cos(delta_theta)) / 2 in [0, 1]."""
    adapter = PhaseCoherenceAdapter(num_regions=4, bands=("theta",))
    # Reset phases directly to controlled values
    phases = [0.0, math.pi / 2, math.pi, 2 * math.pi]
    adapter.reset(initial_phases={"theta": phases})

    # In-phase (0 - 0) -> cos(0) = 1 -> g = 1.0
    g_in_phase = adapter.compute_coherence_gain(0, 0, band="theta")
    assert math.isclose(g_in_phase, 1.0, rel_tol=1e-5)

    # Anti-phase (0 - pi) -> cos(-pi) = -1 -> g = 0.0
    g_anti_phase = adapter.compute_coherence_gain(0, 2, band="theta")
    assert math.isclose(g_anti_phase, 0.0, abs_tol=1e-5)

    # Orthogonal (0 - pi/2) -> cos(-pi/2) = 0 -> g = 0.5
    g_ortho = adapter.compute_coherence_gain(0, 1, band="theta")
    assert math.isclose(g_ortho, 0.5, rel_tol=1e-5)


def test_phase_oscillator_step():
    """Verify stepping advances regional phase angles according to omega * dt."""
    adapter = PhaseCoherenceAdapter(num_regions=3, bands=("theta",), dt=0.01)
    adapter.reset()
    assert adapter.get_phase(0, band="theta") == 0.0
    adapter.step()
    assert adapter.get_phase(0, band="theta") > 0.0


def test_external_coherence_synchronization():
    """Verify external theta coherence phase locks adapter regional phases."""
    adapter = PhaseCoherenceAdapter(num_regions=3, bands=("theta", "gamma"))
    ext = torch.tensor([1.23, 2.34, 3.14], dtype=torch.float32)
    adapter.step(external_coherence_phases=ext)
    assert math.isclose(adapter.get_phase(0, band="theta"), 1.23, rel_tol=1e-4)
    assert math.isclose(adapter.get_phase(1, band="theta"), 2.34, rel_tol=1e-4)


def test_dales_law_compliance():
    """Phase gain must be non-negative so it cannot invert excitatory/inhibitory sign."""
    adapter = PhaseCoherenceAdapter(num_regions=8)
    adapter.reset(initial_phases={"theta": [i * math.pi / 4 for i in range(8)]})
    for i in range(8):
        for j in range(8):
            gain = adapter.compute_coherence_gain(i, j, band="theta")
            assert 0.0 <= gain <= 1.00001


def test_multi_band_gains():
    adapter = PhaseCoherenceAdapter(num_regions=4, bands=("theta", "beta", "gamma"))
    gains = adapter.compute_multi_band_gains(0, 1)
    assert set(gains.keys()) == {"theta", "beta", "gamma"}
    for v in gains.values():
        assert 0.0 <= v <= 1.0


def test_wave_channel_encoding():
    channel = WaveChannel(message_dim=16, num_modes=8, num_regions=4)
    msg = torch.randn(16)
    coeffs, phase_meta = channel.encode_wave_packet_payload(msg, src_region=1, band="theta")
    assert len(coeffs) == 8
    assert len(phase_meta) == 1
    assert all(-1.0 <= c <= 1.0 for c in coeffs)
