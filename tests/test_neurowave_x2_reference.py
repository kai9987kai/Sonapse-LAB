"""Unit tests verifying NumPy Reference Transport engine for Sonapse NeuroWave-X2."""
import math
import numpy as np
import pytest
from synapse.neurowave_x2.packets import NeuralPacket
from synapse.neurowave_x2.reference_transport import (
    ReferenceTransport,
    phase_coherence_gain_numpy,
    quantize_numpy,
    spectral_synthesize_numpy,
)


def test_quantize_numpy_levels():
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    q8 = quantize_numpy(x, bits=8)
    assert q8.shape == x.shape
    np.testing.assert_allclose(q8, x, atol=0.01)

    q4 = quantize_numpy(x, bits=4)
    # 4 bits = 16 levels in [-1, 1], step = 2/15 ~= 0.1333
    np.testing.assert_allclose(q4, x, atol=0.1)


def test_phase_coherence_gain_numpy():
    # Exactly in-phase (d_theta = 0) -> gain = 1.0
    assert math.isclose(phase_coherence_gain_numpy(0.0, 0.0), 1.0, abs_tol=1e-6)
    # Exactly anti-phase (d_theta = pi) -> gain = 0.0
    assert math.isclose(phase_coherence_gain_numpy(0.0, np.pi), 0.0, abs_tol=1e-6)
    # Quadrature (d_theta = pi/2) -> gain = 0.5
    assert math.isclose(phase_coherence_gain_numpy(0.0, np.pi / 2.0), 0.5, abs_tol=1e-6)
    # Non-negative invariant
    for th in np.linspace(-4 * np.pi, 4 * np.pi, 25):
        gain = phase_coherence_gain_numpy(th, 0.0)
        assert 0.0 <= gain <= 1.0


def test_reference_transport_queue_and_zero_gain():
    ref = ReferenceTransport(dimension=8, seed=42)
    pkt = NeuralPacket(0, 1, 0, 0, 1, "vector", (0.5, -0.5, 0.2, 0.1))

    # Send packet
    sent = ref.send(pkt, current_step=0)
    assert sent is True

    # Receive at step 0
    received = ref.receive(current_step=0, receiver_id=1)
    assert len(received) == 1
    assert received[0] == pkt

    # Zero-gain residual test
    base_act = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    weights = np.ones((4, 3), dtype=np.float32)
    h_new, delta = ref.decode_and_inject(received, base_act, weights, gamma=0.0)

    np.testing.assert_array_equal(h_new, base_act)
    np.testing.assert_array_equal(delta, np.zeros(3, dtype=np.float32))
