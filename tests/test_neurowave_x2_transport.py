"""Unit tests verifying PyTorch Transport engine and autograd differentiability."""
import math
import pytest
import torch
from synapse.neurowave_x2.packets import NeuralPacket
from synapse.neurowave_x2.transport import (
    PyTorchTransport,
    phase_coherence_gain_torch,
    quantize_torch,
    spectral_synthesize_torch,
)


def test_quantize_torch_straight_through():
    x = torch.tensor([-0.7, 0.0, 0.45], requires_grad=True)
    q = quantize_torch(x, bits=4)
    loss = torch.sum(q ** 2)
    loss.backward()
    # Gradient should pass straight through
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


def test_phase_coherence_gain_torch():
    src = torch.tensor(0.0)
    dst = torch.tensor(0.0)
    assert math.isclose(float(phase_coherence_gain_torch(src, dst).item()), 1.0, abs_tol=1e-5)
    dst_anti = torch.tensor(math.pi)
    assert math.isclose(float(phase_coherence_gain_torch(src, dst_anti).item()), 0.0, abs_tol=1e-5)


def test_pytorch_transport_differentiable_injection():
    transport = PyTorchTransport(dimension=4, max_aux_current=0.5, seed=42)
    pkt = NeuralPacket(0, 1, 0, 0, 1, "vector", (0.5, -0.2, 0.8, -0.4))

    base_act = torch.randn(6, requires_grad=True)
    dec_w = torch.randn(4, 6, requires_grad=True)

    h_new, delta = transport.decode_and_inject([pkt], base_act, dec_w, gamma=0.5)

    loss = torch.sum(h_new ** 2)
    loss.backward()

    assert base_act.grad is not None
    assert dec_w.grad is not None
    assert not torch.isnan(base_act.grad).any()
    assert not torch.isnan(dec_w.grad).any()
