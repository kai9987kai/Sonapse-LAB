"""Vector-Symbolic Architecture and Holographic Binding.

Implements circular convolution binding and circular correlation unbinding
using Real Fast Fourier Transforms (rfft/irfft):
z = x (*) y = irfft(rfft(x) * rfft(y))
x_approx = z (*) y_inv = irfft(rfft(z) * conj(rfft(y)))

Enables compositional neural symbolic binding of roles, values, and concepts.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.fft


class VectorSymbolicBinder:
    """Holographic vector symbolic operator supporting binding, unbinding, and superposition."""

    @staticmethod
    def bind(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Circular convolution binding: z = x (*) y."""
        dim = x.shape[-1]
        x_fft = torch.fft.rfft(x, n=dim)
        y_fft = torch.fft.rfft(y, n=dim)
        z = torch.fft.irfft(x_fft * y_fft, n=dim)
        # Unit norm normalization
        norm = torch.norm(z, dim=-1, keepdim=True) + 1e-8
        return z / norm

    @staticmethod
    def unbind(z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Circular correlation unbinding: x_approx = z (*) key*."""
        dim = z.shape[-1]
        z_fft = torch.fft.rfft(z, n=dim)
        k_fft = torch.fft.rfft(key, n=dim)
        # Correlation in frequency domain is multiplication by complex conjugate of key
        x_approx = torch.fft.irfft(z_fft * torch.conj(k_fft), n=dim)
        norm = torch.norm(x_approx, dim=-1, keepdim=True) + 1e-8
        return x_approx / norm

    @staticmethod
    def bundle(vectors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """Vector superposition / addition: s = sum w_i * v_i."""
        if not vectors:
            raise ValueError("Empty vector list for bundling.")
        if weights is None:
            weights = [1.0] * len(vectors)

        s = sum(w * v for w, v in zip(weights, vectors))
        norm = torch.norm(s, dim=-1, keepdim=True) + 1e-8
        return s / norm

    @staticmethod
    def similarity(x: torch.Tensor, y: torch.Tensor) -> float:
        """Cosine similarity between two representations."""
        norm_x = torch.norm(x) + 1e-8
        norm_y = torch.norm(y) + 1e-8
        return float((torch.dot(x.flatten(), y.flatten()) / (norm_x * norm_y)).item())
