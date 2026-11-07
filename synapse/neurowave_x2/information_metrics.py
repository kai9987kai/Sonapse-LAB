"""Information-Theoretic Telemetry & Communication Metrics for Sonapse NeuroWave-X2.

Computes message entropy, effective rank, communication sparsity, rate-distortion proxy,
spectral energy distribution, and empirical mutual-information estimators.
Labeled strictly as estimators, not exact analytical Shannon information quantities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_effective_rank(message_matrix: np.ndarray) -> float:
    """Computes effective message rank via singular value entropy (Roy & Vetterli).
    
    Formula:
        p_k = s_k / sum(s)
        H(p) = -sum p_k * log(p_k)
        effective_rank = exp(H(p))
    """
    if message_matrix.shape[0] < 2 or message_matrix.shape[1] < 2:
        return 1.0
    # Center rows
    centered = message_matrix - np.mean(message_matrix, axis=0, keepdims=True)
    try:
        s = np.linalg.svd(centered, compute_uv=False)
    except Exception:
        return 1.0
    s_sum = float(np.sum(s))
    if s_sum <= 1e-12:
        return 1.0
    p = s / s_sum
    p_nonzero = p[p > 1e-12]
    entropy = -float(np.sum(p_nonzero * np.log(p_nonzero)))
    return float(np.exp(entropy))


def compute_message_entropy_estimate(messages: np.ndarray, num_bins: int = 16) -> float:
    """Estimates differential message entropy via histogram binning proxy."""
    if messages.size == 0:
        return 0.0
    flat = messages.flatten()
    counts, _ = np.histogram(flat, bins=num_bins, range=(-1.0, 1.0))
    p = counts / float(np.sum(counts))
    p_nonzero = p[p > 1e-12]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def compute_communication_sparsity(residual_currents: np.ndarray, threshold: float = 1e-3) -> float:
    """Computes the fraction of neural targets receiving near-zero auxiliary communication."""
    if residual_currents.size == 0:
        return 1.0
    near_zero = np.sum(np.abs(residual_currents) < threshold)
    return float(near_zero / residual_currents.size)


def compute_spectral_energy_concentration(mode_coefficients: np.ndarray) -> float:
    """Calculates ratio of energy in top-2 spectral modes relative to total spectral energy."""
    if mode_coefficients.size == 0:
        return 1.0
    pows = np.abs(mode_coefficients) ** 2
    if pows.ndim == 1:
        tot = float(np.sum(pows))
        if tot <= 1e-12:
            return 1.0
        sorted_pows = np.sort(pows)[::-1]
        top2 = float(np.sum(sorted_pows[:2]))
        return float(top2 / tot)
    else:
        tot = float(np.sum(pows))
        if tot <= 1e-12:
            return 1.0
        row_sums = np.sum(pows, axis=0)
        sorted_sums = np.sort(row_sums)[::-1]
        top2 = float(np.sum(sorted_sums[:2]))
        return float(top2 / tot)


def compute_channel_redundancy(message_matrix: np.ndarray) -> float:
    """Estimates cross-channel correlation/redundancy across communication routes."""
    if message_matrix.shape[0] < 2:
        return 0.0
    try:
        corr = np.corrcoef(message_matrix)
        # Average off-diagonal absolute correlation
        N = corr.shape[0]
        off_diag = np.abs(corr[np.triu_indices(N, k=1)])
        return float(np.nanmean(off_diag)) if off_diag.size > 0 else 0.0
    except Exception:
        return 0.0


def compute_rate_distortion_proxy(original: np.ndarray, reconstructed: np.ndarray, bits: int) -> Dict[str, float]:
    """Computes rate-distortion proxy (mean-squared error versus wire bits)."""
    mse = float(np.mean((original - reconstructed) ** 2))
    return {
        "wire_bits": float(bits),
        "distortion_mse": mse,
        "rate_distortion_ratio": float(mse / max(1.0, float(bits))),
    }
