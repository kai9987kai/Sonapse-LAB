"""Scientific Conformance Testing Harness for Sonapse NeuroWave-X2.

Cross-validates PyTorchTransport against independent NumPy ReferenceTransport
under declared numerical tolerances across all communication regimes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .packets import NeuralPacket
from .reference_transport import ReferenceTransport
from .transport import PyTorchTransport


class ConformanceVerificationResult:
    """Detailed receipt of a cross-engine conformance evaluation."""

    def __init__(
        self,
        test_name: str,
        passed: bool,
        max_absolute_error: float,
        declared_tolerance: float,
        details: Dict[str, Any],
    ) -> None:
        self.test_name = str(test_name)
        self.passed = bool(passed)
        self.max_absolute_error = float(max_absolute_error)
        self.declared_tolerance = float(declared_tolerance)
        self.details = dict(details)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "max_absolute_error": self.max_absolute_error,
            "declared_tolerance": self.declared_tolerance,
            "details": self.details,
        }


def run_conformance_suite(seed: int = 12345, tolerance: float = 1e-4) -> List[ConformanceVerificationResult]:
    """Executes the full matrix of conformance tests between PyTorch and NumPy engines."""
    results: List[ConformanceVerificationResult] = []
    rng = np.random.RandomState(seed)

    # 1. Zero-Gain Identity Test
    base_act_np = rng.normal(0.0, 1.0, size=(32,)).astype(np.float32)
    base_act_th = torch.from_numpy(base_act_np)
    weights_np = rng.normal(0.0, 0.5, size=(16, 32)).astype(np.float32)
    weights_th = torch.from_numpy(weights_np)
    pkt = NeuralPacket(0, 1, 0, 0, 1, "vector", tuple(rng.uniform(-1, 1, size=16)))

    ref_engine = ReferenceTransport(dimension=16, seed=seed)
    pt_engine = PyTorchTransport(dimension=16, seed=seed)

    h_np, d_np = ref_engine.decode_and_inject([pkt], base_act_np, weights_np, gamma=0.0)
    h_th, d_th = pt_engine.decode_and_inject([pkt], base_act_th, weights_th, gamma=0.0)

    err = float(np.max(np.abs(h_np - h_th.numpy())))
    passed = err <= tolerance and np.allclose(h_np, base_act_np)
    results.append(ConformanceVerificationResult("zero_gain_identity", passed, err, tolerance, {
        "h_diff": err, "identity_preserved": bool(np.allclose(h_np, base_act_np))
    }))

    # 2. Continuous Vector Communication
    ref_engine.reset(seed)
    pt_engine.reset(seed)
    h_np, d_np = ref_engine.decode_and_inject([pkt], base_act_np, weights_np, gamma=0.25)
    h_th, d_th = pt_engine.decode_and_inject([pkt], base_act_th, weights_th, gamma=0.25)
    err = float(np.max(np.abs(h_np - h_th.numpy())))
    passed = err <= tolerance
    results.append(ConformanceVerificationResult("continuous_vector", passed, err, tolerance, {"max_err": err}))

    # 3. Quantized Communication (8-bit and 4-bit)
    for bits in (8, 4):
        q_pkt = NeuralPacket(0, 1, 0, 1, 1, "quantized", tuple(rng.uniform(-1, 1, size=16)), precision_bits=bits)
        h_np, _ = ref_engine.decode_and_inject([q_pkt], base_act_np, weights_np, gamma=0.3)
        h_th, _ = pt_engine.decode_and_inject([q_pkt], base_act_th, weights_th, gamma=0.3)
        err = float(np.max(np.abs(h_np - h_th.numpy())))
        passed = err <= tolerance
        results.append(ConformanceVerificationResult(f"quantized_{bits}bit", passed, err, tolerance, {"bits": bits, "err": err}))

    # 4. Graph Spectral Communication
    basis_np = np.linalg.qr(rng.randn(16, 16))[0].astype(np.float32)[:8]  # (8, 16)
    basis_th = torch.from_numpy(basis_np)
    spec_pkt = NeuralPacket(0, 1, 0, 2, 1, "spectral", tuple(rng.uniform(-1, 1, size=8)))
    h_np, _ = ref_engine.decode_and_inject([spec_pkt], base_act_np, weights_np, basis=basis_np, gamma=0.2)
    h_th, _ = pt_engine.decode_and_inject([spec_pkt], base_act_th, weights_th, basis=basis_th, gamma=0.2)
    err = float(np.max(np.abs(h_np - h_th.numpy())))
    passed = err <= tolerance
    results.append(ConformanceVerificationResult("graph_spectral", passed, err, tolerance, {"err": err}))

    # 5. Phase Coherence Communication
    phase_src = 0.35 * math.pi
    phase_dst = 0.55 * math.pi
    phase_pkt = NeuralPacket(0, 1, 0, 3, 1, "phase", tuple(rng.uniform(-1, 1, size=16)), phase_metadata=(phase_src,))
    h_np, _ = ref_engine.decode_and_inject([phase_pkt], base_act_np, weights_np, receiver_phase=phase_dst, gamma=0.2)
    h_th, _ = pt_engine.decode_and_inject([phase_pkt], base_act_th, weights_th, receiver_phase=torch.tensor(phase_dst), gamma=0.2)
    err = float(np.max(np.abs(h_np - h_th.numpy())))
    passed = err <= tolerance
    results.append(ConformanceVerificationResult("phase_coherence", passed, err, tolerance, {"err": err}))

    # 6. Combined Phase + Spectral Communication
    combo_pkt = NeuralPacket(0, 1, 0, 4, 1, "phase_spectral", tuple(rng.uniform(-1, 1, size=8)), phase_metadata=(phase_src,))
    h_np, _ = ref_engine.decode_and_inject([combo_pkt], base_act_np, weights_np, basis=basis_np, receiver_phase=phase_dst, gamma=0.2)
    h_th, _ = pt_engine.decode_and_inject([combo_pkt], base_act_th, weights_th, basis=basis_th, receiver_phase=torch.tensor(phase_dst), gamma=0.2)
    err = float(np.max(np.abs(h_np - h_th.numpy())))
    passed = err <= tolerance
    results.append(ConformanceVerificationResult("phase_plus_spectral", passed, err, tolerance, {"err": err}))

    # 7. Queue Latency and Delivery Step Matching
    ref_engine.reset(seed)
    pt_engine.reset(seed)
    ref_engine.max_latency = 2
    pt_engine.max_latency = 2
    for s in range(5):
        p = NeuralPacket(0, 1, s, s, 1, "vector", (float(s),))
        ref_engine.send(p, current_step=s)
        pt_engine.send(p, current_step=s)

    # Receive step-by-step
    queue_match = True
    for s in range(8):
        rec_np = ref_engine.receive(current_step=s, receiver_id=1)
        rec_th = pt_engine.receive(current_step=s, receiver_id=1)
        if len(rec_np) != len(rec_th):
            queue_match = False
            break
        for p1, p2 in zip(rec_np, rec_th):
            if p1.sequence_id != p2.sequence_id:
                queue_match = False
                break
    results.append(ConformanceVerificationResult("queue_and_latency", queue_match, 0.0 if queue_match else 1.0, 0.0, {
        "delivered_ref": len(ref_engine.delivered_packets),
        "delivered_pt": len(pt_engine.delivered_packets),
    }))

    return results
