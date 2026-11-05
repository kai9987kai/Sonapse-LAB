"""Tests for MemoryCarrierObservatory and Topological Persistent Homology."""
import math
import pytest
from poseidon.memory_observatory import (
    MemoryCarrierObservatory,
    TopologicalManifoldAudit,
)


def test_topological_closed_limit_cycle():
    obs = MemoryCarrierObservatory(filtration_epsilon=0.45)
    # Generate circular closed orbit in (energy, hydration) space
    orbit = []
    for step in range(24):
        theta = 2.0 * math.pi * step / 24.0
        e = 0.5 + 0.25 * math.cos(theta)
        w = 0.5 + 0.25 * math.sin(theta)
        orbit.append([1.0, e, w, 0.8])

    audit = obs.analyze_trajectory_topology(orbit)
    assert audit.betti_0 >= 1
    # Circular closed loop produces 1-cycle
    assert audit.betti_1 >= 1
    assert audit.is_homeostatic_cycle is True
    assert len(audit.manifold_signature_sha256) == 64


def test_topological_telemetry_synthesis():
    obs = MemoryCarrierObservatory()
    sample_traj = [[1.0, 0.8, 0.8, 0.8] for _ in range(12)]
    audit = obs.analyze_trajectory_topology(sample_traj)

    frame = obs.synthesize_telemetry_frame(
        audit,
        lyapunov_drift=-0.04,
        quantum_coherence=0.98,
        monologue="Attractor basin stabilized within safe bounds.",
    )
    assert "metrics" in frame
    assert "pcm_base64" in frame
    assert "<svg" in frame["svg_oscilloscope"]
    assert "signature_sha256" in frame
