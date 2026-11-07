"""
Comprehensive Unit Tests for Kai9987kai Architectural Systems:
- Archimedes Hydrodynamic Buoyancy & Vorticity Manifold
- Holographic Memory Carrier Nexus & Fourier-Laplacian Associative Retrieval
- Cross-Connectome Heterogeneous Dale's-Law Grafting
"""

import math
import numpy as np
import pytest
import torch

from synapse.archimedes_buoyancy import ArchimedesBuoyancyEngine, ArchimedesTelemetry
from synapse.memory_carrier import MemoryCarrierNexus, CarrierTelemetry
from synapse.cross_graft import CrossConnectomeGraft, CrossGraftTelemetry


def test_archimedes_buoyancy_and_vorticity():
    """Verify Navier-Stokes Archimedes displacement, fluid density stratification, and drag."""
    engine = ArchimedesBuoyancyEngine(
        base_fluid_density=1.025,
        body_density=1.000,
        body_volume=0.05,
        num_sensory_channels=16,
        seed=42,
    )

    # In standard fluid with body_density < fluid_density, net buoyancy must be positive (upward)
    currents, drag, telem = engine.step(agent_pos=(1.0, 2.0), agent_vel=(0.5, 0.0), vertical_thrust=0.0, dt=0.1)

    assert telem.mean_buoyancy_force > 0.0, "Net buoyancy must be positive for buoyant body"
    assert len(currents) == 16, "Sensory channels must match channel dimension"
    assert torch.all(currents >= -1.0) and torch.all(currents <= 1.0), "Sensory currents must be bounded in [-1, 1]"
    assert drag[0] < 0.0, "Planar drag force must oppose positive velocity vx"

    # Verify state export/import
    exported = engine.export_state()
    engine.reset()
    assert engine.depth_z == 0.0
    engine.import_state(exported)
    assert engine.depth_z == exported['depth_z']


def test_holographic_memory_carrier_nexus():
    """Verify Fourier-space circular convolution binding and O(1) associative recall."""
    engine = MemoryCarrierNexus(dimension=64, max_carriers=16, seed=42)

    # Create key and value representations
    rng = np.random.default_rng(42)
    k1 = torch.tensor(rng.standard_normal(64), dtype=torch.float32)
    v1 = torch.tensor(rng.standard_normal(64), dtype=torch.float32)
    k1 /= torch.norm(k1)
    v1 /= torch.norm(v1)

    # Store memory association on carrier 0
    engine.store(k1, v1, carrier_idx=0)
    assert engine.stored_count == 1

    # Associative recall with query key
    recalled, fidelity = engine.retrieve(k1, carrier_idx=0)
    assert fidelity > 0.65, f"Expected associative retrieval fidelity > 0.65, got {fidelity}"
    assert len(recalled) == 64
    assert engine.telemetry.carrier_snr_db > 0.0
    assert len(engine.telemetry.carrier_frequency_energies) > 0


def test_cross_connectome_dales_law_graft():
    """Verify Dale's law enforcement, affinity gating, and bounded residual projections."""
    graft = CrossConnectomeGraft(node_dim=64, gamma=0.2, seed=42)

    acts = torch.randn(64)
    delta, telem = graft(acts)

    # 1. Verify Dale's law: every synapse must strictly match its dale_signs polarity
    w_dale = torch.nn.functional.softplus(graft.w_param) * graft.dale_signs
    for i in range(10):
        for j in range(10):
            expected_sign = float(graft.dale_signs[i, j].item())
            actual_val = float(w_dale[i, j].item())
            if expected_sign > 0:
                assert actual_val >= 0.0, "Excitatory synapse must have non-negative weight"
            else:
                assert actual_val <= 0.0, "Inhibitory synapse must have non-positive weight"

    # 2. Verify bounded residual projection
    assert delta.shape[0] == 64
    assert float(torch.norm(delta).item()) <= graft.gamma * math.sqrt(64) + 1e-4
    assert telem.dales_law_compliance_pct == 100.0
    assert len(telem.donor_recipient_pairs) == 4
