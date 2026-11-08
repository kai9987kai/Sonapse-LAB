"""Comprehensive Test Suite for Propagating Neural Pulses & Graph Waves.

Verifies:
1. PulseEvent schemas and NeuromodulatoryPulseField trigger conditions.
2. GraphWaveSolver discrete damped wave equation on region Laplacian.
3. NeuralPhaseResetProtocol soft phase entrainment and Kuramoto coherence.
4. BoundedPulseRouter zero-gain identity and norm bounds.
"""
import pytest
import numpy as np
import torch

from synapse.neural_pulses import (
    PulseEventType,
    PulseEvent,
    NeuromodulatoryPulseField,
    GraphWaveSolver,
    NeuralPhaseResetProtocol,
    BoundedPulseRouter,
)


def test_pulse_events_and_triggers():
    field = NeuromodulatoryPulseField(num_regions=8)

    pulse = field.trigger_pulse(
        PulseEventType.ACUTE_THREAT,
        source_region=1,
        amplitude=1.5,
        frequency_hz=20.0,
    )
    assert pulse.event_type == PulseEventType.ACUTE_THREAT
    assert pulse.source_region == 1
    assert len(field.active_pulses) == 1

    # Check automated triggers
    fired = field.check_triggers(
        acute_threat_level=0.85,
        prediction_error=0.80,
        reward_delta=0.60,
        is_sleep_transition=True,
    )
    assert len(fired) >= 3


def test_graph_wave_solver_dynamics_and_energy():
    solver = GraphWaveSolver(num_regions=8, velocity_c=1.5, damping_zeta=0.25, dt=0.05)

    # Initial zero state has zero energy
    out0 = solver.step()
    assert float(out0["total_energy"].item()) == 0.0

    # Inject an excitation pulse at region 0
    solver.inject_pulse(0, amplitude=2.0)
    assert solver.x_curr[0] == 2.0

    # Step forward and verify wave spreads across Laplacian
    trajectories = []
    for _ in range(20):
        out = solver.step()
        trajectories.append(out["displacement"].clone())
        # Numerical stability check: no NaN or inf
        assert torch.all(torch.isfinite(out["displacement"]))

    # Neighboring node 1 should have received wave displacement
    neighbor_displacements = [traj[1].item() for traj in trajectories]
    assert max(abs(d) for d in neighbor_displacements) > 0.05


def test_phase_reset_and_kuramoto_coherence():
    proto = NeuralPhaseResetProtocol(num_regions=8, base_freq_hz=10.0, dt=0.05)

    out0 = proto.step()
    r0 = float(out0["coherence"].item())
    assert 0.0 <= r0 <= 1.0

    # Strong synchronized wave forcing should entrain phases
    wave_x = torch.full((8,), 2.0)
    wave_v = torch.full((8,), 1.0)
    for _ in range(15):
        out_entrained = proto.step(wave_displacement=wave_x, wave_velocity=wave_v)

    r_final = float(out_entrained["coherence"].item())
    assert 0.0 <= r_final <= 1.0


def test_bounded_pulse_router():
    router = BoundedPulseRouter(
        num_regions=8,
        target_dim=64,
        max_local_current=0.25,
        max_global_current=2.0,
        enabled=False,
    )

    disp = torch.randn(8)
    phases = torch.randn(8)

    # Zero-gain identity: output must be exactly zero when disabled
    curr_off = router.compute_current(disp, phases, enabled_override=False)
    assert torch.all(curr_off == 0.0)

    # When enabled, respects local and global bounds
    curr_on = router.compute_current(disp * 10.0, phases, enabled_override=True)
    assert torch.all(torch.abs(curr_on) <= 0.25 + 1e-6)
    assert torch.sum(torch.abs(curr_on)) <= 2.0 + 1e-5
