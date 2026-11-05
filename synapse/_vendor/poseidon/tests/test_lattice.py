"""Tests for Diamond Lattice Entorhinal Neuropil."""
import math
from poseidon.lattice import DiamondLatticeNeuropil


def test_lattice_initialization_and_phases():
    lattice = DiamondLatticeNeuropil(scales=(3.0, 8.0, 20.0))
    telemetry = lattice.telemetry()
    assert len(telemetry["scales"]) == 3
    assert telemetry["estimated_pos"] == [0.0, 0.0, 0.0]
    assert 0.0 <= telemetry["multi_scale_coherence"] <= 1.0
    for act in telemetry["module_activations"]:
        assert 0.0 <= act <= 1.0


def test_lattice_velocity_integration():
    lattice = DiamondLatticeNeuropil(scales=(3.0, 8.0, 20.0))
    # Move right 4 steps, then left 4 steps
    for _ in range(4):
        lattice.update_from_action(4, speed=1.0)
    assert math.isclose(lattice.estimated_pos[0], 4.0, rel_tol=1e-5)

    for _ in range(4):
        lattice.update_from_action(3, speed=1.0)
    assert math.isclose(lattice.estimated_pos[0], 0.0, abs_tol=1e-5)
    # Phases should return near zero
    for phase in lattice.phases:
        assert math.isclose(phase[0], 0.0, abs_tol=1e-5)


def test_lattice_grid_activation_bounds():
    lattice = DiamondLatticeNeuropil()
    for m in range(3):
        act = lattice.grid_activation(m)
        assert 0.0 <= act <= 1.0
    coherence = lattice.multi_scale_coherence()
    assert 0.0 <= coherence <= 1.0


def test_lattice_geodesic_distance():
    lattice = DiamondLatticeNeuropil()
    # Distance to self is zero
    dist_zero = lattice.geodesic_distance_to(lattice.phases)
    assert math.isclose(dist_zero, 0.0, abs_tol=1e-5)

    # Displace and check positive distance
    target_phases = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.0, 0.2]]
    dist_pos = lattice.geodesic_distance_to(target_phases)
    assert dist_pos > 0.0
