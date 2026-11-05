"""Tests for Clifford Cl(3,0) Geometric Algebra and Rotor Dynamics."""
import math
import pytest
from poseidon.geometric_algebra import (
    GeometricKinematicsEngine,
    Multivector3D,
    Rotor3D,
)


def test_geometric_product_basis_identities():
    e1 = Multivector3D.from_vector(1.0, 0.0, 0.0)
    e2 = Multivector3D.from_vector(0.0, 1.0, 0.0)
    e3 = Multivector3D.from_vector(0.0, 0.0, 1.0)

    # e1^2 = +1 (scalar)
    e1_sq = e1 * e1
    assert pytest.approx(e1_sq.s) == 1.0
    assert pytest.approx(e1_sq.v1) == 0.0

    # e1 * e2 = e12 (bivector)
    e12 = e1 * e2
    assert pytest.approx(e12.s) == 0.0
    assert pytest.approx(e12.b12) == 1.0

    # e2 * e1 = -e12 (anticommutativity)
    e21 = e2 * e1
    assert pytest.approx(e21.b12) == -1.0

    # e12^2 = -1
    e12_sq = e12 * e12
    assert pytest.approx(e12_sq.s) == -1.0


def test_rotor_3d_rotation():
    # 90 degree rotation around Z axis (e3)
    rotor_z_90 = Rotor3D.from_axis_angle(0.0, 0.0, 1.0, math.pi / 2.0)
    e1 = Multivector3D.from_vector(1.0, 0.0, 0.0)

    # Rotating e1 around Z by 90 deg should produce e2 = (0, 1, 0)
    rotated = rotor_z_90.rotate_vector(e1)
    assert pytest.approx(rotated.v1, abs=1e-6) == 0.0
    assert pytest.approx(rotated.v2, abs=1e-6) == 1.0
    assert pytest.approx(rotated.v3, abs=1e-6) == 0.0


def test_geometric_kinematics_engine_step_with_vortex():
    engine = GeometricKinematicsEngine(dt=0.1)
    # Apply forward thrust
    vorticity = Multivector3D.from_bivector(b12=0.5, b23=0.0, b31=0.0)
    engine.apply_thrust_and_torque(
        thrust_forward=5.0,
        torque_yaw_b12=0.2,
        fluid_vorticity=vorticity,
    )

    assert engine.vel.magnitude() > 0.0
    assert engine.pos.v1 != 50.0  # moved from starting position
