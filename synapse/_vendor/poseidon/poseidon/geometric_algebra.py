"""POSEIDON GEOMETRIC-ALGEBRA: Clifford Cl(3,0) Multivector Kinematics & Rotor Dynamics.

Synthesizes FLY-DIAMOND-NEXUS & Supermix-Archimedes:
1. Complete 8-dimensional Clifford Geometric Algebra Cl(3,0) multivector calculus:
   - Grade 0: Scalar (vital energy, mass, pressure)
   - Grade 1: Vector (spatial position, linear velocity, hydrodynamic force)
   - Grade 2: Bivector (oriented planes of rotation, torque, fluid vorticity)
   - Grade 3: Pseudoscalar (volumetric dilatation, chirality, topological index)
2. Rotor-based Singularity-free 3D Kinematics:
   R = cos(theta/2) - sin(theta/2) * B,  v' = R * v * ~R
   Eliminates gimbal lock, Euler singularities, and quaternion convention ambiguities.
3. Hydrodynamic Vortex Lie Interaction:
   Vortex circulation bivector fields generate geometric Lorentz forces:
   F_vortex = v . B_vortex via the geometric contraction product.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence


@dataclass
class Multivector3D:
    """Clifford Algebra Cl(3,0) 8-component multivector.
    
    Basis:
    - 1      (scalar s)
    - e1, e2, e3 (vector components v1, v2, v3)
    - e12, e23, e31 (bivector components b12, b23, b31)
    - e123 = I (pseudoscalar p)
    """
    s: float = 0.0     # Grade 0
    v1: float = 0.0    # Grade 1 (e1)
    v2: float = 0.0    # Grade 1 (e2)
    v3: float = 0.0    # Grade 1 (e3)
    b12: float = 0.0   # Grade 2 (e12 = e1^e2)
    b23: float = 0.0   # Grade 2 (e23 = e2^e3)
    b31: float = 0.0   # Grade 2 (e31 = e3^e1)
    p: float = 0.0     # Grade 3 (e123 = I)

    @classmethod
    def from_scalar(cls, val: float) -> Multivector3D:
        return cls(s=float(val))

    @classmethod
    def from_vector(cls, x: float, y: float, z: float) -> Multivector3D:
        return cls(v1=float(x), v2=float(y), v3=float(z))

    @classmethod
    def from_bivector(cls, b12: float, b23: float, b31: float) -> Multivector3D:
        return cls(b12=float(b12), b23=float(b23), b31=float(b31))

    @classmethod
    def from_pseudoscalar(cls, p: float) -> Multivector3D:
        return cls(p=float(p))

    def __add__(self, other: Multivector3D | float) -> Multivector3D:
        if isinstance(other, (int, float)):
            return Multivector3D(self.s + other, self.v1, self.v2, self.v3, self.b12, self.b23, self.b31, self.p)
        return Multivector3D(
            self.s + other.s,
            self.v1 + other.v1,
            self.v2 + other.v2,
            self.v3 + other.v3,
            self.b12 + other.b12,
            self.b23 + other.b23,
            self.b31 + other.b31,
            self.p + other.p,
        )

    def __sub__(self, other: Multivector3D | float) -> Multivector3D:
        if isinstance(other, (int, float)):
            return Multivector3D(self.s - other, self.v1, self.v2, self.v3, self.b12, self.b23, self.b31, self.p)
        return Multivector3D(
            self.s - other.s,
            self.v1 - other.v1,
            self.v2 - other.v2,
            self.v3 - other.v3,
            self.b12 - other.b12,
            self.b23 - other.b23,
            self.b31 - other.b31,
            self.p - other.p,
        )

    def __mul__(self, other: Multivector3D | float) -> Multivector3D:
        """Geometric Product Cl(3,0) multiplication."""
        if isinstance(other, (int, float)):
            return Multivector3D(
                self.s * other, self.v1 * other, self.v2 * other, self.v3 * other,
                self.b12 * other, self.b23 * other, self.b31 * other, self.p * other
            )

        # Full Clifford Cl(3,0) multiplication table:
        # e1^2 = e2^2 = e3^2 = +1
        # e12^2 = e23^2 = e31^2 = -1
        # I^2 = -1
        a, b = self, other

        # Grade 0 (Scalar)
        s = (a.s * b.s +
             a.v1 * b.v1 + a.v2 * b.v2 + a.v3 * b.v3 -
             a.b12 * b.b12 - a.b23 * b.b23 - a.b31 * b.b31 -
             a.p * b.p)

        # Grade 1 (Vectors)
        v1 = (a.s * b.v1 + a.v1 * b.s -
              a.v2 * b.b12 + a.b12 * b.v2 +
              a.v3 * b.b31 - a.b31 * b.v3 -
              a.b23 * b.p - a.p * b.b23)

        v2 = (a.s * b.v2 + a.v2 * b.s +
              a.v1 * b.b12 - a.b12 * b.v1 -
              a.v3 * b.b23 + a.b23 * b.v3 -
              a.b31 * b.p - a.p * b.b31)

        v3 = (a.s * b.v3 + a.v3 * b.s -
              a.v1 * b.b31 + a.b31 * b.v1 +
              a.v2 * b.b23 - a.b23 * b.v2 -
              a.b12 * b.p - a.p * b.b12)

        # Grade 2 (Bivectors)
        b12 = (a.s * b.b12 + a.b12 * b.s +
               a.v1 * b.v2 - a.v2 * b.v1 -
               a.b23 * b.b31 + a.b31 * b.b23 +
               a.v3 * b.p + a.p * b.v3)

        b23 = (a.s * b.b23 + a.b23 * b.s +
               a.v2 * b.v3 - a.v3 * b.v2 -
               a.b31 * b.b12 + a.b12 * b.b31 +
               a.v1 * b.p + a.p * b.v1)

        b31 = (a.s * b.b31 + a.b31 * b.s +
               a.v3 * b.v1 - a.v1 * b.v3 -
               a.b12 * b.b23 + a.b23 * b.b12 +
               a.v2 * b.p + a.p * b.v2)

        # Grade 3 (Pseudoscalar)
        p = (a.s * b.p + a.p * b.s +
             a.v1 * b.b23 + a.b23 * b.v1 +
             a.v2 * b.b31 + a.b31 * b.v2 +
             a.v3 * b.b12 + a.b12 * b.v3)

        return Multivector3D(s, v1, v2, v3, b12, b23, b31, p)

    def __rmul__(self, other: float) -> Multivector3D:
        return self.__mul__(other)

    def reverse(self) -> Multivector3D:
        """Reversion involution (~M): reverses vector ordering in each blade.
        Grade 0: +, Grade 1: +, Grade 2: -, Grade 3: -
        """
        return Multivector3D(
            self.s,
            self.v1, self.v2, self.v3,
            -self.b12, -self.b23, -self.b31,
            -self.p
        )

    def magnitude_squared(self) -> float:
        """Quadratic norm: scalar part of M * ~M."""
        rev = self.reverse()
        prod = self * rev
        return max(0.0, prod.s)

    def magnitude(self) -> float:
        return math.sqrt(self.magnitude_squared())

    def normalized(self) -> Multivector3D:
        m = self.magnitude()
        if m < 1e-9:
            return Multivector3D(1.0, 0, 0, 0, 0, 0, 0, 0)
        return self * (1.0 / m)

    def wedge(self, other: Multivector3D) -> Multivector3D:
        """Exterior wedge product (A ^ B)."""
        # Outer product of vectors gives bivectors
        b12 = self.v1 * other.v2 - self.v2 * other.v1
        b23 = self.v2 * other.v3 - self.v3 * other.v2
        b31 = self.v3 * other.v1 - self.v1 * other.v3
        return Multivector3D(b12=b12, b23=b23, b31=b31)

    def dot_vectors(self, other: Multivector3D) -> float:
        """Standard Euclidean scalar inner product of vector components."""
        return self.v1 * other.v1 + self.v2 * other.v2 + self.v3 * other.v3

    def contract_vector_bivector(self, bivector: Multivector3D) -> Multivector3D:
        """Left contraction of vector v with bivector B: v . B (Grade 1 vector result)."""
        # v . B in Cl(3,0):
        # e1 . e12 = e2,  e2 . e12 = -e1
        # e2 . e23 = e3,  e3 . e23 = -e2
        # e3 . e31 = e1,  e1 . e31 = -e3
        vx = -self.v2 * bivector.b12 + self.v3 * bivector.b31
        vy = self.v1 * bivector.b12 - self.v3 * bivector.b23
        vz = self.v2 * bivector.b23 - self.v1 * bivector.b31
        return Multivector3D.from_vector(vx, vy, vz)

    def to_vector_tuple(self) -> tuple[float, float, float]:
        return (self.v1, self.v2, self.v3)


class Rotor3D:
    """Clifford Cl(3,0) Rotor: R = cos(theta/2) - sin(theta/2) * B_unit."""

    def __init__(self, multivector: Multivector3D | None = None):
        if multivector is None:
            self.mv = Multivector3D(s=1.0)
        else:
            self.mv = multivector.normalized()

    @classmethod
    def from_axis_angle(cls, axis_x: float, axis_y: float, axis_z: float, angle_rad: float) -> Rotor3D:
        """Creates a rotor rotating by angle_rad around the unit axis (using the dual bivector B = I * n)."""
        axis_len = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
        if axis_len < 1e-9:
            return cls()

        nx = axis_x / axis_len
        ny = axis_y / axis_len
        nz = axis_z / axis_len

        half_theta = angle_rad * 0.5
        c = math.cos(half_theta)
        s = math.sin(half_theta)

        # Dual bivector components:
        # B = nx e23 + ny e31 + nz e12
        return cls(Multivector3D(
            s=c,
            b12=-s * nz,
            b23=-s * nx,
            b31=-s * ny,
        ))

    def rotate_vector(self, v: Multivector3D) -> Multivector3D:
        """Applies rotor sandwich transformation: v' = R * v * ~R."""
        rev = self.mv.reverse()
        transformed = self.mv * v * rev
        return Multivector3D.from_vector(transformed.v1, transformed.v2, transformed.v3)

    def compose(self, other: Rotor3D) -> Rotor3D:
        """R_composite = self * other."""
        return Rotor3D(self.mv * other.mv)


class GeometricKinematicsEngine:
    """Continuous 3D Rigid-Fluid Kinematics integrated via Clifford Geometric Algebra."""

    def __init__(
        self,
        mass: float = 1.0,
        drag_coeff: float = 0.8,
        dt: float = 0.2,
    ):
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.dt = dt

        # State multivectors
        self.pos = Multivector3D.from_vector(50.0, 50.0, 3.0)
        self.vel = Multivector3D.from_vector(0.0, 0.0, 0.0)
        self.rotor = Rotor3D()
        self.angular_velocity_bivector = Multivector3D.from_bivector(0.0, 0.0, 0.0)

    def apply_thrust_and_torque(
        self,
        thrust_forward: float,
        torque_yaw_b12: float,
        torque_pitch_b31: float = 0.0,
        fluid_vorticity: Multivector3D | None = None,
    ) -> None:
        """Integrates Newtonian kinematics using Clifford algebra and bivector torque."""
        dt = self.dt

        # 1. Forward direction vector obtained by rotating canonical e1 forward vector
        e1_canonical = Multivector3D.from_vector(1.0, 0.0, 0.0)
        heading_dir = self.rotor.rotate_vector(e1_canonical)

        # Propulsion vector
        f_propulsion = heading_dir * thrust_forward

        # Drag vector
        f_drag = self.vel * (-self.drag_coeff)

        # Hydrodynamic Vortex Force: Lorentz-like contraction v . B_vortex
        f_vortex = Multivector3D.from_vector(0.0, 0.0, 0.0)
        if fluid_vorticity is not None:
            f_vortex = self.vel.contract_vector_bivector(fluid_vorticity) * 0.5

        # Total linear force
        f_total = f_propulsion + f_drag + f_vortex
        accel = f_total * (1.0 / self.mass)

        # Velocity and position integration
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt

        # 2. Rotational dynamics via Bivector rotor integration
        applied_torque_b = Multivector3D.from_bivector(torque_yaw_b12, 0.0, torque_pitch_b31)
        damping_torque_b = self.angular_velocity_bivector * (-0.9)
        total_torque_b = applied_torque_b + damping_torque_b

        # Update angular velocity bivector
        self.angular_velocity_bivector = self.angular_velocity_bivector + total_torque_b * dt

        # Update rotor: dR/dt = -0.5 * Omega * R
        omega_step = self.angular_velocity_bivector * (-0.5 * dt)
        rotor_delta = Rotor3D(Multivector3D(s=1.0) + omega_step)
        self.rotor = self.rotor.compose(rotor_delta)
