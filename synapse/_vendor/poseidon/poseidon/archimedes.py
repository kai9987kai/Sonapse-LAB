"""Archimedes Hydrodynamic & Buoyancy Engine.

Synthesizes hydrodynamic fluid mechanics and buoyancy principles from
supermix-archimedes and Supermix-Expanse into the TidePool survival substrate:
1. Seawater density stratification rho(S, T) based on salinity S and temperature T.
2. Hydrostatic Archimedes buoyancy force F_b = -rho * V_disp * g and net vertical acceleration.
3. Fluid drag dynamic damping F_d = 0.5 * C_d * rho * v^2 * A.
4. Harmonic tidal surge velocity currents u(t) = [U_0 * sin(omega * t), U_0 * cos(omega * t)].
5. Directional hydrodynamic locomotion cost modulation (with-current stamina discount,
   counter-current penalty, neutral buoyancy stabilization).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


class ArchimedesHydrodynamicEngine:
    """Hydrodynamic buoyancy and tidal fluid simulation engine."""

    # Standard physical constants scaled for TidePool dynamics
    GRAVITY: float = 9.81
    BASE_DENSITY: float = 1.025  # kg/L standard seawater density
    SALINITY_COEFF: float = 0.028  # density increase per unit salinity
    THERMAL_EXPANSION: float = 0.003  # density decrease per unit temperature
    DRAG_COEFFICIENT: float = 0.47  # sphere / streamlined body
    AGENT_CROSS_SECTION: float = 0.08  # m^2 projected area
    AGENT_BASE_VOLUME: float = 0.05  # m^3 baseline displacement volume
    AGENT_MASS: float = 0.05125  # kg calibrated for neutral buoyancy at baseline

    def __init__(
        self,
        tide_period_steps: int = 48,
        max_surge_velocity: float = 1.2,
    ) -> None:
        self.tide_period_steps = max(8, tide_period_steps)
        self.max_surge_velocity = max_surge_velocity
        self.step_count: int = 0
        self.vesicle_inflation: float = 0.5  # internal gas bladder in [0, 1]
        self.current_depth: float = 0.5  # vertical depth in [0, 1] (0 = surface, 1 = seabed)
        self.vertical_velocity: float = 0.0  # m/s vertical movement
        self.total_displacement_work: float = 0.0

    def reset(self) -> None:
        """Reset fluid state at the start of an episode."""
        self.step_count = 0
        self.vesicle_inflation = 0.5
        self.current_depth = 0.5
        self.vertical_velocity = 0.0
        self.total_displacement_work = 0.0

    def compute_fluid_density(self, salinity: float, temperature: float) -> float:
        """Compute local fluid density rho(S, T) in kg/L."""
        s = max(0.0, min(1.0, salinity))
        t = max(0.0, min(1.0, temperature))
        density = self.BASE_DENSITY * (1.0 + self.SALINITY_COEFF * s - self.THERMAL_EXPANSION * (t - 0.5))
        return max(0.95, min(1.15, density))

    def compute_tidal_surge(self, step: int) -> Tuple[float, float]:
        """Compute harmonic horizontal tidal current vector (u_x, u_y) in m/s."""
        omega = 2.0 * math.pi / self.tide_period_steps
        phase = omega * float(step)
        u_x = self.max_surge_velocity * math.sin(phase)
        u_y = self.max_surge_velocity * math.cos(phase * 0.5)
        return (round(u_x, 4), round(u_y, 4))

    def compute_buoyancy(
        self,
        density: float,
        vesicle_inflation: float,
    ) -> Tuple[float, float]:
        """Compute buoyant force F_b and net vertical acceleration.

        Returns:
            (F_buoyant, a_vertical)
        """
        # Effective agent volume modulated by gas vesicle inflation
        effective_volume = self.AGENT_BASE_VOLUME * (0.8 + 0.4 * vesicle_inflation)
        # Archimedes principle: F_b = rho * V * g (directed upwards)
        f_buoyant = density * effective_volume * self.GRAVITY
        # Gravity force: F_g = m * g (directed downwards)
        f_gravity = self.AGENT_MASS * self.GRAVITY
        # Net upward force
        net_force = f_buoyant - f_gravity
        # Drag damping on vertical movement
        drag = 0.5 * self.DRAG_COEFFICIENT * density * (self.vertical_velocity ** 2) * self.AGENT_CROSS_SECTION
        if self.vertical_velocity > 0:
            net_force -= drag
        else:
            net_force += drag
        acceleration = net_force / max(0.001, self.AGENT_MASS)
        return (f_buoyant, acceleration)

    def modulate_locomotion_cost(
        self,
        action: int,
        heading_angle: float,
        surge_vector: Tuple[float, float],
    ) -> float:
        """Compute metabolic stamina multiplier based on hydrodynamic currents.

        Moving with current provides up to 35% discount (0.65x cost).
        Moving against current incurs up to 45% penalty (1.45x cost).
        Stationary/resting at neutral buoyancy incurs standard or discounted cost.
        """
        if action == 0:  # Rest / float
            # If vesicle is well-tuned to neutral buoyancy, rest is extra restorative
            depth_error = abs(self.current_depth - 0.5)
            return max(0.70, 1.0 - 0.30 * (1.0 - depth_error))

        if action not in (1, 2, 4, 5):  # forage, drink, explore, flee involve movement
            return 1.0

        u_x, u_y = surge_vector
        surge_speed = math.hypot(u_x, u_y)
        if surge_speed < 1e-4:
            return 1.0

        surge_angle = math.atan2(u_y, u_x)
        # Alignment dot product: cos(theta_heading - theta_surge)
        alignment = math.cos(heading_angle - surge_angle)

        # alignment > 0: moving with current -> lower cost
        # alignment < 0: moving against current -> higher cost
        multiplier = 1.0 - 0.35 * alignment
        return max(0.65, min(1.45, multiplier))

    def step(
        self,
        observation: List[float],
        action: int,
        heading_angle: float = 0.0,
    ) -> Dict[str, float]:
        """Update fluid dynamics for one environmental tick."""
        self.step_count += 1

        # Extract environmental cues from 16-d observation
        # obs indices: 10 = temperature, 9 = severity (salinity proxy), 4 = exposure
        salinity = observation[9] if len(observation) > 9 else 0.5
        temperature = observation[10] if len(observation) > 10 else 0.5
        stamina = observation[3] if len(observation) > 3 else 1.0

        # Autonomous gas vesicle regulation to maintain neutral buoyancy
        target_inflation = 0.5
        if stamina < 0.25:
            # Low stamina: inflate vesicle to float higher with minimal effort
            target_inflation = 0.75
        elif action == 3:  # shelter: deflate vesicle to sink towards protective benthic floor
            target_inflation = 0.25

        # Smooth vesicle adjustment
        self.vesicle_inflation += 0.15 * (target_inflation - self.vesicle_inflation)
        self.vesicle_inflation = max(0.0, min(1.0, self.vesicle_inflation))

        fluid_density = self.compute_fluid_density(salinity, temperature)
        f_buoyant, a_vert = self.compute_buoyancy(fluid_density, self.vesicle_inflation)

        # Numerical integration of vertical position
        dt = 0.2  # step dt
        self.vertical_velocity += a_vert * dt * 0.01
        self.vertical_velocity = max(-0.5, min(0.5, self.vertical_velocity))
        # Note: depth 0 = surface (high buoyancy), depth 1 = seabed
        self.current_depth -= self.vertical_velocity * dt
        self.current_depth = max(0.0, min(1.0, self.current_depth))

        surge_vector = self.compute_tidal_surge(self.step_count)
        stamina_cost_mult = self.modulate_locomotion_cost(action, heading_angle, surge_vector)

        displacement_work = abs(f_buoyant * self.vertical_velocity * dt)
        self.total_displacement_work += displacement_work

        return {
            "fluid_density": round(fluid_density, 4),
            "buoyancy_force": round(f_buoyant, 4),
            "net_vertical_acceleration": round(a_vert, 4),
            "current_depth": round(self.current_depth, 4),
            "vesicle_inflation": round(self.vesicle_inflation, 4),
            "surge_u_x": surge_vector[0],
            "surge_u_y": surge_vector[1],
            "stamina_cost_multiplier": round(stamina_cost_mult, 4),
            "displacement_work": round(self.total_displacement_work, 4),
        }
