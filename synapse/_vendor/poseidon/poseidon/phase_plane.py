"""POSEIDON PHASE-PLANE: Nonlinear Dynamical Systems & Separatrix Stability Engine.

Implements Phase-Plane Analysis (Poincaré-Bendixson & Lyapunov dynamical systems)
for homeostatic organism survival in high-scarcity, non-stationary environments.

Core Formulations:
1. State Space & Phase Trajectory:
   \\mathbf{x}(t) = [E(t), H(t), S(t), V(t)]^T \\in [0, 1]^4
   (Energy, Hydration, Stamina, Vital Health)

2. Phase Velocity Vector Field:
   \\mathbf{f}(\\mathbf{x}, a) = [\\dot{E}, \\dot{H}, \\dot{S}, \\dot{V}]^T
   where \\dot{E}, \\dot{H}, \\dot{S} model coupled non-linear metabolic consumption,
   patch replenishment, and fluid drag dissipation.

3. Attractor Basin & Homeostatic Fixed Point:
   \\mathbf{x}^* = [0.85, 0.85, 0.90, 1.00]^T
   Lyapunov Energy Function:
     L(\\mathbf{x}) = \\frac{1}{2} \\sum_i w_i (x_i - x_i^*)^2
   Orbital Time Derivative (Lie derivative along vector field):
     \\dot{L}(\\mathbf{x}, a) = \\nabla L(\\mathbf{x}) \\cdot \\mathbf{f}(\\mathbf{x}, a)
   - \\dot{L} < 0: Asymptotic convergence towards homeostatic attractor basin.
   - \\dot{L} > 0: Destabilizing divergence towards critical mortality separatrix.

4. Separatrix Margin & Extinction Basin Detection:
   Calculates Euclidean and geodesic distance to the critical extinction hyperplanes
   (E <= 0.05, H <= 0.05, V <= 0.00). Injects stabilizing orbital steering.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .world import ACTIONS, N_ACTIONS, OBS_SIZE


@dataclass
class PhaseStateAudit:
    """Rigorous audit of the agent's phase-plane coordinates and orbital stability."""
    energy: float
    hydration: float
    stamina: float
    health: float
    v_energy: float      # \dot{E} phase velocity
    v_hydration: float   # \dot{H} phase velocity
    v_stamina: float     # \dot{S} phase velocity
    v_health: float      # \dot{V} phase velocity
    lyapunov_energy: float
    lyapunov_drift: float     # \dot{L}
    separatrix_distance: float
    in_critical_basin: bool
    stability_regime: str     # "contracting_attractor" | "transient_orbit" | "divergent_saddle" | "extinction_basin"
    action_phase_drifts: dict[int, float]
    recommended_action: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": round(self.energy, 4),
            "hydration": round(self.hydration, 4),
            "stamina": round(self.stamina, 4),
            "health": round(self.health, 4),
            "phase_velocities": {
                "dE_dt": round(self.v_energy, 5),
                "dH_dt": round(self.v_hydration, 5),
                "dS_dt": round(self.v_stamina, 5),
                "dV_dt": round(self.v_health, 5),
            },
            "lyapunov_energy": round(self.lyapunov_energy, 5),
            "lyapunov_drift": round(self.lyapunov_drift, 5),
            "separatrix_distance": round(self.separatrix_distance, 4),
            "in_critical_basin": self.in_critical_basin,
            "stability_regime": self.stability_regime,
            "action_phase_drifts": {k: round(v, 5) for k, v in self.action_phase_drifts.items()},
            "recommended_action": self.recommended_action,
        }


class PhasePlaneEngine:
    """Nonlinear dynamical systems phase plane analyzer and attractor orbit stabilizer."""

    # Target homeostatic fixed point attractor \mathbf{x}^*
    ATTRACTOR_ENERGY: float = 0.85
    ATTRACTOR_HYDRATION: float = 0.85
    ATTRACTOR_STAMINA: float = 0.90
    ATTRACTOR_HEALTH: float = 1.00

    # Critical separatrix boundaries
    SEPARATRIX_ENERGY: float = 0.12
    SEPARATRIX_HYDRATION: float = 0.12
    SEPARATRIX_HEALTH: float = 0.15

    def __init__(
        self,
        w_energy: float = 3.0,
        w_hydration: float = 3.5,
        w_stamina: float = 1.5,
        w_health: float = 5.0,
        dt: float = 0.20,
    ):
        self.w_energy = float(w_energy)
        self.w_hydration = float(w_hydration)
        self.w_stamina = float(w_stamina)
        self.w_health = float(w_health)
        self.dt = float(dt)

        self.last_state: tuple[float, float, float, float] | None = None
        self.orbital_history: list[dict[str, float]] = []

    def reset(self) -> None:
        self.last_state = None
        self.orbital_history.clear()

    def compute_lyapunov_function(self, e: float, h: float, s: float, v: float) -> float:
        """Computes quadratic Lyapunov distance L(x) from homeostatic fixed point."""
        de = e - self.ATTRACTOR_ENERGY
        dh = h - self.ATTRACTOR_HYDRATION
        ds = s - self.ATTRACTOR_STAMINA
        dv = v - self.ATTRACTOR_HEALTH

        return 0.5 * (
            self.w_energy * (de ** 2)
            + self.w_hydration * (dh ** 2)
            + self.w_stamina * (ds ** 2)
            + self.w_health * (dv ** 2)
        )

    def compute_separatrix_distance(self, e: float, h: float, v: float) -> float:
        """Computes minimum Euclidean distance to critical mortality separatrix."""
        d_e = max(0.0, e - self.SEPARATRIX_ENERGY)
        d_h = max(0.0, h - self.SEPARATRIX_HYDRATION)
        d_v = max(0.0, v - self.SEPARATRIX_HEALTH)

        return math.sqrt(d_e ** 2 + d_h ** 2 + d_v ** 2)

    def estimate_vector_field(
        self,
        obs: Sequence[float],
        candidate_action: int,
    ) -> tuple[float, float, float, float]:
        """Estimates vector field f(x, a) = [dE/dt, dH/dt, dS/dt, dV/dt] for an action."""
        e = float(obs[1])
        h = float(obs[2])
        s = float(obs[3])
        v = float(obs[0])
        food_local = float(obs[6]) if len(obs) > 6 else 0.5
        water_local = float(obs[7]) if len(obs) > 7 else 0.5
        exposure = float(obs[4]) if len(obs) > 4 else 0.1

        # Action-specific metabolic deltas
        if candidate_action == 0:  # Rest
            dE = -0.008
            dH = -0.010
            dS = +0.150
            dV = +0.020 if e > 0.3 and h > 0.3 else 0.0
        elif candidate_action == 1:  # Forage
            dE = (food_local * 0.40) - 0.020
            dH = -0.025
            dS = -0.030
            dV = 0.0
        elif candidate_action == 2:  # Drink
            dE = -0.020
            dH = (water_local * 0.45) - 0.015
            dS = -0.025
            dV = 0.0
        elif candidate_action == 3:  # Shelter
            dE = -0.015
            dH = -0.015
            dS = +0.050
            dV = +0.010
        elif candidate_action in (4, 5):  # Explore / Flee
            dE = -0.035
            dH = -0.035
            dS = -0.070
            dV = 0.0
        else:
            dE, dH, dS, dV = -0.02, -0.02, -0.02, 0.0

        # Environmental drag & exposure dissipation
        if exposure > 0.65:
            dV -= (exposure - 0.65) * 0.5
            dH -= 0.015

        # Starvation / Dehydration metabolic penalty
        if e < self.SEPARATRIX_ENERGY:
            dV -= 0.08
        if h < self.SEPARATRIX_HYDRATION:
            dV -= 0.10

        return dE, dH, dS, dV

    def analyze_phase_space(
        self,
        obs: Sequence[float],
        action_taken: int | None = None,
    ) -> PhaseStateAudit:
        """Performs complete phase plane audit and returns stability diagnostics."""
        v = float(obs[0])
        e = float(obs[1])
        h = float(obs[2])
        s = float(obs[3])

        current_lyapunov = self.compute_lyapunov_function(e, h, s, v)
        sep_dist = self.compute_separatrix_distance(e, h, v)
        in_critical = (e <= self.SEPARATRIX_ENERGY or h <= self.SEPARATRIX_HYDRATION or v <= self.SEPARATRIX_HEALTH)

        # Compute Lie orbital derivative for all candidate actions
        action_drifts: dict[int, float] = {}
        for act in range(N_ACTIONS):
            dE, dH, dS, dV = self.estimate_vector_field(obs, act)

            # Lie derivative \dot{L} = \nabla L \cdot f(x, a)
            dL = (
                self.w_energy * (e - self.ATTRACTOR_ENERGY) * dE
                + self.w_hydration * (h - self.ATTRACTOR_HYDRATION) * dH
                + self.w_stamina * (s - self.ATTRACTOR_STAMINA) * dS
                + self.w_health * (v - self.ATTRACTOR_HEALTH) * dV
            )
            action_drifts[act] = float(dL)

        # Chosen or current action velocities
        current_act = action_taken if action_taken is not None else min(action_drifts, key=action_drifts.get)
        dE, dH, dS, dV = self.estimate_vector_field(obs, current_act)
        current_drift = action_drifts[current_act]

        # Determine stability regime
        if in_critical:
            regime = "extinction_basin"
        elif current_drift < -0.05:
            regime = "contracting_attractor"  # Strongly stabilizing
        elif current_drift <= 0.02:
            regime = "transient_orbit"        # Quasi-stable
        else:
            regime = "divergent_saddle"       # Destabilizing

        # Recommended action minimizes Lyapunov drift \dot{L} (steers back to attractor)
        best_steering_action = min(action_drifts, key=action_drifts.get)

        audit = PhaseStateAudit(
            energy=e,
            hydration=h,
            stamina=s,
            health=v,
            v_energy=dE,
            v_hydration=dH,
            v_stamina=dS,
            v_health=dV,
            lyapunov_energy=current_lyapunov,
            lyapunov_drift=current_drift,
            separatrix_distance=sep_dist,
            in_critical_basin=in_critical,
            stability_regime=regime,
            action_phase_drifts=action_drifts,
            recommended_action=best_steering_action,
        )

        self.last_state = (e, h, s, v)
        self.orbital_history.append({
            "energy": e, "hydration": h, "stamina": s, "health": v,
            "lyapunov": current_lyapunov, "drift": current_drift
        })
        return audit
