"""NexusFlow: Directed Acyclic Causal Flux & Heuristic Routing.

Synthesizing graph-guided heuristic flow, causal flux networks, and coherent
superposition arbitration for the TidePool environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FlowWaypoint:
    node_id: str
    x: float
    y: float
    replenishment: float
    depletion: float
    potential: float


class NexusFlowNetwork:
    """Directed causal flux network over visited spatial waypoints and replenishment corridors."""

    def __init__(
        self,
        decay_tau: float = 2.5,
        depletion_penalty: float = 0.45,
        conservation_epsilon: float = 0.05,
    ):
        self.decay_tau = float(decay_tau)
        self.depletion_penalty = float(depletion_penalty)
        self.conservation_epsilon = float(conservation_epsilon)
        self.waypoints: Dict[str, FlowWaypoint] = {}
        self.flux_matrix: Dict[str, Dict[str, float]] = {}
        self.total_flux_transferred = 0.0
        self.interference_events = 0

    def reset(self) -> None:
        """Reset all registered waypoints and flux matrices."""
        self.waypoints.clear()
        self.flux_matrix.clear()
        self.total_flux_transferred = 0.0
        self.interference_events = 0

    def register_waypoint(
        self,
        node_id: str,
        x: float,
        y: float,
        replenishment: float,
        depletion: float = 0.0,
    ) -> FlowWaypoint:
        potential = max(0.0, float(replenishment) - self.depletion_penalty * float(depletion))
        wp = FlowWaypoint(
            node_id=str(node_id),
            x=round(float(x), 4),
            y=round(float(y), 4),
            replenishment=round(float(replenishment), 4),
            depletion=round(float(depletion), 4),
            potential=round(float(potential), 4),
        )
        self.waypoints[node_id] = wp
        if node_id not in self.flux_matrix:
            self.flux_matrix[node_id] = {}
        return wp

    def compute_causal_flux(self, src_id: str, dst_id: str) -> float:
        """Compute continuous causal flux Phi_{src, dst}."""
        if src_id not in self.waypoints or dst_id not in self.waypoints:
            return 0.0
        if src_id == dst_id:
            return 0.0

        src = self.waypoints[src_id]
        dst = self.waypoints[dst_id]

        dist = math.hypot(dst.x - src.x, dst.y - src.y)
        dist_factor = math.exp(-dist / max(self.decay_tau, 1e-4))
        delta_potential = max(0.0, dst.potential - src.potential)

        flux = dist_factor * delta_potential
        self.flux_matrix[src_id][dst_id] = round(float(flux), 4)
        return flux

    def solve_topological_gradients(self) -> Dict[str, Tuple[float, float]]:
        """Compute potential flow gradient vector for every registered waypoint."""
        gradients: Dict[str, Tuple[float, float]] = {}
        for src_id, src in self.waypoints.items():
            fx, fy = 0.0, 0.0
            for dst_id, dst in self.waypoints.items():
                if src_id == dst_id:
                    continue
                flux = self.compute_causal_flux(src_id, dst_id)
                if flux > 0.0:
                    dist = max(1e-4, math.hypot(dst.x - src.x, dst.y - src.y))
                    ux = (dst.x - src.x) / dist
                    uy = (dst.y - src.y) / dist
                    fx += flux * ux
                    fy += flux * uy
            gradients[src_id] = (round(fx, 4), round(fy, 4))
        return gradients

    def evaluate_superposition_interference(
        self,
        value_a: float,
        value_b: float,
        heading_angle_a: float,
        heading_angle_b: float,
    ) -> Tuple[float, bool]:
        """Compute coherent wave-interference term between two candidate directional policies.
        
        I(a, b) = 2 * sqrt(V_a * V_b) * cos(theta_a - theta_b)
        Returns (interference_term, destructive_conflict_flag).
        """
        va = max(0.0, float(value_a))
        vb = max(0.0, float(value_b))
        delta_theta = float(heading_angle_a) - float(heading_angle_b)
        cos_diff = math.cos(delta_theta)

        interference = 2.0 * math.sqrt(va * vb) * cos_diff
        destructive = bool(cos_diff < -0.2)  # Conflict if divergent by > ~101 degrees
        if destructive:
            self.interference_events += 1

        return round(float(interference), 4), destructive

    def recommend_action_from_flow(
        self,
        current_x: float,
        current_y: float,
        compass_heading: float,
    ) -> Tuple[int, Dict[str, Any]]:
        """Compute recommended directional action from active flow gradients."""
        # Find nearest or active source waypoint
        if not self.waypoints:
            return 0, {"source": "no_waypoints", "flux_magnitude": 0.0}

        best_dst: Optional[FlowWaypoint] = None
        max_flux = -1.0

        for wp in self.waypoints.values():
            dist = math.hypot(wp.x - current_x, wp.y - current_y)
            if dist > 0.05:  # Target other nodes
                flux = math.exp(-dist / self.decay_tau) * wp.potential
                if flux > max_flux:
                    max_flux = flux
                    best_dst = wp

        if best_dst is None or max_flux <= 0.01:
            return 0, {"source": "diffuse_flux", "flux_magnitude": 0.0}

        dx = best_dst.x - current_x
        dy = best_dst.y - current_y
        target_heading = math.atan2(dy, dx)

        # Map target heading to TidePool actions:
        # Actions: 0: rest, 1: UP (-y), 2: DOWN (+y), 3: LEFT (-x), 4: RIGHT (+x), 5: FORAGE
        # In grid coordinates:
        if abs(dx) > abs(dy):
            action = 4 if dx > 0 else 3
        else:
            action = 2 if dy > 0 else 1

        return action, {
            "source": "nexus_gradient",
            "target_node": best_dst.node_id,
            "target_heading": round(float(target_heading), 4),
            "flux_magnitude": round(float(max_flux), 4),
        }

    def receipt_digest(self) -> str:
        """Deterministic digest of network topology and flux configuration."""
        data = {
            "waypoints": {k: asdict(v) for k, v in sorted(self.waypoints.items())},
            "decay_tau": self.decay_tau,
            "depletion_penalty": self.depletion_penalty,
            "interference_events": self.interference_events,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
