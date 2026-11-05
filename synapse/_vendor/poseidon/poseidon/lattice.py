"""Diamond Lattice Entorhinal Neuropil for Poseidon.

Implements multi-scale 3D tetrahedral / diamond grid cell modules
with continuous velocity integration on toroidal phase manifolds,
geodesic distance estimation, and multi-scale resonance coherence.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


class DiamondLatticeNeuropil:
    """Multi-scale 3D Diamond / FCC geodesic entorhinal grid module."""

    # 4 Tetrahedral wave directions for FCC/diamond lattice in 3D
    TETRAHEDRAL_VECTORS = [
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
        (1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
        (-1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    ]

    def __init__(self, scales: Tuple[float, float, float] = (3.0, 8.0, 20.0)):
        self.scales = scales
        # Toroidal phases phi_m = (phi_x, phi_y, phi_z) in [-pi, pi)^3 for each module
        self.phases: List[List[float]] = [[0.0, 0.0, 0.0] for _ in scales]
        # Accumulated estimated position in virtual continuous coordinate space
        self.estimated_pos = [0.0, 0.0, 0.0]
        self.step_count = 0

    def reset(self, initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Reset lattice phases and position."""
        self.estimated_pos = list(initial_pos)
        self.phases = [[0.0, 0.0, 0.0] for _ in self.scales]
        self.step_count = 0

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi)."""
        wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
        return wrapped

    def integrate_velocity(self, vx: float, vy: float, vz: float = 0.0) -> None:
        """Integrate continuous velocity vector into multi-scale toroidal grid phases."""
        self.estimated_pos[0] += vx
        self.estimated_pos[1] += vy
        self.estimated_pos[2] += vz
        self.step_count += 1

        v = (vx, vy, vz)
        for m, scale in enumerate(self.scales):
            k = 2.0 * math.pi / scale
            for d in range(3):
                self.phases[m][d] = self._wrap_angle(self.phases[m][d] + k * v[d])

    def update_from_action(self, action: int, speed: float = 1.0) -> None:
        """Update grid coordinates from discrete TidePool action."""
        # 0: stay, 1: up (0, -1), 2: down (0, 1), 3: left (-1, 0), 4: right (1, 0), 5: interact (0, 0)
        action_vel = {
            0: (0.0, 0.0, 0.0),
            1: (0.0, -speed, 0.0),
            2: (0.0, speed, 0.0),
            3: (-speed, 0.0, 0.0),
            4: (speed, 0.0, 0.0),
            5: (0.0, 0.0, 0.0),
        }
        vx, vy, vz = action_vel.get(action, (0.0, 0.0, 0.0))
        self.integrate_velocity(vx, vy, vz)

    def grid_activation(self, module_index: int) -> float:
        """Compute Diamond lattice activation for a specific module."""
        if not (0 <= module_index < len(self.scales)):
            return 0.0
        phase = self.phases[module_index]
        # Sum projections along the 4 tetrahedral directions
        activations = []
        for k_vec in self.TETRAHEDRAL_VECTORS:
            dot = k_vec[0] * phase[0] + k_vec[1] * phase[1] + k_vec[2] * phase[2]
            activations.append(math.cos(dot))
        return (sum(activations) / 4.0 + 1.0) / 2.0  # normalized to [0, 1]

    def multi_scale_coherence(self) -> float:
        """Harmonic resonance across all multi-scale modules."""
        acts = [self.grid_activation(m) for m in range(len(self.scales))]
        return sum(acts) / len(acts)

    def geodesic_distance_to(self, target_phases: List[List[float]]) -> float:
        """Compute scale-weighted geodesic toroidal distance to target phases."""
        dist = 0.0
        for m, scale in enumerate(self.scales):
            cur_p = self.phases[m]
            tgt_p = target_phases[m]
            # Angular distance on circle
            d_m = 0.0
            for d in range(3):
                diff = abs(self._wrap_angle(cur_p[d] - tgt_p[d]))
                d_m += diff * diff
            dist += (scale / (2.0 * math.pi)) * math.sqrt(d_m)
        return float(dist)

    def telemetry(self) -> Dict[str, Any]:
        """Diagnostic telemetry for Diamond Lattice Neuropil."""
        coherence = self.multi_scale_coherence()
        return {
            "estimated_pos": [round(p, 4) for p in self.estimated_pos],
            "step_count": self.step_count,
            "scales": list(self.scales),
            "multi_scale_coherence": round(coherence, 4),
            "module_activations": [
                round(self.grid_activation(m), 4) for m in range(len(self.scales))
            ],
            "toroidal_phases": [
                [round(d, 4) for d in phase] for phase in self.phases
            ],
        }
