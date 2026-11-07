"""Supermix Ecosystem Adapters.

Provides typed, versioned integration interfaces for the six complementary repositories
in the research ecosystem:
1. PoseidonAdapter: Fluid-dynamic continuous sensory manifold encoding.
2. ExpanseAdapter: Multi-agent spatial coordination & territory partitioning.
3. NeuroGraftAdapter: Heterogeneous cross-species connectome graft bridging.
4. TetherAdapter: Safety boundary enforcement, torque limiting & action clamping.
5. FaultlineAdapter: Dynamic disturbance injection & resilience recovery benchmarking.
6. GhostInTheMachineAdapter: Cognitive latent probe and neural readout interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PoseidonManifoldState:
    vorticity: float
    viscosity: float
    current_vector: List[float]


class PoseidonAdapter:
    """Fluid-dynamic continuous sensory manifold encoder."""

    def __init__(self, sensory_dim: int = 155):
        self.sensory_dim = sensory_dim

    def encode_fluid_currents(self, raw_obs: torch.Tensor, flow_velocity: float = 1.0) -> torch.Tensor:
        """Inject simulated laminar and turbulent flow components into sensory observation."""
        obs = raw_obs.clone().float()
        time_phase = float(torch.sum(obs[:5]).item())
        drift = torch.sin(torch.linspace(0, 2 * math.pi, self.sensory_dim) + time_phase) * (0.05 * flow_velocity)
        return obs + drift


class ExpanseAdapter:
    """Multi-agent spatial coordination and territory partitioning interface."""

    def __init__(self, agent_id: str = "agent_0", num_agents: int = 4):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.spatial_bounds = [-10.0, 10.0]

    def coordinate_territory(self, my_pos: List[float], peer_positions: Dict[str, List[float]]) -> Dict[str, float]:
        """Compute Voronoi-like centroid proximity and collision avoidance vectors."""
        repulsion = [0.0, 0.0]
        for pid, pos in peer_positions.items():
            if pid == self.agent_id:
                continue
            dx = my_pos[0] - pos[0]
            dy = my_pos[1] - pos[1]
            dist = math.sqrt(dx * dx + dy * dy) + 1e-5
            if dist < 2.0:
                repulsion[0] += (dx / dist) * (2.0 - dist)
                repulsion[1] += (dy / dist) * (2.0 - dist)
        return {"repulsion_x": repulsion[0], "repulsion_y": repulsion[1]}


class NeuroGraftAdapter:
    """Heterogeneous cross-species connectome graft bridge protocol."""

    def __init__(self, donor_species: str = "flywire_drosophila", host_species: str = "manc_vnc"):
        self.donor_species = donor_species
        self.host_species = host_species

    def align_feature_spaces(self, donor_activations: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Projects donor activations into target host dimension via pseudo-isometric mapping."""
        src_dim = donor_activations.shape[-1]
        if src_dim == target_dim:
            return donor_activations
        # Deterministic projection
        proj = torch.eye(src_dim, target_dim) if src_dim <= target_dim else torch.eye(target_dim, src_dim).t()
        return torch.matmul(donor_activations, proj)


class TetherAdapter:
    """Safety bounding, torque limiting, and action clamping protocol."""

    def __init__(self, max_torque: float = 1.0, max_delta: float = 0.5):
        self.max_torque = max_torque
        self.max_delta = max_delta
        self.last_action: Optional[torch.Tensor] = None

    def enforce_safety(self, action: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Clamps absolute action to max_torque and rate of change to max_delta."""
        clamped = torch.clamp(action, -self.max_torque, self.max_torque)
        violation = False

        if self.last_action is not None and self.last_action.shape == action.shape:
            delta = clamped - self.last_action
            if torch.max(torch.abs(delta)) > self.max_delta:
                violation = True
                delta_clamped = torch.clamp(delta, -self.max_delta, self.max_delta)
                clamped = self.last_action + delta_clamped

        self.last_action = clamped.clone().detach()
        return clamped, violation


class FaultlineAdapter:
    """Seismic and high-frequency disturbance injection benchmark."""

    def __init__(self, disturbance_scale: float = 0.5, p_fault: float = 0.05):
        self.disturbance_scale = disturbance_scale
        self.p_fault = p_fault

    def inject_disturbance_if_active(self, activations: torch.Tensor, step: int) -> Tuple[torch.Tensor, bool]:
        """Inject high-frequency pulse shockwave if scheduled."""
        if (step % 20 == 0) and (step > 0):
            shock = (torch.rand_like(activations) * 2.0 - 1.0) * self.disturbance_scale
            return activations + shock, True
        return activations, False


class GhostInTheMachineAdapter:
    """Latent cognitive probe and internal activation readout interface."""

    def __init__(self, probe_dim: int = 16):
        self.probe_dim = probe_dim
        self.probe_proj = nn.Linear(64, probe_dim, bias=False)
        nn.init.orthogonal_(self.probe_proj.weight)

    def extract_probe_representation(self, bottleneck_vector: torch.Tensor) -> Dict[str, float]:
        """Read out low-dimensional cognitive summary coordinates."""
        with torch.no_grad():
            flat = bottleneck_vector.view(-1)
            if flat.shape[0] >= 64:
                flat_64 = flat[:64]
            else:
                flat_64 = F.pad(flat, (0, 64 - flat.shape[0]))
            probe = self.probe_proj(flat_64)
            return {f"dim_{i}": round(float(probe[i].item()), 4) for i in range(min(self.probe_dim, 8))}
