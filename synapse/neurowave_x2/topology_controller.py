"""Topology Controller for Sonapse NeuroWave-X2.

Manages active communication routes, computes regional adjacency matrices,
enforces graph invariants (no self-loops, degree caps), and maintains monotonic topology versions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class TopologyInvariantError(ValueError):
    """Raised when a candidate communication topology violates graph invariants."""
    pass


class CommunicationTopologyController:
    """Controls the inter-region communication graph topology."""

    def __init__(
        self,
        num_regions: int = 8,
        max_in_degree: int = 4,
        max_out_degree: int = 4,
        allow_self_loops: bool = False,
    ) -> None:
        self.num_regions = max(1, int(num_regions))
        self.max_in_degree = int(max_in_degree)
        self.max_out_degree = int(max_out_degree)
        self.allow_self_loops = bool(allow_self_loops)

        self.routes: Set[Tuple[int, int]] = set()
        self.topology_version = 1

    def add_route(self, src: int, dst: int) -> bool:
        """Adds a candidate directed route (src -> dst) if it satisfies graph invariants."""
        src, dst = int(src), int(dst)
        if not (0 <= src < self.num_regions):
            raise TopologyInvariantError(f"Source region {src} out of bounds [0, {self.num_regions-1}]")
        if not (0 <= dst < self.num_regions):
            raise TopologyInvariantError(f"Destination region {dst} out of bounds [0, {self.num_regions-1}]")
        if not self.allow_self_loops and src == dst:
            raise TopologyInvariantError(f"Self-loop {src} -> {dst} is not permitted")

        if (src, dst) in self.routes:
            return False  # Already present

        # Check out-degree of src
        out_deg = sum(1 for s, d in self.routes if s == src)
        if out_deg >= self.max_out_degree:
            raise TopologyInvariantError(f"Source region {src} exceeds max_out_degree={self.max_out_degree}")

        # Check in-degree of dst
        in_deg = sum(1 for s, d in self.routes if d == dst)
        if in_deg >= self.max_in_degree:
            raise TopologyInvariantError(f"Destination region {dst} exceeds max_in_degree={self.max_in_degree}")

        self.routes.add((src, dst))
        self.topology_version += 1
        return True

    def remove_route(self, src: int, dst: int) -> bool:
        """Removes route (src -> dst). Returns True if removed."""
        if (src, dst) in self.routes:
            self.routes.remove((src, dst))
            self.topology_version += 1
            return True
        return False

    def get_adjacency_matrix(self) -> np.ndarray:
        """Constructs binary (num_regions, num_regions) adjacency matrix."""
        adj = np.zeros((self.num_regions, self.num_regions), dtype=np.float32)
        for s, d in self.routes:
            adj[s, d] = 1.0
        return adj

    def list_routes(self) -> List[Tuple[int, int]]:
        return sorted(list(self.routes))

    def clone(self) -> "CommunicationTopologyController":
        """Creates an independent clone of the topology controller."""
        c = CommunicationTopologyController(
            num_regions=self.num_regions,
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            allow_self_loops=self.allow_self_loops,
        )
        c.routes = set(self.routes)
        c.topology_version = int(self.topology_version)
        return c

    def restore_from(self, other: "CommunicationTopologyController") -> None:
        """Restores topology state from another controller (e.g., during transaction rollback)."""
        self.num_regions = int(other.num_regions)
        self.max_in_degree = int(other.max_in_degree)
        self.max_out_degree = int(other.max_out_degree)
        self.allow_self_loops = bool(other.allow_self_loops)
        self.routes = set(other.routes)
        self.topology_version = int(other.topology_version)
