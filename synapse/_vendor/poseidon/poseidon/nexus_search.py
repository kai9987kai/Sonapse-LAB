"""NEXUS-SEARCH: Federated Hybrid Associative Vector Search Engine.

Combines high-dimensional Holographic Reduced Representation (HRR) concept vectors,
3D Diamond Lattice geodesic coordinates, and causal action-reward valences into
a constant-memory federated indexing and retrieval engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SearchIndexEntry:
    concept: str
    vector: np.ndarray  # 256-d normalized vector
    coordinate: Tuple[float, float, float]  # 3D lattice position
    action: int
    reward: float
    step: int


class NexusSearchEngine:
    """Federated associative search engine indexing holographic vectors and spatial coordinates."""

    def __init__(
        self,
        dimension: int = 256,
        max_entries: int = 128,
        resonance_threshold: float = 0.20,
    ):
        self.dimension = int(dimension)
        self.max_entries = int(max_entries)
        self.resonance_threshold = float(resonance_threshold)
        self.entries: List[SearchIndexEntry] = []
        self.total_queries = 0
        self.total_hits = 0

    def reset(self) -> None:
        """Clear all indexed entries and telemetry."""
        self.entries.clear()
        self.total_queries = 0
        self.total_hits = 0

    def index_event(
        self,
        concept: str,
        vector: np.ndarray,
        coordinate: Tuple[float, float, float],
        action: int,
        reward: float,
        step: int,
    ) -> None:
        """Index an associative event with normalized vector and coordinate."""
        v = np.array(vector, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v = v / norm

        if len(self.entries) >= self.max_entries:
            # Evict entry with lowest reward or oldest step
            min_idx = min(range(len(self.entries)), key=lambda i: (self.entries[i].reward, -self.entries[i].step))
            self.entries.pop(min_idx)

        self.entries.append(
            SearchIndexEntry(
                concept=str(concept),
                vector=v,
                coordinate=(float(coordinate[0]), float(coordinate[1]), float(coordinate[2])),
                action=int(action),
                reward=float(reward),
                step=int(step),
            )
        )

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Perform cosine resonance associative search across indexed holographic concepts."""
        self.total_queries += 1
        if not self.entries:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(qv))
        if q_norm > 1e-6:
            qv = qv / q_norm

        results = []
        for entry in self.entries:
            dot = float(np.dot(qv, entry.vector))
            if dot >= self.resonance_threshold:
                results.append({
                    "concept": entry.concept,
                    "similarity": round(dot, 4),
                    "action": entry.action,
                    "reward": round(entry.reward, 4),
                    "coordinate": list(entry.coordinate),
                    "step": entry.step,
                })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        top_results = results[:top_k]
        if top_results:
            self.total_hits += 1
        return top_results

    def search_by_spatial_radius(
        self,
        center: Tuple[float, float, float],
        radius: float = 8.0,
    ) -> List[Dict[str, Any]]:
        """Query entries within geodesic lattice Euclidean distance radius."""
        cx, cy, cz = center
        matches = []
        for entry in self.entries:
            ex, ey, ez = entry.coordinate
            dist = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2)
            if dist <= radius:
                matches.append({
                    "concept": entry.concept,
                    "distance": round(dist, 4),
                    "action": entry.action,
                    "reward": round(entry.reward, 4),
                    "coordinate": list(entry.coordinate),
                    "step": entry.step,
                })
        matches.sort(key=lambda m: m["distance"])
        return matches

    def get_highest_reward_concept(self) -> Optional[Dict[str, Any]]:
        """Retrieve highest reward concept encountered in the current lifetime."""
        if not self.entries:
            return None
        best = max(self.entries, key=lambda e: e.reward)
        return {
            "concept": best.concept,
            "reward": round(best.reward, 4),
            "action": best.action,
            "coordinate": list(best.coordinate),
            "step": best.step,
        }

    def stats(self) -> Dict[str, Any]:
        """Return index summary statistics."""
        return {
            "indexed_count": len(self.entries),
            "capacity": self.max_entries,
            "total_queries": self.total_queries,
            "total_hits": self.total_hits,
            "hit_ratio": round(self.total_hits / max(1, self.total_queries), 4),
        }
