"""POSEIDON MEMORY-OBSERVATORY: Topological Manifold Observatory & Holographic Telemetry.

Synthesizes:
1. MEMORY-CARRIER-OBSERVATORY & Mnemorph: Persistent homology simplicial filtration,
   Betti number analysis (\\beta_0, \\beta_1), and Euler characteristic tracking over experience trajectories.
2. FLY-DIAMOND-NEXUS & NexusFlow: Topological phase transition detection identifying shifts
   between stable homeostatic limit cycles and chaotic mortality decay spirals.
3. S-Video & Quibble: Holographic scanline and telemetry packet synthesis for real-time visualization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Sequence

from .svideo import SVideoTelemetryEngine, ScanlineSignal


@dataclass
class TopologicalManifoldAudit:
    """Audit metrics representing the persistent homology of the memory replay manifold."""
    betti_0: int                # Connected components (behavioral attractor regimes)
    betti_1: int                # 1-cycles (homeostatic limit cycles)
    euler_characteristic: int   # \\chi = \\beta_0 - \\beta_1
    topological_entropy: float  # Shannon diversity over simplicial components
    is_homeostatic_cycle: bool  # True if \\beta_1 >= 1 (limit cycle intact)
    phase_transition_detected: bool  # True if topological bifurcation occurred
    regime_name: str
    sample_size: int
    manifold_signature_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "euler_characteristic": self.euler_characteristic,
            "topological_entropy": round(self.topological_entropy, 4),
            "is_homeostatic_cycle": self.is_homeostatic_cycle,
            "phase_transition_detected": self.phase_transition_detected,
            "regime_name": self.regime_name,
            "sample_size": self.sample_size,
            "manifold_signature_sha256": self.manifold_signature_sha256,
        }


class MemoryCarrierObservatory:
    """Topological persistent homology manifold observatory over agent experience replays."""

    def __init__(self, filtration_epsilon: float = 0.35):
        self.filtration_epsilon = filtration_epsilon
        self.audit_history: list[TopologicalManifoldAudit] = []
        self.telemetry_engine = SVideoTelemetryEngine()

    def analyze_trajectory_topology(
        self,
        trajectory: Sequence[Sequence[float]],
    ) -> TopologicalManifoldAudit:
        """Computes persistent homology simplicial complex over a somatic state trajectory."""
        n_points = len(trajectory)
        if n_points < 3:
            # Minimal default for trivial sequences
            audit = TopologicalManifoldAudit(
                betti_0=1,
                betti_1=0,
                euler_characteristic=1,
                topological_entropy=0.0,
                is_homeostatic_cycle=False,
                phase_transition_detected=False,
                regime_name="trivial_point",
                sample_size=n_points,
            )
            audit.manifold_signature_sha256 = self._compute_signature(audit)
            return audit

        # Subsample to maximum 64 points for bounded graph analysis
        step = max(1, n_points // 48)
        pts = [trajectory[i] for i in range(0, n_points, step)]
        v_count = len(pts)

        # Distance matrix & Adjacency Graph (1-skeleton of Vietoris-Rips complex)
        adj: dict[int, set[int]] = {i: set() for i in range(v_count)}
        e_count = 0

        for i in range(v_count):
            for j in range(i + 1, v_count):
                d = self._euclidean_dist(pts[i], pts[j])
                if d <= self.filtration_epsilon:
                    adj[i].add(j)
                    adj[j].add(i)
                    e_count += 1

        # 1. Compute Connected Components (Betti_0) via BFS
        visited = set()
        components = []
        for i in range(v_count):
            if i not in visited:
                comp = set()
                q = [i]
                visited.add(i)
                while q:
                    curr = q.pop(0)
                    comp.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append(comp)

        betti_0 = max(1, len(components))

        # 2. Approximate 1-cycles (Betti_1) using graph fundamental cycles:
        # For a graph, Betti 1 is E - V + Betti_0
        raw_betti_1 = e_count - v_count + betti_0
        betti_1 = max(0, raw_betti_1)

        # Euler Characteristic \\chi = V - E = \\beta_0 - \\beta_1
        chi = betti_0 - betti_1

        # Topological Entropy: Shannon entropy over relative sizes of components
        entropy = 0.0
        for comp in components:
            p = len(comp) / v_count
            if p > 0:
                entropy -= p * math.log(p)

        # Homeostatic cycle exists if \\beta_1 >= 1 (orbit returns upon itself)
        is_homeostatic = betti_1 >= 1

        # Detect phase transition:
        # Multiple fragmented components (betti_0 > 2) or high entropy indicates fragmentation
        phase_transition = betti_0 >= 3 or (len(self.audit_history) > 0 and abs(self.audit_history[-1].betti_1 - betti_1) >= 2)

        regime = "homeostatic_limit_cycle" if is_homeostatic else ("fractured_fragmentation" if phase_transition else "open_transient_orbit")

        audit = TopologicalManifoldAudit(
            betti_0=betti_0,
            betti_1=betti_1,
            euler_characteristic=chi,
            topological_entropy=entropy,
            is_homeostatic_cycle=is_homeostatic,
            phase_transition_detected=phase_transition,
            regime_name=regime,
            sample_size=n_points,
        )
        audit.manifold_signature_sha256 = self._compute_signature(audit)
        self.audit_history.append(audit)
        return audit

    def synthesize_telemetry_frame(
        self,
        audit: TopologicalManifoldAudit,
        lyapunov_drift: float = 0.0,
        quantum_coherence: float = 1.0,
        monologue: str = "",
    ) -> dict[str, Any]:
        """Encodes topological and cognitive metrics into S-Video and holographic telemetry stream."""
        metrics = {
            "betti_0": float(audit.betti_0),
            "betti_1": float(audit.betti_1),
            "euler_chi": float(audit.euler_characteristic),
            "topological_entropy": audit.topological_entropy,
            "lyapunov_drift": lyapunov_drift,
            "quantum_coherence": quantum_coherence,
        }
        vitals = {
            "health": 1.0 if audit.is_homeostatic_cycle else 0.5,
            "energy": max(0.1, min(1.0, 1.0 - audit.topological_entropy)),
            "hydration": 0.8,
            "stamina": 0.85,
            "exposure": 0.1,
        }
        svideo_frame = self.telemetry_engine.encode_telemetry_frame(
            vitals=vitals,
            lattice_pos=[float(audit.betti_0), float(audit.betti_1), float(audit.euler_characteristic)],
            quantum_phases=[quantum_coherence, lyapunov_drift],
        )

        return {
            "metrics": metrics,
            "regime": audit.regime_name,
            "is_homeostatic": audit.is_homeostatic_cycle,
            "monologue_snippet": monologue[:80] if monologue else "Nominal orbital homeostasis",
            "frame_index": svideo_frame.get("frame_index"),
            "pcm_base64": svideo_frame.get("pcm_base64"),
            "svg_oscilloscope": svideo_frame.get("svg_oscillogram"),
            "signature_sha256": audit.manifold_signature_sha256,
        }

    def _euclidean_dist(self, a: Sequence[float], b: Sequence[float]) -> float:
        dim = min(len(a), len(b), 4)  # focus on core somatic dimensions [h, e, w, s]
        return math.sqrt(sum((a[k] - b[k])**2 for k in range(dim)))

    def _compute_signature(self, audit: TopologicalManifoldAudit) -> str:
        blob = {
            "b0": audit.betti_0,
            "b1": audit.betti_1,
            "chi": audit.euler_characteristic,
            "regime": audit.regime_name,
            "size": audit.sample_size,
        }
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()
