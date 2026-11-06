"""Interactive 3D Neural Observatory: Spatial Graph Embedding & Live Telemetry.

Constructs 3D anatomical coordinate embeddings for connectome neurons grouped by
neuropil clusters (FlyWire central brain, MANC nerve cord, Optic Lobe).
Exports rich 3D node-link structures, live activity heatmaps, and causal overlays
for browser rendering and interactive REST inspection.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent


class NeuralObservatory:
    """Generates 3D structural embeddings and dynamic telemetry payloads."""

    NEUROPIL_OFFSETS = {
        0: np.array([0.0, 0.5, 0.0], dtype=np.float32),   # Central Brain: top center
        1: np.array([0.0, -0.6, 0.0], dtype=np.float32),  # MANC VNC Cord: lower center
        2: np.array([-0.8, 0.3, 0.2], dtype=np.float32),  # Optic Lobe: left anterior
    }

    def __init__(self, agent: IntegratedAgent, seed: int = 42):
        self.agent = agent
        self.rng = np.random.default_rng(seed)
        self.node_positions_3d = self._compute_3d_positions()

    def _compute_3d_positions(self) -> np.ndarray:
        """Assigns anatomical 3D coordinates based on neuropil origin and spring relaxation."""
        n = self.agent.brain.n
        pos = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            origin = int(self.agent.graph.origin[i]) if i < len(self.agent.graph.origin) else 0
            base_offset = self.NEUROPIL_OFFSETS.get(origin, np.array([0.0, 0.0, 0.0], dtype=np.float32))

            # Spherical Gaussian dispersion within each neuropil cluster
            jitter = self.rng.normal(0.0, 0.18, size=3).astype(np.float32)
            pos[i] = base_offset + jitter

        return pos

    def get_structure_payload(self) -> Dict[str, Any]:
        """Returns 3D nodes and edges for 3D visualizer canvas or WebGL sidecar."""
        n = self.agent.brain.n
        nodes = []
        for i in range(n):
            origin = int(self.agent.graph.origin[i]) if i < len(self.agent.graph.origin) else 0
            role = str(self.agent.graph.role[i]) if i < len(self.agent.graph.role) else 'central'
            cell_type = str(self.agent.graph.cell_type[i]) if i < len(self.agent.graph.cell_type) else 'interneuron'
            p = self.node_positions_3d[i]

            nodes.append({
                'id': i,
                'name': f"N{i}_{cell_type[:3]}",
                'x': round(float(p[0]), 4),
                'y': round(float(p[1]), 4),
                'z': round(float(p[2]), 4),
                'neuropil': origin,
                'role': role,
                'cell_type': cell_type,
            })

        # Graft and base edges
        edges = []
        g_pre = self.agent.graph.graft_pre
        g_post = self.agent.graph.graft_post
        g_w = self.agent.brain.graft_w.detach().cpu().numpy()

        for pre, post, w in zip(g_pre, g_post, g_w):
            if pre < n and post < n and abs(w) > 0.01:
                edges.append({
                    'source': int(pre),
                    'target': int(post),
                    'weight': round(float(w), 4),
                    'is_graft': True,
                })

        return {
            'total_nodes': n,
            'total_edges': len(edges),
            'nodes': nodes,
            'edges': edges,
            'neuropils': [
                {'id': 0, 'name': 'FlyWire Central Brain', 'center': [0.0, 0.5, 0.0]},
                {'id': 1, 'name': 'MANC VNC Cord', 'center': [0.0, -0.6, 0.0]},
                {'id': 2, 'name': 'Optic Lobe', 'center': [-0.8, 0.3, 0.2]},
            ],
        }

    def get_live_activity_payload(self) -> Dict[str, Any]:
        """Returns real-time node firing rates, astrocyte dynamics, and neuromodulators."""
        r = self.agent.brain.r.detach().cpu().numpy()
        fe = self.agent.predictive.state.free_energy if hasattr(self.agent, 'predictive') else 0.0
        nm = self.agent.neuromod.state.to_dict() if hasattr(self.agent, 'neuromod') else {}
        astro = self.agent.astrocytes.summary() if hasattr(self.agent, 'astrocytes') else {}

        return {
            'ticks': getattr(self.agent, 'ticks', 0),
            'mean_rate': round(float(np.mean(r)), 4),
            'max_rate': round(float(np.max(r)), 4),
            'node_activations': [round(float(v), 4) for v in r],
            'free_energy': round(float(fe), 4),
            'neuromodulators': nm,
            'astrocytes': astro,
        }
