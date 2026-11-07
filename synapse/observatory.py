"""Schematic source/role graph layout and actual executing activity.

Coordinates are generated display positions, never measured anatomy. Catalog
metadata identifies a source; executing provenance identifies the representation
actually loaded. Base adjacency and active graft links remain distinguishable.
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from .agent import IntegratedAgent
from .connectome_datasets import CONNECTOME_CATALOG, ALL_DATASET_IDS


class NeuralObservatory:
    """Generates reproducible display positions and runtime telemetry."""

    # Kept for API compatibility; actual display centers follow configured sources.
    NEUROPIL_OFFSETS: Dict[int, np.ndarray] = {
        0: np.array([0.0, 0.4, 0.0], dtype=np.float32),    # Region 0: Central Brain / FlyWire / MaleCNS
        1: np.array([0.0, -0.6, 0.0], dtype=np.float32),   # Region 1: MANC / VNC Cord / BANC
        2: np.array([-0.8, 0.3, 0.2], dtype=np.float32),   # Region 2: Optic Lobe (left anterior)
        3: np.array([0.3, -0.6, -0.2], dtype=np.float32),  # Region 3: FANC (lower right)
        4: np.array([0.0, 0.7, 0.1], dtype=np.float32),    # Region 4: FlyWire FAFB
        5: np.array([-0.3, -0.6, 0.2], dtype=np.float32),  # Region 5: MANC
        6: np.array([0.6, 0.1, 0.3], dtype=np.float32),    # Region 6: Larval Brain
        7: np.array([0.5, 0.4, -0.2], dtype=np.float32),   # Region 7: Hemibrain
    }

    NEUROPIL_NAMES: Dict[int, str] = {
        0: 'Central Brain / FlyWire / MaleCNS',
        1: 'VNC Nerve Cord / MANC / BANC',
        2: 'FlyEM Optic Lobe',
        3: 'FANC Female Nerve Cord',
        4: 'FlyWire FAFB v783',
        5: 'MANC Male Nerve Cord',
        6: 'Larval Drosophila Brain',
        7: 'Hemibrain Central Complex',
    }

    def __init__(self, agent: IntegratedAgent, seed: int = 42):
        self.agent = agent
        self.rng = np.random.default_rng(seed)
        self.source_names = tuple(getattr(agent.config, 'connectome_sources', ('flywire', 'manc', 'optic-lobe')))
        count = len(self.source_names)
        self.source_centers = {i: np.array([.68 * math.cos(i / count * 2 * math.pi - math.pi / 2),
                                           .68 * math.sin(i / count * 2 * math.pi - math.pi / 2), 0.], dtype=np.float32)
                               if count > 1 else np.zeros(3, dtype=np.float32) for i in range(count)}
        self.node_positions_3d = self._compute_3d_positions()

    def _compute_3d_positions(self) -> np.ndarray:
        """Assign stable schematic coordinates grouped by source and role."""
        n = self.agent.brain.n
        pos = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            origin = int(self.agent.graph.origin[i]) if i < len(self.agent.graph.origin) else 0
            base_offset = self.source_centers.get(origin, np.zeros(3, dtype=np.float32)).copy()
            role = str(self.agent.graph.role[i])
            base_offset[2] += {'sensory': -.25, 'visual': -.17, 'ascending': -.08,
                               'central': 0., 'descending': .14, 'motor': .25}.get(role, 0.)
            jitter = self.rng.normal(0.0, 0.10, size=3).astype(np.float32)
            pos[i] = base_offset + jitter

        return pos

    def get_structure_payload(self) -> Dict[str, Any]:
        """Returns 3D nodes and edges for 3D visualizer canvas or WebGL sidecar."""
        n = self.agent.brain.n
        nodes = []
        source_names = self.source_names
        provenance = getattr(self.agent, 'source_provenance', [])

        for i in range(n):
            origin = int(self.agent.graph.origin[i]) if i < len(self.agent.graph.origin) else 0
            role = str(self.agent.graph.role[i]) if i < len(self.agent.graph.role) else 'central'
            cell_type = str(self.agent.graph.cell_type[i]) if i < len(self.agent.graph.cell_type) else 'interneuron'
            p = self.node_positions_3d[i]

            dataset_id = source_names[origin] if origin < len(source_names) else 'unknown'
            meta = CONNECTOME_CATALOG.get(dataset_id)

            nodes.append({
                'id': i,
                'name': f"N{i}_{cell_type[:3]}",
                'x': round(float(p[0]), 4),
                'y': round(float(p[1]), 4),
                'z': round(float(p[2]), 4),
                'neuropil': origin,
                'dataset': dataset_id,
                'dataset_name': meta.name if meta else dataset_id,
                'license': meta.license if meta else 'unknown',
                'authors': meta.organisation_authors if meta else 'unknown',
                'role': role,
                'cell_type': cell_type,
                'provenance': deepcopy(provenance[origin]) if origin < len(provenance) else {'representation': 'unknown'},
            })

        # Export bounded samples separately, with actual active graft membership.
        edges = []
        graph = self.agent.graph
        base_count = len(graph.base_pre)
        for index in np.linspace(0, base_count-1, min(2000, base_count), dtype=int):
            edges.append({'source': int(graph.base_pre[index]), 'target': int(graph.base_post[index]),
                          'weight': round(float(graph.base_w[index]), 4), 'is_graft': False})
        g_pre = self.agent.graph.graft_pre
        g_post = self.agent.graph.graft_post
        g_w = self.agent.brain.graft_w.detach().cpu().numpy()

        active = self.agent.plasticity.active_mask.detach().cpu().numpy()
        active_graft_count = int(np.count_nonzero(active))
        graft_links = []
        for index, (pre, post, w) in enumerate(zip(g_pre, g_post, g_w)):
            if pre < n and post < n and active[index] and len(graft_links) < 2000:
                link = {
                    'source': int(pre), 'target': int(post),
                    'weight': round(float(w), 4), 'is_graft': True,
                }
                graft_links.append(link)
                edges.append({
                    **link,
                })

        neuropils_list = []
        for i, s_name in enumerate(source_names):
            meta = CONNECTOME_CATALOG.get(s_name)
            neuropils_list.append({
                'id': i,
                'dataset_id': s_name,
                'name': meta.name if meta else s_name,
                'authors': meta.organisation_authors if meta else 'unknown',
                'license': meta.license if meta else 'unknown',
                'center': self.source_centers[i].tolist(),
                'provenance': deepcopy(provenance[i]) if i < len(provenance) else {'representation': 'unknown'},
            })

        # Generate hierarchical Level-Of-Detail (LOD) clusters for scalable WebGL rendering
        lod_clusters = []
        cluster_id = 0
        for origin_idx, s_name in enumerate(source_names):
            source_node_indices = [i for i in range(n) if (int(self.agent.graph.origin[i]) if i < len(self.agent.graph.origin) else 0) == origin_idx]
            if not source_node_indices:
                continue
            # Partition each source into sub-clusters by role and spatial quadrant
            for role_name in ('sensory', 'central', 'motor'):
                role_nodes = [i for i in source_node_indices if (str(self.agent.graph.role[i]) if i < len(self.agent.graph.role) else 'central') == role_name]
                if not role_nodes:
                    continue
                pts = self.node_positions_3d[role_nodes]
                centroid = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if len(pts) > 1 else 0.5
                lod_clusters.append({
                    'id': cluster_id,
                    'dataset': s_name,
                    'role': role_name,
                    'node_count': len(role_nodes),
                    'x': round(float(centroid[0]), 4),
                    'y': round(float(centroid[1]), 4),
                    'z': round(float(centroid[2]), 4),
                    'radius': round(radius, 4),
                    'sample_node_ids': role_nodes[:10],
                })
                cluster_id += 1

        return {
            'total_nodes': n,
            'total_edges': base_count + active_graft_count,
            'displayed_edges': len(edges),
            'base_edges': base_count,
            'active_graft_edges': active_graft_count,
            'graft_links': graft_links,
            'lod_clusters': lod_clusters,
            'num_lod_clusters': len(lod_clusters),
            'layout_scope': 'schematic source/role layout; coordinates are not measured anatomy',
            'nodes': nodes,
            'edges': edges,
            'neuropils': neuropils_list,
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
