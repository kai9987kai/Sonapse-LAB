"""Directional Causal Graph Discovery & Pearlian Path Tracing Engine.

Grounding:
Integrates structural equation discovery (DirectLiNGAM, Shimizu et al., 2006) and
Pearlian interventional path tracing (Pearl 2009; Peters et al. 2017) over
multi-connectome regional activity streams.

Key Capabilities:
1. DirectLiNGAM Causal Ordering: Identifies acyclic causal hierarchies between
   connectome neuropils by iteratively selecting exogenous variables whose residuals
   are maximally independent.
2. Causal Path Tracing: Traces the most influential causal pathways from sensory
   neuropils (Optic Lobe / Antenna) through central neuropils (Central Complex /
   Mushroom Body) to motor output neuropils (MANC).
3. Interventional Transmission Strength: Measures product of path coefficients
   to determine signal throughput under hypothetical counterfactual interventions.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CausalPathTracer:
    """Discovers directional causal DAGs and traces interventional pathways."""

    def __init__(self, region_names: Optional[List[str]] = None, seed: int = 42):
        self.region_names = region_names or [
            "optic-lobe", "antennal_lobe", "mushroom_body", "central_complex", "flywire_central", "manc_vnc"
        ]
        self.num_regions = len(self.region_names)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def fit_direct_lingam(self, time_series_data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Fits DirectLiNGAM to discover causal DAG adjacency matrix B where X = B X + e.

        Args:
            time_series_data: array of shape (T, num_regions)

        Returns:
            (causal_adjacency_matrix, causal_topological_order)
        """
        X = np.asarray(time_series_data, dtype=np.float64)
        T, p = X.shape
        if p != self.num_regions:
            # Adjust to matching dimensions
            p = min(p, self.num_regions)
            X = X[:, :p]

        # Standardize columns
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        stds = np.std(X_centered, axis=0, keepdims=True)
        X_norm = X_centered / np.maximum(stds, 1e-6)

        remaining_indices = list(range(p))
        causal_order = []
        B = np.zeros((p, p), dtype=np.float32)

        # Iterative selection of most exogenous variable
        for _ in range(p):
            best_idx = remaining_indices[0]
            best_score = float("inf")

            for cand in remaining_indices:
                # Regress cand out of all other remaining variables and test non-Gaussianity
                score = 0.0
                x_cand = X_norm[:, cand]
                for other in remaining_indices:
                    if other == cand:
                        continue
                    x_other = X_norm[:, other]
                    # Simple linear regression coefficient
                    slope = np.cov(x_other, x_cand)[0, 1] / max(1e-6, np.var(x_cand))
                    residual = x_other - slope * x_cand
                    # Kurtosis departure from Gaussianity as independence proxy
                    kurt = float(np.mean(residual ** 4) / max(1e-6, np.mean(residual ** 2) ** 2)) - 3.0
                    score += abs(kurt)

                if score < best_score:
                    best_score = score
                    best_idx = cand

            causal_order.append(best_idx)
            remaining_indices.remove(best_idx)

        # Regress each variable on all its causal predecessors to form B
        for i, target in enumerate(causal_order):
            predecessors = causal_order[:i]
            if not predecessors:
                continue
            X_pred = X_norm[:, predecessors]
            y = X_norm[:, target]
            try:
                # Ridge regression
                reg = np.linalg.lstsq(
                    X_pred.T @ X_pred + 1e-4 * np.eye(len(predecessors)),
                    X_pred.T @ y,
                    rcond=None,
                )[0]
                for p_idx, coef in zip(predecessors, reg):
                    B[target, p_idx] = float(np.clip(coef, -2.0, 2.0))
            except Exception:
                pass

        return B, causal_order

    def trace_causal_paths(
        self,
        causal_matrix: np.ndarray,
        source_idx: int = 0,
        target_idx: Optional[int] = None,
        max_depth: int = 4,
    ) -> List[Dict[str, Any]]:
        """Traces top causal pathways from source_idx to target_idx."""
        p = len(causal_matrix)
        target = target_idx if target_idx is not None else (p - 1)

        paths = []

        def dfs(current: int, current_path: List[int], current_gain: float, depth: int):
            if current == target and len(current_path) > 1:
                names = [self.region_names[i] if i < len(self.region_names) else f"node_{i}" for i in current_path]
                paths.append({
                    "path_indices": current_path,
                    "path_names": names,
                    "path_length": len(current_path) - 1,
                    "causal_transmission_gain": round(float(current_gain), 4),
                })
                return
            if depth >= max_depth:
                return

            # Follow outgoing edges (where target has non-zero incoming weight from current)
            for next_node in range(p):
                edge_weight = causal_matrix[next_node, current]
                if abs(edge_weight) > 0.05 and next_node not in current_path:
                    dfs(next_node, current_path + [next_node], current_gain * edge_weight, depth + 1)

        dfs(source_idx, [source_idx], 1.0, 0)
        # Sort paths by absolute causal transmission gain
        paths.sort(key=lambda x: abs(x["causal_transmission_gain"]), reverse=True)
        return paths[:5]

    def run_causal_path_discovery_assay(
        self,
        time_series_data: Optional[np.ndarray] = None,
        steps: int = 64,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Runs complete DirectLiNGAM and causal path tracing assay."""
        if time_series_data is None:
            rng = np.random.default_rng(seed)
            # Generate synthetic 6-region cascading signal
            T = steps
            X = np.zeros((T, self.num_regions), dtype=np.float32)
            # Exogenous source drive
            X[:, 0] = rng.standard_normal(T)
            for t in range(T):
                for j in range(1, self.num_regions):
                    prev = X[t, j - 1] if t > 0 else 0.0
                    X[t, j] = 0.65 * prev + 0.2 * rng.standard_normal()
            time_series_data = X

        B, order = self.fit_direct_lingam(time_series_data)
        top_paths = self.trace_causal_paths(B, source_idx=0, target_idx=min(len(B) - 1, self.num_regions - 1))

        order_names = [self.region_names[i] if i < len(self.region_names) else f"node_{i}" for i in order]

        return {
            "schema": "direct-lingam-causal-path-discovery-v1",
            "status": "completed",
            "num_regions": self.num_regions,
            "causal_topological_order": order_names,
            "num_discovered_paths": len(top_paths),
            "top_pathways": top_paths,
            "max_causal_path_gain": top_paths[0]["causal_transmission_gain"] if top_paths else 0.0,
            "is_strictly_acyclic": True,
        }
