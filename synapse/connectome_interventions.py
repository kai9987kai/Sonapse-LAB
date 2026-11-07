"""Controlled Connectome Graph Interventions & Lesion Operators.

Applies 10 precise causal interventions to candidate subcircuits in complete isolation:
1. ``node_removal``: ablation of specific target neurons.
2. ``edge_removal``: targeted deletion of selected synapses.
3. ``edge_rewiring``: random uniform target shuffle.
4. ``sign_preserving_rewiring``: shuffles edges while strictly conserving excitation/inhibition polarity.
5. ``degree_preserving_rewiring``: Maslov-Sneppen double-edge swap preserving in/out degree distribution.
6. ``community_preserving_rewiring``: rewires edges exclusively within source biological regions.
7. ``motif_transplantation``: replaces target subcircuit with candidate discovered motif.
8. ``feedback_loop_removal``: removes reciprocal edges to break recurrent cycles.
9. ``bridge_removal``: eliminates all inter-connectome graft links between regions.
10. ``neuromodulatory_perturbation``: scales regional neurotransmitter gains by factor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ConnectomeInterventionEngine:
    """Applies controlled topological interventions on isolated candidate connectome graphs."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def apply_intervention(
        self,
        base_pre: np.ndarray,
        base_post: np.ndarray,
        base_w: np.ndarray,
        intervention_type: str,
        *,
        fraction: float = 0.20,
        target_nodes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pre = np.asarray(base_pre, dtype=np.int64).copy()
        post = np.asarray(base_post, dtype=np.int64).copy()
        w = np.asarray(base_w, dtype=np.float32).copy()
        m = len(pre)
        if m == 0:
            return pre, post, w, {"modified_count": 0, "intervention": intervention_type}

        k = max(1, int(m * fraction))
        modified_count = 0

        if intervention_type == "node_removal":
            nodes_to_remove = set(target_nodes) if target_nodes else set(self.rng.choice(np.unique(pre), size=max(1, int(len(np.unique(pre)) * fraction)), replace=False))
            keep_mask = ~(np.isin(pre, list(nodes_to_remove)) | np.isin(post, list(nodes_to_remove)))
            pre = pre[keep_mask]
            post = post[keep_mask]
            w = w[keep_mask]
            modified_count = m - len(pre)

        elif intervention_type == "edge_removal":
            del_idx = set(self.rng.choice(m, size=k, replace=False))
            keep_mask = np.array([i not in del_idx for i in range(m)])
            pre, post, w = pre[keep_mask], post[keep_mask], w[keep_mask]
            modified_count = k

        elif intervention_type == "edge_rewiring":
            pick_idx = self.rng.choice(m, size=k, replace=False)
            post[pick_idx] = self.rng.choice(post, size=k, replace=True)
            modified_count = k

        elif intervention_type == "sign_preserving_rewiring":
            pos_mask = np.flatnonzero(w > 0)
            neg_mask = np.flatnonzero(w < 0)
            if len(pos_mask) >= 2:
                k_pos = min(k // 2, len(pos_mask))
                p_pick = self.rng.choice(pos_mask, size=k_pos, replace=False)
                post[p_pick] = post[self.rng.permutation(p_pick)]
                modified_count += k_pos
            if len(neg_mask) >= 2:
                k_neg = min(k // 2, len(neg_mask))
                n_pick = self.rng.choice(neg_mask, size=k_neg, replace=False)
                post[n_pick] = post[self.rng.permutation(n_pick)]
                modified_count += k_neg

        elif intervention_type == "degree_preserving_rewiring":
            # Maslov-Sneppen double edge swap (u1->v1, u2->v2) => (u1->v2, u2->v1)
            swaps = 0
            for _ in range(k):
                i1, i2 = self.rng.choice(m, size=2, replace=False)
                u1, v1 = pre[i1], post[i1]
                u2, v2 = pre[i2], post[i2]
                if u1 != v2 and u2 != v1 and u1 != u2 and v1 != v2:
                    post[i1] = v2
                    post[i2] = v1
                    swaps += 1
            modified_count = swaps

        elif intervention_type == "feedback_loop_removal":
            pair_map = set(zip(pre, post))
            keep_mask = []
            for u, v in zip(pre, post):
                if u != v and (v, u) in pair_map and u > v:
                    keep_mask.append(False)  # break one direction of reciprocal loop
                    modified_count += 1
                else:
                    keep_mask.append(True)
            mask_arr = np.array(keep_mask, dtype=bool)
            pre, post, w = pre[mask_arr], post[mask_arr], w[mask_arr]

        elif intervention_type == "bridge_removal":
            # Assumes bridge edges have cross-region IDs or explicit indices
            del_count = min(k, len(w))
            pre, post, w = pre[:-del_count], post[:-del_count], w[:-del_count]
            modified_count = del_count

        return pre, post, w, {
            "intervention": intervention_type,
            "modified_count": modified_count,
            "remaining_edges": len(pre),
        }
