"""Sparse Neuropil Arbiter: Kenyon Cell Expansion & Tri-Drive Homeostatic Arbitration.

Inspired by Drosophila mushroom-body sparse coding (APL inhibitory feedback) and descending
neuropil motor arbitration (Harvester, Sentinel, Escaper).
Expands 16-dimensional observation vectors into 128 virtual Kenyon cells with exact
k-winners-take-all (APL inhibition, 6.25% sparsity), and regulates behavior across
tri-drive homeostatic channels:
  1. Metabolic drive (energy / nutrient deficit)
  2. Fatigue drive (stamina exhaustion)
  3. Threat drive (depth hazard / predator proximity / adverse flow)
"""
from __future__ import annotations

import math
from typing import Any, Tuple
import numpy as np


class SparseNeuropilArbiter:
    """128-cell Kenyon sparse projection and tri-drive descending motor arbiter."""

    NUM_KENYON_CELLS = 128
    SPARSITY_K = 8  # 8 out of 128 = 6.25% active

    def __init__(self, obs_dim: int = 16, seed: int = 42) -> None:
        self.obs_dim = obs_dim
        rng = np.random.RandomState(seed)
        # Sparse random projection matrix: each KC samples ~4 random inputs
        self.w_proj = np.zeros((self.NUM_KENYON_CELLS, obs_dim), dtype=np.float64)
        for i in range(self.NUM_KENYON_CELLS):
            inputs = rng.choice(obs_dim, size=4, replace=False)
            self.w_proj[i, inputs] = rng.uniform(0.5, 1.5, size=4)
        
        # Drive weights for descending channels
        self.w_descending = {
            "harvester": rng.uniform(0.1, 0.5, size=self.NUM_KENYON_CELLS),
            "sentinel": rng.uniform(0.1, 0.5, size=self.NUM_KENYON_CELLS),
            "escaper": rng.uniform(0.1, 0.5, size=self.NUM_KENYON_CELLS),
        }

    def compute_kenyon_activity(self, obs: np.ndarray) -> np.ndarray:
        """Project observation through KCs with APL k-winners-take-all inhibition."""
        raw_drive = np.dot(self.w_proj, obs[:self.obs_dim])
        
        # APL inhibitory thresholding: top-k winners fire
        sorted_drives = np.sort(raw_drive)[::-1]
        top_k_indices = np.argsort(raw_drive)[::-1][:self.SPARSITY_K]
        baseline = sorted_drives[self.SPARSITY_K] if self.SPARSITY_K < len(sorted_drives) else 0.0
        
        kc_activity = np.zeros(self.NUM_KENYON_CELLS, dtype=np.float64)
        kc_activity[top_k_indices] = (raw_drive[top_k_indices] - baseline) + 0.1
        
        norm = np.linalg.norm(kc_activity)
        if norm > 1e-8:
            kc_activity = kc_activity / norm
        return kc_activity

    def compute_homeostatic_drives(self, obs: list[float] | np.ndarray) -> dict[str, float]:
        """Compute tri-drive homeostatic state vector from TidePool observation.
        
        TidePool observation layout:
          0: energy (0..1)
          1: stamina (0..1)
          2: nutrients (0..1)
          3: depth (0..1)
          4: flow_x (-1..1)
          5: flow_y (-1..1)
          6: food_distance (0..1)
          7: food_angle (-pi..pi)
          8: hazard_distance (0..1)
          9: hazard_angle (-pi..pi)
          10: predator_distance (0..1)
          11: predator_angle (-pi..pi)
          12: patch_count
          13: nearest_patch_dist
          14: nearest_patch_replenish
          15: step
        """
        energy = float(obs[0]) if len(obs) > 0 else 0.5
        stamina = float(obs[1]) if len(obs) > 1 else 0.5
        nutrients = float(obs[2]) if len(obs) > 2 else 0.5
        depth = float(obs[3]) if len(obs) > 3 else 0.5
        hazard_dist = float(obs[8]) if len(obs) > 8 else 1.0
        predator_dist = float(obs[10]) if len(obs) > 10 else 1.0
        
        # 1. Metabolic drive: hunger/nutrient depletion
        d_metabolic = max(0.0, min(1.0, (1.0 - energy) * 0.6 + (1.0 - nutrients) * 0.4))
        
        # 2. Fatigue drive: stamina exhaustion
        d_fatigue = max(0.0, min(1.0, 1.0 - stamina))
        
        # 3. Threat drive: hazard depth, predator proximity, toxic flow
        depth_threat = max(0.0, (depth - 0.75) / 0.25) if depth > 0.75 else 0.0
        proximity_threat = max(0.0, 1.0 - min(hazard_dist, predator_dist))
        d_threat = max(0.0, min(1.0, 0.4 * depth_threat + 0.6 * proximity_threat))
        
        return {
            "metabolic": round(float(d_metabolic), 4),
            "fatigue": round(float(d_fatigue), 4),
            "threat": round(float(d_threat), 4),
        }

    def arbitrate(self, obs: list[float] | np.ndarray) -> dict[str, Any]:
        """Arbitrate descending motor pathways based on sparse KCs and homeostatic drives."""
        obs_arr = np.array(obs[:self.obs_dim], dtype=np.float64)
        kc_activity = self.compute_kenyon_activity(obs_arr)
        drives = self.compute_homeostatic_drives(obs_arr)
        
        # Base pathway activations from KC sparse representations
        act_harvester = float(np.dot(self.w_descending["harvester"], kc_activity))
        act_sentinel = float(np.dot(self.w_descending["sentinel"], kc_activity))
        act_escaper = float(np.dot(self.w_descending["escaper"], kc_activity))
        
        # Modulate by homeostatic drives
        score_harvester = act_harvester * (1.0 + 2.0 * drives["metabolic"]) * (1.0 - 0.5 * drives["threat"])
        score_sentinel = act_sentinel * (1.0 + 1.5 * drives["fatigue"]) * (1.0 + 0.5 * drives["threat"])
        score_escaper = act_escaper * (1.0 + 4.0 * (drives["threat"] ** 2))
        
        scores = {
            "harvester": max(0.0, score_harvester),
            "sentinel": max(0.0, score_sentinel),
            "escaper": max(0.0, score_escaper),
        }
        
        total = sum(scores.values()) + 1e-8
        probs = {k: v / total for k, v in scores.items()}
        winning_channel = max(scores, key=lambda k: scores[k])
        
        return {
            "active_kc_count": int(np.count_nonzero(kc_activity)),
            "sparsity_ratio": float(np.count_nonzero(kc_activity) / self.NUM_KENYON_CELLS),
            "homeostatic_drives": drives,
            "channel_scores": {k: round(v, 4) for k, v in scores.items()},
            "channel_probs": {k: round(v, 4) for k, v in probs.items()},
            "winning_channel": winning_channel,
            "confidence": round(float(probs[winning_channel]), 4),
        }
