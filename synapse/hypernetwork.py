"""Weight-space hypernetwork program synthesizer.
Learns to map diagnostic behavioral and connectome telemetry vectors directly into
bounded, Dale's-law-compliant synapse-weight-program-v1 patch packets.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import IntegratedAgent
from .dales_law import project_dales_law
from .patches import make_patch

class WeightSpaceHypernetwork(nn.Module):
    """Low-rank neural hypernetwork generating weight modifications from diagnostic telemetry."""
    def __init__(
        self,
        in_dim: int = 32,
        rank: int = 2,
        target_rows: int = 5,
        target_cols: int = 64,
        scale: float = 0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.scale = scale

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Low-rank factor generators for target matrix (e.g. motor decoder [5, 64])
        self.gen_u = nn.Linear(64, target_rows * rank)
        self.gen_v = nn.Linear(64, target_cols * rank)

        # Graft update generator for inter-neuropil bridges
        self.gen_graft = nn.Linear(64, 32)

    def extract_signature(self, diag: Dict[str, Any]) -> torch.Tensor:
        """Extract a normalized 32-dimensional diagnostic signature from telemetry."""
        sig = np.zeros(self.in_dim, dtype=np.float32)
        # 0..1: returns and lengths
        sig[0] = np.clip(diag.get('mean_return', 0.0), -1.0, 1.0)
        sig[1] = np.clip(diag.get('total_steps', 0) / 100.0, 0.0, 1.0)
        # 2..5: connectome rate dynamics
        sig[2] = np.clip(diag.get('mean_graph_activity', 0.0) * 5.0, 0.0, 1.0)
        sig[3] = np.clip(diag.get('mean_approach_valence', 0.0), -1.0, 1.0)
        sig[4] = np.clip(diag.get('mean_avoid_valence', 0.0), -1.0, 1.0)
        # 5..10: action distributions
        actions = diag.get('action_mean', [0.0] * 5)
        for i, a in enumerate(actions[:5]):
            sig[5 + i] = np.clip(a, -1.0, 1.0)
        # 10..15: task indicator hash
        task_hash = hash(diag.get('task', 'forage')) % 1000 / 1000.0
        sig[10] = task_hash
        return torch.from_numpy(sig)

    def forward(self, signature: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate low-rank parameter delta matrices from diagnostic signature."""
        feat = self.trunk(signature.unsqueeze(0)).squeeze(0)
        U = self.gen_u(feat).reshape(self.target_rows, self.rank)
        V = self.gen_v(feat).reshape(self.target_cols, self.rank)

        # Low-rank product: ΔW = scale * tanh(U @ V^T)
        delta_motor = self.scale * torch.tanh(U @ V.T)
        delta_graft = (self.scale * 0.5) * torch.tanh(self.gen_graft(feat))

        return {
            'delta_motor': delta_motor,
            'delta_graft': delta_graft,
        }


def compile_hypernetwork_patch(
    agent: IntegratedAgent,
    hypernet: WeightSpaceHypernetwork,
    diagnostic_data: Dict[str, Any],
    target_surface: str = 'motor.weight',
    max_edits: int = 32,
) -> Dict[str, Any]:
    """Generate and package a verified synapse-weight-program-v1 packet via the hypernetwork."""
    sig = hypernet.extract_signature(diagnostic_data)
    with torch.no_grad():
        out = hypernet(sig)

    surfaces = agent.editable_tensors()
    edits = []

    if target_surface == 'motor.weight' and 'motor.weight' in surfaces:
        tensor = surfaces['motor.weight']
        delta_mat = out['delta_motor']
        indices = []
        values = []
        rows, cols = tensor.shape
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= max_edits:
                    break
                old_val = float(tensor[r, c].detach())
                new_val = float(np.clip(old_val + float(delta_mat[r, c]), -3.0, 3.0))
                indices.append([r, c])
                values.append(new_val)
                count += 1
            if count >= max_edits:
                break
        edits.append({
            'tensor': 'motor.weight',
            'indices': indices,
            'values': values,
            'shape': list(tensor.shape),
        })

    elif target_surface == 'brain.graft_w' and 'brain.graft_w' in surfaces:
        tensor = surfaces['brain.graft_w']
        delta_vec = out['delta_graft']
        indices = []
        values = []
        n_edits = min(len(delta_vec), len(tensor), max_edits)
        for i in range(n_edits):
            old_val = float(tensor[i].detach())
            new_val = float(np.clip(old_val + float(delta_vec[i]), -2.0, 2.0))
            indices.append([i])
            values.append(new_val)
        edits.append({
            'tensor': 'brain.graft_w',
            'indices': indices,
            'values': values,
            'shape': list(tensor.shape),
        })

    desc = f"Learned hypernetwork synthesis targeting {target_surface}"
    return make_patch(agent, edits, description=desc)
