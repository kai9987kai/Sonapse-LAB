"""Heterogeneous Cross-Connectome Synaptic Grafting Engine.

Grounded in Kai9987kai's Supermix-Expanse and NEUROGRAFT architectures:
- Multi-dataset connectome bridging:
    MaleCNS (motor & courtship) <-> Optic-Lobe (retinotopy) <-> Hemibrain (central complex/MB) <-> FlyWire (whole brain)
- Dale's Law Biological Consistency:
    Strict presynaptic sign enforcement: all outgoing projections from presynaptic neuron j
    share identical sign polarity s_j in {+1, -1} across all postsynaptic targets i:
    W_bridge = softplus(W_param) * DaleSigns
- Sparse Edge Provenance & Fan-In Normalization:
    Anatomical region-to-region provenance masks with fan-in scaling (1 / sqrt(k_in))
- Contextual Affinity & Route Gating:
    Dynamic routing weights g(c_t) in [0, 1]^N conditioned on context and neuromodulatory state
- Bounded Residual Injection:
    I_cross,t = clip(gamma * P_target[g(c_t) * tanh(W_eff * r_t)], -B, B)

Zero-gain guarantee:
When gamma = 0.0, cross-connectome grafting produces strictly zero current bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossGraftTelemetry:
    """Live telemetry payload for Cross-Connectome Grafting Engine."""
    num_active_bridges: int = 4
    total_grafted_synapses: int = 2048
    dales_law_compliance_pct: float = 100.0
    mean_cross_affinity: float = 0.76
    current_l2_norm: float = 0.0
    participation_index: float = 1.0
    fan_in_norm_mean: float = 1.0
    sparse_density: float = 0.5
    donor_recipient_pairs: List[Dict[str, Any]] = field(default_factory=list)
    recent_bridge_activity: List[float] = field(default_factory=list)
    scientific_basis: str = "NEUROGRAFT & Supermix-Expanse Heterogeneous Cross-Species Connectome Grafting (Kai Piper)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_active_bridges': int(self.num_active_bridges),
            'total_grafted_synapses': int(self.total_grafted_synapses),
            'dales_law_compliance_pct': round(float(self.dales_law_compliance_pct), 1),
            'mean_cross_affinity': round(float(self.mean_cross_affinity), 4),
            'current_l2_norm': round(float(self.current_l2_norm), 6),
            'participation_index': round(float(self.participation_index), 4),
            'fan_in_norm_mean': round(float(self.fan_in_norm_mean), 4),
            'sparse_density': round(float(self.sparse_density), 4),
            'donor_recipient_pairs': self.donor_recipient_pairs[:8],
            'recent_bridge_activity': [round(float(a), 4) for a in self.recent_bridge_activity[-32:]],
            'scientific_basis': self.scientific_basis,
        }


class CrossConnectomeGraft(nn.Module):
    """
    Connectome-to-connectome synaptic bridge enforcing biological presynaptic Dale's law,
    sparse edge provenance, fan-in normalization, and adaptive contextual routing.
    """
    def __init__(
        self,
        node_dim: int = 64,
        gamma: float = 0.15,
        seed: int = 42,
        max_current: float = 2.0,
        target_dim: Optional[int] = None,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.gamma = float(gamma)
        self.seed = int(seed)
        self.max_current = float(max_current)
        self.target_dim = int(target_dim) if target_dim is not None else int(node_dim)
        self.rng = np.random.default_rng(seed)

        # 4 biological source connectomes
        self.regions = ['MaleCNS', 'OpticLobe', 'Hemibrain', 'FlyWire']
        self.block_size = max(1, self.node_dim // len(self.regions))

        # Biological Dale's law: presynaptic neurons determine sign across ALL outgoing synapses
        # 80% Excitatory (+1), 20% Inhibitory (-1)
        pre_signs_arr = np.where(self.rng.uniform(size=self.node_dim) > 0.2, 1.0, -1.0).astype(np.float32)
        # signs_arr[i, j] = pre_signs_arr[j] (column-wise constant signs)
        signs_matrix = np.tile(pre_signs_arr, (self.node_dim, 1))

        self.register_buffer('pre_signs', torch.tensor(pre_signs_arr, dtype=torch.float32))
        self.register_buffer('dale_signs', torch.tensor(signs_matrix, dtype=torch.float32))

        # Sparse edge provenance mask between regions
        # Construct anatomical connectivity paths:
        # OpticLobe (1) -> Hemibrain (2) [retinotopic visual routing]
        # Hemibrain (2) -> MaleCNS (0) [learned associative behavioral motor command]
        # FlyWire (3) -> MaleCNS (0) [whole-brain descending command]
        # MaleCNS (0) -> Hemibrain (2) [proprioceptive ascending reafference]
        prov_mask = np.zeros((self.node_dim, self.node_dim), dtype=np.float32)
        quad = self.block_size

        def block_slice(idx: int) -> slice:
            start = idx * quad
            end = self.node_dim if idx == len(self.regions) - 1 else (idx + 1) * quad
            return slice(start, end)

        # Primary inter-region cross-bridges
        bridges = [
            (1, 2),  # OpticLobe -> Hemibrain
            (2, 0),  # Hemibrain -> MaleCNS
            (3, 0),  # FlyWire -> MaleCNS
            (0, 2),  # MaleCNS -> Hemibrain
        ]
        for src_reg, dst_reg in bridges:
            s_src = block_slice(src_reg)
            s_dst = block_slice(dst_reg)
            # Density ~ 0.6 within authorized anatomical pathways
            pathway_rand = self.rng.uniform(size=(s_dst.stop - s_dst.start, s_src.stop - s_src.start))
            prov_mask[s_dst, s_src] = (pathway_rand < 0.6).astype(np.float32)

        # Sparse background connectivity (15% across remaining blocks)
        bg_rand = self.rng.uniform(size=(self.node_dim, self.node_dim))
        prov_mask = np.maximum(prov_mask, (bg_rand < 0.15).astype(np.float32))

        # Ensure every post-synaptic neuron receives at least 1 connection
        row_sums = prov_mask.sum(axis=1)
        for i in range(self.node_dim):
            if row_sums[i] == 0:
                j = int(self.rng.integers(0, self.node_dim))
                prov_mask[i, j] = 1.0

        self.register_buffer('provenance_mask', torch.tensor(prov_mask, dtype=torch.float32))

        # Fan-in normalization per row: 1 / sqrt(max(1, k_in))
        k_in = self.provenance_mask.sum(dim=1, keepdim=True)
        fan_in_norm = 1.0 / torch.sqrt(torch.clamp(k_in, min=1.0))
        self.register_buffer('fan_in_norm', fan_in_norm)

        # Trainable raw synaptic parameters
        self.w_param = nn.Parameter(torch.randn(self.node_dim, self.node_dim) * 0.05)

        # Context-dependent route gate: g(c_t) = sigmoid(W_gate c_t + b_gate) in [0, 1]^node_dim
        self.context_gate = nn.Linear(self.node_dim, self.node_dim)
        nn.init.normal_(self.context_gate.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.context_gate.bias)

        # Region-level affinity gate for diagnostic routing
        self.affinity_gate = nn.Linear(self.node_dim, len(self.regions), bias=False)
        nn.init.orthogonal_(self.affinity_gate.weight)

        # Target projection if target_dim differs from node_dim
        if self.target_dim != self.node_dim:
            self.target_proj = nn.Linear(self.node_dim, self.target_dim, bias=False)
            nn.init.orthogonal_(self.target_proj.weight)
        else:
            self.target_proj = None

        self.activity_history: List[float] = []
        self.max_history = 64
        self.telemetry = CrossGraftTelemetry()

    def reset(self):
        """Reset diagnostics."""
        self.activity_history.clear()
        self.telemetry = CrossGraftTelemetry()

    def get_effective_weights(self) -> torch.Tensor:
        """
        Return the Dale's-compliant, sparse-masked, fan-in-normalized projection matrix.
        W_eff = (softplus(W_param) * dale_signs * provenance_mask) * fan_in_norm
        """
        w_mag = F.softplus(self.w_param)
        w_dale = w_mag * self.dale_signs
        w_sparse = w_dale * self.provenance_mask
        w_eff = w_sparse * self.fan_in_norm
        return w_eff

    def verify_dales_law(self, tol: float = 1e-7) -> float:
        """
        Verify Dale's law compliance percentage across all presynaptic columns.
        A column satisfies Dale's law if all its non-zero entries match pre_signs[j].
        """
        w_eff = self.get_effective_weights()
        num_cols = self.node_dim
        valid_cols = 0
        total_active_cols = 0

        for j in range(num_cols):
            col = w_eff[:, j]
            nonzero_mask = col.abs() > tol
            if not nonzero_mask.any():
                continue
            total_active_cols += 1
            col_nonzero = col[nonzero_mask]
            expected_sign = float(self.pre_signs[j].item())
            if expected_sign > 0:
                if (col_nonzero >= -tol).all():
                    valid_cols += 1
            else:
                if (col_nonzero <= tol).all():
                    valid_cols += 1

        if total_active_cols == 0:
            return 100.0
        return float(100.0 * valid_cols / total_active_cols)

    def forward(
        self,
        source_acts: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        neuromod_da: float = 1.0,
    ) -> Tuple[torch.Tensor, CrossGraftTelemetry]:
        """
        Compute cross-connectome synaptic projection with zero-gain identity guarantee.

        Args:
            source_acts: Activation tensor of shape [..., node_dim] or [..., N]
            context: Contextual modulation tensor
            neuromod_da: Dopaminergic neuromodulation scaling

        Returns:
            grafted_currents: Tensor [target_dim] to be injected into recipient neurons
            telemetry: Live state telemetry
        """
        # Strict zero-gain identity guarantee: when gamma = 0.0, output is strictly zeros
        target_size = self.target_dim
        device = source_acts.device
        dtype = source_acts.dtype

        if self.gamma == 0.0:
            zero_delta = torch.zeros(target_size, dtype=dtype, device=device)
            self.telemetry = CrossGraftTelemetry(
                num_active_bridges=0,
                total_grafted_synapses=int(self.provenance_mask.sum().item()),
                dales_law_compliance_pct=100.0,
                mean_cross_affinity=0.0,
                current_l2_norm=0.0,
                participation_index=0.0,
                fan_in_norm_mean=float(self.fan_in_norm.mean().item()),
                sparse_density=float(self.provenance_mask.mean().item()),
                donor_recipient_pairs=[],
                recent_bridge_activity=list(self.activity_history),
            )
            return zero_delta, self.telemetry

        # Align input dimension
        x = source_acts.detach().squeeze()
        if x.dim() > 1:
            x = x.flatten()
        if len(x) < self.node_dim:
            x = F.pad(x, (0, self.node_dim - len(x)))
        elif len(x) > self.node_dim:
            x = x[:self.node_dim]
        x = x.to(device=self.w_param.device, dtype=self.w_param.dtype)

        # Context alignment and gating
        ctx = context.detach().squeeze() if context is not None else x
        if ctx.dim() > 1:
            ctx = ctx.flatten()
        if len(ctx) < self.node_dim:
            ctx = F.pad(ctx, (0, self.node_dim - len(ctx)))
        elif len(ctx) > self.node_dim:
            ctx = ctx[:self.node_dim]
        ctx = ctx.to(device=self.w_param.device, dtype=self.w_param.dtype)

        # 1. Effective weight matrix with Dale's law, sparse mask, and fan-in norm
        w_eff = self.get_effective_weights()

        # 2. Context route gating g(c_t) in [0, 1]^node_dim
        context_gate_out = torch.sigmoid(self.context_gate(ctx))

        # 3. Dynamic affinity gating across regions
        affinities = torch.softmax(self.affinity_gate(ctx) * max(0.01, float(neuromod_da)), dim=-1)
        mean_aff = float(affinities.mean().item())

        # 4. Synaptic integration: u = W_eff * x
        projected = torch.matmul(w_eff, x)

        # 5. Gated non-linear current: delta = gamma * clip(g(c_t) * tanh(u), -max_current, max_current)
        gated_projected = context_gate_out * torch.tanh(projected)
        delta_node = torch.clamp(self.gamma * gated_projected, -self.max_current, self.max_current)

        # 6. Target projection if target_dim != node_dim
        if self.target_proj is not None:
            delta = self.target_proj(delta_node)
        else:
            delta = delta_node

        delta = delta.to(device=device, dtype=dtype)
        act_norm = float(torch.norm(delta).item())
        self.activity_history.append(act_norm)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)

        # Calculate participation index: (sum |delta|)^2 / (N * sum delta^2)
        delta_sq = float((delta ** 2).sum().item())
        delta_abs_sum = float(delta.abs().sum().item())
        participation = (delta_abs_sum ** 2) / (float(len(delta)) * max(1e-8, delta_sq))

        # Region-level donor-recipient pair diagnostics with measured norms and flux
        quad = self.block_size
        def get_block(r_idx: int) -> slice:
            start = r_idx * quad
            end = self.node_dim if r_idx == len(self.regions) - 1 else (r_idx + 1) * quad
            return slice(start, end)

        pairs = [
            {
                'source': 'OpticLobe (LPTC)',
                'target': 'Hemibrain (CX)',
                'weight_norm': round(float(w_eff[get_block(2), get_block(1)].norm().item()), 3),
                'affinity': round(float(affinities[1].item()), 3),
                'current_mag': round(float(delta_node[get_block(2)].norm().item()), 4),
            },
            {
                'source': 'Hemibrain (MB)',
                'target': 'MaleCNS (VNC)',
                'weight_norm': round(float(w_eff[get_block(0), get_block(2)].norm().item()), 3),
                'affinity': round(float(affinities[2].item()), 3),
                'current_mag': round(float(delta_node[get_block(0)].norm().item()), 4),
            },
            {
                'source': 'FlyWire (Central)',
                'target': 'MaleCNS (Motor)',
                'weight_norm': round(float(w_eff[get_block(0), get_block(3)].norm().item()), 3),
                'affinity': round(float(affinities[3].item()), 3),
                'current_mag': round(float(delta_node[get_block(0)].norm().item()), 4),
            },
            {
                'source': 'MaleCNS (Proprio)',
                'target': 'Hemibrain (EB)',
                'weight_norm': round(float(w_eff[get_block(2), get_block(0)].norm().item()), 3),
                'affinity': round(float(affinities[0].item()), 3),
                'current_mag': round(float(delta_node[get_block(2)].norm().item()), 4),
            },
        ]

        total_synapses = int(self.provenance_mask.sum().item())
        compliance = self.verify_dales_law()

        self.telemetry = CrossGraftTelemetry(
            num_active_bridges=len(self.regions),
            total_grafted_synapses=total_synapses,
            dales_law_compliance_pct=compliance,
            mean_cross_affinity=mean_aff,
            current_l2_norm=act_norm,
            participation_index=participation,
            fan_in_norm_mean=float(self.fan_in_norm.mean().item()),
            sparse_density=float(self.provenance_mask.mean().item()),
            donor_recipient_pairs=pairs,
            recent_bridge_activity=list(self.activity_history),
        )

        return delta, self.telemetry

    def export_state(self) -> Dict[str, Any]:
        """Export state for checkpoint persistence."""
        state = {
            'w_param': self.w_param.detach().cpu().numpy().tolist(),
            'gamma': float(self.gamma),
            'node_dim': int(self.node_dim),
            'max_current': float(self.max_current),
            'pre_signs': self.pre_signs.detach().cpu().numpy().tolist(),
            'provenance_mask': self.provenance_mask.detach().cpu().numpy().tolist(),
            'context_gate_weight': self.context_gate.weight.detach().cpu().numpy().tolist(),
            'context_gate_bias': self.context_gate.bias.detach().cpu().numpy().tolist(),
            'activity_history': list(self.activity_history),
        }
        return state

    def import_state(self, state: Dict[str, Any]):
        """Import state from checkpoint with backward compatibility."""
        if 'activity_history' in state:
            self.activity_history = list(state['activity_history'])[-self.max_history:]
        if 'w_param' in state:
            w_t = torch.tensor(state['w_param'], dtype=torch.float32)
            if w_t.shape == self.w_param.shape:
                self.w_param.data.copy_(w_t)
        if 'gamma' in state:
            self.gamma = float(state['gamma'])
        if 'max_current' in state:
            self.max_current = float(state['max_current'])
        if 'pre_signs' in state:
            ps_t = torch.tensor(state['pre_signs'], dtype=torch.float32)
            if ps_t.shape == self.pre_signs.shape:
                self.pre_signs.copy_(ps_t)
                self.dale_signs.copy_(torch.tile(ps_t, (self.node_dim, 1)))
        if 'provenance_mask' in state:
            pm_t = torch.tensor(state['provenance_mask'], dtype=torch.float32)
            if pm_t.shape == self.provenance_mask.shape:
                self.provenance_mask.copy_(pm_t)
                k_in = self.provenance_mask.sum(dim=1, keepdim=True)
                self.fan_in_norm.copy_(1.0 / torch.sqrt(torch.clamp(k_in, min=1.0)))
        if 'context_gate_weight' in state and 'context_gate_bias' in state:
            gw = torch.tensor(state['context_gate_weight'], dtype=torch.float32)
            gb = torch.tensor(state['context_gate_bias'], dtype=torch.float32)
            if gw.shape == self.context_gate.weight.shape:
                self.context_gate.weight.data.copy_(gw)
            if gb.shape == self.context_gate.bias.shape:
                self.context_gate.bias.data.copy_(gb)
