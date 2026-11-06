"""Biological Connectome Compiler: Real Connectome Translation & Null Benchmarking.

Compiles real Janelia Drosophila whole-brain/cord connectome records into
executable sparse PyTorch neural modules preserving cell types, directionality,
neurotransmitter identities (Dale's law), and biological synapse counts.
Includes degree-preserving randomized configuration model null generators to
benchmark biological wiring efficiency against structural random nulls.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .config import VENDOR

DATA_PATH = VENDOR / 'expanse' / 'expanse' / 'data' / 'malecns_types.npz'


class BiologicalConnectomeModule(nn.Module):
    """Executable PyTorch neural module compiled directly from biological connectome records."""

    def __init__(
        self,
        node_names: List[str],
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        weights: np.ndarray,
        signs: np.ndarray,
        neurotransmitters: List[str],
        superclasses: List[str],
        tau: float = 0.8,
    ):
        super().__init__()
        self.node_names = list(node_names)
        self.num_nodes = len(node_names)
        self.neurotransmitters = list(neurotransmitters)
        self.superclasses = list(superclasses)
        self.tau = tau

        # Construct signed weight matrix preserving Dale's law
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for pre, post, w, s in zip(pre_indices, post_indices, weights, signs):
            if pre < self.num_nodes and post < self.num_nodes:
                # Biological weight: normalized synapse count multiplied by sign (+1 excitatory, -1 inhibitory)
                adj[post, pre] += float(s) * float(math.log1p(max(0.0, float(w))))

        # Normalize spectral radius for stable recurrence
        eigvals = np.linalg.eigvals(adj)
        max_eig = float(np.max(np.abs(eigvals))) if len(eigvals) > 0 else 1.0
        self.spectral_radius = max_eig
        if max_eig > 1.2:
            adj = adj * (0.95 / max_eig)

        self.register_buffer('weight_matrix', torch.tensor(adj, dtype=torch.float32))
        self.register_buffer('state', torch.zeros(self.num_nodes, dtype=torch.float32))

        # Identify sensory (in-degree=0 or visual/ascending) and motor (out-degree=0 or vnc_motor) subsets
        in_degrees = (np.abs(adj) > 1e-5).sum(axis=1)
        out_degrees = (np.abs(adj) > 1e-5).sum(axis=0)
        self.sensory_indices = [
            i for i in range(self.num_nodes)
            if 'visual' in self.superclasses[i] or 'ascending' in self.superclasses[i] or in_degrees[i] == 0
        ]
        if not self.sensory_indices:
            self.sensory_indices = list(range(min(4, self.num_nodes)))

        self.motor_indices = [
            i for i in range(self.num_nodes)
            if 'motor' in self.superclasses[i] or out_degrees[i] == 0
        ]
        if not self.motor_indices:
            self.motor_indices = list(range(max(0, self.num_nodes - 4), self.num_nodes))

    def reset_state(self):
        self.state.zero_()

    def forward(self, external_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single time-step discrete recurrent update:

        r(t+1) = tau * r(t) + (1 - tau) * tanh(W r(t) + I(t))
        """
        rec_input = torch.matmul(self.weight_matrix, self.state)
        if external_input is not None:
            if external_input.shape[-1] == len(self.sensory_indices):
                rec_input[self.sensory_indices] += external_input
            elif external_input.shape[-1] == self.num_nodes:
                rec_input += external_input
            else:
                rec_input[:external_input.shape[-1]] += external_input

        new_state = self.tau * self.state + (1.0 - self.tau) * torch.tanh(rec_input)
        self.state.copy_(new_state)
        return new_state


class BiologicalConnectomeCompiler:
    """Compiles Drosophila whole-brain/cord biological records into executable modules."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else DATA_PATH
        self._raw_data: Optional[Dict[str, Any]] = None

    def load_raw_data(self) -> Dict[str, Any]:
        if self._raw_data is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Connectome dataset not found at {self.data_path}")
            npz = np.load(self.data_path, allow_pickle=True)
            self._raw_data = {k: npz[k] for k in npz.files}
        return self._raw_data

    def compile_subcircuit(
        self,
        query: Optional[str] = None,
        max_neurons: int = 128,
        min_weight: float = 1.0,
    ) -> BiologicalConnectomeModule:
        """Extracts an induced biological subcircuit and compiles it into an executable module.

        Args:
            query: substring filter on superclass, cell_class, or neurotransmitter (e.g. 'motor', 'visual', 'gaba')
            max_neurons: upper bound on neuron count
            min_weight: minimum synapse count threshold for inclusion
        """
        data = self.load_raw_data()
        type_names = data['type_names']
        superclasses = data['superclass']
        cell_classes = data['cell_class']
        nt = data['nt']
        signs = data['sign']
        pre = data['pre']
        post = data['post']
        weight = data['weight']

        # Determine qualifying neurons
        n_total = len(type_names)
        if query:
            q_lower = query.lower()
            matching_mask = np.array([
                (q_lower in str(superclasses[i]).lower() or
                 q_lower in str(cell_classes[i]).lower() or
                 q_lower in str(nt[i]).lower() or
                 q_lower in str(type_names[i]).lower())
                for i in range(n_total)
            ], dtype=bool)
            selected_indices = np.where(matching_mask)[0][:max_neurons]
        else:
            selected_indices = np.arange(min(max_neurons, n_total))

        if len(selected_indices) < 2:
            # Fallback to first max_neurons if query was too narrow
            selected_indices = np.arange(min(max_neurons, n_total))

        idx_set = set(selected_indices.tolist())
        idx_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

        # Filter edges where both pre and post are in selected subcircuit
        edge_mask = np.isin(pre, selected_indices) & np.isin(post, selected_indices) & (weight >= min_weight)
        sub_pre = pre[edge_mask]
        sub_post = post[edge_mask]
        sub_w = weight[edge_mask]

        # Remap indices to 0..len(selected)-1
        remap_pre = np.array([idx_remap[p] for p in sub_pre], dtype=np.int64)
        remap_post = np.array([idx_remap[p] for p in sub_post], dtype=np.int64)
        sub_signs = np.array([signs[p] for p in sub_pre], dtype=np.float32)

        return BiologicalConnectomeModule(
            node_names=[str(type_names[i]) for i in selected_indices],
            pre_indices=remap_pre,
            post_indices=remap_post,
            weights=sub_w,
            signs=sub_signs,
            neurotransmitters=[str(nt[i]) for i in selected_indices],
            superclasses=[str(superclasses[i]) for i in selected_indices],
        )

    def generate_degree_preserving_null(
        self,
        bio_module: BiologicalConnectomeModule,
        swaps: int = 500,
        seed: int = 42,
    ) -> BiologicalConnectomeModule:
        """Constructs a randomized degree-preserving configuration model null network.

        Preserves in-degree and out-degree distributions of all neurons while shuffling edges.
        """
        rng = np.random.default_rng(seed)
        W = bio_module.weight_matrix.cpu().numpy().copy()
        n = bio_module.num_nodes

        # Extract directed edges: post is target (row), pre is source (col)
        # In adj matrix: W[post, pre]
        edges = []
        for post in range(n):
            for pre in range(n):
                if abs(W[post, pre]) > 1e-6:
                    edges.append((pre, post, float(W[post, pre])))

        if len(edges) < 4:
            # Trivial network: return identical module
            null_mod = BiologicalConnectomeModule(
                bio_module.node_names, np.array([]), np.array([]), np.array([]), np.array([]),
                bio_module.neurotransmitters, bio_module.superclasses, bio_module.tau
            )
            return null_mod

        # Maslov-Sneppen directed edge swap preserving degrees
        edge_list = [[e[0], e[1], e[2]] for e in edges]
        edge_set = {(e[0], e[1]) for e in edge_list}
        num_edges = len(edge_list)
        successful_swaps = 0

        for _ in range(swaps * 10):
            if successful_swaps >= swaps:
                break
            idx1, idx2 = rng.choice(num_edges, size=2, replace=False)
            u, v, w1 = edge_list[idx1]
            x, y, w2 = edge_list[idx2]

            # Candidate swap: (u -> y) and (x -> v)
            if u != y and x != v and u != x and v != y:
                if (u, y) not in edge_set and (x, v) not in edge_set:
                    edge_set.remove((u, v))
                    edge_set.remove((x, y))
                    edge_set.add((u, y))
                    edge_set.add((x, v))
                    edge_list[idx1] = [u, y, w1]
                    edge_list[idx2] = [x, v, w2]
                    successful_swaps += 1

        pre_null = np.array([e[0] for e in edge_list], dtype=np.int64)
        post_null = np.array([e[1] for e in edge_list], dtype=np.int64)
        w_null = np.array([abs(e[2]) for e in edge_list], dtype=np.float32)
        s_null = np.array([math.copysign(1.0, e[2]) for e in edge_list], dtype=np.float32)

        return BiologicalConnectomeModule(
            node_names=bio_module.node_names,
            pre_indices=pre_null,
            post_indices=post_null,
            weights=w_null,
            signs=s_null,
            neurotransmitters=bio_module.neurotransmitters,
            superclasses=bio_module.superclasses,
            tau=bio_module.tau,
        )

    def benchmark_biological_vs_null(
        self,
        bio_module: BiologicalConnectomeModule,
        null_module: BiologicalConnectomeModule,
        steps: int = 40,
    ) -> Dict[str, Any]:
        """Empirically evaluates signal transmission and dynamical efficiency.

        Compares real biological wiring against degree-preserving randomized null network.
        """
        bio_module.reset_state()
        null_module.reset_state()

        # Step pulse into sensory neurons
        pulse = torch.ones(len(bio_module.sensory_indices), dtype=torch.float32)

        bio_signals = []
        null_signals = []
        bio_motor_resp = []
        null_motor_resp = []

        for t in range(steps):
            inp = pulse if t == 0 else None
            b_s = bio_module(inp)
            n_s = null_module(inp)
            bio_signals.append(float(b_s.abs().mean().item()))
            null_signals.append(float(n_s.abs().mean().item()))
            bio_motor_resp.append(float(b_s[bio_module.motor_indices].abs().mean().item()))
            null_motor_resp.append(float(n_s[null_module.motor_indices].abs().mean().item()))

        # Calculate metrics
        bio_signal_reach = float(np.mean(bio_signals[5:]))
        null_signal_reach = float(np.mean(null_signals[5:]))
        bio_motor_reach = float(np.max(bio_motor_resp))
        null_motor_reach = float(np.max(null_motor_resp))

        bio_w = bio_module.weight_matrix.cpu().numpy()
        null_w = null_module.weight_matrix.cpu().numpy()

        bio_density = float(np.count_nonzero(bio_w)) / float(bio_w.size)
        null_density = float(np.count_nonzero(null_w)) / float(null_w.size)

        return {
            'biological': {
                'num_neurons': bio_module.num_nodes,
                'spectral_radius': float(bio_module.spectral_radius),
                'density': bio_density,
                'sustained_activity': bio_signal_reach,
                'motor_peak_response': bio_motor_reach,
            },
            'degree_preserving_null': {
                'num_neurons': null_module.num_nodes,
                'spectral_radius': float(null_module.spectral_radius),
                'density': null_density,
                'sustained_activity': null_signal_reach,
                'motor_peak_response': null_motor_reach,
            },
            'biological_signal_advantage': float(bio_signal_reach - null_signal_reach),
            'motor_responsiveness_advantage': float(bio_motor_reach - null_motor_reach),
        }
