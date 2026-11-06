"""9-Arm Connectome Causal Controls and Inductive Bias Benchmark.

Grounding:
Transitions from "does a fly graph help?" to "which structural motifs improve
which computations, relative to degree-matched, sign-matched, community-matched,
and compute-matched controls?"
Implements the 9-Arm controlled graph generator:
1. bio: Real biological connectome subcircuit (Janelia male-CNS).
2. degree_rewired: Degree-preserving directed Maslov-Sneppen rewiring.
3. sign_rewired: Sign-preserving shuffled graph (testing Dale's law).
4. community_rewired: Community-preserving modular rewiring.
5. random_sparse: Erdős-Rényi random sparse graph with equal edge count & density.
6. learned_sparse: Continuous learned sparse weight matrix with equal compute budget.
7. unsigned: Real graph with signs stripped (all positive excitatory).
8. local_only: Real graph with long-range edges pruned (testing hierarchical transmission).
9. brain_cord_motif: Local sensory-motor loops with sparse supervisory hub (2026 brain-cord abstraction).

Computes formal causal metrics:
- BioAdvantage = S(G_bio) - mean(S(G_control))
- Cohen's d: EffectSize = (mean_bio - mean_control) / s_pooled
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from .connectome_compiler import BiologicalConnectomeModule, BiologicalConnectomeCompiler


class ControlledConnectomeAssay:
    """Generates the 9 matched control topologies and runs the causal assay."""

    def __init__(self, compiler: Optional[BiologicalConnectomeCompiler] = None, seed: int = 42):
        self.compiler = compiler or BiologicalConnectomeCompiler()
        self.rng = np.random.default_rng(seed)

    def generate_all_controls(
        self,
        bio_module: BiologicalConnectomeModule,
        seed: Optional[int] = None
    ) -> Dict[str, BiologicalConnectomeModule]:
        """Generates all 8 matched control topologies for the reference biological module."""
        rng = np.random.default_rng(seed if seed is not None else 42)
        adj_bio = bio_module.weight_matrix.cpu().numpy()
        N = bio_module.num_nodes
        controls: Dict[str, BiologicalConnectomeModule] = {'bio': bio_module}

        # Extract non-zero edges from biological adjacency: (post, pre, w)
        posts, pres = np.nonzero(adj_bio)
        weights = adj_bio[posts, pres]
        num_edges = len(weights)

        # Helper to wrap adjacency into a module
        def _make_mod(adj_mat: np.ndarray, name: str) -> BiologicalConnectomeModule:
            p_post, p_pre = np.nonzero(adj_mat)
            w = np.abs(adj_mat[p_post, p_pre])
            s = np.sign(adj_mat[p_post, p_pre])
            s[s == 0] = 1.0
            return BiologicalConnectomeModule(
                node_names=bio_module.node_names,
                pre_indices=p_pre.astype(np.int64),
                post_indices=p_post.astype(np.int64),
                weights=w.astype(np.float32),
                signs=s.astype(np.float32),
                neurotransmitters=bio_module.neurotransmitters,
                superclasses=bio_module.superclasses,
                tau=bio_module.tau,
            )

        # 1. Degree-Preserving Rewired (Maslov-Sneppen)
        edge_list = [[pres[i], posts[i], weights[i]] for i in range(num_edges)]
        edge_set = {(e[0], e[1]) for e in edge_list}
        swaps = num_edges * 4
        succ = 0
        for _ in range(swaps * 5):
            if succ >= swaps or num_edges < 2:
                break
            i1, i2 = rng.choice(num_edges, size=2, replace=False)
            u, v, w1 = edge_list[i1]
            x, y, w2 = edge_list[i2]
            if u != y and x != v and u != x and v != y:
                if (u, y) not in edge_set and (x, v) not in edge_set:
                    edge_set.remove((u, v))
                    edge_set.remove((x, y))
                    edge_set.add((u, y))
                    edge_set.add((x, v))
                    edge_list[i1] = [u, y, w1]
                    edge_list[i2] = [x, v, w2]
                    succ += 1

        adj_deg = np.zeros((N, N), dtype=np.float32)
        for u, v, w in edge_list:
            adj_deg[v, u] = w
        controls['degree_rewired'] = _make_mod(adj_deg, 'degree_rewired')

        # 2. Sign-Preserving Rewired (Shuffles edge connections only between same-sign pairs)
        pos_edges = [e for e in edge_list if e[2] > 0]
        neg_edges = [e for e in edge_list if e[2] < 0]
        adj_sign = np.zeros((N, N), dtype=np.float32)
        for edges in [pos_edges, neg_edges]:
            if len(edges) > 1:
                p_nodes = [e[0] for e in edges]
                q_nodes = [e[1] for e in edges]
                rng.shuffle(q_nodes)
                for u, v, orig_e in zip(p_nodes, q_nodes, edges):
                    adj_sign[v, u] = orig_e[2]
        controls['sign_rewired'] = _make_mod(adj_sign, 'sign_rewired')

        # 3. Community-Preserving Rewired (Preserves 4 partitioned clusters)
        comm_size = max(1, N // 4)
        comm_ids = np.array([min(3, i // comm_size) for i in range(N)])
        adj_comm = np.zeros((N, N), dtype=np.float32)
        for u, v, w in edge_list:
            if comm_ids[u] == comm_ids[v]:
                # Rewire within community
                target = rng.choice(np.where(comm_ids == comm_ids[u])[0])
                adj_comm[target, u] = w
            else:
                # Inter-community edge
                target = rng.choice(np.where(comm_ids != comm_ids[u])[0])
                adj_comm[target, u] = w
        controls['community_rewired'] = _make_mod(adj_comm, 'community_rewired')

        # 4. Erdős-Rényi Random Sparse (Equal edge count & equal weight distribution)
        adj_rand = np.zeros((N, N), dtype=np.float32)
        all_possible = [(u, v) for u in range(N) for v in range(N) if u != v]
        chosen = rng.choice(len(all_possible), size=min(num_edges, len(all_possible)), replace=False)
        shuffled_w = rng.permutation(weights)
        for idx, edge_idx in enumerate(chosen):
            u, v = all_possible[edge_idx]
            adj_rand[v, u] = shuffled_w[idx]
        controls['random_sparse'] = _make_mod(adj_rand, 'random_sparse')

        # 5. Learned Sparse (Simulated dense-optimized initialization with equal L1 norm)
        adj_learned = rng.normal(0, 0.1, size=(N, N)).astype(np.float32)
        np.fill_diagonal(adj_learned, 0.0)
        # Sparsify to top edge count
        threshold = np.percentile(np.abs(adj_learned), 100.0 * (1.0 - num_edges / float(N * N)))
        adj_learned[np.abs(adj_learned) < threshold] = 0.0
        # Normalize L1 norm to match bio
        if np.sum(np.abs(adj_learned)) > 0:
            adj_learned *= (np.sum(np.abs(adj_bio)) / np.sum(np.abs(adj_learned)))
        controls['learned_sparse'] = _make_mod(adj_learned, 'learned_sparse')

        # 6. Unsigned Biological (Absolute weights only, no Dale inhibition)
        adj_unsigned = np.abs(adj_bio.copy())
        controls['unsigned'] = _make_mod(adj_unsigned, 'unsigned')

        # 7. Local Only (Pruning long-range hierarchical connections |u - v| > N/3)
        adj_local = adj_bio.copy()
        for u in range(N):
            for v in range(N):
                if abs(u - v) > (N // 3):
                    adj_local[v, u] = 0.0
        controls['local_only'] = _make_mod(adj_local, 'local_only')

        # 8. Brain-Cord Motif (Local sensory-motor feedback loops with 2 central supervisory hub nodes)
        adj_bcm = np.zeros((N, N), dtype=np.float32)
        hub_nodes = [0, N - 1]
        for i in range(N):
            # Local sensory-motor neighbor ring
            next_i = (i + 1) % N
            prev_i = (i - 1) % N
            adj_bcm[next_i, i] = 0.25
            adj_bcm[prev_i, i] = -0.15
            # Connections to supervisory hub
            adj_bcm[hub_nodes[0], i] = 0.1
            adj_bcm[i, hub_nodes[1]] = 0.15
        controls['brain_cord_motif'] = _make_mod(adj_bcm, 'brain_cord_motif')

        return controls

    def run_causal_assay(
        self,
        bio_module: BiologicalConnectomeModule,
        steps: int = 30,
        trials: int = 5,
    ) -> Dict[str, Any]:
        """Runs the multi-trial causal assay across all 9 arms and computes formal effect sizes."""
        controls = self.generate_all_controls(bio_module)
        scores: Dict[str, List[float]] = {k: [] for k in controls.keys()}

        sensory_dim = len(bio_module.sensory_indices)
        motor_dim = len(bio_module.motor_indices)

        for trial in range(trials):
            pulse = torch.randn(sensory_dim, dtype=torch.float32)

            for arm_name, mod in controls.items():
                mod.reset_state()
                motor_energy = 0.0
                sustained_act = 0.0

                for t in range(steps):
                    inp = pulse if t < 3 else None
                    state = mod(inp)
                    sustained_act += float(state.abs().mean().item())
                    motor_act = state[mod.motor_indices[:motor_dim]]
                    motor_energy += float(motor_act.pow(2).sum().item())

                # Task capability score: signal transmission efficiency penalizing explosion or collapse
                score = (sustained_act / float(steps)) * math.exp(-abs(sustained_act / float(steps) - 0.4))
                score += 0.5 * math.log1p(motor_energy)
                scores[arm_name].append(score)

        # Compute summary statistics
        summary: Dict[str, Dict[str, float]] = {}
        for arm_name, vals in scores.items():
            summary[arm_name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'min': float(np.min(vals)),
                'max': float(np.max(vals))
            }

        bio_mean = summary['bio']['mean']
        bio_std = summary['bio']['std']

        # Control group pooled (excluding bio)
        control_means = [summary[k]['mean'] for k in controls.keys() if k != 'bio']
        mean_control = float(np.mean(control_means))

        control_stds = [summary[k]['std'] for k in controls.keys() if k != 'bio']
        pooled_std = math.sqrt(float(np.mean([s**2 for s in [bio_std] + control_stds]))) + 1e-6

        # Formal Metrics
        bio_advantage = float(bio_mean - mean_control)
        cohens_d = float(bio_advantage / pooled_std)

        pairwise_effects: Dict[str, float] = {}
        for k in controls.keys():
            if k != 'bio':
                diff = bio_mean - summary[k]['mean']
                pair_sd = math.sqrt(0.5 * (bio_std**2 + summary[k]['std']**2)) + 1e-6
                pairwise_effects[f'bio_vs_{k}'] = round(float(diff / pair_sd), 4)

        return {
            'status': 'ok',
            'num_arms': len(controls),
            'trials': trials,
            'summary_by_arm': summary,
            'bio_mean': round(bio_mean, 4),
            'mean_control': round(mean_control, 4),
            'bio_advantage': round(bio_advantage, 4),
            'cohens_d_effect_size': round(cohens_d, 4),
            'pairwise_cohens_d': pairwise_effects,
            'scope': '9-arm connectome causal controls & inductive bias assay'
        }
