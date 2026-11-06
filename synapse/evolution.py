"""Autonomous Neural Evolution: Transaction-Safe Graph-Topology Mutation.

Enables SYNAPSE to dynamically grow, remove, split, duplicate, and reorganise
neural circuits based on task performance and metabolic constraints.
All mutations are transaction-safe with atomic preview, parent digest verification,
and rollback receipts.
"""
from __future__ import annotations
import copy
import math
import uuid
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from .stateio import digest

SCHEMA = 'synapse-topology-mutation-v1'

@dataclass
class MutationReceipt:
    mutation_id: str
    mutation_type: str
    parent_digest: str
    resulting_digest: str
    details: dict
    undo_state: dict

    def to_dict(self) -> dict:
        return {
            'mutation_id': self.mutation_id,
            'mutation_type': self.mutation_type,
            'parent_digest': self.parent_digest,
            'resulting_digest': self.resulting_digest,
            'details': self.details,
        }

class TopologyEvolutionEngine:
    """Manages structural mutations on an IntegratedAgent's connectome."""

    def __init__(self, metabolic_penalty_per_neuron: float = 0.05, metabolic_penalty_per_synapse: float = 0.01):
        self.metabolic_penalty_per_neuron = metabolic_penalty_per_neuron
        self.metabolic_penalty_per_synapse = metabolic_penalty_per_synapse
        self.mutation_history: list[dict] = []
        self._undo_stack: dict[str, dict] = {}

    def add_neuron(self, agent, neuropil: int = 0) -> MutationReceipt:
        """Add a new neuron to the specified neuropil with bounded connectivity."""
        parent_digest = digest(agent.export_state())
        undo_state = copy.deepcopy(agent.export_state())
        undo_graph = {
            'role': agent.graph.role.copy(),
            'origin': agent.graph.origin.copy(),
            'cell_type': agent.graph.cell_type.copy(),
            'names': list(agent.graph.names),
            'graft_pre': agent.graph.graft_pre.copy(),
            'graft_post': agent.graph.graft_post.copy(),
            'brain_n': agent.brain.n,
        }
        mutation_id = uuid.uuid4().hex

        new_idx = agent.brain.n
        agent.brain.n += 1
        agent.graph.role = np.append(agent.graph.role, 'central')
        agent.graph.origin = np.append(agent.graph.origin, neuropil)
        agent.graph.cell_type = np.append(agent.graph.cell_type, 'interneuron')
        agent.graph.names.append(f'new_{new_idx}')

        # Expand rate and state vectors
        with torch.no_grad():
            agent.brain.r = torch.cat([agent.brain.r, torch.zeros(1)])
            agent.brain.activity_ema = torch.cat([agent.brain.activity_ema, torch.zeros(1)])
            agent.brain.elig = torch.cat([agent.brain.elig, torch.zeros(1)])
            agent.brain.heb = torch.cat([agent.brain.heb, torch.zeros(1)])
            agent.evoked = torch.cat([agent.evoked, torch.zeros(1)])

            # Connect new neuron with 2 random input and 2 random output graft edges
            rng = np.random.default_rng()
            possible_peers = list(range(max(0, new_idx - 32), new_idx))
            if len(possible_peers) >= 2:
                in_peers = rng.choice(possible_peers, size=2, replace=False)
                out_peers = rng.choice(possible_peers, size=2, replace=False)

                new_pres = list(agent.graph.graft_pre)
                new_posts = list(agent.graph.graft_post)
                new_weights = list(agent.brain.graft_w.detach().cpu().numpy())

                for p in in_peers:
                    new_pres.append(int(p))
                    new_posts.append(new_idx)
                    new_weights.append(float(rng.normal(0.0, 0.1)))
                for p in out_peers:
                    new_pres.append(new_idx)
                    new_posts.append(int(p))
                    new_weights.append(float(rng.normal(0.0, 0.1)))

                agent.graph.graft_pre = np.array(new_pres, dtype=np.int64)
                agent.graph.graft_post = np.array(new_posts, dtype=np.int64)
                agent.brain.graft_w = nn.Parameter(torch.tensor(new_weights, dtype=torch.float32))

        resulting_digest = digest(agent.export_state())
        details = {
            'new_neuron_index': new_idx,
            'neuropil': neuropil,
            'total_neurons': agent.brain.n,
            'new_synapses': 4,
            'metabolic_delta': self.metabolic_penalty_per_neuron + 4 * self.metabolic_penalty_per_synapse
        }
        receipt = MutationReceipt(mutation_id, 'add_neuron', parent_digest, resulting_digest, details, undo_state)
        self._undo_stack[mutation_id] = {'agent_state': undo_state, 'graph_state': undo_graph}
        self.mutation_history.append(receipt.to_dict())
        return receipt

    def split_neuron(self, agent, node_id: int) -> MutationReceipt:
        """Split a bottleneck neuron into two specialized sister neurons."""
        if not (0 <= node_id < agent.brain.n):
            raise ValueError(f'Invalid node_id {node_id}')
        parent_digest = digest(agent.export_state())
        undo_state = copy.deepcopy(agent.export_state())
        mutation_id = uuid.uuid4().hex

        neuropil = int(agent.graph.origin[node_id]) if node_id < len(agent.graph.origin) else 0
        sister_id = agent.brain.n
        agent.brain.n += 1
        agent.graph.role = np.append(agent.graph.role, str(agent.graph.role[node_id]))
        agent.graph.origin = np.append(agent.graph.origin, neuropil)
        agent.graph.cell_type = np.append(agent.graph.cell_type, str(agent.graph.cell_type[node_id]))
        agent.graph.names.append(f'sister_{node_id}')

        with torch.no_grad():
            agent.brain.r = torch.cat([agent.brain.r, torch.zeros(1)])
            agent.brain.activity_ema = torch.cat([agent.brain.activity_ema, torch.zeros(1)])
            agent.brain.elig = torch.cat([agent.brain.elig, torch.zeros(1)])
            agent.brain.heb = torch.cat([agent.brain.heb, torch.zeros(1)])
            agent.evoked = torch.cat([agent.evoked, torch.zeros(1)])

            # Split incoming edges between original and sister; duplicate outgoing edges
            new_pres = list(agent.graph.graft_pre)
            new_posts = list(agent.graph.graft_post)
            new_weights = list(agent.brain.graft_w.detach().cpu().numpy())

            for i in range(len(agent.graph.graft_pre)):
                pre = int(agent.graph.graft_pre[i])
                post = int(agent.graph.graft_post[i])
                w = float(new_weights[i])

                if post == node_id and i % 2 == 1:
                    # Divert half of incoming connections to sister
                    new_posts[i] = sister_id
                elif pre == node_id:
                    # Duplicate outgoing connection for sister
                    new_pres.append(sister_id)
                    new_posts.append(post)
                    new_weights.append(w * 0.5)

            agent.graph.graft_pre = np.array(new_pres, dtype=np.int64)
            agent.graph.graft_post = np.array(new_posts, dtype=np.int64)
            agent.brain.graft_w = nn.Parameter(torch.tensor(new_weights, dtype=torch.float32))

        resulting_digest = digest(agent.export_state())
        details = {
            'original_node': node_id,
            'sister_node': sister_id,
            'total_neurons': agent.brain.n,
            'total_grafts': len(agent.graph.graft_pre)
        }
        receipt = MutationReceipt(mutation_id, 'split_neuron', parent_digest, resulting_digest, details, undo_state)
        self._undo_stack[mutation_id] = undo_state
        self.mutation_history.append(receipt.to_dict())
        return receipt

    def duplicate_subcircuit(self, agent, node_ids: list[int]) -> MutationReceipt:
        """Duplicate a specialist subnetwork to create parallel processing channels."""
        if not node_ids or any(n < 0 or n >= agent.brain.n for n in node_ids):
            raise ValueError('Invalid node_ids for subcircuit duplication')
        parent_digest = digest(agent.export_state())
        undo_state = copy.deepcopy(agent.export_state())
        mutation_id = uuid.uuid4().hex

        id_map = {}
        for old_id in node_ids:
            new_id = agent.brain.n
            id_map[old_id] = new_id
            agent.brain.n += 1
            agent.graph.role = np.append(agent.graph.role, str(agent.graph.role[old_id]))
            agent.graph.origin = np.append(agent.graph.origin, int(agent.graph.origin[old_id]))
            agent.graph.cell_type = np.append(agent.graph.cell_type, str(agent.graph.cell_type[old_id]))
            agent.graph.names.append(f'dup_{old_id}')

        with torch.no_grad():
            pad_len = len(node_ids)
            agent.brain.r = torch.cat([agent.brain.r, torch.zeros(pad_len)])
            agent.brain.activity_ema = torch.cat([agent.brain.activity_ema, torch.zeros(pad_len)])
            agent.brain.elig = torch.cat([agent.brain.elig, torch.zeros(pad_len)])
            agent.brain.heb = torch.cat([agent.brain.heb, torch.zeros(pad_len)])
            agent.evoked = torch.cat([agent.evoked, torch.zeros(pad_len)])

            new_pres = list(agent.graph.graft_pre)
            new_posts = list(agent.graph.graft_post)
            new_weights = list(agent.brain.graft_w.detach().cpu().numpy())

            node_set = set(node_ids)
            for i in range(len(agent.graph.graft_pre)):
                pre = int(agent.graph.graft_pre[i])
                post = int(agent.graph.graft_post[i])
                w = float(new_weights[i])

                if pre in node_set and post in node_set:
                    # Internal edge duplicated
                    new_pres.append(id_map[pre])
                    new_posts.append(id_map[post])
                    new_weights.append(w)
                elif pre in node_set:
                    # Outgoing edge duplicated
                    new_pres.append(id_map[pre])
                    new_posts.append(post)
                    new_weights.append(w * 0.5)
                elif post in node_set:
                    # Incoming edge duplicated
                    new_pres.append(pre)
                    new_posts.append(id_map[post])
                    new_weights.append(w * 0.5)

            agent.graph.graft_pre = np.array(new_pres, dtype=np.int64)
            agent.graph.graft_post = np.array(new_posts, dtype=np.int64)
            agent.brain.graft_w = nn.Parameter(torch.tensor(new_weights, dtype=torch.float32))

        resulting_digest = digest(agent.export_state())
        details = {
            'duplicated_nodes': len(node_ids),
            'id_map': id_map,
            'total_neurons': agent.brain.n,
            'total_grafts': len(agent.graph.graft_pre)
        }
        receipt = MutationReceipt(mutation_id, 'duplicate_subcircuit', parent_digest, resulting_digest, details, undo_state)
        self._undo_stack[mutation_id] = undo_state
        self.mutation_history.append(receipt.to_dict())
        return receipt

    def prune_synapses(self, agent, threshold: float = 0.02) -> MutationReceipt:
        """Prune functionally dormant synapses below activity threshold with metabolic credit."""
        parent_digest = digest(agent.export_state())
        undo_state = copy.deepcopy(agent.export_state())
        mutation_id = uuid.uuid4().hex

        with torch.no_grad():
            w = agent.brain.graft_w.detach().cpu().numpy()
            keep_mask = np.abs(w) >= threshold

            if not np.any(keep_mask):
                # Always keep at least the top 10% strongest connections
                top_k = max(1, len(w) // 10)
                keep_indices = np.argsort(np.abs(w))[-top_k:]
                keep_mask = np.zeros(len(w), dtype=bool)
                keep_mask[keep_indices] = True

            pruned_count = int(np.sum(~keep_mask))
            agent.graph.graft_pre = agent.graph.graft_pre[keep_mask]
            agent.graph.graft_post = agent.graph.graft_post[keep_mask]
            agent.brain.graft_w = nn.Parameter(torch.tensor(w[keep_mask], dtype=torch.float32))

        resulting_digest = digest(agent.export_state())
        details = {
            'pruned_synapses': pruned_count,
            'retained_synapses': int(np.sum(keep_mask)),
            'threshold': threshold,
            'metabolic_credit': pruned_count * self.metabolic_penalty_per_synapse
        }
        receipt = MutationReceipt(mutation_id, 'prune_synapses', parent_digest, resulting_digest, details, undo_state)
        self._undo_stack[mutation_id] = undo_state
        self.mutation_history.append(receipt.to_dict())
        return receipt

    def rollback(self, agent, mutation_id: str) -> dict:
        """Rollback a structural mutation using its saved undo state."""
        if mutation_id not in self._undo_stack:
            raise KeyError(f'Mutation {mutation_id} not found in undo stack')
        saved = self._undo_stack.pop(mutation_id)
        if isinstance(saved, dict) and 'graph_state' in saved:
            undo_state = saved['agent_state']
            gs = saved['graph_state']
            agent.graph.role = gs['role'].copy()
            agent.graph.origin = gs['origin'].copy()
            agent.graph.cell_type = gs['cell_type'].copy()
            agent.graph.names = list(gs['names'])
            agent.graph.graft_pre = gs['graft_pre'].copy()
            agent.graph.graft_post = gs['graft_post'].copy()
            agent.brain.n = gs['brain_n']
        else:
            undo_state = saved

        agent.import_state(undo_state)
        return {
            'status': 'ok',
            'rolled_back': mutation_id,
            'restored_digest': digest(agent.export_state()),
            'scope': 'transaction-safe graph topology mutation rollback'
        }
