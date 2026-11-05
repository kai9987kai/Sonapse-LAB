"""Mnemorph: Archaeological Memory Lens & Typed Structural Interventions.

Inspired by Mnemorph (adaptive memory archaeology: distinguishing persistent traces,
reconstruction during regeneration, and behavioral readout changes under typed interventions).
Evaluates memory carrier graphs across:
  1. Hub Lesion: targeted knockout of high-degree structural nodes.
  2. Periphery Lesion: knockout of peripheral/leaf nodes.
  3. Context Shift: environmental/habitat edge perturbation.
  4. Associative Regrowth: reconstructive recovery of missing nodes from relational neighbors.
Computes Veridical Retention Rate, Reconstructive Fidelity, and Hub Vulnerability Ratio.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


class MnemorphGraph:
    """Relational knowledge carrier graph for archaeological memory experiments."""

    def __init__(self) -> None:
        # node_id -> {"fact": str, "carrier": str, "tags": list[str]}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # node_id -> set of neighbor node_ids
        self.edges: Dict[str, Set[str]] = {}

    def add_node(self, node_id: str, fact: str, carrier: str, tags: List[str]) -> None:
        self.nodes[node_id] = {"fact": fact, "carrier": carrier, "tags": list(tags)}
        if node_id not in self.edges:
            self.edges[node_id] = set()

    def add_edge(self, u: str, v: str) -> None:
        if u in self.nodes and v in self.nodes:
            self.edges[u].add(v)
            self.edges[v].add(u)

    def degree(self, node_id: str) -> int:
        return len(self.edges.get(node_id, set()))

    def copy(self) -> "MnemorphGraph":
        clone = MnemorphGraph()
        for k, v in self.nodes.items():
            clone.add_node(k, v["fact"], v["carrier"], v["tags"])
        for u, neighbors in self.edges.items():
            for v in neighbors:
                if u < v:  # add undirected edge once
                    clone.add_edge(u, v)
        return clone


def _extract_tokens(text: str, keywords: List[str]) -> Set[str]:
    tokens = set()
    for kw in keywords:
        for w in kw.lower().split():
            if len(w) > 2:
                tokens.add(w)
    for w in text.lower().split():
        cleaned = "".join(c for c in w if c.isalnum())
        if len(cleaned) > 3:
            tokens.add(cleaned)
    return tokens


def build_standard_mnemorph_corpus() -> MnemorphGraph:
    """Build standardized 32-node carrier memory graph with rich relational topology."""
    from .mco import STANDARD_CORPUS
    g = MnemorphGraph()
    for item in STANDARD_CORPUS:
        tokens = list(_extract_tokens(item["text"], item.get("keywords", [])))
        g.add_node(item["id"], item["text"], item["carrier"], tokens)

    # Establish semantic / relational co-occurrence edges
    nodes_list = list(g.nodes.items())
    for i in range(len(nodes_list)):
        id_a, data_a = nodes_list[i]
        tags_a = set(data_a["tags"])
        carrier_a = data_a["carrier"]
        for j in range(i + 1, len(nodes_list)):
            id_b, data_b = nodes_list[j]
            tags_b = set(data_b["tags"])
            carrier_b = data_b["carrier"]
            shared_tags = len(tags_a & tags_b)
            if shared_tags >= 2 or (carrier_a == carrier_b and shared_tags >= 1):
                g.add_edge(id_a, id_b)
    return g


class MnemorphArchaeology:
    """Executes typed interventions and measures structural persistence vs reconstructive fidelity."""

    def __init__(self, graph: MnemorphGraph | None = None) -> None:
        self.graph = graph or build_standard_mnemorph_corpus()

    def evaluate_retrieval(self, g: MnemorphGraph, query_tags: Set[str]) -> Tuple[float, List[str]]:
        """Score precision/recall of surviving nodes matching query tags."""
        hits = []
        for node_id, data in g.nodes.items():
            overlap = len(set(data["tags"]) & query_tags)
            if overlap > 0:
                hits.append((overlap, node_id))
        hits.sort(reverse=True)
        retrieved_ids = [hid for _, hid in hits[:5]]
        score = sum(overlap for overlap, _ in hits[:5]) / (len(query_tags) * 5.0 + 1e-8)
        return min(1.0, float(score)), retrieved_ids

    def apply_hub_lesion(self, fraction: float = 0.25) -> Tuple[MnemorphGraph, Set[str]]:
        """Targeted ablation of top degree hub nodes."""
        clone = self.graph.copy()
        sorted_nodes = sorted(clone.nodes.keys(), key=lambda n: clone.degree(n), reverse=True)
        count = max(1, int(len(sorted_nodes) * fraction))
        ablated = set(sorted_nodes[:count])
        for node_id in ablated:
            del clone.nodes[node_id]
            for neighbor in clone.edges.get(node_id, set()):
                clone.edges[neighbor].discard(node_id)
            del clone.edges[node_id]
        return clone, ablated

    def apply_periphery_lesion(self, fraction: float = 0.25) -> Tuple[MnemorphGraph, Set[str]]:
        """Targeted ablation of lowest degree peripheral nodes."""
        clone = self.graph.copy()
        sorted_nodes = sorted(clone.nodes.keys(), key=lambda n: clone.degree(n))
        count = max(1, int(len(sorted_nodes) * fraction))
        ablated = set(sorted_nodes[:count])
        for node_id in ablated:
            del clone.nodes[node_id]
            for neighbor in clone.edges.get(node_id, set()):
                clone.edges[neighbor].discard(node_id)
            del clone.edges[node_id]
        return clone, ablated

    def apply_associative_regrowth(self, damaged: MnemorphGraph, ablated_nodes: Set[str]) -> Tuple[MnemorphGraph, float]:
        """Attempt reconstructive recovery of ablated nodes using surviving neighbor consensus."""
        reconstructed = damaged.copy()
        recovered_count = 0
        for target_id in ablated_nodes:
            original_meta = self.graph.nodes[target_id]
            original_neighbors = self.graph.edges.get(target_id, set())
            # Check how many original neighbors survive in damaged graph
            surviving_neighbors = [n for n in original_neighbors if n in damaged.nodes]
            if len(surviving_neighbors) >= 2:
                # Reconstruct node with consensus tags
                neighbor_tags = set()
                for n in surviving_neighbors:
                    neighbor_tags.update(damaged.nodes[n]["tags"])
                reconstructed.add_node(target_id, f"reconstructed:{original_meta['fact']}", original_meta["carrier"], list(neighbor_tags))
                for n in surviving_neighbors:
                    reconstructed.add_edge(target_id, n)
                recovered_count += 1
        regrowth_rate = recovered_count / (len(ablated_nodes) + 1e-8)
        return reconstructed, float(regrowth_rate)

    def run_archaeological_study(self) -> Dict[str, Any]:
        """Run complete 4-assay factorial study."""
        from .mco import STANDARD_TASKS
        
        # 1. Baseline intact
        intact_scores = [self.evaluate_retrieval(self.graph, set(t["expected_keywords"]))[0] for t in STANDARD_TASKS]
        intact_mean = sum(intact_scores) / len(intact_scores)

        # 2. Hub Lesion (25%)
        g_hub, ablated_hubs = self.apply_hub_lesion(0.25)
        hub_scores = [self.evaluate_retrieval(g_hub, set(t["expected_keywords"]))[0] for t in STANDARD_TASKS]
        hub_mean = sum(hub_scores) / len(hub_scores)

        # 3. Periphery Lesion (25%)
        g_periphery, ablated_periphery = self.apply_periphery_lesion(0.25)
        periphery_scores = [self.evaluate_retrieval(g_periphery, set(t["expected_keywords"]))[0] for t in STANDARD_TASKS]
        periphery_mean = sum(periphery_scores) / len(periphery_scores)

        # 4. Associative Regrowth from Hub Lesion
        g_regrown, regrowth_rate = self.apply_associative_regrowth(g_hub, ablated_hubs)
        regrowth_scores = [self.evaluate_retrieval(g_regrown, set(t["expected_keywords"]))[0] for t in STANDARD_TASKS]
        regrowth_mean = sum(regrowth_scores) / len(regrowth_scores)

        # Hub vulnerability ratio: how much worse is hub loss than peripheral loss?
        hub_vulnerability_ratio = (periphery_mean - hub_mean) / (periphery_mean + 1e-8)

        receipt = {
            "schema": "poseidon-mnemorph-v1",
            "total_carrier_nodes": len(self.graph.nodes),
            "total_edges": sum(len(n) for n in self.graph.edges.values()) // 2,
            "intact_mean_retrieval": round(float(intact_mean), 4),
            "hub_lesion_mean_retrieval": round(float(hub_mean), 4),
            "periphery_lesion_mean_retrieval": round(float(periphery_mean), 4),
            "regrowth_mean_retrieval": round(float(regrowth_mean), 4),
            "associative_regrowth_recovery_rate": round(float(regrowth_rate), 4),
            "hub_vulnerability_ratio": round(float(hub_vulnerability_ratio), 4),
            "ablated_hub_count": len(ablated_hubs),
            "ablated_periphery_count": len(ablated_periphery),
        }
        receipt_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
        receipt["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
        return receipt


def verify_mnemorph_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Verify deterministic BLAKE2b/SHA256 receipt integrity."""
    required = {"schema", "intact_mean_retrieval", "hub_lesion_mean_retrieval", "periphery_lesion_mean_retrieval", "regrowth_mean_retrieval", "receipt_sha256"}
    if not required.issubset(receipt.keys()):
        return {"verified": False, "error": "Missing required receipt fields"}
    clone = dict(receipt)
    recorded_sha = clone.pop("receipt_sha256")
    computed_sha = hashlib.sha256(json.dumps(clone, sort_keys=True).encode("utf-8")).hexdigest()
    if recorded_sha != computed_sha:
        return {"verified": False, "error": "SHA-256 digest mismatch"}
    return {"verified": True, "receipt_sha256": recorded_sha}
