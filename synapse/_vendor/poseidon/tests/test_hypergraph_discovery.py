"""Tests for InductiveHypergraphEngine and Symbolic Theorem Discovery."""
import pytest
from poseidon.hypergraph_discovery import (
    InductiveHypergraphEngine,
    SymbolicTheorem,
)


def test_inductive_theorem_discovery_and_ratification():
    engine = InductiveHypergraphEngine(confidence_threshold=0.70)

    # Generate synthetic experience trace confirming rest recovers stamina
    experiences = []
    for _ in range(10):
        experiences.append({
            "action": 0,  # rest
            "stamina": 0.5,
            "next_stamina": 0.55,
            "in_patch": False,
            "is_submerged": False,
        })
    # Add a couple explore actions
    for _ in range(5):
        experiences.append({
            "action": 4,  # explore
            "stamina": 0.8,
            "next_stamina": 0.77,
            "in_patch": False,
            "is_submerged": False,
        })

    theorems = engine.ingest_trajectory(experiences)
    assert len(theorems) >= 4

    rest_thm = next(t for t in theorems if t.theorem_id == "thm_rest_recovers_stamina")
    assert rest_thm.support_count == 10
    assert rest_thm.refutation_count == 0
    assert rest_thm.ratified is True
    assert len(rest_thm.signature_sha256) == 64


def test_jsonld_knowledge_graph_export():
    engine = InductiveHypergraphEngine()
    dummy_exp = [{"action": 0, "stamina": 0.4, "next_stamina": 0.45}]
    engine.ingest_trajectory(dummy_exp)

    kg = engine.export_jsonld_knowledge_graph()
    assert "@context" in kg
    assert "@graph" in kg
    assert len(kg["@graph"]) >= 4
