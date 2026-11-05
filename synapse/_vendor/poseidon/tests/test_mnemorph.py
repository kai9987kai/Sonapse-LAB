import pytest

from poseidon.mnemorph import (
    MnemorphGraph,
    build_standard_mnemorph_corpus,
    MnemorphArchaeology,
    verify_mnemorph_receipt,
)


def test_mnemorph_graph_construction():
    graph = build_standard_mnemorph_corpus()
    assert len(graph.nodes) == 32
    # Ensure graph has edges
    total_edges = sum(len(neighbors) for neighbors in graph.edges.values()) // 2
    assert total_edges > 20


def test_mnemorph_hub_vs_periphery_lesions():
    arch = MnemorphArchaeology()
    g_hub, ablated_hubs = arch.apply_hub_lesion(0.25)
    g_periph, ablated_periph = arch.apply_periphery_lesion(0.25)
    
    assert len(ablated_hubs) == 8
    assert len(ablated_periph) == 8
    assert len(g_hub.nodes) == 24
    assert len(g_periph.nodes) == 24
    
    # Hub degree should be higher than peripheral degree
    hub_deg_avg = sum(arch.graph.degree(n) for n in ablated_hubs) / 8.0
    periph_deg_avg = sum(arch.graph.degree(n) for n in ablated_periph) / 8.0
    assert hub_deg_avg > periph_deg_avg


def test_mnemorph_study_and_receipt_verification():
    arch = MnemorphArchaeology()
    receipt = arch.run_archaeological_study()
    
    assert receipt["schema"] == "poseidon-mnemorph-v1"
    assert receipt["intact_mean_retrieval"] > receipt["hub_lesion_mean_retrieval"]
    assert receipt["associative_regrowth_recovery_rate"] > 0.0
    assert "receipt_sha256" in receipt
    
    verification = verify_mnemorph_receipt(receipt)
    assert verification["verified"]
