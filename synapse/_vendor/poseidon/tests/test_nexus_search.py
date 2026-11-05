"""Tests for NexusSearch Federated Hybrid Associative Search Engine."""
import numpy as np
import pytest
from poseidon.nexus_search import NexusSearchEngine


def test_nexus_search_indexing_and_retrieval():
    engine = NexusSearchEngine(dimension=16, max_entries=10, resonance_threshold=0.10)
    assert engine.stats()["indexed_count"] == 0

    # Index 2 distinct concepts
    v1 = np.array([1.0] * 8 + [0.0] * 8, dtype=np.float32)
    v2 = np.array([0.0] * 8 + [1.0] * 8, dtype=np.float32)

    engine.index_event("food_source_a", v1, (1.0, 2.0, 3.0), action=2, reward=0.85, step=1)
    engine.index_event("shelter_b", v2, (5.0, 5.0, 5.0), action=4, reward=0.60, step=2)

    assert engine.stats()["indexed_count"] == 2

    # Query with vector close to v1
    query_v1 = np.array([0.9] * 8 + [0.1] * 8, dtype=np.float32)
    hits = engine.search_by_vector(query_v1, top_k=1)
    assert len(hits) == 1
    assert hits[0]["concept"] == "food_source_a"
    assert hits[0]["action"] == 2


def test_nexus_search_spatial_radius():
    engine = NexusSearchEngine(dimension=16)
    v = np.zeros(16, dtype=np.float32)

    engine.index_event("near_point", v, (1.0, 1.0, 1.0), action=0, reward=0.5, step=1)
    engine.index_event("far_point", v, (20.0, 20.0, 20.0), action=1, reward=0.5, step=2)

    nearby = engine.search_by_spatial_radius((1.0, 1.0, 1.0), radius=3.0)
    assert len(nearby) == 1
    assert nearby[0]["concept"] == "near_point"


def test_nexus_search_highest_reward():
    engine = NexusSearchEngine(dimension=16)
    v = np.zeros(16, dtype=np.float32)

    engine.index_event("low_yield", v, (0.0, 0.0, 0.0), action=1, reward=0.2, step=1)
    engine.index_event("high_yield", v, (2.0, 2.0, 2.0), action=2, reward=0.95, step=2)

    best = engine.get_highest_reward_concept()
    assert best is not None
    assert best["concept"] == "high_yield"
    assert best["reward"] == 0.95
