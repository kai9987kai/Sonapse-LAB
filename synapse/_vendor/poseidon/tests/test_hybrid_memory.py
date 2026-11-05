from poseidon.memory import MemoryBank


def test_hybrid_rrf_retrieves_semantically_related_entries():
    bank = MemoryBank()
    bank.remember("Freshwater filtration unit is operational in the habitat module", "habitat")
    bank.remember("Dune crossing depletes stamina rapidly", "body")
    bank.remember("Mira signaled an incoming sandstorm warning", "social")

    # Pure lexical BM25
    lexical_hits = bank.retrieve("water drinking", mode="bm25")
    assert len(lexical_hits) == 0  # no exact token 'water' or 'drinking' in text

    # Hybrid RRF (dense semantic features + BM25)
    hybrid_hits = bank.retrieve("water supply in habitat", mode="hybrid")
    assert len(hybrid_hits) > 0
    assert hybrid_hits[0]["carrier"] == "habitat"
    assert hybrid_hits[0]["provenance"]["retrieval"] == "hybrid-rrf-v1"
    assert "bm25_rank" in hybrid_hits[0]["provenance"] or "dense_rank" in hybrid_hits[0]["provenance"]


def test_dense_mode_retrieves_based_on_cosine():
    bank = MemoryBank()
    bank.remember("Solar batteries store power during clear days", "habitat")
    dense_hits = bank.retrieve("solar energy storage", mode="dense")
    assert len(dense_hits) > 0
    assert dense_hits[0]["provenance"]["retrieval"] == "dense-hashing-v1"
    assert dense_hits[0]["provenance"]["dense_cosine"] > 0.08
