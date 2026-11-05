"""Tests for Project NOOSPHERE (Stage 17): Creative Bisociation, Concept Blending, and Dialectical Tribunal."""
from __future__ import annotations

import copy
import pytest

from poseidon.creative_synthesis import (
    ARCHETYPE_LIBRARY,
    SCHEMA,
    ConceptBlender,
    DialecticalTribunal,
    StructuralMorphismMapper,
    verify_creative_receipt,
)


def test_archetype_library_completeness() -> None:
    assert len(ARCHETYPE_LIBRARY) == 12
    slugs = [a["slug"] for a in ARCHETYPE_LIBRARY]
    assert len(set(slugs)) == 12
    for a in ARCHETYPE_LIBRARY:
        assert len(a["structural_signature"]) == 8
        assert len(a["semantic_vector"]) == 8
        for role in ("state", "generator", "invariant", "dual", "lyapunov"):
            assert role in a["operators"]


def test_structural_morphism_mapper_bisociation_ranking() -> None:
    mapper = StructuralMorphismMapper()
    src = mapper.select_source_archetype("Analyze quantum entanglement concurrence and Bell state")
    assert src["slug"] == "hilbert_quantum_entanglement"
    top = mapper.rank_bisociations(src["slug"], top_k=3)
    assert len(top) == 3
    assert top[0]["koestler_bisociation_score"] >= top[1]["koestler_bisociation_score"] >= top[2]["koestler_bisociation_score"]
    assert top[0]["structural_isomorphism"] > 0.7
    assert top[0]["semantic_distance"] > 0.2
    assert "state" in top[0]["functorial_morphism"]


def test_concept_blender_numerical_verification() -> None:
    mapper = StructuralMorphismMapper()
    src = mapper.by_slug["symplectic_hamiltonian"]
    bis = mapper.rank_bisociations("symplectic_hamiltonian", top_k=1)[0]
    tgt = mapper.by_slug[bis["target_slug"]]
    blend = ConceptBlender.blend(src, tgt, bis)
    assert blend["numerical_verification"]["verified"] is True
    assert blend["numerical_verification"]["contraction_ratio"] < 1.0
    assert blend["numerical_verification"]["spectral_radius"] < 1.0
    assert blend["numerical_verification"]["max_unitary_symplectic_residual"] < 1e-9

    # Falsifiable check: unstable coupling gain >= 2.0 must fail verification
    unstable_blend = ConceptBlender.blend(src, tgt, bis, coupling_gain=2.5)
    assert unstable_blend["numerical_verification"]["verified"] is False
    assert unstable_blend["numerical_verification"]["contraction_ratio"] > 1.0
    assert unstable_blend["numerical_verification"]["spectral_radius"] > 1.0


def test_dialectical_tribunal_and_receipt_verification() -> None:
    tribunal = DialecticalTribunal()
    receipt = tribunal.deliberate(
        "Synthesize a creative cross-domain solution for Lotka-Volterra predator-prey oscillations with alpha=1.2, beta=0.4, gamma=0.8, delta=0.2",
        seed=42,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["verified"] is True
    assert len(receipt["dialectical_tribunal"]) == 4
    assert len(receipt["creative_portfolio"]) == 3
    assert receipt["exact_solver_grounding"]["grounded"] is True
    assert receipt["exact_solver_grounding"]["domain"] == "evolutionary_ecology"

    ver = verify_creative_receipt(receipt)
    assert ver["verified"] is True
    assert ver["receipt_sha256"] == receipt["receipt_sha256"]

    tampered = copy.deepcopy(receipt)
    tampered["creative_portfolio"][0]["hcri_score"] = 0.1234
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_creative_receipt(tampered)


def test_dialectical_tribunal_open_ended_creative_prompt() -> None:
    tribunal = DialecticalTribunal()
    receipt = tribunal.deliberate(
        "Invent a bio-inspired distributed consensus protocol that resists Byzantine sybil storms using neuromorphic ring attractors",
        seed=99,
    )
    assert receipt["verified"] is True
    assert receipt["source_archetype"]["slug"] == "hebbian_ring_neuropil_memory"
    assert verify_creative_receipt(receipt)["verified"] is True
