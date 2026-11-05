"""Unit tests for Project DA VINCI: Deep Computational Creativity Engine."""

import pytest
from poseidon.creativity_engine import (
    CreativityEngine,
    CANONICAL_MATRICES,
    TRIZ_PRINCIPLES,
)


def test_list_matrices_and_principles():
    engine = CreativityEngine()
    matrices = engine.list_matrices()
    principles = engine.list_triz_principles()
    assert len(matrices) >= 8
    assert len(principles) >= 11
    assert any(m["name"] == "Quantum Mechanics" for m in matrices)
    assert any(p["id"] == 1 and p["name"] == "Segmentation" for p in principles)


def test_bisociation_synthesis():
    engine = CreativityEngine()
    synth = engine.bisociate("quantum_mechanics", "biological_immunology")
    assert synth.matrix_a == "Quantum Mechanics"
    assert synth.matrix_b == "Biological Immunology"
    assert len(synth.analogical_isomorphisms) > 0
    assert synth.novelty_score >= 0.75
    assert len(synth.falsifiable_predictions) >= 3


def test_triz_contradiction_resolution():
    engine = CreativityEngine()
    res = engine.resolve_triz("speed", "precision")
    assert res.improving_parameter == "speed"
    assert res.deteriorating_parameter == "precision"
    assert res.contradiction_resolved is True
    assert len(res.recommended_principles) > 0


def test_autonomous_invention_dossier():
    engine = CreativityEngine()
    dossier = engine.invent(
        problem_statement="Ultra-low-latency consensus in adversarial distributed swarms",
        domain_a="distributed_consensus",
        domain_b="mycology_fungal_networks",
        improving_param="capacity",
        deteriorating_param="latency",
    )
    assert "Project DA VINCI" in dossier.title
    assert dossier.pareto_score > 0.8
    assert len(dossier.receipt_sha256) == 64
    assert dossier.triz_resolution is not None
    assert dossier.triz_resolution.contradiction_resolved is True
