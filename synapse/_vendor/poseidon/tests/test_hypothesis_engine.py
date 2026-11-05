"""Unit tests for Project OLYMPUS: Scientific Hypothesis & Falsification Engine."""
import pytest
from poseidon.hypothesis_engine import (
    ScientificHypothesis,
    CrucialExperiment,
    HypothesisFalsificationReceipt,
    FalsificationOutcome,
    HypothesisEngine,
)


def test_hypothesis_formulation_and_demarcation():
    engine = HypothesisEngine()
    hyp = engine.formulate_hypothesis(
        conjecture="General Relativity: Light rays bend around massive bodies with angle theta = 4GM / (c^2 * r).",
        domain="astrophysics",
        hard_core_assumptions=["Equivalence Principle", "Pseudo-Riemannian Metric Curvature"],
        protective_belt=["PPN parameters", "Solar oblateness bounds"],
        falsification_criteria=["Solar eclipse deflection angle is <= 2GM / (c^2 * r) within 3-sigma"],
        novel_predictions=["1.75 arcsecond deflection at solar limb", "Gravitational time dilation in GPS orbits"],
        prior=0.5,
    )
    assert hyp.id.startswith("hyp-")
    assert hyp.domain == "astrophysics"
    assert hyp.status == "PROPOSED"
    assert len(hyp.hard_core_assumptions) == 2
    assert len(hyp.falsification_criteria) == 1

    # Popperian demarcation: Must specify falsification criteria
    with pytest.raises(ValueError, match="Popperian demarcation"):
        engine.formulate_hypothesis(
            conjecture="Unfalsifiable metaphysical dogma",
            domain="metaphysics",
            hard_core_assumptions=["Dogma"],
            protective_belt=[],
            falsification_criteria=[],
            novel_predictions=[],
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        engine.formulate_hypothesis(
            conjecture="",
            domain="physics",
            hard_core_assumptions=[],
            protective_belt=[],
            falsification_criteria=["x > 0"],
            novel_predictions=[],
        )


def test_crucial_experiment_design_and_shannon_information():
    engine = HypothesisEngine()
    h_einstein = engine.formulate_hypothesis(
        conjecture="Einstein GR: Deflection theta = 1.75 arcsec",
        domain="gravity",
        hard_core_assumptions=["Curvature"],
        protective_belt=[],
        falsification_criteria=["Deflection < 1.5 arcsec"],
        novel_predictions=["1.75 arcsec deflection"],
        prior=0.5,
    )
    h_newton = engine.formulate_hypothesis(
        conjecture="Corpuscular Newtonian: Deflection theta = 0.875 arcsec",
        domain="gravity",
        hard_core_assumptions=["Flat Euclidean Space", "Inverse Square Law"],
        protective_belt=[],
        falsification_criteria=["Deflection > 1.2 arcsec"],
        novel_predictions=["0.875 arcsec deflection"],
        prior=0.5,
    )

    exp = engine.design_crucial_experiment(
        hypothesis_a=h_einstein,
        hypothesis_b=h_newton,
        name="1919 Eddington Solar Eclipse Experiment",
        description="Measure starlight deflection angle around solar limb during total solar eclipse in Sobral and Príncipe.",
        discriminator_variable="deflection_arcsec",
        prediction_a={"point": 1.75, "tolerance": 0.20},
        prediction_b={"point": 0.875, "tolerance": 0.20},
        severity_score=0.92,
    )

    assert exp.id.startswith("exp-")
    assert exp.discriminator_variable == "deflection_arcsec"
    assert exp.expected_shannon_information_bits > 0.8
    assert exp.severity_score == 0.92


def test_crucial_experiment_execution_and_receipt_verification():
    engine = HypothesisEngine()
    h1 = engine.formulate_hypothesis(
        conjecture="H1: Superconductivity mediated by Cooper pairs at critical temperature Tc.",
        domain="condensed_matter",
        hard_core_assumptions=["BCS Electron-Phonon Coupling"],
        protective_belt=["Isotope effect exponent alpha ~ 0.5"],
        falsification_criteria=["Isotope coefficient alpha is exactly 0.0"],
        novel_predictions=["Energy gap Delta(0) = 1.764 k_B Tc"],
        prior=0.5,
    )
    h2 = engine.formulate_hypothesis(
        conjecture="H2: Classical Drude-Lorentz electron transport without gap.",
        domain="condensed_matter",
        hard_core_assumptions=["Classical scattering"],
        protective_belt=[],
        falsification_criteria=["Resistance drops discontinuously to 0"],
        novel_predictions=["Resistance is linear in T down to 0K"],
        prior=0.5,
    )

    exp = engine.design_crucial_experiment(
        hypothesis_a=h1,
        hypothesis_b=h2,
        name="Tunneling Density of States Energy Gap Measurement",
        description="Giaever tunneling junction conductance assay to detect spectral gap.",
        discriminator_variable="energy_gap_ratio",
        prediction_a={"range": [1.70, 1.82]},
        prediction_b={"range": [0.0, 0.1]},
        severity_score=0.90,
    )

    # Observed outcome matching H1
    outcome = engine.conduct_test(exp, observed_data={"energy_gap_ratio": 1.765})
    assert outcome.falsification_triggered is False
    assert outcome.decisive is True
    assert outcome.lakatos_assessment == "PROGRESSIVE"
    assert outcome.posterior_confidence > 0.80
    assert h1.status == "CORROBORATED"

    # Verify cryptographic SHA-256 receipt
    receipt = outcome.receipt
    assert receipt.verify() is True

    # Tampering test: altering receipt payload breaks signature
    receipt.observed_value = 999.999
    assert receipt.verify() is False


def test_falsification_decisive_refutation():
    engine = HypothesisEngine()
    h1 = engine.formulate_hypothesis(
        conjecture="Luminiferous Aether: Earth moves relative to stationary aether frame producing fringe shift delta >= 0.4.",
        domain="optics",
        hard_core_assumptions=["Aether Medium"],
        protective_belt=[],
        falsification_criteria=["Fringe shift delta < 0.05"],
        novel_predictions=["Seasonal fringe shift fluctuation of 0.4 fringes"],
        prior=0.7,
    )
    h2 = engine.formulate_hypothesis(
        conjecture="Special Relativity: Speed of light is isotropic in all inertial frames (delta = 0).",
        domain="optics",
        hard_core_assumptions=["Invariance of c"],
        protective_belt=[],
        falsification_criteria=["Fringe shift delta > 0.1"],
        novel_predictions=["Null fringe shift delta < 0.01"],
        prior=0.3,
    )

    exp = engine.design_crucial_experiment(
        hypothesis_a=h1,
        hypothesis_b=h2,
        name="1887 Michelson-Morley Interferometer",
        description="Measure optical interference fringe shift between perpendicular arms as Earth rotates.",
        discriminator_variable="fringe_shift",
        prediction_a={"range": [0.35, 0.45]},
        prediction_b={"range": [-0.02, 0.02]},
        severity_score=0.95,
    )

    # Observation: Null result (fringe_shift = 0.005)
    outcome = engine.conduct_test(exp, observed_data={"fringe_shift": 0.005})
    assert outcome.falsification_triggered is True
    assert outcome.decisive is True
    assert outcome.lakatos_assessment == "REFUTED"
    assert outcome.posterior_confidence < 0.15
    assert h1.status == "FALSIFIED"
    assert outcome.receipt.verify() is True


def test_research_programme_evaluation_and_paradigm_synthesis():
    engine = HypothesisEngine()
    h1 = engine.formulate_hypothesis(
        conjecture="Turing Morphogenesis: Reaction-diffusion produces spot/stripe periodicity.",
        domain="biophysics",
        hard_core_assumptions=["Inhibitor diffusion Dv >> Du", "Nonlinear activator feedback"],
        protective_belt=["Boundary geometric constraints"],
        falsification_criteria=["Spatial pattern forms when Dv <= Du"],
        novel_predictions=["Wavelength scales with sqrt(Du * Dv)"],
        prior=0.6,
    )
    h_null = engine.formulate_hypothesis(
        conjecture="Uniform Morphogen Gradient: French Flag model without self-organization.",
        domain="biophysics",
        hard_core_assumptions=["Positional information source-sink"],
        protective_belt=[],
        falsification_criteria=["Spontaneous pattern emergence without external gradient"],
        novel_predictions=["Linear spatial concentration decay"],
        prior=0.4,
    )

    exp = engine.design_crucial_experiment(
        hypothesis_a=h1,
        hypothesis_b=h_null,
        name="CIMA Chemical Reaction Morphogenesis Test",
        description="Chlorite-Iodide-Malonic Acid pattern formation in gel reactor.",
        discriminator_variable="pattern_type",
        prediction_a={"exact": "periodic_turing_spots"},
        prediction_b={"exact": "uniform_gradient"},
        severity_score=0.90,
    )

    outcome = engine.conduct_test(exp, observed_data={"pattern_type": "periodic_turing_spots"})
    assert outcome.lakatos_assessment == "PROGRESSIVE"

    eval_prog = engine.evaluate_research_programme("biophysics")
    assert eval_prog["programme_status"] == "THEORETICALLY_AND_EMPIRICALLY_PROGRESSIVE"
    assert eval_prog["corroborations"] == 1
    assert eval_prog["falsifications"] == 0

    synthesis = engine.generate_paradigm_synthesis("biophysics")
    assert len(synthesis["corroborated_principles"]) == 1
    assert synthesis["epistemic_integrity_receipts_verified"] is True
    assert "Scientific Paradigm" in synthesis["summary"]
