"""Unit and integration tests for Project CARTAN (poseidon/cartan.py)."""
from __future__ import annotations

import math
from poseidon.cartan import (
    CartanExteriorSystemEngine,
    NonHermitianPTEngine,
    Octonion,
    ExceptionalJordanAlgebraEngine,
    HolographicEntanglementEngine,
    MorphogeneticTuringEngine,
    CartanEngine,
)


def test_cartan_exterior_structure_equations():
    engine = CartanExteriorSystemEngine(dim=3)
    res = engine.structure_equations()
    assert res["manifold_dimension"] == 3
    assert res["metric_compatible"] is True
    assert res["is_involutive_system"] is True
    assert res["cartan_test_dimension"] > 0
    assert res["structure_equation_status"] == "Q.E.D. VERIFIED"


def test_non_hermitian_pt_system_unbroken_and_exceptional_point():
    engine = NonHermitianPTEngine()

    # Unbroken PT: coupling g > gamma
    res_unbroken = engine.analyze_pt_system(delta=0.0, gamma=1.0, coupling_g=2.0)
    assert res_unbroken["phase"] == "Unbroken PT Symmetry (Real Spectrum)"
    assert res_unbroken["pt_broken"] is False
    assert abs(res_unbroken["eigenvalue_plus"]["imag"]) < 1e-6
    assert abs(res_unbroken["eigenvalue_plus"]["real"] - math.sqrt(3.0)) < 1e-4

    # Exceptional Point: coupling g == gamma
    res_ep = engine.analyze_pt_system(delta=0.0, gamma=1.0, coupling_g=1.0)
    assert res_ep["is_exceptional_point"] is True
    assert res_ep["jordan_block_defective"] is True
    assert res_ep["petermann_divergence_factor"] >= 1e4

    # Broken PT: coupling g < gamma
    res_broken = engine.analyze_pt_system(delta=0.0, gamma=1.5, coupling_g=0.5)
    assert res_broken["pt_broken"] is True
    assert abs(res_broken["eigenvalue_plus"]["real"]) < 1e-6
    assert abs(res_broken["eigenvalue_plus"]["imag"]) > 0.0


def test_exceptional_jordan_algebra_and_octonions():
    engine = ExceptionalJordanAlgebraEngine()

    # Octonion norm composition Hurwitz theorem
    a = (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    res = engine.test_octonion_algebra(a, b)
    assert res["hurwitz_norm_identity_verified"] is True
    assert res["algebra_dimension"] == 8
    assert res["albert_algebra_dimension"] == 27
    assert res["automorphism_group"] == "F4 (dimension 52)"

    # Freudenthal cubic determinant
    f_res = engine.freudenthal_determinant(diag=(2.0, 3.0, 1.0), off_norms=(0.5, 0.4, 0.3))
    assert f_res["jordan_trace"] == 6.0
    assert f_res["freudenthal_determinant"] > 0.0
    assert f_res["rank"] == 3


def test_holographic_entanglement_and_thermofield_double():
    engine = HolographicEntanglementEngine(c_central_charge=12.0, total_length_L=100.0, uv_cutoff_epsilon=0.1)

    # Ryu-Takayanagi single interval
    res = engine.single_interval_entropy(interval_length_ell=25.0)
    assert res["central_charge_c"] == 12.0
    assert res["entanglement_entropy_nats"] > 0.0
    assert res["entanglement_entropy_bits"] > 0.0
    assert res["bulk_geodesic_length"] > 0.0

    # Thermofield Double state
    tfd_res = engine.thermofield_double_horizon(temperature_T=1.0, time_t=5.0)
    assert tfd_res["thermal_bekenstein_hawking_entropy"] > 0.0
    assert tfd_res["page_transition_time"] > 0.0
    assert tfd_res["einstein_rosen_bridge_active"] is True


def test_morphogenetic_turing_instability():
    engine = MorphogeneticTuringEngine(diffusivity_u=0.01, diffusivity_v=0.50)

    # Standard Turing conditions: fu > 0, gv < 0, tr < 0, det > 0
    res = engine.analyze_turing_instability(fu=0.6, fv=-0.8, gu=1.0, gv=-1.1)
    assert res["condition_1_homogeneous_trace_negative"] is True
    assert res["condition_2_homogeneous_det_positive"] is True
    assert res["turing_instability_active"] is True
    assert res["critical_wavenumber_kc"] > 0.0
    assert res["critical_wavelength_lambda_c"] != "infinity"
    assert "Spots" in res["predicted_morphology"] or "Stripes" in res["predicted_morphology"]


def test_cartan_engine_synthesis():
    engine = CartanEngine(seed=42)
    receipt = engine.synthesize(seed=42)
    assert receipt["project"] == "CARTAN"
    assert receipt["stage"] == 27
    assert receipt["version"] == "v3.8.0-CARTAN-75"
    assert receipt["verified"] is True
    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["cartan_exterior"]["structure_equation_status"] == "Q.E.D. VERIFIED"
    assert receipt["non_hermitian_pt"]["verified"] is True
    assert receipt["exceptional_jordan"]["algebra_dimension"] == 8
    assert receipt["holographic_entanglement"]["verified"] is True
    assert receipt["morphogenetic_turing"]["verified"] is True
