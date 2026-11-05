"""Tests for Project NOETHER: Stage 26 Invariant Symmetries,
Modular Tensor Categories, Clifford STA, Batalin-Vilkovisky Quantization,
and Nonequilibrium Active Matter.
"""
from __future__ import annotations

import math
from poseidon.noether import (
    NoetherSymmetryEngine,
    ModularTensorCategoryEngine,
    CliffordSpacetimeAlgebraEngine,
    BatalinVilkoviskyEngine,
    ActiveMatterEngine,
    NoetherEngine,
)


def test_noether_symmetry_engine() -> None:
    engine = NoetherSymmetryEngine()
    res = engine.evaluate_symmetry(
        potential_type="harmonic",
        charge_type="angular_momentum",
        mass=2.0,
        omega=1.5,
        r0=2.0,
        v_theta=4.0,
    )
    assert res["verified"] is True
    assert res["symmetry_group"] == "SO(2)"
    assert res["noether_conserved_charge"] > 0.0
    assert abs(res["time_derivative_charge"]) < 1e-12
    assert res["poisson_bracket_bracket_zero"] is True
    assert res["noether_second_theorem_identity_residual"] < 1e-12


def test_modular_tensor_category_fibonacci() -> None:
    engine = ModularTensorCategoryEngine()
    res = engine.evaluate_fibonacci_mtc()
    assert res["verified"] is True
    assert res["category"] == "Fibonacci MTC"
    assert res["s_matrix_unitary"] is True
    assert res["s_matrix_symmetric"] is True
    assert res["verlinde_algebra_valid"] is True
    assert res["fusion_coefficients"]["N_tau_tau^1"] == 1
    assert res["fusion_coefficients"]["N_tau_tau^tau"] == 1
    assert res["total_quantum_dimension_D"] > 1.9


def test_modular_tensor_category_ising() -> None:
    engine = ModularTensorCategoryEngine()
    res = engine.evaluate_ising_mtc()
    assert res["verified"] is True
    assert res["category"] == "Ising MTC"
    assert res["s_matrix_unitary"] is True
    assert res["s_matrix_symmetric"] is True
    assert res["verlinde_algebra_valid"] is True
    assert res["fusion_coefficients"]["N_sigma_sigma^1"] == 1
    assert res["fusion_coefficients"]["N_sigma_sigma^psi"] == 1
    assert res["fusion_coefficients"]["N_sigma_sigma^sigma"] == 0
    assert res["total_quantum_dimension_D"] == 2.0


def test_clifford_spacetime_algebra() -> None:
    engine = CliffordSpacetimeAlgebraEngine()
    res = engine.evaluate_em_bivector_lorentz_invariants(
        e_field=(2.0, 3.0, 1.0),
        b_field=(0.5, 1.0, 2.0),
        boost_beta=0.8,
    )
    assert res["verified"] is True
    assert res["lorentz_invariance_verified"] is True
    assert res["residual_invariant_1"] < 1e-10
    assert res["residual_invariant_2"] < 1e-10


def test_batalin_vilkovisky_engine() -> None:
    engine = BatalinVilkoviskyEngine()
    res = engine.evaluate_bv_gauge_fixing(gauge_parameter=1.5)
    assert res["verified"] is True
    assert res["cme_satisfied"] is True
    assert res["qme_anomaly_free"] is True
    assert res["classical_master_equation_residual"] < 1e-12
    assert res["quantum_bv_laplacian_anomaly"] < 1e-12


def test_active_matter_engine() -> None:
    engine = ActiveMatterEngine()
    res = engine.evaluate_active_brownian_particle(
        swim_speed=15.0,
        rotational_diffusion=2.0,
        translational_diffusion=0.2,
    )
    assert res["verified"] is True
    assert res["persistence_length_lp"] == 7.5
    assert res["effective_long_time_diffusion_Deff"] > 0.2
    assert res["short_time_ballistic_fractional_residual"] < 0.05
    assert res["long_time_diffusive_fractional_residual"] < 0.01


def test_noether_unified_engine() -> None:
    engine = NoetherEngine()
    res = engine.synthesize(seed=42)
    assert res["project"] == "NOETHER"
    assert res["stage"] == 26
    assert res["version"] == "v3.7.0-NOETHER-70"
    assert res["verified"] is True
    assert len(res["receipt_sha256"]) == 64
    assert "[Project NOETHER Stage 26 Synthesis Certified]" in res["text"]
