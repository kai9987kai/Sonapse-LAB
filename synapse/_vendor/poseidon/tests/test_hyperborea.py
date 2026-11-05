"""Unit tests for Stage 22 Project HYPERBOREA (`poseidon/hyperborea.py`)."""
from __future__ import annotations

import math

from poseidon.hyperborea import (
    HyperboreaEngine,
    PortHamiltonianDiracEngine,
    RenormalizationGroupEngine,
    SheafCohomologyEngine,
    TopologicalAnyonBraidingEngine,
)


def test_sheaf_cohomology_consistent_and_frustrated():
    engine = SheafCohomologyEngine()
    # 1. Consistent (trivial) 2D connection sheaf on a 3-cycle: dim(H^0) = 2, dim(H^1) = 2
    res_consistent = engine.analyze_sheaf(
        num_vertices=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        stalk_dim=2,
        edge_angles_rad=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
    )
    assert res_consistent["verified"] is True
    assert res_consistent["dim_H0_global_sections"] == 2
    assert res_consistent["dim_H1_obstructions"] == 2
    assert res_consistent["topologically_frustrated"] is False
    assert res_consistent["euler_poincare_verified"] is True

    # 2. Frustrated 1D signed sheaf on a 3-cycle with one negative edge (-1 holonomy): dim(H^0) = 0, dim(H^1) = 0
    res_frustrated = engine.analyze_sheaf(
        num_vertices=3,
        edges=[(0, 1), (1, 2), (0, 2)],
        stalk_dim=1,
        edge_signs=[(1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
    )
    assert res_frustrated["verified"] is True
    assert res_frustrated["dim_H0_global_sections"] == 0
    assert res_frustrated["topologically_frustrated"] is True
    assert res_frustrated["spectral_gap_lambda_min"] > 0.5


def test_port_hamiltonian_dirac_and_ida_pbc():
    engine = PortHamiltonianDiracEngine()
    res = engine.analyze_and_control()
    assert res["verified"] is True
    assert res["dirac_skew_residual"] == 0.0
    assert res["dirac_power_conservation_error"] < 1e-12
    assert res["passivity_certified"] is True
    assert res["lyapunov_stability_certified"] is True
    assert res["shaped_hamiltonian_final"] < res["shaped_hamiltonian_initial"]


def test_renormalization_group_migdal_kadanoff_criticality():
    engine = RenormalizationGroupEngine()
    res = engine.analyze_rg_flow(K_initial=0.5, spatial_dim=2, rescaling_b=2)
    assert res["verified"] is True
    assert abs(res["critical_coupling_Kc"] - 0.60937786) < 1e-6
    assert abs(res["thermal_eigenvalue_lambda_T"] - 1.67857351) < 1e-6
    assert abs(res["correlation_length_exponent_nu"] - 1.338268) < 1e-5
    assert res["tribonacci_polynomial_residual"] < 1e-12
    assert res["basin_separation_verified"] is True
    assert "paramagnetic" in res["infrared_phase"]


def test_topological_anyon_braiding_pentagon_and_hexagon():
    engine = TopologicalAnyonBraidingEngine()
    res = engine.evaluate_braid_word(braid_word=["sigma_1", "sigma_2", "sigma_1"], num_anyons=4)
    assert res["verified"] is True
    assert abs(res["golden_ratio_quantum_dim_d_tau"] - (1.0 + math.sqrt(5.0)) / 2.0) < 1e-7
    assert res["pentagon_identity_residual"] < 1e-12
    assert res["hexagon_artin_braid_residual"] < 1e-12
    assert res["unitarity_residual"] < 1e-12
    assert res["fusion_space_dim_charge_tau"] == 3
    assert res["fusion_space_dim_charge_1"] == 2


def test_hyperborea_orchestrator_synthesize():
    orchestrator = HyperboreaEngine()
    status = orchestrator.status()
    assert status["ready"] is True
    assert status["cellular_sheaf_cohomology"] is True
    assert status["port_hamiltonian_dirac_control"] is True
    assert status["renormalization_group_criticality"] is True
    assert status["topological_anyon_braiding"] is True

    syn = orchestrator.synthesize()
    assert syn["verified"] is True
    assert syn["project"] == "HYPERBOREA"
    assert len(syn["receipt_sha256"]) == 64
