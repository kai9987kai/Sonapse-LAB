"""Unit tests for Project AETHER (Stage 24):
Gauge Field Holonomy, Jarzynski Thermodynamics, Information Geometry, Symplectic Contact Mechanics & Categorical Lenses.
"""

import pytest
import math
from poseidon.aether import (
    GaugeFieldHolonomyEngine,
    NonEquilibriumThermodynamicsEngine,
    InformationGeometryFisherEngine,
    SymplecticContactMechanicsEngine,
    CategoricalLens,
    CategoricalOpenGameLensEngine,
    AetherEngine,
)


def test_gauge_field_holonomy():
    engine = GaugeFieldHolonomyEngine(grid_size=4, g_coupling=1.0, seed=42)
    plaq = engine.compute_plaquette(0, 0)

    assert plaq.plaquette_coords == (0, 0)
    assert -math.pi <= plaq.u1_flux_angle_rad <= math.pi
    assert -1.0 <= plaq.u1_holonomy_trace <= 1.0
    assert -2.0 <= plaq.su2_holonomy_trace <= 2.0
    assert plaq.yang_mills_action >= 0.0
    assert plaq.gauge_invariant is True
    assert len(plaq.su2_holonomy_matrix) == 2

    # Test loop holonomy along a closed rectangular path: (0,0,0,1) -> (1,0,1,1) -> (1,1,0,-1) -> (0,1,1,-1)
    loop = [
        (0, 0, 0, 1),
        (1, 0, 1, 1),
        (0, 1, 0, -1),
        (0, 0, 1, -1),
    ]
    wilson = engine.compute_wilson_loop(loop)
    assert wilson["loop_length"] == 4
    assert -1.0 <= wilson["u1_wilson_loop"] <= 1.0
    assert wilson["unitarity_preserved"] is True


def test_jarzynski_thermodynamics():
    thermo = NonEquilibriumThermodynamicsEngine(beta=1.5, spring_k=2.0)
    res = thermo.evaluate_protocol(
        center_start=0.0,
        center_end=1.5,
        num_trajectories=200,
        seed=42,
    )

    assert res.num_trajectories == 200
    assert res.beta_inverse_temp == 1.5
    assert res.free_energy_delta_exact == 0.0
    # Jarzynski estimate should be close to 0.0 within sampling fluctuation
    assert abs(res.free_energy_delta_jarzynski - res.free_energy_delta_exact) < 0.5
    assert res.dissipated_work >= 0.0
    assert res.entropy_production_nats >= 0.0
    assert res.second_law_satisfied is True
    assert res.tur_bound_satisfied is True
    assert res.crooks_relative_error < 0.15


def test_information_geometry_fisher():
    fisher_eng = InformationGeometryFisherEngine()
    res = fisher_eng.compute_gaussian_manifold(
        mu=1.0,
        sigma=2.0,
        loss_grad=[0.4, -0.2],
        mu_ref=0.0,
        sigma_ref=1.0,
    )

    assert res.mu == 1.0
    assert res.sigma == 2.0
    # g11 = 1 / 4 = 0.25, g22 = 2 / 4 = 0.5
    assert abs(res.fisher_metric[0][0] - 0.25) < 1e-4
    assert abs(res.fisher_metric[1][1] - 0.50) < 1e-4
    assert res.gaussian_curvature == -0.5
    # Natural gradient: g^-1 grad = [4 * 0.4, 2 * -0.2] = [1.6, -0.4]
    assert abs(res.natural_gradient[0] - 1.6) < 1e-4
    assert abs(res.natural_gradient[1] - (-0.4)) < 1e-4
    # Geodesic distance must be strictly positive and finite
    assert res.geodesic_distance_to_ref > 0.0
    assert res.riemannian_isometry_verified is True


def test_symplectic_contact_mechanics():
    # 1. Conservative (gamma = 0) Störmer-Verlet preserves energy with small drift
    engine = SymplecticContactMechanicsEngine(mass=1.0, spring_k=4.0, gamma_damping=0.0)
    res = engine.simulate(q0=1.0, p0=0.0, dt=0.05, steps=100)

    assert res.steps == 100
    assert res.dt == 0.05
    assert res.initial_energy > 0.0
    assert res.energy_drift < 0.02
    assert res.symplectic_2form_conserved is True
    assert res.shadow_hamiltonian_residual < 0.005

    # 2. Dissipative contact mechanics (gamma > 0) dissipates energy
    engine_damped = SymplecticContactMechanicsEngine(mass=1.0, spring_k=4.0, gamma_damping=0.5)
    res_damped = engine_damped.simulate(q0=1.0, p0=0.0, dt=0.05, steps=100)
    assert res_damped.final_energy < res_damped.initial_energy


def test_categorical_open_game_lenses():
    l1 = CategoricalLens("SensorFilter", state_dim=3, action_dim=2, feedback_dim=2)
    l2 = CategoricalLens("ActuatorMotor", state_dim=2, action_dim=2, feedback_dim=2)

    # Sequential composition
    seq = CategoricalOpenGameLensEngine.compose_sequential(l1, l2)
    assert seq["composition"] == "sequential"
    assert seq["category_associativity_verified"] is True
    assert len(seq["composed_view"]) == 2

    # Parallel tensor product
    par = CategoricalOpenGameLensEngine.compose_parallel_tensor(l1, l2)
    assert par["composition"] == "monoidal_tensor"
    assert par["bifunctoriality_verified"] is True
    assert len(par["tensor_view"]) == 4

    # Nash equilibrium in bimatrix open game (Matching Pennies / Coordination)
    p1 = [[1.0, 0.0], [0.0, 2.0]]
    p2 = [[2.0, 0.0], [0.0, 1.0]]
    nash = CategoricalOpenGameLensEngine.solve_open_game_nash(p1, p2)
    assert nash["has_pure_equilibrium"] is True
    assert len(nash["pure_nash_equilibria"]) == 2


def test_aether_engine_end_to_end():
    engine = AetherEngine(seed=42)
    status = engine.status()
    assert status["ready"] is True
    assert status["stage"] == 24
    assert status["total_verified_domains"] == 60

    synth = engine.synthesize()
    assert synth["project"] == "AETHER"
    assert synth["stage"] == "v3.5.0-AETHER-60"
    assert synth["verified"] is True
    assert "gauge_holonomy" in synth
    assert "non_equilibrium_thermodynamics" in synth
    assert "information_geometry" in synth
    assert "symplectic_mechanics" in synth
    assert "categorical_lenses" in synth
    assert len(synth["receipt_sha256"]) == 64
