"""Unit tests for Project POINCARÉ Engine & The 100-Domain Problem Forge Milestone (Stage 32).

Tests:
1. Symplectic Poincaré Return Map & Chirikov Standard Map (pairing sum lambda_1 + lambda_2 = 0, KAM stability).
2. Hyperbolic Poincaré Disk & Möbius Gyrovector Metric (geodesic distance, Möbius addition, Gromov delta-hyperbolicity).
3. Algebraic Topology Poincaré Duality & Morse Complex (boundary nilpotency d^2 = 0, Betti duality beta_k = beta_{n-k}).
4. Hamilton-Perelman Normalized Surface Ricci Flow (Gauss-Bonnet conservation, curvature variance reduction, W-entropy).
5. Unified Quartet Cognitive Coordinator & Closed-Loop Decision Shield.
6. Full PoincareEngine deterministic synthesis pipeline.
"""
from __future__ import annotations

import math
import pytest

from poseidon.poincare_engine import (
    PoincareReturnMapEngine,
    HyperbolicPoincareDiskEngine,
    PoincareDualityMorseEngine,
    RicciFlowPerelmanEngine,
    UnifiedQuartetCognitiveEngine,
    PoincareEngine,
    SCHEMA_ID,
)


def test_poincare_return_map_symplectic_pairing():
    # 1. Conservative KAM Torus Regime (stochastic K < K_c ≈ 0.971635)
    engine_kam = PoincareReturnMapEngine(stochastic_parameter_k=0.5, seed=42)
    step_res = engine_kam.step(theta=1.0, p=0.5)
    assert 0.0 <= step_res["theta_next"] <= 2.0 * math.pi
    assert 0.0 <= step_res["p_next"] <= 2.0 * math.pi
    assert step_res["jacobian_det"] == 1.0  # Exact symplectic volume preservation

    analysis_kam = engine_kam.compute_lyapunov_spectrum(iterations=300)
    assert analysis_kam["area_preserving_verified"] is True
    assert abs(analysis_kam["pairing_sum"]) < 1e-4  # lambda_1 + lambda_2 ≈ 0
    assert analysis_kam["regime"] == "kam_invariant_torus"

    # 2. Chaotic Regime (stochastic K > K_c)
    engine_chaos = PoincareReturnMapEngine(stochastic_parameter_k=2.5, seed=42)
    analysis_chaos = engine_chaos.compute_lyapunov_spectrum(iterations=300)
    assert analysis_chaos["area_preserving_verified"] is True
    assert abs(analysis_chaos["pairing_sum"]) < 1e-4
    assert analysis_chaos["lambda_1"] > 0.05
    assert analysis_chaos["regime"] == "stochastic_sea_chaotic"


def test_hyperbolic_poincare_disk_geometry():
    engine = HyperbolicPoincareDiskEngine(curvature_c=1.0)

    # 1. Distance Symmetry and Zero at origin
    u = [0.2, 0.3]
    v = [-0.4, 0.1]
    origin = [0.0, 0.0]

    assert engine.poincare_distance(origin, origin) == 0.0
    dist_uv = engine.poincare_distance(u, v)
    dist_vu = engine.poincare_distance(v, u)
    assert abs(dist_uv - dist_vu) < 1e-6
    assert dist_uv > 0.0

    # 2. Triangle inequality: d(u, w) <= d(u, v) + d(v, w)
    w = [0.1, -0.4]
    dist_uw = engine.poincare_distance(u, w)
    dist_vw = engine.poincare_distance(v, w)
    assert dist_uw <= dist_uv + dist_vw + 1e-6

    # 3. Möbius Gyrovector Addition: u (+) 0 = u
    sum_origin = engine.mobius_addition(u, origin)
    assert abs(sum_origin[0] - u[0]) < 1e-6
    assert abs(sum_origin[1] - u[1]) < 1e-6

    mob_uv = engine.mobius_addition(u, v)
    assert len(mob_uv) == 2
    norm_sq = mob_uv[0] ** 2 + mob_uv[1] ** 2
    assert norm_sq < 1.0  # Remains strictly inside the unit disk

    # 4. Gromov 4-point condition (delta-hyperbolicity)
    quad = [u, v, w, origin]
    gromov_res = engine.check_gromov_four_point_condition(quad)
    assert gromov_res["delta_hyperbolicity"] >= 0.0
    assert gromov_res["is_delta_hyperbolic"] is True


def test_poincare_duality_and_morse_complex():
    engine = PoincareDualityMorseEngine()

    # 1. Torus T^2: Betti numbers [1, 2, 1], chi = 0
    torus_res = engine.verify_poincare_duality(surface_type="torus")
    assert torus_res["surface"] == "torus"
    assert torus_res["betti_numbers"] == [1, 2, 1]
    assert torus_res["euler_characteristic"] == 0
    assert torus_res["boundary_nilpotency_verified"] is True  # d^2 = 0
    assert torus_res["poincare_duality_verified"] is True    # beta_0 = beta_2, beta_1 = beta_1

    # 2. Sphere S^2: Betti numbers [1, 0, 1], chi = 2
    sphere_res = engine.verify_poincare_duality(surface_type="sphere")
    assert sphere_res["surface"] == "sphere"
    assert sphere_res["betti_numbers"] == [1, 0, 1]
    assert sphere_res["euler_characteristic"] == 2
    assert sphere_res["boundary_nilpotency_verified"] is True
    assert sphere_res["poincare_duality_verified"] is True


def test_ricci_flow_perelman_entropy():
    engine = RicciFlowPerelmanEngine(num_vertices=6, euler_characteristic=2)

    # Initial inhomogeneous curvatures
    curvatures = [1.2, 0.4, 2.5, 3.1, 1.8, 3.56637]
    # Sum ≈ 4*pi ≈ 12.56637 for chi=2

    flow_res = engine.run_normalized_ricci_flow(curvatures=curvatures, steps=30, dt=0.05)
    assert flow_res["gauss_bonnet_conserved"] is True
    # Normalized Ricci flow contracts curvature variance toward uniform sphere
    assert flow_res["variance_reduction"] > 0.0
    assert len(flow_res["entropy_trajectory"]) == 30
    assert flow_res["verified"] is True


def test_unified_quartet_cognitive_engine():
    engine = UnifiedQuartetCognitiveEngine(seed=42)

    decision = engine.synthesize_quartet_decision(
        observation_16d=[0.85, 0.90] + [0.1] * 14,
        base_policy_action=0,
        intended_action=1,
        hazard_action=4,
        num_oracle_members=5,
        panel_size=3,
    )

    assert decision["project"] == "POINCARE-QUARTET"
    assert "receipt_sha256" in decision
    assert decision["final_dispatched_action"] in (0, 1)
    assert isinstance(decision["override_admitted"], bool)
    assert len(decision["faultline_active_panel"]) == 3
    assert decision["lyapunov_homeostatic_stable"] is True
    assert decision["verified"] is True


def test_poincare_engine_synthesize():
    engine = PoincareEngine(seed=42)
    res = engine.synthesize(
        seed=42,
        stochastic_k=0.85,
        iterations=200,
        curvature_c=1.0,
    )

    assert res["schema"] == SCHEMA_ID
    assert res["project"] == "POINCARE"
    assert res["status"] == "completed"
    assert res["total_domains_verified"] is None
    assert res["diagnostics_checked"] == 5
    assert res["diagnostics_passed"] == 5
    assert "poincare_return_map" in res
    assert "hyperbolic_poincare_disk" in res
    assert "poincare_duality_morse" in res
    assert "ricci_flow_perelman" in res
    assert "unified_quartet_coordinator" in res
    assert len(res["receipt_sha256"]) == 64
    assert res["verified"] is True


@pytest.mark.parametrize("matrix,rank", [([[0, 1]], 1), ([[0, 0, 1], [0, 1, 0]], 2),
                                         ([[1, 2], [2, 4], [3, 6]], 1)])
def test_matrix_rank_handles_rectangular_matrices(matrix, rank):
    assert PoincareDualityMorseEngine.matrix_rank(matrix) == rank


def test_unsupported_surface_cannot_receive_a_duality_certificate():
    with pytest.raises(ValueError):
        PoincareDualityMorseEngine().verify_poincare_duality("garbage")


@pytest.mark.parametrize("u,v", [([0], [0, .5]), ([], []), ([1, 0], [0, 0]),
                                ([2, 0], [-2, 0]), ([float("nan"), 0], [0, 0])])
def test_disk_operations_reject_invalid_points(u, v):
    disk = HyperbolicPoincareDiskEngine()
    for operation in (disk.poincare_distance, disk.mobius_addition):
        with pytest.raises(ValueError):
            operation(u, v)


@pytest.mark.parametrize("curvature", [0, -1, float("nan"), float("inf"), True])
def test_disk_rejects_nonpositive_or_nonfinite_curvature(curvature):
    with pytest.raises(ValueError):
        HyperbolicPoincareDiskEngine(curvature)


def test_disk_radial_distance_matches_closed_form_and_scaled_geometry():
    for curvature in (.25, 1.0, 4.0):
        disk = HyperbolicPoincareDiskEngine(curvature)
        point = [.2 / math.sqrt(curvature), 0]
        assert disk.poincare_distance([0, 0], point) == pytest.approx(
            2 * math.atanh(.2) / math.sqrt(curvature))
        inverse = disk.mobius_addition(point, [-point[0], 0])
        assert inverse == pytest.approx([0, 0])


def test_synthesis_honors_parameters_and_seed_independently_of_object_history():
    reused = PoincareEngine(seed=42)
    result = reused.synthesize(seed=19, iterations=17, curvature_c=4.0, ricci_steps=8)
    assert result["poincare_return_map"]["iterations"] == 17
    assert result["hyperbolic_poincare_disk"]["curvature_c"] == 4.0
    assert len(result["ricci_flow_perelman"]["entropy_trajectory"]) == 8
    assert result == PoincareEngine(seed=19).synthesize(iterations=17, curvature_c=4.0, ricci_steps=8)


def test_synthesis_rejects_unknown_parameters_instead_of_silently_discarding_them():
    with pytest.raises(TypeError):
        PoincareEngine().synthesize(iteratons=17)


@pytest.mark.parametrize("settings", [dict(iterations=0), dict(iterations=-1), dict(iterations=True),
                                     dict(iterations=2.5), dict(p0=float("nan"))])
def test_return_map_rejects_invalid_inputs(settings):
    with pytest.raises(ValueError):
        PoincareReturnMapEngine().compute_lyapunov_spectrum(**settings)


@pytest.mark.parametrize("settings", [dict(curvatures=[]), dict(curvatures=[1, 2]),
                                     dict(curvatures=[float("nan")] * 6), dict(steps=0),
                                     dict(steps=-1), dict(steps=True), dict(dt=0),
                                     dict(dt=float("inf")), dict(dt=8)])
def test_curvature_relaxation_rejects_invalid_inputs(settings):
    with pytest.raises(ValueError):
        RicciFlowPerelmanEngine().run_discrete_ricci_flow(**settings)


def test_curvature_diagnostics_describe_relaxation_and_check_actual_entropy_path():
    result = RicciFlowPerelmanEngine().run_discrete_ricci_flow(steps=30)
    assert result["model"] == "mean_curvature_relaxation"
    assert result["perelman_entropy_verified"] is False
    path = [result["initial_entropy_proxy"]] + result["entropy_trajectory"]
    assert result["perelman_entropy_monotonic"] == all(
        b >= a - 1e-6 for a, b in zip(path, path[1:]))
    assert result["verified"] == (result["gauss_bonnet_conserved"] and result["uniformization_verified"])


@pytest.mark.parametrize("settings", [dict(base_policy_action=-1), dict(base_policy_action=20),
                                     dict(intended_action=True), dict(intended_action=9),
                                     dict(hazard_action=-1), dict(panel_size=0),
                                     dict(num_oracle_members=0)])
def test_quartet_rejects_invalid_actions_and_panel_inputs(settings):
    with pytest.raises(ValueError):
        UnifiedQuartetCognitiveEngine().synthesize_quartet_decision([.8] * 16, **settings)


@pytest.mark.parametrize("observation", [[], [.8] * 15, [.8] * 15 + [float("inf")]])
def test_quartet_requires_a_finite_16d_observation(observation):
    with pytest.raises(ValueError):
        UnifiedQuartetCognitiveEngine().synthesize_quartet_decision(observation)


def test_failed_diagnostics_cannot_report_all_checks_passed():
    result = PoincareEngine().synthesize(observation_16d=[0.0] * 16)
    assert result["verified"] is False
    assert result["diagnostics_passed"] < result["diagnostics_checked"]
    assert result["total_domains_verified"] is None


def test_disk_distance_is_accurate_near_origin_and_ball_boundary():
    disk = HyperbolicPoincareDiskEngine()
    assert disk.poincare_distance([0, 0], [1e-10, 0]) == pytest.approx(2e-10, rel=1e-12, abs=0)
    near_edge = 1 - 1e-9
    assert disk.poincare_distance([near_edge, 0], [-near_edge, 0]) == pytest.approx(
        4 * math.atanh(near_edge), rel=1e-9)
    assert disk.mobius_addition([near_edge, 0], [-near_edge, 0]) == pytest.approx([0, 0])


@pytest.mark.parametrize("matrix", [[[1], [1, 2]], [[float("nan")]], [[float("inf")]]])
def test_matrix_rank_rejects_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        PoincareDualityMorseEngine.matrix_rank(matrix)


@pytest.mark.parametrize("settings", [dict(epsilon=0), dict(epsilon=float("nan")),
                                     dict(max_steps=0), dict(max_steps=True), dict(theta0=float("inf"))])
def test_recurrence_rejects_invalid_parameters(settings):
    with pytest.raises(ValueError):
        PoincareReturnMapEngine().compute_poincare_recurrence(**settings)


def test_recurrence_uses_toroidal_initial_coordinates():
    first = PoincareReturnMapEngine().compute_poincare_recurrence(p0=.5, theta0=.5, max_steps=50)
    translated = PoincareReturnMapEngine().compute_poincare_recurrence(
        p0=.5 + 8 * math.pi, theta0=.5 - 12 * math.pi, max_steps=50)
    assert first == translated


@pytest.mark.parametrize("seed", [True, 1.5, float("nan")])
def test_synthesis_rejects_invalid_seed_without_partial_computation(seed):
    with pytest.raises(ValueError):
        PoincareEngine().synthesize(seed=seed)


@pytest.mark.parametrize("curvature", [1e200, 1e-300])
def test_disk_operations_preserve_curvature_scaling_for_extreme_finite_units(curvature):
    disk = HyperbolicPoincareDiskEngine(curvature)
    scale = math.sqrt(curvature)
    point = [.2 / scale, .1 / scale]
    assert disk.mobius_addition(point, [0, 0]) == pytest.approx(point, abs=0, rel=1e-12)
    assert disk.poincare_distance([0, 0], point) * scale == pytest.approx(
        2 * math.atanh(math.sqrt(.05)))


def test_uniform_curvature_is_a_verified_fixed_point():
    result = RicciFlowPerelmanEngine().run_discrete_ricci_flow(curvatures=[2 * math.pi / 3] * 6)
    assert result["final_curvature_variance"] == pytest.approx(0)
    assert result["gauss_bonnet_conserved"] is True
    assert result["uniformization_verified"] is True
    assert result["verified"] is True


@pytest.mark.parametrize("engine", [PoincareReturnMapEngine, RicciFlowPerelmanEngine])
def test_component_seed_rejects_silent_boolean_and_fractional_coercion(engine):
    for seed in (True, 1.5, float("nan")):
        with pytest.raises(ValueError):
            engine(seed=seed)


def test_standard_map_rejects_unrepresentable_finite_parameter():
    with pytest.raises(ValueError):
        PoincareReturnMapEngine(stochastic_parameter_k=1e308).step(theta=math.pi / 2, p=1e308)


@pytest.mark.parametrize("curvatures", [[1e308] * 6, [1e200, -1e200] * 3])
def test_curvature_relaxation_rejects_unrepresentable_finite_moments(curvatures):
    with pytest.raises(ValueError):
        RicciFlowPerelmanEngine().run_discrete_ricci_flow(curvatures=curvatures, steps=1)


@pytest.mark.parametrize("matrix", [[[], [1]], [[], [float("nan")]]])
def test_matrix_rank_validates_later_rows_before_zero_column_shortcut(matrix):
    with pytest.raises(ValueError):
        PoincareDualityMorseEngine.matrix_rank(matrix)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf"), True])
def test_matrix_rank_requires_positive_finite_tolerance(tolerance):
    with pytest.raises(ValueError):
        PoincareDualityMorseEngine.matrix_rank([[0]], tol=tolerance)


@pytest.mark.parametrize("matrix", [[], [[]], [[], []]])
def test_matrix_rank_preserves_rectangular_zero_column_complexes(matrix):
    assert PoincareDualityMorseEngine.matrix_rank(matrix) == 0
