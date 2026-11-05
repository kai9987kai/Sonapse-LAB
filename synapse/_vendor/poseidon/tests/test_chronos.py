"""Unit tests for Project CHRONOS Stage 25 theoretical engines."""
from poseidon.chronos import (
    CausalDynamicalTriangulationsEngine,
    NoncommutativeSpectralGeometryEngine,
    TropicalAlgebraicGeometryEngine,
    StiefelGrassmannianOptimizationEngine,
    CoalgebraicBisimulationEngine,
    ChronosEngine,
)


def test_causal_dynamical_triangulation():
    engine = CausalDynamicalTriangulationsEngine(time_slices=4, spatial_nodes_per_slice=4)
    res = engine.evaluate_triangulation()
    assert res["verified"] is True
    assert res["euler_characteristic"] == 0
    assert res["total_vertices_n0"] == 16
    assert res["total_simplices_n2"] == 32
    assert len(res["volume_profile"]) == 4
    assert res["deficit_angle_curvature"] == 0.0


def test_noncommutative_spectral_geometry():
    engine = NoncommutativeSpectralGeometryEngine()
    res = engine.evaluate_spectral_triple(node_count=3, dirac_couplings=[1.0, 2.0])
    assert res["verified"] is True
    assert res["trace_dirac"] == 0.0
    assert res["spectral_action_tr_d2"] > 0
    assert "d(p0,p1)" in res["connes_spectral_distances"]
    assert res["triangle_inequality_preserved"] is True


def test_tropical_algebraic_geometry():
    engine = TropicalAlgebraicGeometryEngine()
    res = engine.evaluate_tropical_polynomial(monomials=[(0.0, 2.0), (3.0, 1.0), (5.0, 0.0)], eval_point=2.0)
    assert res["verified"] is True
    assert res["tropical_value"] == 5.0  # max(0+4, 3+2, 5+0) = 5
    assert len(res["corner_locus_roots"]) > 0
    assert len(res["tropical_linear_solution"]) == 2


def test_stiefel_grassmannian_optimization():
    engine = StiefelGrassmannianOptimizationEngine()
    res = engine.optimize_rayleigh_stiefel(n=3, p=2, iterations=5, step_size=0.1)
    assert res["verified"] is True
    assert res["orthogonality_residual"] < 1e-9
    assert res["optimized_objective"] >= res["initial_objective"] - 1e-9


def test_coalgebraic_bisimulation():
    engine = CoalgebraicBisimulationEngine()
    res = engine.evaluate_behavioral_pseudometric(discount=0.8, iterations=10)
    assert res["verified"] is True
    assert res["dist_s0_s1"] < res["dist_s0_s2"]
    assert res["triangle_inequality_preserved"] is True
    assert res["behavioral_hierarchy_valid"] is True


def test_chronos_engine_synthesis():
    engine = ChronosEngine()
    synth = engine.synthesize(seed=42)
    assert synth["project"] == "CHRONOS"
    assert synth["stage"] == 25
    assert synth["version"] == "v3.6.0-CHRONOS-65"
    assert synth["verified"] is True
    assert len(synth["receipt_sha256"]) == 64
    assert "Project CHRONOS Stage 25 Synthesis Certified" in synth["text"]
