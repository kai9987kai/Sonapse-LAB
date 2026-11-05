"""Tests for Archimedes Hydrodynamic & Buoyancy Engine."""
from poseidon.archimedes import ArchimedesHydrodynamicEngine


def test_archimedes_density_and_buoyancy():
    engine = ArchimedesHydrodynamicEngine()
    engine.reset()

    # Fresh water (salinity 0) vs saline ocean (salinity 1)
    fresh_density = engine.compute_fluid_density(salinity=0.0, temperature=0.5)
    ocean_density = engine.compute_fluid_density(salinity=1.0, temperature=0.5)
    assert ocean_density > fresh_density

    # Buoyancy with deflated vs inflated vesicle
    f_b_deflated, a_deflated = engine.compute_buoyancy(ocean_density, vesicle_inflation=0.0)
    f_b_inflated, a_inflated = engine.compute_buoyancy(ocean_density, vesicle_inflation=1.0)
    assert f_b_inflated > f_b_deflated
    assert a_inflated > a_deflated


def test_archimedes_tidal_surge_and_cost_modulation():
    engine = ArchimedesHydrodynamicEngine(tide_period_steps=48, max_surge_velocity=1.0)
    engine.reset()

    # Surge at step 0 vs step 12
    u_x_0, u_y_0 = engine.compute_tidal_surge(0)
    u_x_12, u_y_12 = engine.compute_tidal_surge(12)
    assert abs(u_x_0) < 1e-4  # sin(0) = 0
    assert abs(u_x_12 - 1.0) < 1e-4  # sin(pi/2) = 1.0

    # Moving with current vs against current
    # Current moving East (angle 0)
    surge = (1.0, 0.0)
    cost_with = engine.modulate_locomotion_cost(action=4, heading_angle=0.0, surge_vector=surge)
    cost_against = engine.modulate_locomotion_cost(action=4, heading_angle=3.14159, surge_vector=surge)
    assert cost_with < 1.0  # discount
    assert cost_against > 1.0  # penalty


def test_archimedes_step_telemetry():
    engine = ArchimedesHydrodynamicEngine()
    engine.reset()
    dummy_obs = [0.8, 0.7, 0.6, 0.5, 0.1, 0.0, 0.5, 0.5, 0.5, 0.7, 0.4, 0.5, 0.0, 0.5, 0.0, 0.1]
    info = engine.step(dummy_obs, action=1, heading_angle=0.0)

    assert "fluid_density" in info
    assert "buoyancy_force" in info
    assert 0.0 <= info["current_depth"] <= 1.0
    assert 0.0 <= info["vesicle_inflation"] <= 1.0
    assert info["stamina_cost_multiplier"] > 0.0
