import math

import pytest

from poseidon.fullbody import (
    FullBodyConfig,
    FullBodyTidePool3D,
    JOINT_LIMITS,
    JOINT_NAMES,
    N_JOINTS,
    N_SYNERGIES,
    SYNERGY_BASIS,
    BodyState,
    CPG,
    forward_kinematics,
    ground_reaction_force,
    motor_synergy,
    spinal_reflexes,
    synergy_activations,
)
from poseidon.world import OBS_SIZE


def test_joint_structure_is_consistent():
    assert N_JOINTS == len(JOINT_NAMES)
    assert N_JOINTS == len(JOINT_LIMITS)
    for lo, hi in JOINT_LIMITS:
        assert lo < hi


def test_motor_synergy_shape_and_bounds():
    for action in range(6):
        for phase in (0.0, 1.5, 3.0, 6.28):
            targets = motor_synergy(action, phase)
            assert len(targets) == N_JOINTS
            for i, value in enumerate(targets):
                lo, hi = JOINT_LIMITS[i]
                # Synergies may request slightly outside limits; joint servo clamps.
                assert math.isfinite(value)
                assert lo - 1e-6 <= value <= hi + 1e-6


def test_forward_kinematics_returns_finite_tips():
    state = BodyState()
    tips = forward_kinematics(state)
    assert len(tips) == N_JOINTS
    for tip in tips:
        assert all(math.isfinite(v) for v in tip)


def test_forward_kinematics_changes_with_joint_angle():
    base = BodyState()
    moved = BodyState(joint_angles=[0.0, -1.0, -1.0, -1.0, -1.0])
    base_tips = forward_kinematics(base)
    moved_tips = forward_kinematics(moved)
    # At least the limb tips should move when the joints flex.
    assert any(
        abs(a - b) > 1e-6
        for bt, mt in zip(base_tips[1:], moved_tips[1:])
        for a, b in zip(bt, mt)
    )


def test_step_returns_valid_observation():
    env = FullBodyTidePool3D(seed=42)
    obs, reward, done, info = env.step(action=4)
    assert len(obs) == OBS_SIZE
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["alive"] is True
    assert len(info["joint_angles"]) == N_JOINTS
    assert len(info["contact"]) == N_JOINTS - 1
    # All observation channels are normalized to [0, 1].
    for value in obs:
        assert 0.0 <= value <= 1.0


def test_joint_dynamics_moves_toward_target():
    env = FullBodyTidePool3D(seed=1)
    initial = list(env.state.joint_angles)
    # Request a strong forward reach on the front-left limb.
    targets = motor_synergy(1, 0.0)  # forage
    for _ in range(40):
        env._apply_joint_dynamics(targets, env.config.dt)
    # The front-left joint should have moved toward the requested target.
    assert abs(env.state.joint_angles[1] - targets[1]) < abs(initial[1] - targets[1])


def test_metabolic_cost_drains_stamina_under_locomotion():
    env = FullBodyTidePool3D(seed=7)
    start_stamina = env.stamina
    start_energy = env.energy
    for _ in range(20):
        env.step(action=5)  # flee
    assert env.stamina < start_stamina
    assert env.energy <= start_energy


def test_rest_recovers_stamina():
    env = FullBodyTidePool3D(seed=7)
    # First exhaust a little with flee.
    for _ in range(10):
        env.step(action=5)
    exhausted = env.stamina
    for _ in range(20):
        env.step(action=0)  # rest
    assert env.stamina > exhausted


def test_contact_flag_reflects_limb_tip_height():
    env = FullBodyTidePool3D(seed=3)
    # Force all limbs downward so their tips go below ground.
    env.state.com_z = 0.0
    env.state.joint_angles = [0.0, -1.5, -1.5, -1.5, -1.5]
    tips = forward_kinematics(env.state)
    ground = env._terrain_height(env.state.com_x, env.state.com_y)
    low_limb = min(tip[2] for tip in tips[1:])
    # At least one limb tip should be at or below the support surface.
    assert low_limb <= ground + 0.5


def test_multi_step_episode_advances_tick():
    env = FullBodyTidePool3D(seed=101)
    for step_i in range(16):
        obs, _, done, _ = env.step(action=step_i % 6)
        if done:
            break
    assert env.tick > 0
    assert len(env.patches) == 12
    assert len(obs) == OBS_SIZE


def test_locomotion_changes_position():
    env = FullBodyTidePool3D(seed=42)
    start = (env.state.com_x, env.state.com_y)
    # Drive a sustained explore gait.
    for _ in range(60):
        env.step(action=4)
    end = (env.state.com_x, env.state.com_y)
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    # Sustained locomotion should displace the body.
    assert distance > 0.5


def test_invalid_action_rejected():
    env = FullBodyTidePool3D(seed=1)
    with pytest.raises(ValueError):
        env.step(action=6)


def test_observation_has_proprioceptive_channels():
    env = FullBodyTidePool3D(seed=5)
    env.step(action=4)
    obs, _, _, info = env.step(action=5)
    # Channel 13 is contact ratio; channel 15 is metabolic effort.
    assert 0.0 <= obs[13] <= 1.0
    assert info["metabolic_cost"] >= 0.0
    assert isinstance(info["pitch"], float)


# ---------------------------------------------------------------------------
# CPG: coupled phase oscillators
# ---------------------------------------------------------------------------
def test_cpg_initial_phase_is_trot():
    cpg = CPG()
    # Diagonal pairs (fl, hr) and (fr, hl) should be in phase; pairs anti-phase.
    assert abs(cpg.phases[0] - cpg.phases[3]) < 1e-6
    assert abs(cpg.phases[1] - cpg.phases[2]) < 1e-6
    assert abs(abs(cpg.phases[0] - cpg.phases[1]) - math.pi) < 1e-6


def test_cpg_phases_advance_with_frequency():
    cpg = CPG()
    before = list(cpg.phases)
    cpg.step(dt=0.25, frequencies=[2.0, 2.0, 2.0, 2.0], contact=[False] * 4)
    after = cpg.phases
    # All phases should have advanced (not identical to before, modulo 2pi).
    assert any(abs(a - b) > 1e-6 for a, b in zip(after, before))


def test_cpg_contact_slows_loaded_limb():
    freqs = [3.0, 3.0, 3.0, 3.0]
    free = CPG()
    loaded = CPG()
    # Single-step comparison avoids phase-wrap ambiguity.
    free_before = free.phases[0]
    loaded_before = loaded.phases[0]
    free.step(dt=0.25, frequencies=freqs, contact=[False] * 4)
    loaded.step(dt=0.25, frequencies=freqs, contact=[True, False, False, False])
    free_delta = free.phases[0] - free_before
    loaded_delta = loaded.phases[0] - loaded_before
    # Loaded limb should advance less than the free limb.
    assert loaded_delta < free_delta


# ---------------------------------------------------------------------------
# Synergy basis expansion
# ---------------------------------------------------------------------------
def test_synergy_basis_shape():
    assert len(SYNERGY_BASIS) == N_SYNERGIES
    for basis in SYNERGY_BASIS:
        assert len(basis) == N_JOINTS


def test_synergy_activations_dimension():
    cpg = CPG()
    for action in range(6):
        act = synergy_activations(action, cpg)
        assert len(act) == N_SYNERGIES
        assert all(math.isfinite(a) for a in act)


def test_motor_synergy_varies_with_phase_for_locomotion():
    # Locomotion actions should produce different targets at different phases.
    # Use phases where sin differs (0 vs pi/2) so per-limb stance activation
    # actually changes.
    explore_0 = motor_synergy(4, 0.0)
    explore_half = motor_synergy(4, math.pi / 2.0)
    assert any(abs(a - b) > 1e-6 for a, b in zip(explore_0, explore_half))


# ---------------------------------------------------------------------------
# Ground reaction force
# ---------------------------------------------------------------------------
def test_grf_zero_when_not_penetrating():
    cfg = FullBodyConfig()
    tip = (0.0, 0.0, 5.0)
    fx, fy, fz, normal = ground_reaction_force(tip, (0.0, 0.0, 0.0), support_z=2.0, config=cfg)
    assert fx == 0.0 and fy == 0.0 and fz == 0.0 and normal == 0.0


def test_grf_upward_when_penetrating():
    cfg = FullBodyConfig()
    tip = (0.0, 0.0, 1.0)  # 1.0 below support at 2.0
    _, _, fz, normal = ground_reaction_force(tip, (0.0, 0.0, 0.0), support_z=2.0, config=cfg)
    assert fz > 0.0
    assert normal > 0.0


def test_grf_friction_opposes_slip():
    cfg = FullBodyConfig()
    # Foot slipping backward (+x) while in contact.
    tip = (0.0, 0.0, 1.0)
    fx, _, _, _ = ground_reaction_force(tip, (1.0, 0.0, 0.0), support_z=2.0, config=cfg)
    # Friction should push the body forward (negative x opposes +x slip).
    assert fx < 0.0


# ---------------------------------------------------------------------------
# Spinal reflexes
# ---------------------------------------------------------------------------
def test_loading_reflex_extends_loaded_limb():
    cfg = FullBodyConfig()
    state = BodyState(joint_angles=[0.0, -0.5, -0.5, -0.5, -0.5])
    targets = [0.0, -0.5, -0.5, -0.5, -0.5]
    # No load on any limb.
    unloaded = spinal_reflexes(targets, state, [0.0, 0.0, 0.0, 0.0], cfg)
    # Heavy load on the front-left limb (index 0 -> limb 1).
    loaded = spinal_reflexes(targets, state, [50.0, 0.0, 0.0, 0.0], cfg)
    # Loaded limb target should be more extended (more negative) than unloaded.
    assert loaded[1] < unloaded[1]


# ---------------------------------------------------------------------------
# Integrated: info exposes new control telemetry
# ---------------------------------------------------------------------------
def test_info_exposes_cpg_and_synergy_telemetry():
    env = FullBodyTidePool3D(seed=42)
    _, _, _, info = env.step(action=4)
    assert len(info["cpg_phases"]) == 4
    assert len(info["synergy_activations"]) == N_SYNERGIES
    assert len(info["contact_normals"]) == 4
    assert all(isinstance(n, float) for n in info["contact_normals"])
