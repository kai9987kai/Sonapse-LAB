"""POSEIDON FULLBODY: Articulated embodied agent for the continuous 3D tidepool.

Advances the point-mass agent of :mod:`poseidon.genesis_3d` into a full
articulated body: a torso, a head and four limbs connected by revolute joints.
Control follows a three-layer bio-inspired hierarchy grounded in recent
neuromechanical and embodied-AI work:

1. **Synergy basis (latent action space)**. Each macro-action selects a small
   vector of *synergy activations* that is expanded through a fixed motor
   synergy basis into joint targets. This realizes the *synergy expansion
   hypothesis* (PNAS 2025): the high-DOF body is controlled through a
   low-dimensional coordinative structure rather than raw joint commands.

2. **Coupled central pattern generators (CPGs)**. Four limb phase oscillators,
   weakly coupled to produce coordinated gaits (trot by default), replace a
   single global phase. Contact/loading feedback modulates each oscillator's
   phase, matching the two-layer rhythm-generator / pattern-formation CPG
   model used in vertebrate locomotion (Refai et al., 2024; eLife 2025).

3. **Spinal reflexes + ground reaction forces**. Ia stretch reflex (muscle
   spindle length/velocity) and Ib loading reflex (Golgi tendon organ force)
   correct the synergy targets. Foot contact generates spring-damper normal
   forces and Coulomb friction, so propulsion emerges from stance-limb push-off
   rather than an applied thrust.

Movement has a biomechanical metabolic cost (muscle activation + mechanical
power) that drains stamina and energy, giving the cognitive system a real
embodied budget. Deterministic, CPU-only toy model; not a validated physical or
biological simulator.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .world import OBS_SIZE

# Joint order: head pitch, front-left, front-right, hind-left, hind-right.
JOINT_NAMES: tuple[str, ...] = ("head", "fl", "fr", "hl", "hr")
N_JOINTS: int = len(JOINT_NAMES)
N_LIMBS: int = N_JOINTS - 1  # head is not a limb

# Per-joint mechanical limits (radians) and inertia (arbitrary scaled units).
JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),    # head pitch
    (-2.2, 0.4),    # front-left limb: forward reach to backward fold
    (-2.2, 0.4),    # front-right limb
    (-2.2, 0.4),    # hind-left limb
    (-2.2, 0.4),    # hind-right limb
)
JOINT_INERTIA: tuple[float, ...] = (0.6, 1.0, 1.0, 1.0, 1.0)

# Segment geometry (lengths in world units).
TORSO_LENGTH: float = 2.4
TORSO_WIDTH: float = 1.4
HEAD_LENGTH: float = 0.9
LIMB_LENGTH: float = 2.0

# PD gains for the joint servo.
JOINT_KP: float = 22.0
JOINT_KD: float = 3.2

# ---------------------------------------------------------------------------
# Motor synergy basis
# ---------------------------------------------------------------------------
# A small set of fixed coordinative patterns (columns of the basis). Macro
# actions select low-dimensional activation weights; their weighted combination
# produces the target joint posture. This is the synergy-expansion hypothesis:
# control lives in a low-D latent space, not in raw joint space.
# Four synergies: 0=stance_extend, 1=swing_retract, 2=reach_down, 3=compact.
N_SYNERGIES: int = 4
SYNERGY_BASIS: tuple[tuple[float, ...], ...] = (
    # stance_extend: all limbs extended downward for weight support
    (0.0, -1.4, -1.4, -1.4, -1.4),
    # swing_retract: limbs fold back/up to clear the ground
    (0.0, 0.3, 0.3, 0.3, 0.3),
    # reach_down: head and front limbs reach toward the ground (forage/drink)
    (0.9, -1.7, -1.7, -0.4, -0.4),
    # compact: fully tucked posture for shelter/rest
    (-0.4, 0.2, 0.2, 0.2, 0.2),
)


def _synergy_expand(activations: Sequence[float]) -> list[float]:
    """Expand low-D synergy activations into N_JOINTS target angles."""
    targets = [0.0] * N_JOINTS
    for s in range(N_SYNERGIES):
        w = float(activations[s])
        basis = SYNERGY_BASIS[s]
        for j in range(N_JOINTS):
            targets[j] += w * basis[j]
    return [_clamp_joint(j, targets[j]) for j in range(N_JOINTS)]


def joint_targets(action: int, cpg: CPG) -> list[float]:
    """Compute per-joint target angles for a macro-action given the CPG state.

    Static actions expand the synergy basis uniformly. Locomotion actions
    blend the stance_extend and swing_retract synergies *per limb* by that
    limb's own CPG stance activation, so diagonal pairs alternate and the
    gait is not washed out by a global average.
    """
    if action in (4, 5):
        targets = [0.0] * N_JOINTS
        reach = 0.3 if action == 5 else 0.1
        # Head leans forward during locomotion.
        targets[0] = reach * SYNERGY_BASIS[2][0]
        stance_basis = SYNERGY_BASIS[0]
        retract_basis = SYNERGY_BASIS[1]
        for limb in range(N_LIMBS):
            stance = cpg.activation(limb)
            j = limb + 1
            targets[j] = stance * stance_basis[j] + (1.0 - stance) * retract_basis[j]
            if action == 5:
                targets[j] += reach * SYNERGY_BASIS[2][j]
        return [_clamp_joint(j, targets[j]) for j in range(N_JOINTS)]
    return _synergy_expand(synergy_activations(action, cpg))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FullBodyConfig:
    world_size: float = 100.0
    gravity: float = 9.81
    fluid_density: float = 1.0
    air_density: float = 0.02
    tide_period: float = 64.0
    tide_amplitude: float = 3.0
    base_water_level: float = 2.0
    drag_coefficient: float = 0.85
    dt: float = 0.25
    # CPG: intrinsic oscillator frequencies (rad/s) per macro-action.
    cpg_freq_rest: float = 0.4
    cpg_freq_explore: float = 3.0
    cpg_freq_flee: float = 5.0
    cpg_coupling: float = 1.2
    # Reflex gains.
    stretch_gain: float = 6.0          # Ia: muscle spindle length/velocity
    loading_gain: float = 5.0          # Ib: Golgi tendon organ force
    # Ground reaction force (spring-damper foot + Coulomb friction).
    # Tuned so peak ground reaction is ~1-2x body weight for stable walking,
    # not an unrealistic launch impulse.
    grf_stiffness: float = 25.0
    grf_damping: float = 5.0
    foot_friction: float = 0.9
    grf_force_cap: float = 18.0
    # Metabolic cost: activation (|torque|) + mechanical power (|torque*vel|).
    metabolic_activation: float = 0.005
    metabolic_power: float = 0.014
    # Locomotion coupling gain retained for propulsive friction scaling.
    contact_gain: float = 1.0
    metabolic_scale: float = 0.018


@dataclass
class BodyState:
    """Full articulated state of the embodied agent."""

    com_x: float = 50.0
    com_y: float = 50.0
    com_z: float = 3.0
    heading: float = 0.0     # yaw around world z
    pitch: float = 0.0       # body pitch
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    yaw_rate: float = 0.0
    joint_angles: list[float] = field(default_factory=lambda: [0.0] * N_JOINTS)
    joint_velocities: list[float] = field(default_factory=lambda: [0.0] * N_JOINTS)
    contact: list[bool] = field(default_factory=lambda: [False] * N_LIMBS)


def _clamp_joint(index: int, value: float) -> float:
    lo, hi = JOINT_LIMITS[index]
    return max(lo, min(hi, value))


def _wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


# ---------------------------------------------------------------------------
# CPG: four weakly coupled phase oscillators
# ---------------------------------------------------------------------------
class CPG:
    """Coupled limb phase oscillators with contact-feedback modulation.

    Default phase offsets encode a trot (diagonal limb pairs in phase):
    fl=0, fr=pi, hl=pi, hr=0. Coupling pulls each oscillator toward its
    neighbors' preferred relative phase. Contact loading pulls the oscillator
    toward the stance phase (slowing it), matching the biological observation
    that limb loading extends stance.
    """

    # Preferred phase offsets (l -> k) for trot coordination.
    _PREFERRED: tuple[tuple[float, ...], ...] = (
        (0.0, math.pi, math.pi, 0.0),   # fl
        (math.pi, 0.0, 0.0, math.pi),   # fr
        (math.pi, 0.0, 0.0, math.pi),   # hl
        (0.0, math.pi, math.pi, 0.0),   # hr
    )

    def __init__(self, coupling: float = 1.2):
        self.phases: list[float] = [0.0, math.pi, math.pi, 0.0]
        self.coupling = coupling

    def step(self, dt: float, frequencies: Sequence[float], contact: Sequence[bool]) -> None:
        """Advance the four oscillators. ``frequencies`` is per-limb rad/s;
        ``contact`` flags slow loaded limbs (stance extension)."""
        for i in range(N_LIMBS):
            dphi = float(frequencies[i])
            # Coupling to all other oscillators.
            for j in range(N_LIMBS):
                if i == j:
                    continue
                phase_diff = self.phases[j] - self.phases[i] - self._PREFERRED[i][j]
                dphi += self.coupling * math.sin(phase_diff) / (N_LIMBS - 1)
            # Contact feedback: loaded limb lingers in stance (phase ~ pi/2).
            if contact[i]:
                dphi *= 0.5
            self.phases[i] = _wrap_angle(self.phases[i] + dphi * dt)

    def activation(self, limb: int) -> float:
        """0..1 stance activation for a limb: high around phase pi/2."""
        return (math.sin(self.phases[limb]) + 1.0) * 0.5


# ---------------------------------------------------------------------------
# Synergy activations (latent action -> low-D weights)
# ---------------------------------------------------------------------------
def synergy_activations(action: int, cpg: CPG) -> list[float]:
    """Low-dimensional synergy activation weights for a macro-action.

    Locomotion actions (explore/flee) modulate the stance/retract synergies by
    each limb's CPG stance activation, so diagonally opposite limbs alternate.
    Static actions use fixed activations.
    """
    w = [0.0] * N_SYNERGIES
    if action == 0:  # rest: mostly compact, slight stance
        return [0.15, 0.0, 0.0, 0.85]
    if action == 1:  # forage
        return [0.4, 0.0, 0.6, 0.0]
    if action == 2:  # drink
        return [0.3, 0.0, 0.7, 0.0]
    if action == 3:  # shelter
        return [0.0, 0.0, 0.0, 1.0]
    if action in (4, 5):  # explore / flee: alternating stance vs retract
        # Use a single shared phase for the two diagonal pairs; per-limb stance
        # activation drives how much each limb contributes to stance/retract.
        stance_fl = cpg.activation(0)
        stance_fr = cpg.activation(1)
        stance_hl = cpg.activation(2)
        stance_hr = cpg.activation(3)
        # Average stance activation sets the global stance synergy weight; the
        # retract synergy is its complement. This is expanded limb-agnostically
        # but gives a net alternating posture because diagonal pairs are
        # out of phase and their contributions partially cancel/alternate.
        stance_avg = (stance_fl + stance_fr + stance_hl + stance_hr) * 0.25
        w[0] = stance_avg          # stance_extend
        w[1] = 1.0 - stance_avg    # swing_retract
        # Flee leans slightly into reach_down for a forward-leaning sprint.
        w[2] = 0.3 if action == 5 else 0.1
        return w
    return [0.15, 0.0, 0.0, 0.85]


def motor_synergy(action: int, phase: float) -> list[float]:
    """Backward-compatible joint targets from a single scalar phase.

    Kept so existing callers/tests obtain a joint target vector directly.
    Internally builds a CPG snapshot at ``phase`` (trot phase offsets) and
    computes per-limb targets via :func:`joint_targets`.
    """
    cpg = CPG()
    cpg.phases = [phase, phase + math.pi, phase + math.pi, phase]
    return joint_targets(action, cpg)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def forward_kinematics(state: BodyState) -> list[tuple[float, float, float]]:
    """World positions of segment tips: head, fl, fr, hl, hr."""
    hx = math.cos(state.heading)
    hy = math.sin(state.heading)
    rx = -hy
    ry = hx

    half = TORSO_LENGTH * 0.5
    head_base = (state.com_x + hx * half, state.com_y + hy * half, state.com_z)
    head_tip = (
        head_base[0] + hx * math.cos(state.joint_angles[0]) * HEAD_LENGTH,
        head_base[1] + hy * math.cos(state.joint_angles[0]) * HEAD_LENGTH,
        head_base[2] + math.sin(state.joint_angles[0]) * HEAD_LENGTH,
    )

    w = TORSO_WIDTH * 0.5
    limb_anchors = (
        (state.com_x + hx * half + rx * w, state.com_y + hy * half + ry * w),
        (state.com_x + hx * half - rx * w, state.com_y + hy * half - ry * w),
        (state.com_x - hx * half + rx * w, state.com_y - hy * half + ry * w),
        (state.com_x - hx * half - rx * w, state.com_y - hy * half - ry * w),
    )
    tips = [head_tip]
    for i, (ax, ay) in enumerate(limb_anchors, start=1):
        ang = state.joint_angles[i]
        forward_proj = math.cos(ang)
        vertical = math.sin(ang)
        tips.append((
            ax + hx * forward_proj * LIMB_LENGTH,
            ay + hy * forward_proj * LIMB_LENGTH,
            state.com_z + vertical * LIMB_LENGTH,
        ))
    return tips


def _tip_velocity(
    state: BodyState, prev_state: BodyState, index: int, dt: float
) -> tuple[float, float, float]:
    if dt <= 0.0:
        return (0.0, 0.0, 0.0)
    tip = forward_kinematics(state)[index]
    prev = forward_kinematics(prev_state)[index]
    return ((tip[0] - prev[0]) / dt, (tip[1] - prev[1]) / dt, (tip[2] - prev[2]) / dt)


# ---------------------------------------------------------------------------
# Ground reaction force (spring-damper foot + Coulomb friction)
# ---------------------------------------------------------------------------
def ground_reaction_force(
    tip_pos: tuple[float, float, float],
    tip_vel: tuple[float, float, float],
    support_z: float,
    config: FullBodyConfig,
) -> tuple[float, float, float, float]:
    """Return (fx, fy, fz, normal_force) on the body from a foot contact.

    Normal force is a spring-damper resisting penetration below the support
    surface. Tangential friction opposes the foot's slip velocity, converting
    stance-limb backward motion into forward propulsion.
    """
    penetration = support_z - tip_pos[2]
    if penetration <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    # Cap effective penetration so a deeply swung limb cannot produce an
    # unrealistically large ground reaction impulse.
    penetration = min(penetration, 1.0)
    normal = config.grf_stiffness * penetration - config.grf_damping * tip_vel[2]
    normal = max(0.0, min(normal, config.grf_force_cap))
    # Tangential slip velocity.
    slip = math.hypot(tip_vel[0], tip_vel[1])
    fx = -config.foot_friction * normal * (tip_vel[0] / (slip + 1e-6))
    fy = -config.foot_friction * normal * (tip_vel[1] / (slip + 1e-6))
    fz = normal
    return (fx, fy, fz, normal)


# ---------------------------------------------------------------------------
# Spinal reflexes
# ---------------------------------------------------------------------------
def spinal_reflexes(
    targets: Sequence[float],
    state: BodyState,
    contact_normals: Sequence[float],
    config: FullBodyConfig,
) -> list[float]:
    """Apply Ia stretch and Ib loading reflexes to the synergy targets.

    - Ia stretch reflex: if a joint is being pulled away from its target
      (joint velocity opposes the target error), add restoring torque bias.
    - Ib loading reflex: a loaded limb (high contact normal force) extends
      further to support weight and push off.
    Returns the corrected target angles.
    """
    corrected = list(targets)
    for i in range(1, N_JOINTS):
        error = targets[i] - state.joint_angles[i]
        vel = state.joint_velocities[i]
        # Stretch reflex: assist motion toward target when moving toward it.
        stretch = config.stretch_gain * error * 0.05
        corrected[i] += max(0.0, stretch if vel * error >= 0 else 0.0)
        # Loading reflex: loaded stance limb extends (more negative angle).
        normal = contact_normals[i - 1]
        if normal > 0.0:
            corrected[i] -= config.loading_gain * 0.02 * (normal / 50.0)
        corrected[i] = _clamp_joint(i, corrected[i])
    return corrected


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class FullBodyTidePool3D:
    """Continuous 3D tidepool inhabited by a full articulated embodied agent.

    Exposes the same ``step(action) -> (obs, reward, done, info)`` contract as
    :class:`poseidon.genesis_3d.ContinuousTidePool3D` with a 16-D observation,
    so it can stand in for cognitive experiments. Locomotion arises from CPG +
    reflex + ground-reaction dynamics; vitals drain from biomechanical effort.
    """

    def __init__(self, config: FullBodyConfig | None = None, seed: int = 42):
        self.config = config or FullBodyConfig()
        self.rng = random.Random(seed)
        self.time: float = 0.0
        self.tick: int = 0
        self.cpg = CPG(coupling=self.config.cpg_coupling)
        self.state = BodyState()
        # Start the body resting just above the terrain so the first contact
        # is not an unrealistic spring impulse.
        self.state.com_z = self._terrain_height(self.state.com_x, self.state.com_y) + 0.2
        # Previous-step contact normal forces for one-step-delayed reflexes.
        self._last_contact_normals: list[float] = [0.0] * N_LIMBS
        self.health: float = 1.0
        self.energy: float = 0.85
        self.hydration: float = 0.85
        self.stamina: float = 0.90
        self.oxygen: float = 1.0
        self.patches: list[tuple[float, float, float, float, float, float]] = []
        self._init_patches()

    def _init_patches(self, count: int = 12) -> None:
        self.patches = []
        for _ in range(count):
            x = self.rng.uniform(10.0, self.config.world_size - 10.0)
            y = self.rng.uniform(10.0, self.config.world_size - 10.0)
            z = self._terrain_height(x, y)
            self.patches.append((
                x, y, z,
                self.rng.uniform(4.0, 8.0),
                self.rng.uniform(0.3, 0.9),
                self.rng.uniform(0.4, 0.9),
            ))

    def _terrain_height(self, x: float, y: float) -> float:
        w1 = math.sin(x * 0.05) * math.cos(y * 0.05) * 3.0
        w2 = math.sin(x * 0.12 + 1.0) * math.cos(y * 0.12 + 0.5) * 1.5
        return max(0.0, w1 + w2 + 1.0)

    def _tide_height(self) -> float:
        phase = (2.0 * math.pi * self.time) / self.config.tide_period
        return self.config.base_water_level + self.config.tide_amplitude * math.sin(phase)

    def _nearest_patch(self) -> tuple[float, float, float, float, float, float]:
        px, py = self.state.com_x, self.state.com_y
        return min(self.patches, key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)

    def _cpg_frequencies(self, action: int) -> list[float]:
        if action == 5:
            base = self.config.cpg_freq_flee
        elif action == 4:
            base = self.config.cpg_freq_explore
        else:
            base = self.config.cpg_freq_rest
        return [base] * N_LIMBS

    def _apply_joint_dynamics(self, targets: Sequence[float], dt: float) -> float:
        """Servo joints toward targets with PD control.

        Returns total metabolic cost = activation(|torque|) + power(|torque*vel|).
        """
        cost = 0.0
        for i in range(N_JOINTS):
            target = _clamp_joint(i, float(targets[i]))
            angle = self.state.joint_angles[i]
            vel = self.state.joint_velocities[i]
            torque = JOINT_KP * (target - angle) - JOINT_KD * vel
            accel = torque / JOINT_INERTIA[i]
            new_vel = vel + accel * dt
            new_angle = _clamp_joint(i, angle + new_vel * dt)
            if new_angle == angle and (new_vel * (target - angle)) > 0:
                new_vel = 0.0
            self.state.joint_angles[i] = new_angle
            self.state.joint_velocities[i] = new_vel
            cost += (self.config.metabolic_activation * abs(torque)
                     + self.config.metabolic_power * abs(torque * new_vel))
        return cost

    def _contact_forces(
        self, prev_state: BodyState, dt: float, support_z: float
    ) -> tuple[float, float, float, float, list[float]]:
        """Sum ground-reaction forces over contacting limbs.

        Returns (fx, fy, fz, yaw_torque, per-limb normal forces).
        """
        tips = forward_kinematics(self.state)
        hx = math.cos(self.state.heading)
        hy = math.sin(self.state.heading)
        fx = fy = fz = 0.0
        yaw = 0.0
        normals = [0.0] * N_LIMBS
        for i in range(1, N_JOINTS):
            tip = tips[i]
            contacting = tip[2] <= support_z + 0.05
            self.state.contact[i - 1] = contacting
            if not contacting:
                continue
            vx, vy, vz = _tip_velocity(self.state, prev_state, i, dt)
            lfx, lfy, lfz, normal = ground_reaction_force(tip, (vx, vy, vz), support_z, self.config)
            fx += lfx
            fy += lfy
            fz += lfz
            normals[i - 1] = normal
            # Lateral offset of the foot from COM creates a yaw moment.
            side = 1.0 if i in (1, 3) else -1.0
            yaw += side * (-lfx) * 0.04
        return fx, fy, fz, yaw, normals

    def step(self, action: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        if not 0 <= action <= 5:
            raise ValueError("action must be in [0, 5]")
        cfg = self.config
        dt = cfg.dt
        self.time += dt
        self.tick += 1

        prev_state = BodyState(
            com_x=self.state.com_x, com_y=self.state.com_y, com_z=self.state.com_z,
            heading=self.state.heading, pitch=self.state.pitch,
            vel_x=self.state.vel_x, vel_y=self.state.vel_y, vel_z=self.state.vel_z,
            joint_angles=list(self.state.joint_angles),
            joint_velocities=list(self.state.joint_velocities),
        )

        # 1. CPG step (uses previous contact state for feedback).
        self.cpg.step(dt, self._cpg_frequencies(action), prev_contact := self.state.contact)

        # 2. Synergy expansion: latent action -> per-limb joint targets.
        activations = synergy_activations(action, self.cpg)
        base_targets = joint_targets(action, self.cpg)

        # 3. Spinal reflexes (one-step-delayed contact normals).
        targets = spinal_reflexes(base_targets, self.state, self._last_contact_normals, cfg)

        # 4. Joint servo.
        metabolic = self._apply_joint_dynamics(targets, dt)

        tide_h = self._tide_height()
        ground_h = self._terrain_height(self.state.com_x, self.state.com_y)
        # Feet push off the terrain only; water provides buoyancy and drag, not
        # a rigid support surface. Using the water surface as support produced
        # unrealistic launch impulses when the body started submerged.
        support = ground_h
        submerged = self.state.com_z < tide_h

        # 5. Ground reaction forces from contacting limbs.
        fx, fy, fz, yaw_torque, contact_normals = self._contact_forces(prev_state, dt, support)
        self._last_contact_normals = contact_normals

        # Biomechanical effort drains stamina and energy.
        effort = metabolic * cfg.metabolic_scale
        self.stamina = max(0.0, self.stamina - effort - (0.02 if action in (4, 5) else 0.0))
        self.energy = max(0.0, self.energy - effort * 0.4)
        if action == 0:
            self.stamina = min(1.0, self.stamina + 0.03)

        density = cfg.fluid_density if submerged else cfg.air_density
        drag = density * cfg.drag_coefficient

        ax = fx * cfg.contact_gain - drag * self.state.vel_x
        ay = fy * cfg.contact_gain - drag * self.state.vel_y
        az = fz - cfg.gravity + (cfg.gravity * 1.1 if submerged else 0.0) - drag * self.state.vel_z

        self.state.yaw_rate += (yaw_torque - 0.5 * self.state.yaw_rate) * dt
        self.state.heading = _wrap_angle(self.state.heading + self.state.yaw_rate * dt)
        pitch_target = max(-0.5, min(0.5, -fz * 0.01 + self.state.joint_angles[0] * 0.3))
        self.state.pitch += (pitch_target - self.state.pitch) * min(1.0, dt * 4.0)

        self.state.vel_x += ax * dt
        self.state.vel_y += ay * dt
        self.state.vel_z += az * dt

        self.state.com_x = max(0.0, min(cfg.world_size, self.state.com_x + self.state.vel_x * dt))
        self.state.com_y = max(0.0, min(cfg.world_size, self.state.com_y + self.state.vel_y * dt))
        # Vertical ceiling at 15 world units (matches observation normalization);
        # the body cannot leave the tidepool's bounded volume.
        self.state.com_z = max(ground_h, min(15.0, self.state.com_z + self.state.vel_z * dt))

        self.energy = max(0.0, self.energy - 0.006 * dt)
        self.hydration = max(0.0, self.hydration - 0.008 * dt)
        if self.energy < 0.10:
            self.health = max(0.0, self.health - 0.04)
        if self.hydration < 0.10:
            self.health = max(0.0, self.health - 0.05)

        nearest = self._nearest_patch()
        dist = math.hypot(self.state.com_x - nearest[0], self.state.com_y - nearest[1])
        in_patch = dist <= nearest[3]

        reward = 0.05
        if action == 1 and in_patch and nearest[4] > 0.05:
            taken = min(0.3, nearest[4])
            self.energy = min(1.0, self.energy + taken * 0.7)
            nl = list(nearest); nl[4] -= taken
            self.patches[self.patches.index(nearest)] = tuple(nl)
            reward += 0.25
        elif action == 2 and in_patch and nearest[5] > 0.05:
            taken = min(0.35, nearest[5])
            self.hydration = min(1.0, self.hydration + taken * 0.8)
            nl = list(nearest); nl[5] -= taken
            self.patches[self.patches.index(nearest)] = tuple(nl)
            reward += 0.25

        done = self.health <= 0.0 or self.tick >= 256
        obs = self._build_observation(in_patch, nearest, submerged, metabolic)
        return obs, reward, done, {
            "tick": self.tick,
            "pos": (round(self.state.com_x, 2), round(self.state.com_y, 2), round(self.state.com_z, 2)),
            "heading": round(self.state.heading, 4),
            "pitch": round(self.state.pitch, 4),
            "joint_angles": [round(a, 4) for a in self.state.joint_angles],
            "contact": list(self.state.contact),
            "cpg_phases": [round(p, 4) for p in self.cpg.phases],
            "synergy_activations": [round(a, 4) for a in activations],
            "contact_normals": [round(n, 4) for n in contact_normals],
            "metabolic_cost": round(metabolic, 6),
            "is_submerged": submerged,
            "tide_height": round(tide_h, 2),
            "health": self.health,
            "alive": self.health > 0.0,
        }

    def _build_observation(
        self,
        in_patch: bool,
        patch: tuple[float, float, float, float, float, float],
        submerged: bool,
        metabolic: float,
    ) -> list[float]:
        """16-D observation. Channels 0-8 vitals/patch; 9-15 proprioceptive
        body state (pitch, heading, COM height, speed, contact ratio, mean
        joint angle, metabolic effort)."""
        speed = math.hypot(self.state.vel_x, self.state.vel_y)
        contact_ratio = sum(self.state.contact) / N_LIMBS
        mean_joint = sum(self.state.joint_angles) / N_JOINTS
        obs = [0.0] * OBS_SIZE
        obs[0] = float(self.health)
        obs[1] = float(self.energy)
        obs[2] = float(self.hydration)
        obs[3] = float(self.stamina)
        obs[4] = 0.8 if submerged else 0.1
        obs[5] = 0.05
        obs[6] = float(patch[4]) if in_patch else 0.05
        obs[7] = float(patch[5]) if in_patch else 0.05
        obs[8] = 0.5 if in_patch else 0.1
        obs[9] = float((self.state.pitch + math.pi) / (2.0 * math.pi))
        obs[10] = float((math.sin(self.state.heading) + 1.0) / 2.0)
        obs[11] = float(self.state.com_z / 15.0)
        obs[12] = float(min(1.0, speed / 10.0))
        obs[13] = float(contact_ratio)
        obs[14] = float((mean_joint + 2.2) / 4.4)
        obs[15] = float(min(1.0, metabolic * self.config.metabolic_scale))
        return obs
