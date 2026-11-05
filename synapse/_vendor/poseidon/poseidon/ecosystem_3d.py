"""POSEIDON ECOSYSTEM-3D: Multi-Agent Continuous 3D Trophic Dynamics & Bio-Pulse Signaling.

Synthesizes:
1. GenesisEngine & 3D-Animal-Simulator: Continuous multi-agent 3D Newtonian kinematics,
   Archimedean fluid hydrodynamics, tidal current oscillations, and volumetric spatial raycasts.
2. Flipper-Zero-Periodic-NTAG-Emulator: Autonomous periodic duty-cycled bio-electromagnetic/acoustic
   beacon pulses with frequency modulation, inverse-square attenuation, Doppler shift, and
   telemetry payload demodulation.
3. Supermix-Expanse & Causeway: Trophic food-web biomass transfer (primary grazers, apex predators,
   mutualistic symbionts), metabolic dissipation, and ecological niche partitioning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Sequence

from .genesis_3d import ContinuousResourcePatch, PhysicsVector3
from .world import ACTIONS, N_ACTIONS, OBS_SIZE


# Trophic Phenotypic Roles
TROPHIC_GRAZER = "grazer"
TROPHIC_PREDATOR = "predator"
TROPHIC_SYMBIONT = "symbiont"
TROPHIC_ROLES = (TROPHIC_GRAZER, TROPHIC_PREDATOR, TROPHIC_SYMBIONT)


@dataclass
class PeriodicPulseBeacon:
    """Duty-cycled bio-pulse beacon inspired by Flipper-Zero Periodic NTAG emulation."""
    emitter_id: str
    trophic_role: str
    carrier_frequency_khz: float = 13.56  # Analogous to NFC/bio-acoustic MHz/kHz
    duty_cycle: float = 0.25              # Active signal transmission ratio
    period_ticks: int = 8                 # Transmission cycle duration
    tx_power: float = 1.0                 # Emitter source amplitude
    payload: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [distress, food, threat, fitness]

    def is_transmitting(self, tick: int) -> bool:
        """Determines if the beacon is actively in its transmission phase of the duty cycle."""
        active_window = max(1, int(self.period_ticks * self.duty_cycle))
        return (tick % self.period_ticks) < active_window


@dataclass
class DemodulatedPulse:
    """Received and demodulated pulse signal received by an organism."""
    emitter_id: str
    trophic_role: str
    received_power: float
    doppler_frequency_khz: float
    bearing: PhysicsVector3
    distance: float
    payload: list[float]


@dataclass
class OrganismAgentState:
    """State vector of an individual embodied organism in the 3D ecosystem."""
    agent_id: str
    trophic_role: str
    pos: PhysicsVector3
    vel: PhysicsVector3
    heading: float
    pitch: float
    health: float = 1.0
    energy: float = 0.85
    hydration: float = 0.85
    stamina: float = 0.90
    biomass: float = 1.0
    beacon: PeriodicPulseBeacon = field(default_factory=lambda: PeriodicPulseBeacon("agent", TROPHIC_GRAZER))
    last_action: int = 0
    alive: bool = True
    strikes_delivered: int = 0
    symbiotic_services: int = 0


@dataclass
class Ecosystem3DConfig:
    world_size: float = 120.0
    gravity: float = 9.81
    fluid_density: float = 1.0
    air_density: float = 0.02
    tide_period: float = 64.0
    tide_amplitude: float = 3.2
    base_water_level: float = 2.5
    drag_coefficient: float = 0.85
    speed_of_sound_water: float = 1500.0  # m/s for Doppler propagation
    acoustic_attenuation_alpha: float = 0.015
    predator_strike_radius: float = 4.0
    symbiont_cleaning_radius: float = 5.5
    trophic_efficiency: float = 0.65
    dt: float = 0.25


class ContinuousEcosystem3D:
    """Continuous 3D multi-agent tidepool ecosystem with trophic co-evolution and pulse signaling."""

    def __init__(self, config: Ecosystem3DConfig | None = None, seed: int = 42):
        self.config = config or Ecosystem3DConfig()
        self.rng = random.Random(seed)
        self.time: float = 0.0
        self.tick: int = 0

        # Population
        self.agents: dict[str, OrganismAgentState] = {}
        # Resource patches
        self.patches: list[ContinuousResourcePatch] = []
        self._init_patches()

        # Telemetry aggregates
        self.total_trophic_transfers: int = 0
        self.total_pulses_exchanged: int = 0
        self.last_demodulated: dict[str, list[DemodulatedPulse]] = {}

    def _init_patches(self, count: int = 14) -> None:
        self.patches = []
        for i in range(count):
            x = self.rng.uniform(12.0, self.config.world_size - 12.0)
            y = self.rng.uniform(12.0, self.config.world_size - 12.0)
            z = self.get_terrain_height(x, y)
            self.patches.append(
                ContinuousResourcePatch(
                    id=i,
                    pos=PhysicsVector3(x, y, z),
                    radius=self.rng.uniform(5.0, 9.0),
                    food_amount=self.rng.uniform(0.5, 1.0),
                    water_amount=self.rng.uniform(0.6, 1.0),
                    shelter_quality=self.rng.uniform(0.3, 0.9),
                )
            )

    def spawn_agent(
        self,
        agent_id: str,
        trophic_role: str = TROPHIC_GRAZER,
        pos: PhysicsVector3 | None = None,
        beacon_period: int = 8,
        beacon_duty_cycle: float = 0.25,
    ) -> OrganismAgentState:
        """Spawns an agent into the 3D ecosystem."""
        if pos is None:
            x = self.rng.uniform(20.0, self.config.world_size - 20.0)
            y = self.rng.uniform(20.0, self.config.world_size - 20.0)
            z = max(self.get_terrain_height(x, y) + 1.0, 3.0)
            pos = PhysicsVector3(x, y, z)

        freq = 13.56 if trophic_role == TROPHIC_GRAZER else (18.2 if trophic_role == TROPHIC_PREDATOR else 9.4)
        beacon = PeriodicPulseBeacon(
            emitter_id=agent_id,
            trophic_role=trophic_role,
            carrier_frequency_khz=freq,
            duty_cycle=beacon_duty_cycle,
            period_ticks=beacon_period,
            tx_power=1.2 if trophic_role == TROPHIC_PREDATOR else 0.8,
            payload=[0.0, 0.0, 0.0, 0.0],
        )

        initial_biomass = 1.0 if trophic_role == TROPHIC_GRAZER else (1.8 if trophic_role == TROPHIC_PREDATOR else 0.6)
        agent = OrganismAgentState(
            agent_id=agent_id,
            trophic_role=trophic_role,
            pos=pos,
            vel=PhysicsVector3(0.0, 0.0, 0.0),
            heading=self.rng.uniform(0.0, 2.0 * math.pi),
            pitch=0.0,
            health=1.0,
            energy=0.85,
            hydration=0.85,
            stamina=0.90,
            biomass=initial_biomass,
            beacon=beacon,
        )
        self.agents[agent_id] = agent
        return agent

    def spawn_default_colony(self, n_grazers: int = 3, n_predators: int = 1, n_symbionts: int = 1) -> None:
        """Initializes a balanced trophic community."""
        for i in range(n_grazers):
            self.spawn_agent(f"grazer_{i}", TROPHIC_GRAZER, beacon_period=6, beacon_duty_cycle=0.33)
        for i in range(n_predators):
            self.spawn_agent(f"predator_{i}", TROPHIC_PREDATOR, beacon_period=10, beacon_duty_cycle=0.20)
        for i in range(n_symbionts):
            self.spawn_agent(f"symbiont_{i}", TROPHIC_SYMBIONT, beacon_period=4, beacon_duty_cycle=0.50)

    def get_terrain_height(self, x: float, y: float) -> float:
        w1 = math.sin(x * 0.04) * math.cos(y * 0.04) * 3.2
        w2 = math.sin(x * 0.11 + 0.8) * math.cos(y * 0.11 + 0.3) * 1.6
        return max(0.0, w1 + w2 + 1.2)

    def get_tide_height(self) -> float:
        phase = (2.0 * math.pi * self.time) / self.config.tide_period
        return self.config.base_water_level + self.config.tide_amplitude * math.sin(phase)

    def get_tidal_current(self) -> PhysicsVector3:
        """Oscillating tidal ocean flow vector."""
        phase = (2.0 * math.pi * self.time) / self.config.tide_period
        curr_x = math.cos(phase) * 1.2
        curr_y = math.sin(phase) * 0.7
        return PhysicsVector3(curr_x, curr_y, 0.0)

    def _propagate_pulses(self) -> dict[str, list[DemodulatedPulse]]:
        """Propagates periodic duty-cycled pulses through the aquatic medium with Doppler shift and inverse-square decay."""
        reception: dict[str, list[DemodulatedPulse]] = {aid: [] for aid in self.agents}
        active_emitters = [
            a for a in self.agents.values()
            if a.alive and a.beacon.is_transmitting(self.tick)
        ]

        for emitter in active_emitters:
            for receiver_id, receiver in self.agents.items():
                if receiver_id == emitter.agent_id or not receiver.alive:
                    continue

                dx = emitter.pos.x - receiver.pos.x
                dy = emitter.pos.y - receiver.pos.y
                dz = emitter.pos.z - receiver.pos.z
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                if dist < 1e-4:
                    continue

                # Attenuation
                geometric_decay = 1.0 / (1.0 + self.config.acoustic_attenuation_alpha * (dist**2))
                rx_power = emitter.beacon.tx_power * geometric_decay
                if rx_power < 0.01:
                    continue  # Below sensory receiver noise threshold

                # Doppler shift
                # Relative velocity projected onto sightline
                dir_x, dir_y, dir_z = dx / dist, dy / dist, dz / dist
                v_rel_x = emitter.vel.x - receiver.vel.x
                v_rel_y = emitter.vel.y - receiver.vel.y
                v_rel_z = emitter.vel.z - receiver.vel.z
                v_radial = v_rel_x * dir_x + v_rel_y * dir_y + v_rel_z * dir_z

                doppler_ratio = 1.0 + (v_radial / self.config.speed_of_sound_water)
                f_rx = emitter.beacon.carrier_frequency_khz * doppler_ratio

                bearing = PhysicsVector3(dir_x, dir_y, dir_z)
                demod = DemodulatedPulse(
                    emitter_id=emitter.agent_id,
                    trophic_role=emitter.trophic_role,
                    received_power=round(rx_power, 4),
                    doppler_frequency_khz=round(f_rx, 3),
                    bearing=bearing,
                    distance=round(dist, 2),
                    payload=list(emitter.beacon.payload),
                )
                reception[receiver_id].append(demod)
                self.total_pulses_exchanged += 1

        return reception

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, list[float]], dict[str, float], dict[str, bool], dict[str, Any]]:
        """Executes one continuous simulation time step across the entire multi-agent ecosystem."""
        dt = self.config.dt
        self.time += dt
        self.tick += 1

        tide_h = self.get_tide_height()
        current = self.get_tidal_current()

        rewards: dict[str, float] = {aid: 0.02 for aid in self.agents}  # baseline vitality reward
        dones: dict[str, bool] = {aid: False for aid in self.agents}

        # 1. Update beacon telemetry payloads based on somatic states
        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            distress = max(0.0, 1.0 - agent.health)
            food_hunger = max(0.0, 1.0 - agent.energy)
            threat_urgency = 0.8 if agent.trophic_role == TROPHIC_GRAZER and self._has_nearby_predator(agent) else 0.05
            fitness = agent.energy * agent.health
            agent.beacon.payload = [distress, food_hunger, threat_urgency, fitness]

        # 2. Propagate periodic pulse signals
        self.last_demodulated = self._propagate_pulses()

        # 3. Apply physics and individual actions
        for aid, agent in self.agents.items():
            if not agent.alive:
                dones[aid] = True
                continue

            act = actions.get(aid, 0)
            agent.last_action = act

            ground_h = self.get_terrain_height(agent.pos.x, agent.pos.y)
            is_submerged = agent.pos.z < tide_h

            thrust = 0.0
            torque = 0.0
            vz_thrust = 0.0

            # Action execution
            if act == 4:  # explore / forward thrust
                thrust = 8.5
                agent.stamina = max(0.0, agent.stamina - 0.015)
            elif act == 5:  # flee / sprint
                thrust = 15.0
                torque = 0.9
                agent.stamina = max(0.0, agent.stamina - 0.035)
            elif act == 0:  # rest
                thrust = 0.0
                agent.stamina = min(1.0, agent.stamina + 0.04)
                agent.energy = max(0.0, agent.energy - 0.002)
                agent.hydration = max(0.0, agent.hydration - 0.002)

            # Ingestion / Foraging in resource patches
            nearest_patch = self._get_nearest_patch(agent.pos)
            dist_p = math.sqrt((agent.pos.x - nearest_patch.pos.x)**2 + (agent.pos.y - nearest_patch.pos.y)**2)
            in_patch = dist_p <= nearest_patch.radius

            if act == 1:  # forage
                if in_patch and nearest_patch.food_amount > 0.05 and agent.trophic_role != TROPHIC_PREDATOR:
                    taken = min(0.35, nearest_patch.food_amount)
                    agent.energy = min(1.0, agent.energy + taken * 0.75)
                    nearest_patch.food_amount -= taken
                    rewards[aid] += 0.30
                agent.stamina = max(0.0, agent.stamina - 0.008)

            elif act == 2:  # drink
                if in_patch and nearest_patch.water_amount > 0.05:
                    taken = min(0.40, nearest_patch.water_amount)
                    agent.hydration = min(1.0, agent.hydration + taken * 0.85)
                    nearest_patch.water_amount -= taken
                    rewards[aid] += 0.30
                agent.stamina = max(0.0, agent.stamina - 0.008)

            # Physics integration
            density = self.config.fluid_density if is_submerged else self.config.air_density
            drag = density * self.config.drag_coefficient

            agent.heading += torque * dt
            # Hydrodynamic forces + tidal ocean currents
            ax = math.cos(agent.heading) * thrust - drag * (agent.vel.x - (current.x if is_submerged else 0.0))
            ay = math.sin(agent.heading) * thrust - drag * (agent.vel.y - (current.y if is_submerged else 0.0))

            buoyancy = (self.config.gravity * 1.1) if is_submerged else 0.0
            az = buoyancy - self.config.gravity + vz_thrust - drag * agent.vel.z

            agent.vel.x += ax * dt
            agent.vel.y += ay * dt
            agent.vel.z += az * dt

            agent.pos.x = max(0.0, min(self.config.world_size, agent.pos.x + agent.vel.x * dt))
            agent.pos.y = max(0.0, min(self.config.world_size, agent.pos.y + agent.vel.y * dt))
            agent.pos.z = max(ground_h, agent.pos.z + agent.vel.z * dt)

            # Baseline vital decays
            agent.energy = max(0.0, agent.energy - 0.005 * dt)
            agent.hydration = max(0.0, agent.hydration - 0.007 * dt)

            if agent.energy < 0.10:
                agent.health = max(0.0, agent.health - 0.035)
            if agent.hydration < 0.10:
                agent.health = max(0.0, agent.health - 0.045)

            if agent.health <= 0.0:
                agent.alive = False
                dones[aid] = True
                rewards[aid] -= 1.0

        # 4. Trophic Biomass Transfer & Interactions
        self._resolve_trophic_interactions(rewards)

        # 5. Build observations for each agent
        obs_dict: dict[str, list[float]] = {}
        for aid, agent in self.agents.items():
            obs_dict[aid] = self._build_agent_obs(agent, tide_h)

        telemetry = {
            "tick": self.tick,
            "time": round(self.time, 2),
            "tide_height": round(tide_h, 2),
            "alive_counts": {
                role: sum(1 for a in self.agents.values() if a.alive and a.trophic_role == role)
                for role in TROPHIC_ROLES
            },
            "total_trophic_transfers": self.total_trophic_transfers,
            "total_pulses_exchanged": self.total_pulses_exchanged,
        }

        return obs_dict, rewards, dones, telemetry

    def _resolve_trophic_interactions(self, rewards: dict[str, float]) -> None:
        """Computes predation, trophic energy transfer, and mutualistic symbiont cleaning."""
        predators = [a for a in self.agents.values() if a.alive and a.trophic_role == TROPHIC_PREDATOR]
        grazers = [a for a in self.agents.values() if a.alive and a.trophic_role == TROPHIC_GRAZER]
        symbionts = [a for a in self.agents.values() if a.alive and a.trophic_role == TROPHIC_SYMBIONT]

        # A. Predation: Predator attacks Grazer
        for pred in predators:
            for grz in grazers:
                if not grz.alive:
                    continue
                dx = grz.pos.x - pred.pos.x
                dy = grz.pos.y - pred.pos.y
                dz = grz.pos.z - pred.pos.z
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                if dist <= self.config.predator_strike_radius:
                    # Strike executed
                    strike_damage = 0.28
                    grz.health = max(0.0, grz.health - strike_damage)
                    transferred_energy = strike_damage * self.config.trophic_efficiency * 0.9
                    pred.energy = min(1.0, pred.energy + transferred_energy)
                    pred.strikes_delivered += 1
                    self.total_trophic_transfers += 1
                    rewards[pred.agent_id] += 0.45
                    rewards[grz.agent_id] -= 0.35

                    if grz.health <= 0.0:
                        grz.alive = False
                        pred.biomass += 0.15

        # B. Mutualistic Symbiosis: Symbiont cleans Predator
        for sym in symbionts:
            for pred in predators:
                dx = pred.pos.x - sym.pos.x
                dy = pred.pos.y - sym.pos.y
                dz = pred.pos.z - sym.pos.z
                dist = math.sqrt(dx**2 + dy**2 + dz**2)

                if dist <= self.config.symbiont_cleaning_radius:
                    # Mutualism exchange:
                    # Predator gets minor health restoration and parasite reduction
                    pred.health = min(1.0, pred.health + 0.02)
                    # Symbiont assimilates metabolic biofilm scraps
                    sym.energy = min(1.0, sym.energy + 0.035)
                    sym.symbiotic_services += 1
                    rewards[sym.agent_id] += 0.15
                    rewards[pred.agent_id] += 0.05

    def _has_nearby_predator(self, agent: OrganismAgentState, radius: float = 18.0) -> bool:
        for other in self.agents.values():
            if other.alive and other.trophic_role == TROPHIC_PREDATOR:
                d = math.sqrt((other.pos.x - agent.pos.x)**2 + (other.pos.y - agent.pos.y)**2)
                if d <= radius:
                    return True
        return False

    def _get_nearest_patch(self, pos: PhysicsVector3) -> ContinuousResourcePatch:
        return min(self.patches, key=lambda p: (p.pos.x - pos.x)**2 + (p.pos.y - pos.y)**2)

    def _build_agent_obs(self, agent: OrganismAgentState, tide_h: float) -> list[float]:
        """Encodes state + demodulated bio-pulses into normalized 16-D Poseidon observation vector."""
        obs = [0.0] * OBS_SIZE
        obs[0] = float(agent.health)
        obs[1] = float(agent.energy)
        obs[2] = float(agent.hydration)
        obs[3] = float(agent.stamina)

        is_sub = agent.pos.z < tide_h
        obs[4] = 0.8 if is_sub else 0.1  # exposure

        # Threat perception from pulses or proximity
        threat_level = 0.05
        nearest_patch = self._get_nearest_patch(agent.pos)
        dist_p = math.sqrt((agent.pos.x - nearest_patch.pos.x)**2 + (agent.pos.y - nearest_patch.pos.y)**2)
        in_patch = dist_p <= nearest_patch.radius

        pulses = self.last_demodulated.get(agent.agent_id, [])
        predator_pulses = [p for p in pulses if p.trophic_role == TROPHIC_PREDATOR]
        if predator_pulses:
            strongest = max(predator_pulses, key=lambda p: p.received_power)
            threat_level = min(1.0, strongest.received_power * 1.5)

        obs[5] = float(threat_level)
        obs[6] = float(nearest_patch.food_amount) if in_patch else 0.05
        obs[7] = float(nearest_patch.water_amount) if in_patch else 0.05
        obs[8] = float(nearest_patch.shelter_quality) if in_patch else 0.1
        obs[9] = float(agent.pos.x / self.config.world_size)
        obs[10] = float(agent.pos.y / self.config.world_size)
        obs[11] = float(agent.pos.z / 15.0)

        # Pulse communication & beacon sensory channel
        if pulses:
            # Aggregate received signal power and mean Doppler velocity shift
            mean_power = sum(p.received_power for p in pulses) / len(pulses)
            obs[12] = min(1.0, float(mean_power))
            obs[13] = float(len(pulses) / 6.0)  # Pulse traffic density
        else:
            obs[12] = 0.0
            obs[13] = 0.0

        obs[14] = float((math.sin(agent.heading) + 1.0) / 2.0)
        obs[15] = float((math.cos(agent.heading) + 1.0) / 2.0)
        return obs
