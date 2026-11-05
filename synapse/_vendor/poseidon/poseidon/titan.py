"""Experimental controller for the synthetic TidePool task.

TITAN composes selected software mechanisms from the reviewed project
portfolio around the frozen Tidal policy. Every added physical-system analogy
is a local simulation; Causeway selects actions on a classical computer. The
controller is opt-in and is not a newly trained model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np

from .aura import AuraController
from .causeway import CausewaySuperpositionEngine
from .chronos import ChronosBeaconClock, GhostTraceAuditor
from .core import CoreRuntime
from .hologram import HolographicAssociativeMemory
from .intermittent import NtagHarvestEngine
from .lattice import DiamondLatticeNeuropil
from .molt import InstarStage, MoltEngine
from .morpheus import MorpheusDreamEngine
from .nexus_search import NexusSearchEngine
from .nexusflow import NexusFlowNetwork
from .prometheus import PrometheusPlasticityEngine
from .tessera import TesseraMacroCommons
from .archimedes import ArchimedesHydrodynamicEngine
from .genesis import GenesisEcosystemEngine
from .svideo import SVideoTelemetryEngine
from .modder import UniversalModderEngine
from .quantumbot import QuantumBotCircuitEngine


class TitanController:
    """Unified sovereign frontier super-controller synthesizing all 24 projects."""

    def __init__(
        self,
        core: CoreRuntime,
        aura: Optional[AuraController] = None,
        molt: Optional[MoltEngine] = None,
        flow: Optional[NexusFlowNetwork] = None,
        chronos: Optional[ChronosBeaconClock] = None,
        ghost: Optional[GhostTraceAuditor] = None,
        lattice: Optional[DiamondLatticeNeuropil] = None,
        hologram: Optional[HolographicAssociativeMemory] = None,
        causeway: Optional[CausewaySuperpositionEngine] = None,
        tessera: Optional[TesseraMacroCommons] = None,
        morpheus: Optional[MorpheusDreamEngine] = None,
        prometheus: Optional[PrometheusPlasticityEngine] = None,
        intermittent: Optional[NtagHarvestEngine] = None,
        nexus_search: Optional[NexusSearchEngine] = None,
        archimedes: Optional[ArchimedesHydrodynamicEngine] = None,
        genesis: Optional[GenesisEcosystemEngine] = None,
        svideo: Optional[SVideoTelemetryEngine] = None,
        modder: Optional[UniversalModderEngine] = None,
        quantumbot: Optional[QuantumBotCircuitEngine] = None,
    ):
        self.core = core
        self.tessera = tessera or TesseraMacroCommons()
        self.aura = aura or AuraController(core=core, tessera=self.tessera)
        self.molt = molt or MoltEngine()
        self.flow = flow or NexusFlowNetwork()
        self.chronos = chronos or ChronosBeaconClock(period=8)
        self.ghost = ghost or GhostTraceAuditor()
        self.lattice = lattice or DiamondLatticeNeuropil()
        self.hologram = hologram or HolographicAssociativeMemory()
        self.causeway = causeway or CausewaySuperpositionEngine(num_actions=6)
        self.morpheus = morpheus or MorpheusDreamEngine()
        self.prometheus = prometheus or PrometheusPlasticityEngine()
        self.intermittent = intermittent or NtagHarvestEngine()
        self.nexus_search = nexus_search or NexusSearchEngine()
        self.archimedes = archimedes or ArchimedesHydrodynamicEngine(tide_period_steps=48)
        self.genesis = genesis or GenesisEcosystemEngine()
        self.svideo = svideo or SVideoTelemetryEngine()
        self.modder = modder or UniversalModderEngine()
        self.quantumbot = quantumbot or QuantumBotCircuitEngine()

        self.last_action = 0
        self.step_count = 0
        self.consecutive_rests = 0
        self._pending_transition = None

    def reset(self, scarcity: float = 1.0, seed: int = 0) -> None:
        """Reset state across all constituent subsystems."""
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2^63")
        self.aura.reset(scarcity)
        self.molt.reset()
        self.flow.reset()
        self.chronos.reset()
        self.ghost.reset()
        self.lattice.reset()
        self.hologram = HolographicAssociativeMemory()
        self.causeway.reset()
        self.tessera.reset()
        self.morpheus.reset()
        self.prometheus.reset()
        self.intermittent.reset()
        self.nexus_search.reset()
        self.archimedes.reset()
        self.genesis.reset(seed=seed & 0xFFFFFFFF)
        self.svideo.reset()
        self.modder.reset()
        self.quantumbot.reset()
        self.last_action = 0
        self.step_count = 0
        self.consecutive_rests = 0
        self._pending_transition = None

    def plan(self, observation: List[float] | np.ndarray, step: Optional[int] = None) -> Dict[str, Any]:
        if self._pending_transition is not None:
            raise RuntimeError("observe_transition must record the previous action outcome before planning again")
        raw_obs = [float(x) for x in observation]
        if len(raw_obs) < 16:
            raise ValueError(f"Expected at least 16 observation features, got {len(raw_obs)}")

        self.step_count += 1
        food_level = float(raw_obs[0])
        water_level = float(raw_obs[1])
        stamina_level = float(raw_obs[2])
        health_level = float(raw_obs[3])
        severity = float(raw_obs[9]) if len(raw_obs) > 9 else 0.0
        daylight = float(raw_obs[11]) if len(raw_obs) > 11 else 0.5

        # 1. Cyclic Chronos Beacon Clock Tick
        pulse = self.chronos.step()
        if pulse.syn_pulse and self.tessera:
            self.tessera.retire_expired()

        # 2. Archimedes Hydrodynamics & Buoyancy Update
        archimedes_info = self.archimedes.step(raw_obs, action=self.last_action, heading_angle=0.0)

        # 3. Genesis Planetary Ecology & Evolution
        genesis_info = self.genesis.step(daylight=daylight, severity=severity, step_reward=0.0)

        # 4. Diamond Lattice velocity update from previous action
        self.lattice.update_from_action(self.last_action, speed=1.0)
        lattice_coherence = self.lattice.multi_scale_coherence()
        lattice_pos = list(self.lattice.estimated_pos)

        # 5. Holographic Associative Memory Encoding
        self.hologram.encode_observation(raw_obs)
        food_query, food_sim = self.hologram.query("ROLE_FOOD")
        water_query, water_sim = self.hologram.query("ROLE_WATER")

        # 6. Prometheus Meta-Plasticity Evaluation
        prom_info = self.prometheus.evaluate(raw_obs)

        # 7. QuantumBot Bell State Entanglement & Cognitive Teleportation
        quantumbot_info = self.quantumbot.step(raw_obs, action=self.last_action)

        # 8. Intermittent RF Carrier Harvesting & Checkpoint State
        phases = [
            math.atan2(i, r) for r, i in zip(self.causeway.real, self.causeway.imag)
        ]
        intermittent_info = self.intermittent.step(
            step=self.step_count,
            obs=raw_obs,
            action=self.last_action,
            phases=phases,
            lattice_pos=lattice_pos,
        )

        # 9. NexusSearch Indexing and Associative Vector Query
        concept_label = f"c_{int(food_level*10)}_{int(water_level*10)}"
        search_hits = self.nexus_search.search_by_vector(self.hologram.trace[:16], top_k=2)

        # 10. MOLT Developmental Instar Update & Sensory Filter
        exuvia = self.molt.update_development(raw_obs)
        filtered_obs = self.molt.filter_observation(raw_obs)

        # 11. AURA Biomimetic Central Complex & Mushroom Body Neuropil
        aura_plan = self.aura.plan(filtered_obs)
        aura_action = int(aura_plan["action"])
        pva_heading = float(aura_plan.get("pva_heading", 0.0))
        pva_coherence = float(aura_plan.get("pva_coherence", 0.5))
        winning_channel = aura_plan.get("winning_channel", "sentinel")
        drives = aura_plan.get("homeostatic_drives", {})

        # 12. NexusFlow Spatial Registration & Hydraulic Gradient
        pos_x = raw_obs[10] if len(raw_obs) > 10 else 0.5
        pos_y = raw_obs[11] if len(raw_obs) > 11 else 0.5
        node_id = f"node_{int(pos_x * 10)}_{int(pos_y * 10)}"
        replenishment = raw_obs[8] if len(raw_obs) > 8 else 0.5
        depletion = 1.0 - raw_obs[9] if len(raw_obs) > 9 else 0.0

        self.flow.register_waypoint(node_id, pos_x, pos_y, replenishment, depletion)
        flow_action, flow_info = self.flow.recommend_action_from_flow(pos_x, pos_y, pva_heading)
        flux_mag = float(flow_info.get("flux_magnitude", 0.0))
        target_heading = float(flow_info.get("target_heading", pva_heading))

        interf_val, destructive_conflict = self.flow.evaluate_superposition_interference(
            value_a=float(aura_plan.get("confidence", 0.5)),
            value_b=flux_mag,
            heading_angle_a=pva_heading,
            heading_angle_b=target_heading,
        )

        # 13. Construct Hamiltonian Potentials for Causeway Quantum Superposition
        if destructive_conflict and drives.get("threat", 0.0) > 0.4:
            provisional_action = 5  # flee
        elif flux_mag > 0.6 and drives.get("metabolic", 0.0) > 0.5:
            provisional_action = flow_action
        else:
            provisional_action = aura_action

        potentials = [0.05] * 6
        potentials[provisional_action] += 2.8

        # Holographic resonance bias
        if food_query == "STATE_CRITICAL":
            potentials[1] += 0.5
        if water_query == "STATE_CRITICAL":
            potentials[2] += 0.5

        # Prometheus Homeostatic drive modulation
        if prom_info["dominant_drive"] == "fatigue" and stamina_level < 0.20:
            potentials[0] += 1.0
        elif prom_info["dominant_drive"] == "hunger" and food_level < 0.25:
            potentials[1] += 0.8

        # Archimedes hydrodynamic modulation:
        stamina_cost_mult = archimedes_info["stamina_cost_multiplier"]
        if stamina_cost_mult < 0.85:
            potentials[1] += 0.4
            potentials[4] += 0.4
        elif stamina_cost_mult > 1.20:
            potentials[0] += 0.6
            potentials[3] += 0.4

        # Genesis trophic richness modulation
        if genesis_info["trophic_richness"] > 1.2:
            potentials[1] += 0.3
        elif genesis_info["predators_biomass"] > 0.8 and drives.get("threat", 0.0) > 0.5:
            potentials[5] += 0.7

        # Intermittent energy boost preference
        if intermittent_info["resuscitated"]:
            potentials[0] += 1.2

        # Directional phase shifts
        phase_shifts = [0.0] * 6
        cardinal_angles = {1: -math.pi / 2.0, 2: math.pi / 2.0, 3: math.pi, 4: 0.0}
        for a in range(6):
            if a in cardinal_angles:
                phase_shifts[a] = cardinal_angles[a] - target_heading

        # 14. Causeway Unitary Evolution and Born Rule Measurement
        self.causeway.evolve(potentials, phase_shifts)
        collapsed_action, causeway_telemetry = self.causeway.collapse(seed=None)

        # 15. MOLT Life-Stage Action Modulation
        final_action, molt_source = self.molt.modulate_action(collapsed_action, drives)

        # 16. Universal Modder Safety Trampoline Hooks
        final_action, modder_override = self.modder.apply_pre_action_trampolines(
            final_action, raw_obs, {}
        )

        if final_action == 0:
            self.consecutive_rests += 1
        else:
            self.consecutive_rests = 0

        # 17. Morpheus Offline Sleep & Memory Consolidation
        morpheus_telemetry = self.morpheus.process_cycle(
            causeway=self.causeway,
            lattice=self.lattice,
            hologram=self.hologram,
            action=final_action,
            stamina=stamina_level,
            consecutive_rests=self.consecutive_rests,
        )
        # 18. Ghost Counterfactual SDI Introspection against Frozen Core Baseline
        core_action = (
            int(self.core.act(raw_obs))
            if hasattr(self.core, "act")
            else int(raw_obs[12]) if len(raw_obs) > 12 and isinstance(raw_obs[12], (int, float)) else 0
        )
        sdi_val = self.ghost.record_step(final_action, core_action)

        self.last_action = final_action
        self._pending_transition = {
            "step": self.step_count,
            "observation": raw_obs,
            "action": final_action,
            "lattice_pos": list(lattice_pos),
            "predicted_reward": float(np.mean(potentials)),
            "concept": concept_label,
            "vector": self.hologram.trace[:16].copy(),
        }

        # 19. S-Video Scanline Telemetry Encoding
        vitals_dict = {"health": health_level, "energy": food_level, "hydration": water_level, "stamina": stamina_level}
        svideo_info = self.svideo.encode_telemetry_frame(
            vitals=vitals_dict,
            lattice_pos=lattice_pos,
            quantum_phases=phases,
        )

        # Combined joint coherence metrics
        bell_fid = float(quantumbot_info["bell_fidelity"])
        joint_titan_coherence = round(pva_coherence * lattice_coherence * bell_fid, 4)

        action_source = f"titan:{winning_channel}:{self.molt.stage.value}:{morpheus_telemetry['sleep_state']}"
        if molt_source != collapsed_action:
            action_source = f"{action_source}:{molt_source}"
        if modder_override:
            action_source = f"{action_source}:{modder_override}"

        return {
            "action": final_action,
            "core_baseline_action": core_action,
            "action_source": action_source,
            "instar_stage": self.molt.stage.value,
            "exuvia_shed": exuvia.digest if exuvia else None,
            "beacon_phase": pulse.phase,
            "beacon_cycle": pulse.cycle,
            "syn_pulse": pulse.syn_pulse,
            "pva_heading": round(pva_heading, 4),
            "pva_coherence": round(pva_coherence, 4),
            "winning_channel": winning_channel,
            "lattice_coherence": round(lattice_coherence, 4),
            "titan_coherence": joint_titan_coherence,
            "lattice_pos": [round(p, 4) for p in self.lattice.estimated_pos],
            "quantum_entropy": causeway_telemetry["von_neumann_entropy"],
            "quantum_coherence_length": causeway_telemetry["coherence_length"],
            "destructive_conflicts_count": causeway_telemetry["destructive_conflicts_count"],
            "wave_interference": round(interf_val, 4),
            "holographic_capacity_load": round(self.hologram.item_count / self.hologram.dimension, 4),
            "spectral_divergence": round(sdi_val, 4),
            "confidence": round(causeway_telemetry["probabilities"][collapsed_action], 4),
            # Archimedes telemetry
            "buoyancy_force": archimedes_info["buoyancy_force"],
            "fluid_density": archimedes_info["fluid_density"],
            "stamina_cost_multiplier": archimedes_info["stamina_cost_multiplier"],
            # Genesis telemetry
            "trophic_richness": genesis_info["trophic_richness"],
            "generation": genesis_info["generation"],
            "lineage_digest": self.genesis.lineage_digest,
            # QuantumBot telemetry
            "bell_fidelity": bell_fid,
            "total_teleportations": quantumbot_info["total_teleportations"],
            # S-Video telemetry
            "svideo_line_number": svideo_info["line_number"],
            # Morpheus telemetry
            "sleep_state": morpheus_telemetry["sleep_state"],
            "rem_replays": morpheus_telemetry["rem_replays"],
            "phase_gain": morpheus_telemetry["phase_gain"],
            # Prometheus telemetry
            "scarcity_entropy": prom_info["scarcity_entropy"],
            "adapted_gain": prom_info["adapted_gain"],
            "dominant_drive": prom_info["dominant_drive"],
            # Intermittent telemetry
            "virtual_capacitor": intermittent_info["virtual_capacitor"],
            "total_harvested_energy": intermittent_info["total_harvested"],
            # Universal Modder telemetry
            "modder_override": modder_override,
            "total_modder_overrides": self.modder.total_overrides,
        }

    def act(self, observation: List[float] | np.ndarray) -> int:
        """Execute plan and return action integer for world rollouts."""
        return int(self.plan(observation)["action"])

    def observe_transition(self, observation, action: int, reward: float, next_observation, info=None) -> None:
        """Record actual TidePool feedback after the selected action is executed."""
        pending = self._pending_transition
        if pending is None:
            raise RuntimeError("observe_transition called without a pending plan")
        if int(action) != pending["action"]:
            raise ValueError("observed action does not match the pending plan")
        if not np.array_equal(np.asarray(observation, dtype=np.float64), np.asarray(pending["observation"], dtype=np.float64)):
            raise ValueError("observed state does not match the pending plan")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        next_obs = [float(value) for value in next_observation]
        if len(next_obs) < 16 or not all(math.isfinite(value) for value in next_obs):
            raise ValueError("next_observation must contain at least 16 finite values")

        self.morpheus.record_waking_step(
            step=pending["step"], observation=pending["observation"], action=pending["action"],
            reward=float(reward), next_observation=next_obs, lattice_pos=pending["lattice_pos"],
            predicted_reward=pending["predicted_reward"],
        )
        self.genesis.record_observed_reward(float(reward))
        self.nexus_search.index_event(
            concept=pending["concept"], vector=pending["vector"],
            coordinate=tuple(pending["lattice_pos"]), action=pending["action"],
            reward=float(reward), step=pending["step"],
        )
        self._pending_transition = None

    def __call__(self, observation: List[float] | np.ndarray) -> int:
        return self.act(observation)
