"""HYPERION: Unified Frontier Super-Controller for Poseidon.

Harmonizes:
1. Frozen Tidal DAgger Core
2. AURA (16-wedge CX ring attractor compass + 128-KC sparse neuropil arbiter)
3. MOLT (Metamorphic developmental instars: larval, pupa, imago & ecdysis)
4. NexusFlow (Directional hydraulic potential flux gradients)
5. Chronos & Ghost Auditor (Hardware cyclic beacon sync & counterfactual SDI)
6. Diamond Lattice Neuropil (Multi-scale 3D tetrahedral geodesic grid modules)
7. Holographic Associative Memory (Circular convolution HRR concept binding)
8. Causeway Engine (Quantum-inspired unitary superposition decision manifold)
9. Morpheus (Offline oneiric sleep engine & hippocampal memory consolidation)
10. Prometheus (Online meta-plasticity & homeostatic drive self-regulation)
11. Intermittent (Passive ambient RF energy harvesting & NTAG ephemeral resuscitation)
12. NexusSearch (Federated hybrid associative vector and spatial search)
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


class HyperionController:
    """Unified frontier super-controller synthesizing quantum superposition, holographic concept memory,

    3D diamond lattice, offline dream consolidation, meta-plasticity, and intermittent RF harvesting.
    """

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

        self.last_action = 0
        self.step_count = 0
        self.consecutive_rests = 0

    def reset(self, scarcity: float = 1.0) -> None:
        """Reset state across all constituent subsystems."""
        self.aura.reset(scarcity)
        self.molt.reset()
        self.flow.reset()
        self.chronos.reset()
        self.ghost.reset()
        self.lattice.reset()
        self.hologram = HolographicAssociativeMemory()
        self.causeway.reset()
        self.morpheus.reset()
        self.prometheus.reset()
        self.intermittent.reset()
        self.nexus_search.reset()
        self.last_action = 0
        self.step_count = 0
        self.consecutive_rests = 0

    def plan(self, observation: List[float] | np.ndarray, step: Optional[int] = None) -> Dict[str, Any]:
        """Execute unified multi-system decision pipeline and return chosen action with telemetry."""
        raw_obs = [float(x) for x in observation]
        if len(raw_obs) < 16:
            raise ValueError(f"Expected at least 16 observation features, got {len(raw_obs)}")

        self.step_count += 1
        food_level = float(raw_obs[0])
        water_level = float(raw_obs[1])
        stamina_level = float(raw_obs[2])
        health_level = float(raw_obs[3])

        # 1. Cyclic Chronos Beacon Clock Tick
        pulse = self.chronos.step()
        if pulse.syn_pulse and self.tessera:
            self.tessera.retire_expired()

        # 2. Diamond Lattice velocity update from previous action
        self.lattice.update_from_action(self.last_action, speed=1.0)
        lattice_coherence = self.lattice.multi_scale_coherence()
        lattice_pos = list(self.lattice.estimated_pos)

        # 3. Holographic Associative Memory Encoding
        self.hologram.encode_observation(raw_obs)
        food_query, food_sim = self.hologram.query("ROLE_FOOD")
        water_query, water_sim = self.hologram.query("ROLE_WATER")

        # 4. Prometheus Meta-Plasticity Evaluation
        prom_info = self.prometheus.evaluate(raw_obs)

        # 5. Intermittent RF Carrier Harvesting & Checkpoint State
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

        # 6. NexusSearch Indexing and Associative Vector Query
        concept_label = f"c_{int(food_level*10)}_{int(water_level*10)}"
        self.nexus_search.index_event(
            concept=concept_label,
            vector=self.hologram.trace[:16],
            coordinate=(lattice_pos[0], lattice_pos[1], lattice_pos[2]),
            action=self.last_action,
            reward=food_level + water_level,
            step=self.step_count,
        )
        search_hits = self.nexus_search.search_by_vector(self.hologram.trace[:16], top_k=2)

        # 7. MOLT Developmental Instar Update & Sensory Filter
        exuvia = self.molt.update_development(raw_obs)
        filtered_obs = self.molt.filter_observation(raw_obs)

        # 8. AURA Biomimetic Central Complex & Mushroom Body Neuropil
        aura_plan = self.aura.plan(filtered_obs)
        aura_action = int(aura_plan["action"])
        pva_heading = float(aura_plan.get("pva_heading", 0.0))
        pva_coherence = float(aura_plan.get("pva_coherence", 0.5))
        winning_channel = aura_plan.get("winning_channel", "sentinel")
        drives = aura_plan.get("homeostatic_drives", {})

        # 9. NexusFlow Spatial Registration & Hydraulic Gradient
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

        # 10. Construct Hamiltonian Potentials for Causeway Quantum Superposition
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
            potentials[1] += 0.5  # forage
        if water_query == "STATE_CRITICAL":
            potentials[2] += 0.5  # drink

        # Homeostatic drive modulation from Prometheus
        if prom_info["dominant_drive"] == "fatigue" and stamina_level < 0.20:
            potentials[0] += 1.0  # stay / rest
        elif prom_info["dominant_drive"] == "hunger" and food_level < 0.25:
            potentials[1] += 0.8  # forage

        # Intermittent energy boost preference
        if intermittent_info["resuscitated"]:
            potentials[0] += 1.2

        # Phase shifts derived from directional alignments
        phase_shifts = [0.0] * 6
        cardinal_angles = {1: -math.pi / 2.0, 2: math.pi / 2.0, 3: math.pi, 4: 0.0}
        for a in range(6):
            if a in cardinal_angles:
                phase_shifts[a] = cardinal_angles[a] - target_heading

        # 11. Causeway Unitary Evolution and Born Rule Measurement
        self.causeway.evolve(potentials, phase_shifts)
        collapsed_action, causeway_telemetry = self.causeway.collapse(seed=None)

        # 12. MOLT Life-Stage Action Modulation
        final_action, molt_source = self.molt.modulate_action(collapsed_action, drives)

        if final_action == 0:
            self.consecutive_rests += 1
        else:
            self.consecutive_rests = 0

        # 13. Morpheus Offline Sleep & Memory Consolidation
        morpheus_telemetry = self.morpheus.process_cycle(
            causeway=self.causeway,
            lattice=self.lattice,
            hologram=self.hologram,
            action=final_action,
            stamina=stamina_level,
            consecutive_rests=self.consecutive_rests,
        )

        # Record waking step into Morpheus
        self.morpheus.record_waking_step(
            step=self.step_count,
            observation=raw_obs,
            action=final_action,
            reward=food_level + water_level,
            next_observation=raw_obs,
            lattice_pos=lattice_pos,
            predicted_reward=float(np.mean(potentials)),
        )

        # 14. Ghost Counterfactual SDI Introspection against Frozen Core Baseline
        core_action = (
            int(self.core.act(raw_obs))
            if hasattr(self.core, "act")
            else int(raw_obs[12]) if len(raw_obs) > 12 and isinstance(raw_obs[12], (int, float)) else 0
        )
        sdi_val = self.ghost.record_step(final_action, core_action)

        self.last_action = final_action

        action_source = f"hyperion:{winning_channel}:{self.molt.stage.value}:{morpheus_telemetry['sleep_state']}"
        if molt_source != collapsed_action:
            action_source = f"{action_source}:{molt_source}"

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
            "hyperion_coherence": round(pva_coherence * lattice_coherence, 4),
            "lattice_pos": [round(p, 4) for p in self.lattice.estimated_pos],
            "quantum_entropy": causeway_telemetry["von_neumann_entropy"],
            "quantum_coherence_length": causeway_telemetry["coherence_length"],
            "destructive_conflicts_count": causeway_telemetry["destructive_conflicts_count"],
            "wave_interference": round(interf_val, 4),
            "holographic_capacity_load": round(self.hologram.item_count / self.hologram.dimension, 4),
            "spectral_divergence": round(sdi_val, 4),
            "confidence": round(causeway_telemetry["probabilities"][collapsed_action], 4),
            # Morpheus telemetry
            "sleep_state": morpheus_telemetry["sleep_state"],
            "total_sleep_cycles": morpheus_telemetry["total_cycles"],
            "consolidated_memories": morpheus_telemetry["consolidated_count"],
            "rem_replays": morpheus_telemetry["rem_replays"],
            "phase_gain": morpheus_telemetry["phase_gain"],
            # Prometheus telemetry
            "scarcity_entropy": prom_info["scarcity_entropy"],
            "adapted_gain": prom_info["adapted_gain"],
            "adapted_decay": prom_info["adapted_decay"],
            "dominant_drive": prom_info["dominant_drive"],
            "homeostatic_urgency": prom_info["homeostatic_urgency"],
            # Intermittent telemetry
            "virtual_capacitor": intermittent_info["virtual_capacitor"],
            "total_harvested_energy": intermittent_info["total_harvested"],
            "total_resuscitations": intermittent_info["total_resuscitations"],
            # NexusSearch telemetry
            "search_hits_count": len(search_hits),
            "top_search_similarity": search_hits[0]["similarity"] if search_hits else 0.0,
        }

    def act(self, observation: List[float] | np.ndarray) -> int:
        """Execute plan and return action integer for world rollouts."""
        return int(self.plan(observation)["action"])

    def __call__(self, observation: List[float] | np.ndarray) -> int:
        return self.act(observation)
