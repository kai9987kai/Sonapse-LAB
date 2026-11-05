"""METAMORPH: Unified Metamorphic Biomimetic Causal Controller.

Synthesizing:
- AURA: 16-wedge Central Complex ring attractor compass + 128-Kenyon cell Mushroom Body neuropil
- MOLT: Life-stage morphological ecdysis and developmental gain modulation
- NexusFlow: Directed acyclic causal potential flux and superposition wave interference
- Chronos & Ghost: Periodic beacon clock synchronization and counterfactual phantom trace auditing
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
import numpy as np

from .aura import AuraController
from .chronos import ChronosBeaconClock, GhostTraceAuditor
from .molt import InstarStage, MoltEngine
from .nexusflow import NexusFlowNetwork
from .tessera import TesseraMacroCommons


class MetamorphController:
    """Unified controller orchestrating biomimetic, developmental, and causal flow dynamics."""

    def __init__(
        self,
        core: Any = None,
        aura: Optional[AuraController] = None,
        molt: Optional[MoltEngine] = None,
        flow: Optional[NexusFlowNetwork] = None,
        beacon: Optional[ChronosBeaconClock] = None,
        ghost: Optional[GhostTraceAuditor] = None,
        tessera: Optional[TesseraMacroCommons] = None,
    ):
        self.core = core
        self.tessera = tessera or TesseraMacroCommons()
        self.aura = aura or AuraController(core=core, tessera=self.tessera)
        self.molt = molt or MoltEngine()
        self.flow = flow or NexusFlowNetwork()
        self.beacon = beacon or ChronosBeaconClock()
        self.ghost = ghost or GhostTraceAuditor()
        self.last_action = 0

    def reset(self, scarcity: float = 1.0) -> None:
        self.aura.reset(scarcity=scarcity)
        self.molt.reset()
        self.beacon.reset()
        self.ghost.reset()
        self.last_action = 0

    def plan(self, observation: List[float] | np.ndarray) -> Dict[str, Any]:
        """Compute integrated decision through Chronos, MOLT, AURA, and NexusFlow."""
        raw_obs = [float(x) for x in observation]

        # 1. Chronos beacon pulse tick
        pulse = self.beacon.step()
        if pulse.syn_pulse and self.tessera:
            self.tessera.retire_expired()

        # 2. MOLT developmental update & sensory modulation
        exuvia = self.molt.update_development(raw_obs)
        filtered_obs = self.molt.filter_observation(raw_obs)

        # 3. AURA biomimetic plan (CX ring attractor + MB sparse neuropil)
        aura_decision = self.aura.plan(filtered_obs)
        base_action = int(aura_decision["action"])
        drives = aura_decision.get("homeostatic_drives", {})
        compass_heading = float(aura_decision.get("pva_heading", 0.0))

        # 4. NexusFlow spatial registration & gradient evaluation
        pos_x = raw_obs[10] if len(raw_obs) > 10 else 0.5
        pos_y = raw_obs[11] if len(raw_obs) > 11 else 0.5
        node_id = f"node_{int(pos_x * 10)}_{int(pos_y * 10)}"
        replenishment = raw_obs[8] if len(raw_obs) > 8 else 0.5
        depletion = 1.0 - raw_obs[9] if len(raw_obs) > 9 else 0.0

        self.flow.register_waypoint(node_id, pos_x, pos_y, replenishment, depletion)
        flow_action, flow_info = self.flow.recommend_action_from_flow(pos_x, pos_y, compass_heading)

        # Check superposition wave interference between AURA heading and Flow target heading
        target_heading = flow_info.get("target_heading", compass_heading)
        interf_val, destructive_conflict = self.flow.evaluate_superposition_interference(
            value_a=aura_decision.get("confidence", 0.5),
            value_b=flow_info.get("flux_magnitude", 0.5),
            heading_angle_a=compass_heading,
            heading_angle_b=target_heading,
        )

        # 5. MOLT life-stage physical action modulation
        # If in destructive spatial conflict and threat is high, escape takes precedence
        if destructive_conflict and drives.get("threat", 0.0) > 0.4:
            provisional_action = 1  # Vertical escape
            action_source = "nexus_conflict_escape"
        elif flow_info.get("flux_magnitude", 0.0) > 0.6 and drives.get("metabolic", 0.0) > 0.5:
            provisional_action = flow_action
            action_source = "nexus_flux_routing"
        else:
            provisional_action = base_action
            action_source = aura_decision.get("action_source", "aura_neuropil")

        final_action, molt_source = self.molt.modulate_action(provisional_action, drives)
        if molt_source != provisional_action:
            action_source = f"{action_source}:{molt_source}"

        # 6. Ghost auditor records actual chosen action vs baseline core counterfactual
        core_counterfactual = (
            int(self.core.act(raw_obs))
            if hasattr(self.core, "act")
            else int(raw_obs[12]) if len(raw_obs) > 12 and isinstance(raw_obs[12], (int, float)) else 0
        )
        sdi = self.ghost.record_step(actual_action=final_action, phantom_action=core_counterfactual)

        self.last_action = final_action

        return {
            "action": int(final_action),
            "action_source": str(action_source),
            "instar_stage": self.molt.stage.value,
            "beacon_phase": pulse.phase,
            "beacon_cycle": pulse.cycle,
            "syn_pulse": pulse.syn_pulse,
            "pva_heading": aura_decision.get("pva_heading"),
            "pva_coherence": aura_decision.get("pva_coherence"),
            "winning_channel": aura_decision.get("winning_channel"),
            "homeostatic_drives": drives,
            "causal_flux_magnitude": flow_info.get("flux_magnitude", 0.0),
            "wave_interference": interf_val,
            "destructive_conflict": destructive_conflict,
            "spectral_divergence": sdi,
            "exuvia_shed": exuvia.to_dict() if exuvia else None,
            "confidence": aura_decision.get("confidence", 0.5),
        }

    def act(self, observation: List[float] | np.ndarray) -> int:
        return int(self.plan(observation)["action"])
