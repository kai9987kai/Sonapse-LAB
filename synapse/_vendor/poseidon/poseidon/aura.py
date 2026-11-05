"""AURA: Attractor Neuropil Ring Architecture.

Unified controller combining:
  1. CXRingCompass: 16-wedge recurrent ring attractor with PVA heading and optomotor stabilization reflex.
  2. SparseNeuropilArbiter: 128-cell Kenyon sparse projection with APL k-winners-take-all inhibition and tri-drive homeostatic arbitration (Harvester, Sentinel, Escaper).
  3. TesseraMacroCommons: independent-lineage ratified macro-action execution with finite-domain safety verification.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional
import numpy as np

from .cx_ring import CXRingCompass
from .neuropil import SparseNeuropilArbiter
from .tessera import TesseraMacroCommons


class AuraController:
    """Integrated biomimetic controller uniting ring attractor compass, neuropil arbiter, and ratified macro-actions."""

    def __init__(self, core: Any, tessera: Optional[TesseraMacroCommons] = None) -> None:
        self.core = core
        self.compass = CXRingCompass(num_wedges=16)
        self.arbiter = SparseNeuropilArbiter(obs_dim=16)
        self.tessera = tessera or TesseraMacroCommons()
        self.active_macro_sequence: List[int] = []
        self.current_scarcity = 2.5
        self.last_action = 0

    def reset(self, scarcity: float = 2.5) -> None:
        self.current_scarcity = float(scarcity)
        self.compass.reset(heading=0.0)
        self.active_macro_sequence = []
        self.last_action = 0

    def plan(self, observation: list[float] | np.ndarray) -> dict[str, Any]:
        """Compute integrated decision given TidePool observation."""
        obs = list(observation)
        flow_x = obs[4] if len(obs) > 4 else 0.0
        flow_y = obs[5] if len(obs) > 5 else 0.0
        stamina = obs[1] if len(obs) > 1 else 1.0
        depth = obs[3] if len(obs) > 3 else 0.5
        
        # 1. Update compass ring attractor
        flow_angle = math.atan2(flow_y, flow_x)
        # Angular velocity from last action: 3=left (+), 4=right (-)
        ang_vel = 0.5 if self.last_action == 3 else (-0.5 if self.last_action == 4 else 0.0)
        pva_heading, pva_coherence = self.compass.step(angular_velocity=ang_vel)
        disoriented = self.compass.is_disoriented()
        
        # 2. Update neuropil arbitration
        arb_res = self.arbiter.arbitrate(obs)
        winning_channel = arb_res["winning_channel"]
        drives = arb_res["homeostatic_drives"]
        
        selected_action = 0
        action_source = "core_policy"
        active_opcode = None
        
        # 3. Check optomotor reflex if disoriented
        if disoriented:
            orientation_error = math.atan2(math.sin(flow_angle - pva_heading), math.cos(flow_angle - pva_heading))
            selected_action = self.compass.optomotor_reflex_action(self.last_action, orientation_error)
            action_source = "optomotor_reflex"
        # 4. Check active macro-sequence from Tessera
        elif self.active_macro_sequence:
            selected_action = self.active_macro_sequence.pop(0)
            action_source = "tessera_macro"
        # 5. Check if we should initiate a ratified Tessera macro-action
        elif self.tessera.ratified and stamina > 0.40 and not disoriented:
            # Pick best ratified opcode that satisfies safety
            best_op = None
            best_delta = -1.0
            for op_id, op_data in self.tessera.ratified.items():
                if op_data["delta_r_mean"] > best_delta:
                    if self.tessera.validate_safety(op_data["actions"], stamina, depth):
                        best_op = op_data
                        best_delta = op_data["delta_r_mean"]
            if best_op is not None and len(best_op["actions"]) > 0:
                self.active_macro_sequence = list(best_op["actions"])
                selected_action = self.active_macro_sequence.pop(0)
                action_source = "tessera_macro"
                active_opcode = best_op["opcode_id"]
        
        # 6. Fallback to descending neuropil bias over core policy
        if action_source == "core_policy":
            core_action = int(self.core.act(obs))
            # Neuropil arbitration modulation
            if winning_channel == "escaper" and drives["threat"] > 0.5:
                # Upwards escape towards shallower water
                selected_action = 1
                action_source = "escaper_override"
            elif winning_channel == "sentinel" and drives["fatigue"] > 0.8:
                # Rest recovery
                selected_action = 0
                action_source = "sentinel_rest"
            elif winning_channel == "harvester" and drives["metabolic"] > 0.6 and stamina > 0.25:
                # Forage action
                selected_action = 5
                action_source = "harvester_forage"
            else:
                selected_action = core_action
                action_source = "core_policy"

        self.last_action = selected_action
        
        return {
            "action": int(selected_action),
            "action_source": str(action_source),
            "winning_channel": str(winning_channel),
            "pva_heading": round(float(pva_heading), 4),
            "pva_coherence": round(float(pva_coherence), 4),
            "disoriented": bool(disoriented),
            "homeostatic_drives": drives,
            "active_opcode": active_opcode,
            "confidence": arb_res["confidence"],
            "step": int(obs[15]) if len(obs) > 15 else 0,
        }

    def act(self, observation: list[float] | np.ndarray) -> int:
        return int(self.plan(observation)["action"])
