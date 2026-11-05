"""Project PROMETHEUS Discovery Lab: Causal Experiments & Structural Causal Interventions.

Implements Causeway-inspired SCM hypothesis generation and do-calculus evaluation:
Evaluates counterfactual queries P(outcome | do(action)) against empirical snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .object_physics import PhysicalObject, MaterialType, AGENT_MASS


@dataclass
class CausalHypothesis:
    hypothesis_id: str
    target_action: int
    statement: str
    expected_outcome: str
    empirical_evidence_count: int = 0
    falsified_count: int = 0
    supported_count: int = 0
    verdict: str = "UNTESTED"  # "SUPPORTED", "FALSIFIED", "UNTESTED"
    receipt_hash: str = ""

    def evaluate(self, realized_outcome: str) -> None:
        self.empirical_evidence_count += 1
        if realized_outcome == self.expected_outcome:
            self.supported_count += 1
        else:
            self.falsified_count += 1

        if self.supported_count > self.falsified_count * 2 and self.empirical_evidence_count >= 2:
            self.verdict = "SUPPORTED"
        elif self.falsified_count >= 2:
            self.verdict = "FALSIFIED"
        else:
            self.verdict = "TESTING"

        payload = {
            "hypothesis_id": self.hypothesis_id,
            "action": self.target_action,
            "evidence": self.empirical_evidence_count,
            "supported": self.supported_count,
            "falsified": self.falsified_count,
            "verdict": self.verdict,
        }
        self.receipt_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "target_action": self.target_action,
            "statement": self.statement,
            "expected_outcome": self.expected_outcome,
            "evidence_count": self.empirical_evidence_count,
            "supported_count": self.supported_count,
            "falsified_count": self.falsified_count,
            "verdict": self.verdict,
            "receipt_hash": self.receipt_hash,
        }


class DiscoveryCausalEngine:
    """Manages active causal interventions and physical property hypothesis discovery."""

    def __init__(self):
        self.hypotheses: Dict[str, CausalHypothesis] = {}
        self._initialize_core_hypotheses()

    def _initialize_core_hypotheses(self) -> None:
        self.hypotheses["H1_stone_buoyancy"] = CausalHypothesis(
            hypothesis_id="H1_stone_buoyancy",
            target_action=6,  # RELEASE
            statement="Releasing granite stone in river channel will cause it to float",
            expected_outcome="floating",
        )
        self.hypotheses["H2_log_buoyancy"] = CausalHypothesis(
            hypothesis_id="H2_log_buoyancy",
            target_action=6,  # RELEASE
            statement="Releasing pine log in river channel will cause it to float",
            expected_outcome="floating",
        )
        self.hypotheses["H3_stepping_stone_support"] = CausalHypothesis(
            hypothesis_id="H3_stepping_stone_support",
            target_action=2,  # MOVE_EAST
            statement="Stepping onto placed stone in shallow water supports agent mass",
            expected_outcome="supported",
        )
        self.hypotheses["H4_raft_assemblage_support"] = CausalHypothesis(
            hypothesis_id="H4_raft_assemblage_support",
            target_action=9,  # CONNECT
            statement="Connecting multiple logs into a raft provides sufficient buoyant payload for crossing",
            expected_outcome="supported",
        )
        self.hypotheses["H5_enzyme_reinforcement"] = CausalHypothesis(
            hypothesis_id="H5_enzyme_reinforcement",
            target_action=10, # CATALYZE
            statement="Synthesizing and applying bio-resin increases assemblage bond strength to > 0.90",
            expected_outcome="cured",
        )

    def record_intervention(self, action: int, event_info: Dict[str, Any]) -> None:
        """Updates hypotheses matching the performed action."""
        event = event_info.get("event", "")
        if action == 6:  # RELEASE
            if "floating" in event:
                self.hypotheses["H2_log_buoyancy"].evaluate("floating")
                self.hypotheses["H1_stone_buoyancy"].evaluate("sunk")
            elif "settled" in event:
                self.hypotheses["H1_stone_buoyancy"].evaluate("sunk")

        elif action in (0, 1, 2, 3):  # MOVE
            if "moved_onto_stepping_stone" in event:
                self.hypotheses["H3_stepping_stone_support"].evaluate("supported")
            elif "moved_onto_buoyant_raft" in event:
                self.hypotheses["H4_raft_assemblage_support"].evaluate("supported")
            elif "fell_in_deep_water" in event:
                self.hypotheses["H3_stepping_stone_support"].evaluate("unsupported")

        elif action == 9:  # CONNECT
            if "connected" in event:
                self.hypotheses["H4_raft_assemblage_support"].evaluate("supported")

        elif action == 10:  # CATALYZE
            if "catalytic_curing" in event:
                self.hypotheses["H5_enzyme_reinforcement"].evaluate("cured")

    def get_summary(self) -> Dict[str, Any]:
        return {
            hid: hyp.to_dict() for hid, hyp in self.hypotheses.items()
        }
