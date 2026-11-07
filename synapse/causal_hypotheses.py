"""Causal Hypotheses & Mechanism Disentanglement Representations.

Distinguishes between:
1. Observational prediction: P(s_{t+1} | s_t, a_t)
2. Interventional prediction: P(s_{t+1} | do(a_t = a), s_t)
3. Counterfactual explanation: P(s'_{t+1} | do(a'_t), s_t, a_t, s_{t+1})

Represents discrete competing causal hypotheses for environmental mechanics:
- H1: "Object movement caused by agent physical thrust contact"
- H2: "Object movement caused by passive surface conveyor current"
- H3: "Gate open state caused by pressure plate depression"
- H4: "Sensor discrepancy caused by actuator slip/fault vs external collision"
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class CausalHypothesis:
    hypothesis_id: str
    description: str
    mechanism_type: str  # 'agent_induced' | 'environmental_conveyor' | 'interactive_mechanism' | 'actuator_fault'
    prior_probability: float
    posterior_probability: float = 0.5
    evidence_count: int = 0
    intervention_signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CausalHypothesisBank:
    """Manages Bayesian belief updates across competing causal explanations."""

    def __init__(self, seed: int = 42, **kwargs):
        self.hypotheses: Dict[str, CausalHypothesis] = {
            "H_thrust_contact": CausalHypothesis(
                hypothesis_id="H_thrust_contact",
                description="Object acceleration caused by agent body contact thrust",
                mechanism_type="agent_induced",
                prior_probability=0.50,
            ),
            "H_conveyor_current": CausalHypothesis(
                hypothesis_id="H_conveyor_current",
                description="Object movement driven by passive terrain current or sliding conveyor",
                mechanism_type="environmental_conveyor",
                prior_probability=0.25,
            ),
            "H_plate_mechanism": CausalHypothesis(
                hypothesis_id="H_plate_mechanism",
                description="Barrier gate state controlled by button / pressure plate trigger",
                mechanism_type="interactive_mechanism",
                prior_probability=0.75,
            ),
            "H_actuator_fault": CausalHypothesis(
                hypothesis_id="H_actuator_fault",
                description="Trajectory deviation caused by internal actuator motor slip rather than external perturbation",
                mechanism_type="actuator_fault",
                prior_probability=0.20,
            ),
        }

    def update_beliefs(self, observed_transition: Dict[str, Any]) -> Dict[str, float]:
        agent_thrust = float(np.linalg.norm(observed_transition.get("action", [0.0])[:2]))
        d_pos = float(np.linalg.norm(observed_transition.get("delta_pos", [0.0])))
        near_plate = bool(observed_transition.get("near_plate", False))
        gate_opened = bool(observed_transition.get("gate_opened", False))

        # Likelihood updates
        # 1. Thrust contact vs conveyor
        p_thrust = self.hypotheses["H_thrust_contact"].posterior_probability
        if agent_thrust > 0.2 and d_pos > 0.05:
            p_thrust = min(0.98, p_thrust * 1.35)
        elif agent_thrust < 0.05 and d_pos > 0.05:
            p_thrust = max(0.05, p_thrust * 0.45)
        self.hypotheses["H_thrust_contact"].posterior_probability = p_thrust

        p_conveyor = 1.0 - p_thrust
        self.hypotheses["H_conveyor_current"].posterior_probability = p_conveyor

        # 2. Gate plate mechanism
        p_plate = self.hypotheses["H_plate_mechanism"].posterior_probability
        if near_plate and gate_opened:
            p_plate = min(0.99, p_plate * 1.5)
        self.hypotheses["H_plate_mechanism"].posterior_probability = p_plate

        return {k: round(h.posterior_probability, 4) for k, h in self.hypotheses.items()}
