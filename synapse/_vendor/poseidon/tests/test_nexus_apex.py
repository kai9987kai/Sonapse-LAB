"""Test Suite for Stage 12 POSEIDON NEXUS-APEX Omni-Adaptive Architecture.

Covers:
- NonstationaryPhaseFieldResonator (Wavelets, dual-rate drift, boundary latch)
- ContrastiveCausalMemoryFabric (24-dim belief, counterfactual advantage, predictive vitality guard)
- ApexReasoningEngine (Metacognitive monologue, Causeway do-calculus, Prometheus curiosity, NexusSearch lattice)
- NexusApexEngine (Master omni-adaptive orchestrator)
- NexusOmegaAgent Apex Integration (act_apex, plan, get_apex_telemetry)
- Runtime & Server Endpoints (apex-state, reason)
"""

import math
import numpy as np
import pytest

from poseidon.nexus_omega.contracts import (
    CognitiveMonologue,
    ApexDecision,
    MacroIntent,
)
from poseidon.nexus_omega.phase_field import (
    NonstationaryPhaseFieldResonator,
    NonstationaryPhaseState,
)
from poseidon.nexus_omega.contrastive_memory import (
    ContrastiveCausalMemoryFabric,
    ContrastiveMemoryDecision,
)
from poseidon.nexus_omega.reasoning import (
    ApexReasoningEngine,
)
from poseidon.nexus_omega.apex import (
    NexusApexEngine,
)
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.runtime import Poseidon


def test_nonstationary_phase_field_resonator():
    resonator = NonstationaryPhaseFieldResonator(window_size=16, num_frequency_bins=8)

    # Ingest baseline 8-step hazard pulse at t=0, 8, 16
    state = None
    for t in range(17):
        pulse = 0.95 if (t % 8 == 0) else 0.05
        state = resonator.update(pulse, current_tick=t)

    assert isinstance(state, NonstationaryPhaseState)
    assert state.estimated_pulse_interval > 0.0
    assert 0.0 <= state.storm_resonance_probability <= 1.0
    assert state.spectral_entropy >= 0.0
    assert len(state.fourier_amplitudes) == 8

    # Test instantaneous lowering upon interval shortening: pulse at t=20 (4 ticks after t=16)
    for t in range(17, 20):
        resonator.update(0.05, current_tick=t)
    state_short = resonator.update(0.95, current_tick=20)
    assert state_short.estimated_pulse_interval <= 5.0

    # Test drift velocity calculation
    assert isinstance(state_short.frequency_drift_velocity, float)

    # Test dictionary serialization
    d = state_short.to_dict()
    assert "estimated_pulse_interval" in d
    assert "frequency_drift_velocity" in d
    assert "phase_boundary_latch_active" in d


def test_contrastive_causal_memory_fabric():
    fabric = ContrastiveCausalMemoryFabric(d_obs=8, d_working=16)

    sample_obs = np.array([0.9, 0.8, 0.7, 0.85, 0.1, 0.2, 0.0, 0.0], dtype=np.float32)
    decision = fabric.evaluate(sample_obs, last_action=0, last_reward=0.2)

    assert isinstance(decision, ContrastiveMemoryDecision)
    assert decision.recommended_action in (0, 1, 2, 3)
    assert isinstance(decision.counterfactual_advantage, float)
    assert decision.predicted_vitality_energy >= 0.0
    assert decision.predicted_vitality_hydration >= 0.0
    assert isinstance(decision.within_deadband, bool)
    assert isinstance(decision.guard_triggered, bool)

    # Serialization
    d = decision.to_dict()
    assert "counterfactual_advantage" in d
    assert "recommended_action" in d
    assert "guard_triggered" in d

    # Test vitality guard triggering when energy and hydration drop low
    low_vitality_obs = np.array([0.2, 0.15, 0.14, 0.9, 0.1, 0.2, 0.0, 0.0], dtype=np.float32)
    guard_decision = fabric.evaluate(low_vitality_obs, last_action=0, last_reward=-0.5)
    assert guard_decision.guard_triggered is True
    # When guard triggers on low reserves, shelter or rest is mandated
    assert guard_decision.recommended_action in (1, 3)


def test_apex_reasoning_engine():
    engine = ApexReasoningEngine(seed=42)

    sample_obs = np.array([0.85, 0.75, 0.80, 0.05, 0.1, 0.05, 0.0, 0.0], dtype=np.float32)
    monologue = engine.deliberate(sample_obs, last_action=0, last_reward=0.4)

    assert isinstance(monologue, CognitiveMonologue)
    assert len(monologue.thought_chain) >= 2
    assert len(monologue.causal_hypotheses) == 4  # one for each action
    assert monologue.epistemic_curiosity_gain >= 0.0
    assert len(monologue.associative_lattice_matches) > 0
    assert "energy" in monologue.somatic_proprioception
    assert "hydration" in monologue.somatic_proprioception
    assert isinstance(monologue.predicted_regime, str)

    # Serialization
    d = monologue.to_dict()
    assert "thought_chain" in d
    assert "causal_hypotheses" in d
    assert "epistemic_curiosity_gain" in d
    assert "predicted_regime" in d


def test_nexus_apex_engine():
    engine = NexusApexEngine(seed=123)

    sample_obs = np.array([1.0, 0.9, 0.9, 0.1, 0.05, 0.05, 0.0, 0.0], dtype=np.float32)
    apex_dec = engine.decide(sample_obs, last_action=0, last_reward=0.3)

    assert isinstance(apex_dec, ApexDecision)
    assert apex_dec.final_action in (0, 1, 2, 3)
    assert apex_dec.action == apex_dec.final_action
    assert apex_dec.final_action_name in ("forage", "shelter", "explore", "rest")
    assert isinstance(apex_dec.arbitration_source, str)
    assert apex_dec.quantum_state is not None
    assert apex_dec.nonstationary_phase is not None
    assert apex_dec.contrastive_memory is not None
    assert apex_dec.cognitive_monologue is not None
    assert apex_dec.thermodynamic_state is not None

    d = apex_dec.to_dict()
    assert "final_action" in d
    assert "arbitration_source" in d
    assert "cognitive_monologue" in d
    assert "nonstationary_phase" in d
    assert "contrastive_memory" in d


def test_nexus_omega_agent_apex_integration():
    agent = NexusOmegaAgent(
        use_hierarchical_efe=True,
        use_ultima_engine=True,
        use_apex_engine=True,
        base_seed=42,
    )

    sample_obs = [1.0, 0.9, 0.85, 0.05, 0.1, 0.0, 0.0, 0.0]

    # act() delegates to act_apex() when use_apex_engine=True
    action = agent.act(sample_obs)
    assert action in (0, 1, 2, 3)
    assert agent.last_apex_decision is not None

    # plan() returns dictionary with action and Stage 12 decisions
    plan_dict = agent.plan(sample_obs)
    assert isinstance(plan_dict, dict)
    assert "action" in plan_dict
    assert "cognitive_monologue" in plan_dict
    assert "arbitration_source" in plan_dict

    # get_apex_telemetry()
    telem = agent.get_apex_telemetry(sample_obs)
    assert isinstance(telem, ApexDecision)

    # agent reset
    agent.reset()
    assert agent.last_apex_decision is None


def test_runtime_apex_endpoints():
    runtime = Poseidon(".")

    status = runtime.nexus_omega_status()
    assert status.get("nexus_apex") is True
    assert "nonstationary_phase_field_resonator" in status.get("subsystems", [])
    assert "contrastive_causal_memory_fabric" in status.get("subsystems", [])
    assert "metacognitive_reasoning_engine" in status.get("subsystems", [])
    assert "nexus_apex_engine" in status.get("subsystems", [])

    apex_state = runtime.nexus_omega_apex_state(seed=42)
    assert isinstance(apex_state, dict)
    assert "action" in apex_state
    assert "decision" in apex_state
    assert "cognitive_monologue" in apex_state
    assert "nonstationary_phase" in apex_state
    assert "contrastive_memory" in apex_state
    assert "arbitration_source" in apex_state

    reasoning = runtime.nexus_omega_reason(seed=42)
    assert isinstance(reasoning, dict)
    assert "thought_chain" in reasoning
    assert "causal_hypotheses" in reasoning
    assert "epistemic_curiosity_gain" in reasoning
    assert "predicted_regime" in reasoning
