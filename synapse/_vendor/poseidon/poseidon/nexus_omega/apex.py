"""Stage 12 POSEIDON NEXUS-APEX Master Cognitive Orchestrator.

Unifies:
1. Quantum Causal Superposition & Berry Phase Holonomy (C^4 Hilbert Space)
2. Nonstationary Phase-Field Resonator (Dual-Rate Frequency & Schedule Drift Tracking)
3. Contrastive Causal Memory Fabric (Counterfactual Advantage & Predictive Vitality Guard)
4. Metacognitive SCM Reasoning Monologue (GhostInTheMachine + Causeway do-calculus)
5. Metamorphic Bytecode Weaver & LTL Safety Shield (Formal Buchi Automata Verification)
6. Non-equilibrium Autopoiesis Thermodynamic Field (Prigogine Entropy Production)
7. Morpheus Dream Consolidation Engine (Offline Counterfactual Memory Replay)
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

from .contracts import (
    ApexDecision,
    CognitiveMonologue,
    HarmonicSpectralProfile,
    HierarchicalEFEDecision,
    MacroIntent,
    QuantumCausalState,
    SynthesizedOpcode,
    ThermodynamicState,
    DreamConsolidationReceipt,
)
from .quantum_causal import QuantumCausalKernel
from .phase_field import NonstationaryPhaseFieldResonator, NonstationaryPhaseState
from .contrastive_memory import ContrastiveCausalMemoryFabric, ContrastiveMemoryDecision
from .reasoning import ApexReasoningEngine
from .bytecode_weaver import MetamorphicBytecodeWeaver, LTLSafetyShield
from .autopoiesis_field import AutopoiesisThermodynamicField
from .morpheus_dream import MorpheusDreamEngine


class NexusApexEngine:
    """Master Stage 12 Omni-Adaptive Metamorphic Phase-Field Cognitive Engine."""

    ACTION_NAMES = ("forage", "shelter", "explore", "rest")

    def __init__(
        self,
        seed: int = 42,
        initial_interval: float = 8.0,
        reserve_floor: float = 0.27,
    ) -> None:
        self.seed = int(seed)
        self.quantum_kernel = QuantumCausalKernel(tunneling_rate=0.18)
        self.phase_field = NonstationaryPhaseFieldResonator(initial_interval=initial_interval)
        self.contrastive_memory = ContrastiveCausalMemoryFabric(reserve_floor=reserve_floor)
        self.reasoning_engine = ApexReasoningEngine(seed=seed)
        self.bytecode_weaver = MetamorphicBytecodeWeaver()
        self.ltl_shield = LTLSafetyShield()
        self.autopoiesis_field = AutopoiesisThermodynamicField()
        self.morpheus_dream = MorpheusDreamEngine()

        self.last_decision: Optional[ApexDecision] = None
        self.last_action: int = 0
        self.last_reward: float = 0.0
        self.step_count: int = 0

    def reset(self) -> None:
        self.quantum_kernel.reset()
        self.phase_field.reset()
        self.contrastive_memory.reset()
        self.autopoiesis_field.reset()
        self.last_decision = None
        self.last_action = 0
        self.last_reward = 0.0
        self.step_count = 0

    def decide(
        self,
        obs: Sequence[float],
        base_decision: Optional[HierarchicalEFEDecision] = None,
        last_action: Optional[int] = None,
        last_reward: float = 0.0,
    ) -> ApexDecision:
        """Executes full Stage 12 arbitration cycle."""
        self.step_count += 1
        obs_arr = np.asarray(obs, dtype=np.float32)
        cue = float(obs_arr[3]) if len(obs_arr) > 3 else 0.0
        is_pulse = cue > 0.8

        # 1. Nonstationary Phase-Field Resonator
        phase_state = self.phase_field.step(cue_or_exposure=cue, is_pulse_observed=is_pulse)

        # 2. Determine base proposed action
        if base_decision is not None:
            proposed_micro = base_decision.micro_action
            macro_intent = base_decision.macro_intent
        else:
            proposed_micro = 0
            macro_intent = MacroIntent.FORAGE_EXPEDITION

        # 3. Contrastive Causal Memory arbitration
        # Candidate action can come from memory or proposed action
        contrastive_dec = self.contrastive_memory.arbitrate(
            obs=obs_arr,
            candidate_action=proposed_micro,
            baseline_action=proposed_micro,
            predicted_advantage=0.10 if not is_pulse else -0.50,
        )

        # 4. Quantum Causal Superposition over Macro-Intents
        free_energy_landscape = [0.2, 0.4, 0.5, 0.3]
        if phase_state.phase_boundary_latch_active:
            # Shift free energy in favor of diapause / emergency shelter
            free_energy_landscape = [0.8, 0.2, 0.7, 0.1]
        quantum_state = self.quantum_kernel.evolve_step(free_energy_landscape, dt=0.20)

        # 5. Non-equilibrium Autopoiesis Thermodynamic Field
        thermo_state = self.autopoiesis_field.update(
            obs=obs_arr,
            epistemic_uncertainty=0.05,
        )

        # 6. Metamorphic Bytecode Weaver & LTL Shield
        active_opcode = self.bytecode_weaver.synthesize(
            intent=macro_intent,
            current_obs=obs_arr,
        )
        ltl_safe = active_opcode.ltl_safe if active_opcode is not None else True

        # 7. Final Action Arbitration
        arbitration_source = "hierarchical_efe"
        final_action = contrastive_dec.final_action

        if phase_state.phase_boundary_latch_active:
            # Proactive boundary latch enforces Rest/Shelter before hazard strikes
            energy = float(obs_arr[1]) if len(obs_arr) > 1 else 1.0
            hydration = float(obs_arr[2]) if len(obs_arr) > 2 else 1.0
            final_action = 3 if hydration < 0.98 or energy < 0.75 else 1
            arbitration_source = "phase_boundary_latch"
        elif contrastive_dec.predictive_guard_override:
            arbitration_source = "predictive_vitality_guard"
        elif contrastive_dec.balance_deadband_override:
            arbitration_source = "risk_adaptive_balance_deadband"
        elif contrastive_dec.memory_override:
            arbitration_source = "contrastive_memory_advantage"
        elif not ltl_safe:
            final_action = 3  # Safe rest fallback under LTL violation
            arbitration_source = "ltl_safety_shield"
        elif quantum_state.probabilities[3] > 0.6:  # Strong emergency tunneling
            final_action = 1
            arbitration_source = "quantum_tunneling"

        # 8. Metacognitive Reasoning Monologue (GhostInTheMachine + Causeway SCM)
        monologue = self.reasoning_engine.reason(
            obs=obs_arr,
            last_action=last_action if last_action is not None else self.last_action,
            last_reward=last_reward,
            phase_drift_velocity=phase_state.drift_velocity,
            estimated_interval=phase_state.estimated_interval,
            phase_latch_active=phase_state.phase_boundary_latch_active,
            predictive_guard_active=contrastive_dec.predictive_guard_active,
            balance_deadband_active=contrastive_dec.balance_deadband_active,
            selected_action=final_action,
        )

        # Construct dummy HierarchicalEFEDecision if none provided
        if base_decision is None:
            base_decision = HierarchicalEFEDecision(
                macro_intent=macro_intent,
                macro_intent_remaining_ticks=4,
                micro_action=final_action,
                micro_action_name=self.ACTION_NAMES[final_action],
                macro_free_energy=0.25,
                micro_free_energy=0.15,
                conformal_lcb=0.10,
                conformal_override=False,
            )

        decision = ApexDecision(
            quantum_state=quantum_state,
            nonstationary_phase=phase_state.to_dict(),
            contrastive_memory=contrastive_dec.to_dict(),
            cognitive_monologue=monologue,
            thermodynamic_state=thermo_state,
            hierarchical_decision=base_decision,
            active_opcode=active_opcode,
            dream_receipt=None,
            final_action=final_action,
            final_action_name=self.ACTION_NAMES[final_action],
            arbitration_source=arbitration_source,
            timestamp=time.time(),
        )

        self.last_decision = decision
        self.last_action = final_action
        self.last_reward = last_reward
        return decision
