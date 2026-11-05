"""Test Suite for Stage 11 POSEIDON NEXUS-ULTIMA Omni-Resonant Autopoietic Cognitive Architecture.

Covers:
- BioharmonicResonator (Fourier wavelet spectral analysis)
- QuantumCausalKernel (Unitary evolution, Berry phase holonomy, Born collapse)
- MetamorphicBytecodeWeaver & LTLSafetyShield (LTL safety automata & synthesis)
- AutopoiesisThermodynamicField (Nonequilibrium entropy production & PAD affect)
- MorpheusDreamEngine (Offline generative replay consolidation)
- NexusOmegaAgent integration (use_ultima_engine=True)
- AutonomousDiscoveryEngine Stage 11 hypotheses
"""

import math
import numpy as np
import pytest

from poseidon.nexus_omega.contracts import (
    FactoredState,
    MacroIntent,
    QuantumCausalState,
    HarmonicSpectralProfile,
    SynthesizedOpcode,
    ThermodynamicState,
    DreamConsolidationReceipt,
    UltimaDecision,
)
from poseidon.nexus_omega.spectral_resonator import BioharmonicResonator
from poseidon.nexus_omega.quantum_causal import QuantumCausalKernel
from poseidon.nexus_omega.bytecode_weaver import MetamorphicBytecodeWeaver, LTLSafetyShield
from poseidon.nexus_omega.autopoiesis_field import AutopoiesisThermodynamicField
from poseidon.nexus_omega.morpheus_dream import MorpheusDreamEngine
from poseidon.nexus_omega.causal_world_model import FactoredCausalWorldModel
from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.discovery import AutonomousDiscoveryEngine, verify_discovery_receipt


def test_bioharmonic_resonator():
    resonator = BioharmonicResonator(window_size=16, num_frequency_bins=8)
    
    # Ingest synthetic wave
    profile = None
    for t in range(24):
        # 12-step periodic hazard/storm oscillation
        storm_val = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / 12.0)
        profile = resonator.update(storm_val)
    
    assert isinstance(profile, HarmonicSpectralProfile)
    assert profile.dominant_period > 0.0
    assert 0.0 <= profile.storm_resonance_probability <= 1.0
    assert profile.spectral_entropy >= 0.0
    assert len(profile.fourier_amplitudes) == 8
    
    # Dict serialization
    d = profile.to_dict()
    assert "dominant_period" in d
    assert "storm_resonance_probability" in d


def test_quantum_causal_kernel():
    kernel = QuantumCausalKernel(tunneling_rate=0.15, dephasing_rate=0.02)
    
    # Time evolution with EFE vector for 4 intents
    efe_vector = [-1.2, -0.5, -0.8, -2.0]
    next_state = kernel.evolve_step(efe_vector, dt=0.25)
    assert isinstance(next_state, QuantumCausalState)
    assert len(next_state.complex_amplitudes) == 4
    assert len(next_state.born_probabilities) == 4
    assert pytest.approx(sum(next_state.born_probabilities), abs=1e-4) == 1.0
    assert -math.pi <= next_state.berry_phase_holonomy <= math.pi
    assert next_state.purity > 0.0
    assert next_state.von_neumann_entropy >= 0.0
    
    # Intent collapse
    intent, idx = kernel.collapse_state(force_index=2)
    assert idx == 2
    assert intent == MacroIntent.HABITAT_RELOCATION
    
    d = next_state.to_dict()
    assert "purity" in d
    assert "berry_phase_holonomy" in d


def test_bytecode_weaver_and_ltl_shield():
    weaver = MetamorphicBytecodeWeaver()
    world_model = FactoredCausalWorldModel()
    
    sample_obs = [1.0, 0.9, 0.8, 0.7, 0.05, 0.1, 0.0, 0.0]
    factored_state = world_model.get_factored_state(sample_obs)
    
    # Test opcode synthesis for all macro intents
    for intent in MacroIntent:
        opcode = weaver.synthesize_opcode(intent, factored_state, world_model)
        assert isinstance(opcode, SynthesizedOpcode)
        assert len(opcode.execution_steps) > 0
        assert opcode.ltl_safe is True
        assert len(opcode.verification_certificate) == 64
        d = opcode.to_dict()
        assert "opcode_name" in d
        assert "execution_steps" in d
    
    # Test LTL shield under critical hazard
    shield = LTLSafetyShield(min_health_threshold=0.20)
    unsafe_states = [
        np.array([0.05, 0.1, 0.1, 0.9, 0.9, 0.9, 0.0, 0.0], dtype=np.float32)
    ]
    is_safe, violation_msg = shield.verify_trace(unsafe_states, actions=[0])
    assert is_safe is False
    assert "Violation" in violation_msg


def test_autopoiesis_thermodynamic_field():
    field = AutopoiesisThermodynamicField(base_curiosity_temp=0.80)
    
    # Healthy observation update
    healthy_obs = [1.0, 0.95, 0.90, 0.85, 0.02, 0.05, 0.0, 0.0]
    healthy_state = field.evaluate_state(healthy_obs, epistemic_surprise=0.02, empowerment=0.60)
    assert math.isfinite(healthy_state.internal_entropy_production)
    assert healthy_state.heat_dissipation_rate >= 0.0
    assert healthy_state.autopoietic_viability >= 0.8
    assert len(healthy_state.affect_valence_pad) == 3

    
    # Acute distress observation update
    distress_obs = [0.1, 0.05, 0.05, 0.1, 0.9, 0.9, 1.0, 0.0]
    distress_state = field.evaluate_state(distress_obs, epistemic_surprise=0.80, empowerment=0.10)
    assert distress_state.autopoietic_viability < healthy_state.autopoietic_viability
    assert distress_state.affect_valence_pad[0] < healthy_state.affect_valence_pad[0]
    
    d = healthy_state.to_dict()
    assert "internal_entropy_production" in d
    assert "affect_valence_pad" in d


def test_morpheus_dream_engine():
    dream_engine = MorpheusDreamEngine(dream_horizon=6, synaptic_pruning_threshold=0.05)
    
    agent = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, base_seed=42)
    agent.act([1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0])
    
    recent_transitions = [
        {"obs": [1.0, 0.9, 0.9, 0.8], "action": 1, "reward": 0.5, "regret": 0.4},
        {"obs": [1.0, 0.85, 0.9, 0.8], "action": 0, "reward": 0.2, "regret": 0.3},
    ]
    
    receipt = dream_engine.consolidate(agent.memory, recent_transitions, num_dream_rollouts=2)
    assert isinstance(receipt, DreamConsolidationReceipt)
    assert receipt.replayed_transitions > 0
    assert receipt.mean_regret_mitigation >= 0.0
    assert 0.0 <= receipt.synaptic_pruning_ratio <= 1.0
    assert receipt.hub_resilience_gain >= 0.0
    assert len(receipt.consolidation_sha256) == 64
    
    d = receipt.to_dict()
    assert "consolidation_sha256" in d
    assert "replayed_transitions" in d


def test_nexus_omega_agent_ultima_engine():
    # Backward compatibility: use_ultima_engine=False
    agent_classic = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=False, base_seed=42)
    obs = [1.0, 0.9, 0.9, 0.8, 0.05, 0.1, 0.0, 0.0]
    act_classic = agent_classic.act(obs)
    assert act_classic in (0, 1, 2, 3)
    assert agent_classic.get_last_ultima_decision() is None
    
    # Stage 11: use_ultima_engine=True
    agent_ultima = NexusOmegaAgent(use_hierarchical_efe=True, use_ultima_engine=True, base_seed=42)
    act_ultima = agent_ultima.act(obs)
    assert act_ultima in (0, 1, 2, 3)
    
    ultima_dec = agent_ultima.get_last_ultima_decision()
    assert isinstance(ultima_dec, UltimaDecision)
    assert ultima_dec.quantum_state is not None
    assert ultima_dec.spectral_profile is not None
    assert ultima_dec.thermodynamic_state is not None
    assert ultima_dec.active_opcode is not None
    assert ultima_dec.hierarchical_decision is not None
    assert ultima_dec.action == act_ultima
    
    # Check accessors
    assert agent_ultima.get_quantum_state() is not None
    assert agent_ultima.get_spectral_profile() is not None
    
    # Test dream consolidation during diapause
    receipt = agent_ultima.dream_consolidate(num_rollouts=2)
    assert isinstance(receipt, DreamConsolidationReceipt)
    assert receipt.replayed_transitions > 0
    
    d = ultima_dec.to_dict()
    assert "quantum_state" in d
    assert "spectral_profile" in d
    assert "thermodynamic_state" in d
    assert "active_opcode" in d


def test_discovery_engine_stage11_hypotheses():
    engine = AutonomousDiscoveryEngine(max_steps=8)
    hypotheses = engine.generate_hypotheses()
    
    # Verify that the 4 Stage 11 hypotheses are registered
    hyp_ids = [h.hypothesis_id for h in hypotheses]
    assert "hyp_quantum_holonomic_coherence" in hyp_ids
    assert "hyp_bioharmonic_fourier_resonance" in hyp_ids
    assert "hyp_ltl_metamorphic_bytecode" in hyp_ids
    assert "hyp_morpheus_dream_consolidation" in hyp_ids
    
    # Test interventional trials on Stage 11 hypotheses
    for h_id in [
        "hyp_quantum_holonomic_coherence",
        "hyp_bioharmonic_fourier_resonance",
        "hyp_ltl_metamorphic_bytecode",
        "hyp_morpheus_dream_consolidation",
    ]:
        hyp = next(h for h in hypotheses if h.hypothesis_id == h_id)
        res = engine.test_hypothesis(hyp, seed_base=101, num_seeds=2)
        assert res.hypothesis.hypothesis_id == h_id
        assert len(res.seeds_evaluated) == 2
        assert res.verdict in ("ACCEPTED", "REFUTED", "INCONCLUSIVE")
