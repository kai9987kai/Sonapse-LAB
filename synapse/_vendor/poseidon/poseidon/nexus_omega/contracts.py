"""PROJECT NEXUS-Ω: Core Interfaces, Data Contracts, and Formal Protocols.

Defines typed schemas for observations, memory carriers, interventions,
world models, audit receipts, and scientific hypotheses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np


class InterventionType(str, Enum):
    INTACT = "intact"
    ABLATE_CARRIER = "ablate_carrier"
    SHUFFLE_CARRIER = "shuffle_carrier"
    TRANSPLANT_CARRIER = "transplant_carrier"
    RECONSTRUCT_CARRIER = "reconstruct_carrier"
    CLAMP_INPUT = "clamp_input"


class MemoryCarrierType(str, Enum):
    WORKING = "working"         # Short-horizon bounded scratchpad
    EPISODIC = "episodic"       # Key-value trajectory store
    STRUCTURAL = "structural"   # Latent topology / causal adjacency
    PARAMETRIC = "parametric"   # Slow synaptic / weight reserves


@dataclass
class StructuredObservation:
    """Layer 1: Structured Observation Contract."""
    source_id: str
    step: int
    data: np.ndarray
    uncertainty: np.ndarray
    timestamp: float = field(default_factory=time.time)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            h = hashlib.sha256()
            h.update(self.source_id.encode("utf-8"))
            h.update(str(self.step).encode("utf-8"))
            h.update(self.data.tobytes())
            self.provenance_hash = h.hexdigest()[:16]


@dataclass
class CarrierState:
    """Layer 2: Multi-substrate memory carrier state."""
    working_state: np.ndarray        # Shape: (d_working,)
    episodic_keys: List[np.ndarray] = field(default_factory=list)
    episodic_values: List[np.ndarray] = field(default_factory=list)
    structural_graph: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # Adjacency matrix
    step_count: int = 0
    intervention_log: List[str] = field(default_factory=list)

    def clone(self) -> CarrierState:
        return CarrierState(
            working_state=self.working_state.copy(),
            episodic_keys=[k.copy() for k in self.episodic_keys],
            episodic_values=[v.copy() for v in self.episodic_values],
            structural_graph=self.structural_graph.copy(),
            step_count=self.step_count,
            intervention_log=list(self.intervention_log),
        )


@dataclass
class PredictionOutput:
    """Layer 4: World model transition prediction with epistemic uncertainty."""
    predicted_obs: np.ndarray
    predicted_reward: float
    predicted_done: float
    epistemic_uncertainty: float
    ensemble_variance: np.ndarray
    comparator_error: Optional[np.ndarray] = None


@dataclass
class InterventionSpec:
    """Layer 5: Formal Causal Intervention Specification."""
    intervention_type: InterventionType
    target_carrier: Optional[MemoryCarrierType] = None
    target_indices: Optional[List[int]] = None
    donor_state: Optional[np.ndarray] = None
    description: str = ""


@dataclass
class HypothesisSpec:
    """Scientific hypothesis with explicit falsification conditions."""
    hypothesis_id: str
    statement: str
    null_hypothesis: str
    independent_variable: str
    dependent_variable: str
    alpha: float = 0.05
    minimum_effect_size: float = 0.05
    falsification_condition: str = ""


@dataclass
class AuditReceipt:
    """Layer 7: Independent Scientific Audit Receipt."""
    audit_id: str
    hypothesis_id: str
    timestamp: str
    seeds_evaluated: List[int]
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    p_value: float
    effect_size: float
    retained_in_library: bool
    verdict: str  # "ACCEPTED", "FALSIFIED", "INCONCLUSIVE"
    receipt_hash: str = ""

    def seal(self) -> str:
        payload = {
            "audit_id": self.audit_id,
            "hypothesis_id": self.hypothesis_id,
            "timestamp": self.timestamp,
            "seeds": self.seeds_evaluated,
            "baseline": self.baseline_metrics,
            "candidate": self.candidate_metrics,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "verdict": self.verdict,
        }
        self.receipt_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.receipt_hash


class DevelopmentalStage(str, Enum):
    PROTOCELL = "protocell"
    LARVAL_FORAGER = "larval_forager"
    PUPAL_DIAPAUSE = "pupal_diapause"
    IMAGO_NAVIGATOR = "imago_navigator"
    SENESCENT_ARCHIVE = "senescent_archive"


@dataclass(frozen=True)
class EpigeneticProfile:
    stage: DevelopmentalStage
    basal_metabolism: float        # Multiplier on vital decay rates
    sensory_gain: float            # Multiplier on resource / gradient cues
    risk_penalty_scale: float      # Horizon-X lambda risk factor
    plasticity_rate: float         # Weight/adapter adaptation learning rate
    mobility_efficiency: float     # Stamina translation efficiency
    receptive_horizon: float       # Spatial perception radius scale


@dataclass
class ExuviaShell:
    """Cryptographic trace of an ecdysis event shedding an old morphological profile."""
    from_stage: DevelopmentalStage
    to_stage: DevelopmentalStage
    step: int
    biomass_accumulated: float
    health_at_ecdysis: float
    stamina_at_ecdysis: float
    shell_sha256: str = ""

    def __post_init__(self):
        if not self.shell_sha256:
            payload = {
                "from_stage": self.from_stage.value,
                "to_stage": self.to_stage.value,
                "step": self.step,
                "biomass": round(float(self.biomass_accumulated), 4),
                "health": round(float(self.health_at_ecdysis), 4),
                "stamina": round(float(self.stamina_at_ecdysis), 4),
            }
            self.shell_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class CounterfactualQuery:
    factual_state: np.ndarray
    factual_action: int
    factual_next_state: np.ndarray
    counterfactual_action: int
    exogenous_noise: Optional[np.ndarray] = None


@dataclass
class CounterfactualResult:
    query: CounterfactualQuery
    counterfactual_predicted_state: np.ndarray
    regret: float
    preventable_by_policy: bool
    attribution: str  # "policy_error", "exogenous_shock", "unavoidable_attrition"


@dataclass
class DiscoveryResult:
    hypothesis: HypothesisSpec
    baseline_reward: float
    candidate_reward: float
    reward_advantage: float
    p_value: float
    effect_size: float
    verdict: str  # "ACCEPTED", "REFUTED", or "INCONCLUSIVE"
    seeds_evaluated: List[int]
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    adjusted_p_value: Optional[float] = None
    paired_outcomes: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class DiscoveryLedger:
    ledger_id: str
    timestamp: str
    ratified_laws: List[Dict[str, Any]] = field(default_factory=list)
    refutation_graveyard: List[Dict[str, Any]] = field(default_factory=list)
    ledger_sha256: str = ""

    def seal(self) -> str:
        payload = {
            "ledger_id": self.ledger_id,
            "timestamp": self.timestamp,
            "ratified": self.ratified_laws,
            "refutations": self.refutation_graveyard,
        }
        self.ledger_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.ledger_sha256


@dataclass(frozen=True)
class FactoredState:
    """Factored Causal Latent Space separating physiological and environmental invariants."""
    endogenous_vitals: np.ndarray      # Shape (4,): health, energy, hydration, stamina
    exogenous_field: np.ndarray        # Shape (4,): exposure, storm_cue, food_scarcity, water_scarcity
    causal_regime: np.ndarray          # Shape (2,): storm_persistence, replenishment_rate


@dataclass
class FactoredPredictionOutput:
    """Prediction output with partitioned epistemic uncertainty."""
    predicted_obs: np.ndarray
    predicted_reward: float
    endogenous_pred: np.ndarray
    exogenous_pred: np.ndarray
    regime_pred: np.ndarray
    epistemic_uncertainty_total: float
    endogenous_uncertainty: float
    exogenous_uncertainty: float
    regime_uncertainty: float
    comparator_error: Optional[np.ndarray] = None


@dataclass
class CausalEFEDecision:
    """Decision telemetry for active inference and heuristic safety actions."""
    action: int
    action_name: str
    free_energy_objective: float
    pragmatic_value: float
    epistemic_value: float
    epistemic_weight: float
    empowerment_value: float
    conformal_lcb: float
    conformal_override: bool
    factored_state: Optional[Dict[str, List[float]]] = None
    timestamp: float = field(default_factory=time.time)
    action_scores: List[Dict[str, Any]] = field(default_factory=list)
    recovery_admission: bool = False
    hazard_guard_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": int(self.action),
            "action_name": str(self.action_name),
            "free_energy_objective": round(float(self.free_energy_objective), 6),
            "pragmatic_value": round(float(self.pragmatic_value), 6),
            "epistemic_value": round(float(self.epistemic_value), 6),
            "epistemic_weight": round(float(self.epistemic_weight), 6),
            "empowerment_value": round(float(self.empowerment_value), 6),
            "conformal_lcb": round(float(self.conformal_lcb), 6),
            "conformal_override": bool(self.conformal_override),
            "recovery_admission": bool(self.recovery_admission),
            "hazard_guard_override": bool(self.hazard_guard_override),
            "factored_state": self.factored_state or {},
            "timestamp": round(float(self.timestamp), 4),
            "action_scores": self.action_scores,
        }


class MacroIntent(str, Enum):
    FORAGE_EXPEDITION = "forage_expedition"         # Aggressive nutrient acquisition
    DIAPAUSE_HIBERNATION = "diapause_hibernation"   # Metabolic depression during acute hazard
    HABITAT_RELOCATION = "habitat_relocation"       # Spatial exploration to escape local scarcity
    EMERGENCY_FORTIFICATION = "emergency_fortification" # Defensive shelter reinforcement


@dataclass
class ImaginedTrajectory:
    """Matched raw and heuristic-guided rollouts over one action sequence."""
    horizon: int
    states: List[np.ndarray]              # Shape (H, d_obs)
    actions: List[int]                   # Length H
    rewards: List[float]                 # Length H
    epistemic_uncertainties: List[float] # Length H
    cumulative_reward: float
    conformal_trajectory_lcb: float
    score_guidance_norm: float
    macro_intent: MacroIntent = MacroIntent.FORAGE_EXPEDITION
    unguided_states: List[np.ndarray] = field(default_factory=list)
    unguided_rewards: List[float] = field(default_factory=list)
    unguided_epistemic_uncertainties: List[float] = field(default_factory=list)
    direct_guidance_adjustments: List[np.ndarray] = field(default_factory=list)
    branch_state_deltas: List[np.ndarray] = field(default_factory=list)
    branch_divergence_l2: List[float] = field(default_factory=list)
    unguided_cumulative_reward: float = 0.0
    unguided_return_penalty: float = 0.0
    predictor_evaluations: int = 0
    ensemble_member_forwards: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "poseidon-matched-rollout-v1",
            "predictor_status": "untrained_experimental_ensemble",
            "horizon": int(self.horizon),
            "macro_intent": self.macro_intent.value if hasattr(self.macro_intent, "value") else str(self.macro_intent),
            "states": [[round(float(x), 4) for x in s] for s in self.states],
            "actions": [int(a) for a in self.actions],
            "rewards": [round(float(r), 4) for r in self.rewards],
            "epistemic_uncertainties": [round(float(u), 4) for u in self.epistemic_uncertainties],
            "cumulative_reward": round(float(self.cumulative_reward), 4),
            "conformal_trajectory_lcb": round(float(self.conformal_trajectory_lcb), 4),
            "conformal_return_lcb": round(float(self.conformal_trajectory_lcb), 4),
            "guided_return_penalty": round(float(self.conformal_trajectory_lcb), 4),
            "uncertainty_calibrated": False,
            "score_guidance_norm": round(float(self.score_guidance_norm), 4),
            "unguided_states": [[round(float(x), 4) for x in s] for s in self.unguided_states],
            "unguided_rewards": [round(float(r), 4) for r in self.unguided_rewards],
            "unguided_epistemic_uncertainties": [round(float(u), 4) for u in self.unguided_epistemic_uncertainties],
            "direct_guidance_adjustments": [[round(float(x), 5) for x in row] for row in self.direct_guidance_adjustments],
            "branch_state_deltas": [[round(float(x), 5) for x in row] for row in self.branch_state_deltas],
            "branch_divergence_l2": [round(float(x), 5) for x in self.branch_divergence_l2],
            "max_branch_divergence_l2": round(max(self.branch_divergence_l2, default=0.0), 5),
            "unguided_cumulative_reward": round(float(self.unguided_cumulative_reward), 4),
            "unguided_return_penalty": round(float(self.unguided_return_penalty), 4),
            "unguided_uncertainty_calibrated": False,
            "guided_return_difference": round(float(self.cumulative_reward - self.unguided_cumulative_reward), 4),
            "predictor_evaluations": int(self.predictor_evaluations),
            "ensemble_member_forwards": int(self.ensemble_member_forwards),
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class HierarchicalEFEDecision:
    """Rich hierarchical decision across macro-intentional and micro-action levels."""
    macro_intent: MacroIntent
    macro_intent_remaining_ticks: int
    micro_action: int
    micro_action_name: str
    macro_free_energy: float
    micro_free_energy: float
    conformal_lcb: float
    conformal_override: bool
    imagined_trajectory: Optional[ImaginedTrajectory] = None
    factored_state: Optional[Dict[str, List[float]]] = None
    timestamp: float = field(default_factory=time.time)
    action_scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_intent": self.macro_intent.value,
            "macro_intent_remaining_ticks": int(self.macro_intent_remaining_ticks),
            "micro_action": int(self.micro_action),
            "micro_action_name": str(self.micro_action_name),
            "macro_free_energy": round(float(self.macro_free_energy), 6),
            "micro_free_energy": round(float(self.micro_free_energy), 6),
            "conformal_lcb": round(float(self.conformal_lcb), 6),
            "conformal_override": bool(self.conformal_override),
            "imagined_trajectory": self.imagined_trajectory.to_dict() if self.imagined_trajectory else None,
            "factored_state": self.factored_state or {},
            "timestamp": round(float(self.timestamp), 4),
            "action_scores": self.action_scores,
        }


# ============================================================================
# STAGE 11: POSEIDON NEXUS-ULTIMA (OMNI-RESONANCE COGNITIVE ARCHITECTURE)
# ============================================================================

@dataclass
class QuantumCausalState:
    """Holonomic quantum superposition state over macro-intent Hilbert space C^4."""
    amplitudes: List[complex]              # 4 complex amplitudes (c_0, c_1, c_2, c_3)
    probabilities: List[float]             # Born rule probabilities |c_m|^2
    phases: List[float]                    # Phase angles theta_m in [-pi, pi]
    berry_phase: float                     # Geometric holonomic Berry phase accumulated along trajectory loop
    purity: float                          # Tr(rho^2) state purity in [0.25, 1.0]
    von_neumann_entropy: float             # -Tr(rho log rho)
    tunneling_rate: float                  # Inter-intent superposition tunneling amplitude
    timestamp: float = field(default_factory=time.time)

    @property
    def berry_phase_holonomy(self) -> float:
        return self.berry_phase

    @property
    def complex_amplitudes(self) -> List[complex]:
        return self.amplitudes

    @property
    def born_probabilities(self) -> List[float]:
        return self.probabilities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amplitudes_real": [round(float(c.real), 5) for c in self.amplitudes],
            "amplitudes_imag": [round(float(c.imag), 5) for c in self.amplitudes],
            "probabilities": [round(float(p), 5) for p in self.probabilities],
            "born_probabilities": [round(float(p), 5) for p in self.probabilities],
            "phases": [round(float(th), 5) for th in self.phases],
            "berry_phase": round(float(self.berry_phase), 6),
            "berry_phase_holonomy": round(float(self.berry_phase), 6),
            "purity": round(float(self.purity), 5),
            "von_neumann_entropy": round(float(self.von_neumann_entropy), 5),
            "tunneling_rate": round(float(self.tunneling_rate), 5),
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class HarmonicSpectralProfile:
    """Bio-Harmonic Fourier spectral frequency profile for cyclical pulse forecasting."""
    dominant_frequency: float              # Primary cyclical frequency omega_dom (rad/tick)
    dominant_period: float                 # Estimated period T = 2*pi / omega_dom (ticks)
    spectral_power: List[float]            # Power spectral density across frequency bins
    current_phase: float                   # Current cyclical phase in [-pi, pi]
    predicted_phase_advance: float         # Anticipated phase advance over horizon H
    spectral_entropy: float                # Spectral disorder / coherence measure
    storm_resonance_probability: float     # Resonant anticipation probability for upcoming hazard pulse
    timestamp: float = field(default_factory=time.time)

    @property
    def fourier_amplitudes(self) -> List[float]:
        return self.spectral_power

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_frequency": round(float(self.dominant_frequency), 5),
            "dominant_period": round(float(self.dominant_period), 3),
            "spectral_power": [round(float(p), 5) for p in self.spectral_power],
            "fourier_amplitudes": [round(float(p), 5) for p in self.spectral_power],
            "current_phase": round(float(self.current_phase), 5),
            "predicted_phase_advance": round(float(self.predicted_phase_advance), 5),
            "spectral_entropy": round(float(self.spectral_entropy), 5),
            "storm_resonance_probability": round(float(self.storm_resonance_probability), 5),
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class SynthesizedOpcode:
    """Metamorphic composite bytecode instruction formally verified under LTL."""
    opcode_id: str
    opcode_name: str
    micro_action_sequence: List[int]       # Concrete macro-unrolled micro actions
    ltl_formula: str                       # e.g., "G(health > 0.20) & (storm -> F[<=2] shelter)"
    ltl_verified: bool                     # True if verified by safety automaton
    verification_hash: str                 # Cryptographic verification receipt SHA-256
    execution_horizon: int
    homeostatic_cost: float
    timestamp: float = field(default_factory=time.time)

    @property
    def execution_steps(self) -> List[int]:
        return self.micro_action_sequence

    @property
    def ltl_safe(self) -> bool:
        return self.ltl_verified

    @property
    def verification_certificate(self) -> str:
        return self.verification_hash

    @property
    def mutated(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opcode_id": self.opcode_id,
            "opcode_name": self.opcode_name,
            "micro_action_sequence": [int(a) for a in self.micro_action_sequence],
            "execution_steps": [int(a) for a in self.micro_action_sequence],
            "ltl_formula": self.ltl_formula,
            "ltl_verified": bool(self.ltl_verified),
            "ltl_safe": bool(self.ltl_verified),
            "verification_hash": self.verification_hash,
            "verification_certificate": self.verification_hash,
            "mutated": False,
            "execution_horizon": int(self.execution_horizon),
            "homeostatic_cost": round(float(self.homeostatic_cost), 4),
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class ThermodynamicState:
    """Nonequilibrium autopoietic dissipation and PAD affect valence."""
    entropy_production_rate: float         # dS/dt internal entropy production
    heat_dissipation_rate: float           # Prigogine dissipative rate
    pleasure: float                        # P in [-1, 1]: vital reserve surplus
    arousal: float                         # A in [-1, 1]: epistemic prediction error intensity
    dominance: float                       # D in [-1, 1]: empowerment / agency controllability
    effective_curiosity_temp: float        # Beta_curiosity modulated by PAD
    effective_risk_aversion: float         # Lambda_risk modulated by PAD
    autopoietic_integrity: float           # Viability distance from fatal boundary
    timestamp: float = field(default_factory=time.time)

    @property
    def internal_entropy_production(self) -> float:
        return self.entropy_production_rate

    @property
    def autopoietic_viability(self) -> float:
        return self.autopoietic_integrity

    @property
    def affect_valence_pad(self) -> Tuple[float, float, float]:
        return (self.pleasure, self.arousal, self.dominance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entropy_production_rate": round(float(self.entropy_production_rate), 5),
            "internal_entropy_production": round(float(self.entropy_production_rate), 5),
            "heat_dissipation_rate": round(float(self.heat_dissipation_rate), 5),
            "pleasure": round(float(self.pleasure), 4),
            "arousal": round(float(self.arousal), 4),
            "dominance": round(float(self.dominance), 4),
            "affect_valence_pad": [round(float(self.pleasure), 4), round(float(self.arousal), 4), round(float(self.dominance), 4)],
            "effective_curiosity_temp": round(float(self.effective_curiosity_temp), 4),
            "effective_risk_aversion": round(float(self.effective_risk_aversion), 4),
            "autopoietic_integrity": round(float(self.autopoietic_integrity), 4),
            "autopoietic_viability": round(float(self.autopoietic_integrity), 4),
        }


@dataclass

class DreamConsolidationReceipt:
    """Morpheus dream-state sleep consolidation trace during diapause."""
    episodes_dreamed: int
    replayed_transitions: int
    mean_regret_mitigation: float
    synaptic_pruning_ratio: float
    hub_resilience_gain: float
    consolidation_sha256: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes_dreamed": int(self.episodes_dreamed),
            "replayed_transitions": int(self.replayed_transitions),
            "mean_regret_mitigation": round(float(self.mean_regret_mitigation), 5),
            "synaptic_pruning_ratio": round(float(self.synaptic_pruning_ratio), 4),
            "hub_resilience_gain": round(float(self.hub_resilience_gain), 4),
            "consolidation_sha256": self.consolidation_sha256,
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class UltimaDecision:
    """The unified Stage 11 decision spanning quantum, spectral, hierarchical, and metamorphic planes."""
    quantum_state: QuantumCausalState
    harmonic_profile: HarmonicSpectralProfile
    thermodynamic_state: ThermodynamicState
    hierarchical_decision: HierarchicalEFEDecision
    active_opcode: Optional[SynthesizedOpcode] = None
    dream_receipt: Optional[DreamConsolidationReceipt] = None
    final_action: int = 0
    final_action_name: str = "forage"
    holonomic_override: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def spectral_profile(self) -> HarmonicSpectralProfile:
        return self.harmonic_profile

    @property
    def dream_consolidation(self) -> Optional[DreamConsolidationReceipt]:
        return self.dream_receipt

    @property
    def action(self) -> int:
        return self.final_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantum_state": self.quantum_state.to_dict(),
            "harmonic_profile": self.harmonic_profile.to_dict(),
            "spectral_profile": self.harmonic_profile.to_dict(),
            "thermodynamic_state": self.thermodynamic_state.to_dict(),
            "hierarchical_decision": self.hierarchical_decision.to_dict(),
            "active_opcode": self.active_opcode.to_dict() if self.active_opcode else None,
            "dream_receipt": self.dream_receipt.to_dict() if self.dream_receipt else None,
            "dream_consolidation": self.dream_receipt.to_dict() if self.dream_receipt else None,
            "final_action": int(self.final_action),
            "action": int(self.final_action),
            "final_action_name": str(self.final_action_name),
            "holonomic_override": bool(self.holonomic_override),
            "timestamp": round(float(self.timestamp), 4),
        }

# ============================================================================
# STAGE 12: POSEIDON NEXUS-APEX (OMNI-ADAPTIVE METAMORPHIC PHASE-FIELD ARCHITECTURE)
# ============================================================================

@dataclass
class CognitiveMonologue:
    """Metacognitive internal dialogue and structural causal DAG reasoning."""
    thought_chain: List[str]
    causal_hypotheses: List[Dict[str, Any]]
    epistemic_curiosity_gain: float
    associative_lattice_matches: List[str]
    somatic_proprioception: Dict[str, float]
    predicted_regime: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought_chain": self.thought_chain,
            "causal_hypotheses": self.causal_hypotheses,
            "epistemic_curiosity_gain": round(float(self.epistemic_curiosity_gain), 5),
            "associative_lattice_matches": self.associative_lattice_matches,
            "somatic_proprioception": {k: round(float(v), 4) for k, v in self.somatic_proprioception.items()},
            "predicted_regime": self.predicted_regime,
            "timestamp": round(float(self.timestamp), 4),
        }


@dataclass
class ApexDecision:
    """The unified Stage 12 omni-adaptive decision spanning quantum, nonstationary phase-field, contrastive memory, and metacognitive planes."""
    quantum_state: QuantumCausalState
    nonstationary_phase: Dict[str, Any]
    contrastive_memory: Dict[str, Any]
    cognitive_monologue: CognitiveMonologue
    thermodynamic_state: ThermodynamicState
    hierarchical_decision: HierarchicalEFEDecision
    active_opcode: Optional[SynthesizedOpcode] = None
    dream_receipt: Optional[DreamConsolidationReceipt] = None
    final_action: int = 0
    final_action_name: str = "forage"
    arbitration_source: str = "quantum_tunneling"
    timestamp: float = field(default_factory=time.time)

    @property
    def action(self) -> int:
        return self.final_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantum_state": self.quantum_state.to_dict(),
            "nonstationary_phase": self.nonstationary_phase,
            "contrastive_memory": self.contrastive_memory,
            "cognitive_monologue": self.cognitive_monologue.to_dict(),
            "thermodynamic_state": self.thermodynamic_state.to_dict(),
            "hierarchical_decision": self.hierarchical_decision.to_dict(),
            "active_opcode": self.active_opcode.to_dict() if self.active_opcode else None,
            "dream_receipt": self.dream_receipt.to_dict() if self.dream_receipt else None,
            "final_action": int(self.final_action),
            "action": int(self.final_action),
            "final_action_name": str(self.final_action_name),
            "arbitration_source": str(self.arbitration_source),
            "timestamp": round(float(self.timestamp), 4),
        }
