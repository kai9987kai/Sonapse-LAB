"""POSEIDON HYPER-AGENT: Apex v3.0 Causal-Quantum Morphogenetic Cognitive Agent.

Synthesizes the complete portfolio breakthrough lineages:
1. CAUSEWAY: Structural Causal Model (SCM) & Pearl's do-calculus interventional projections.
2. QUANTUM-BOT: Quantum superposition amplitude tree planning with wave-function interference.
3. MOLT: Morphogenetic neuroplastic adapter with online neurogenesis & synaptic pruning.
4. ACTIVE INFERENCE: Friston Free Energy Principle (FEP) homeostatic precision evaluation.
5. HORIZON-X & METACOGNITION: Multi-horizon reserve-sensitive risk modulation.
6. TESSERA: Formal safety contracts & non-negotiable vital invariant guardrails.
7. GENESIS-3D & WORLDLAB: Bridges both discrete POMDPs and continuous embodied 3D physics.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .active_inference import ActiveInferenceConfig, ActiveInferenceEvaluator
from .causal_scm import CausalAttribution, StructuralCausalModel
from .core import CoreRuntime, active_core_path
from .genesis_3d import ContinuousTidePool3D
from .ghost_monologue import GhostIntrospectionEngine, IntrospectiveAudit, LatentDrives
from .guardrails import FormalSafetyGuard, GuardrailAudit
from .morpheus_replay import MorpheusSleepEngine, OneiricSleepReport
from .phase_plane import PhasePlaneEngine, PhaseStateAudit
from .horizon_x import HorizonXConfig, HorizonXPlanner
from .metacognition import MetacognitiveMonitor, MetacognitiveState
from .mnemosyne import MnemosyneConfig, MnemosyneMemory
from .molt import MoltEngine
from .morphogenesis import MorphogeneticAdapter
from .oracle import OracleConfig, OracleEnsemble
from .research.oracle_adapter import ResearchOracleAdapter, oracle_status
from .quantum_planner import QuantumSuperpositionPlanner
from .world import ACTIONS, N_ACTIONS, OBS_SIZE
from .worldlab import POMDPTidePool


@dataclass
class HyperDecision:
    """Rich audit decision output from the HyperAgent."""
    action: int
    action_name: str
    base_action: int
    base_action_name: str
    override: bool
    margin: float
    quantum_coherence: float
    interference_gain: float
    causal_effect: float
    preventable_regret: bool
    free_energy_objective: float
    active_neurons: int
    morphogenesis_state: str
    alert_level: str
    guardrail_veto: bool
    candidates: list[dict[str, Any]]
    phase_regime: str = "transient_orbit"
    lyapunov_drift: float = 0.0
    separatrix_distance: float = 1.0
    dominant_drive: str = "nominal"
    monologue: str = ""
    sleep_cycles: int = 0
    planning_enabled: bool = False
    fallback_reason: str | None = None
    oracle: dict[str, Any] = field(default_factory=dict)

    def __int__(self) -> int:
        return self.action

    def __index__(self) -> int:
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "base_action": self.base_action,
            "base_action_name": self.base_action_name,
            "override": self.override,
            "margin": round(self.margin, 6),
            "quantum_coherence": round(self.quantum_coherence, 4),
            "interference_gain": round(self.interference_gain, 4),
            "causal_effect": round(self.causal_effect, 6),
            "preventable_regret": self.preventable_regret,
            "free_energy_objective": round(self.free_energy_objective, 6),
            "active_neurons": self.active_neurons,
            "morphogenesis_state": self.morphogenesis_state,
            "alert_level": self.alert_level,
            "guardrail_veto": self.guardrail_veto,
            "phase_regime": self.phase_regime,
            "lyapunov_drift": round(self.lyapunov_drift, 5),
            "separatrix_distance": round(self.separatrix_distance, 4),
            "dominant_drive": self.dominant_drive,
            "monologue": self.monologue,
            "sleep_cycles": self.sleep_cycles,
            "candidates": self.candidates,
            "planning_enabled": self.planning_enabled,
            "fallback_reason": self.fallback_reason,
            "oracle": self.oracle,
        }


class HyperAgent:
    """Supermix Poseidon v3.0 Apex Causal-Quantum Morphing Cognitive Agent."""

    SCHEMA = "poseidon-hyper-agent-v3"

    def __init__(
        self,
        core: CoreRuntime | None = None,
        active_inference_config: ActiveInferenceConfig | None = None,
        horizon_config: HorizonXConfig | None = None,
        oracle_checkpoint: str | Path | None = None,
    ):
        self.core = core or CoreRuntime(active_core_path())

        # Subsystems
        self.oracle = (ResearchOracleAdapter(oracle_checkpoint) if oracle_checkpoint is not None
                       else OracleEnsemble(OracleConfig(ensemble_size=3, hidden_size=64)))
        self.mnemosyne = MnemosyneMemory(MnemosyneConfig())
        self.planner = HorizonXPlanner(core=self.core, oracle=self.oracle, config=horizon_config)
        self.active_inference = ActiveInferenceEvaluator(active_inference_config)
        self.metacognition = MetacognitiveMonitor()
        self.guardrails = FormalSafetyGuard()

        # Portfolio Breakthroughs
        self.causeway = StructuralCausalModel()
        self.quantum = QuantumSuperpositionPlanner()
        self.molt = MorphogeneticAdapter(in_dim=OBS_SIZE, out_dim=N_ACTIONS, initial_hidden=64)
        self.phase_plane = PhasePlaneEngine()
        self.ghost = GhostIntrospectionEngine()
        self.morpheus = MorpheusSleepEngine()

        # Runtime transition cache
        self.last_observation: list[float] | None = None
        self.predicted_next_obs: list[float] | None = None
        self.last_causal_attribution: CausalAttribution | None = None
        self.step_counter: int = 0

    def reset(self) -> None:
        """Resets the agent's internal episodic state."""
        self.mnemosyne.reset()
        self.metacognition = MetacognitiveMonitor()
        self.phase_plane.reset()
        self.ghost.reset()
        self.morpheus.reset()
        self.planner.override_history.clear()
        self.planner.step_counter = 0
        if isinstance(self.oracle, ResearchOracleAdapter):
            self.oracle.reset_counters()
        self.last_observation = None
        self.predicted_next_obs = None
        self.last_causal_attribution = None
        self.step_counter = 0

    def act(self, observation: Sequence[float]) -> HyperDecision:
        """Computes action decision through integrated causal-quantum cognitive hierarchy."""
        obs_list = list(observation)
        obs_t = torch.tensor(obs_list, dtype=torch.float32).reshape(1, -1)
        self.step_counter += 1

        plan_out = self.planner.plan(obs_list)
        if not plan_out["planning_enabled"]:
            self.last_observation = obs_list
            return HyperDecision(
                action=plan_out["base_action"], action_name=plan_out["base_action_name"],
                base_action=plan_out["base_action"], base_action_name=plan_out["base_action_name"],
                override=False, margin=0.0, quantum_coherence=0.0, interference_gain=0.0,
                causal_effect=0.0, preventable_regret=False, free_energy_objective=0.0,
                active_neurons=self.molt.current_hidden, morphogenesis_state="frozen",
                alert_level="nominal", guardrail_veto=False,
                candidates=plan_out["candidate_evaluations"], planning_enabled=False,
                fallback_reason=plan_out["fallback_reason"], oracle=oracle_status(self.oracle))

        # 1. Memory, Metacognitive State & Phase-Plane Analysis
        mem_info = self.mnemosyne.step_observation(obs_list, self.predicted_next_obs)
        meta_state = self.metacognition.evaluate_state(obs_list)
        phase_audit = self.phase_plane.analyze_phase_space(obs_list)

        # 2. MOLT Morphogenesis Trigger
        # Frozen evaluation: adaptation needs its own matched experiment.
        morph_state = "frozen"

        # 3. Base Core Model & MOLT Residual Logits
        core_out = self.core.model(["Survive"], obs_t)
        latent_state = core_out["memory"]
        molt_delta = self.molt(obs_t)

        # 4. Multi-Horizon Risk-Sensitive Planning
        base_action = plan_out["base_action"]

        # 5. Causeway do(a) Interventional Predictions
        causeway_predictions = []
        for a in range(N_ACTIONS):
            causeway_pred = self.causeway.interventional_prediction(obs_list, do_action=a)
            causeway_predictions.append(causeway_pred)

        # 6. Oracle Trajectory Predictions & Uncertainty
        predicted_outcomes = []
        predicted_healths = {}
        calibrated_uncs = []
        with torch.no_grad():
            for act_idx in range(N_ACTIONS):
                a_tens = torch.tensor([act_idx], dtype=torch.long)
                wm_res = self.oracle(latent_state, obs_t, a_tens)
                d = wm_res["mean_delta"].squeeze(0)
                pred_s = torch.clamp(obs_t.squeeze(0) + d, 0.0, 1.0).tolist()
                predicted_outcomes.append(pred_s)
                predicted_healths[act_idx] = pred_s[0]
                unc_a = float(wm_res["calibrated_uncertainty"].item())
                calibrated_uncs.append(unc_a)

        # 7. Quantum Superposition Amplitude Planning
        quantum_res = self.quantum.plan_superposition(
            candidate_evaluations=plan_out["candidate_evaluations"],
            calibrated_uncertainties=calibrated_uncs,
            base_action=base_action,
        )

        # 8. Active Inference Expected Free Energy
        empowerment = self.active_inference.compute_empowerment(predicted_outcomes)
        efe_evals = []
        for act_idx in range(N_ACTIONS):
            efe_val = self.active_inference.evaluate_expected_free_energy(
                current_obs=obs_list,
                predicted_obs=predicted_outcomes[act_idx],
                calibrated_uncertainty=calibrated_uncs[act_idx],
                empowerment_value=empowerment,
            )
            efe_evals.append(efe_val)

        # 9. Unified Multi-Plane Candidate Scoring Synthesis
        # S(a) = S_horizon(a) + alpha_efe * EFE(a) + gamma_quantum * Gain_quantum(a) - beta_phase * dL/dt(a)
        alpha_efe = self.active_inference.config.active_inference_blend
        gamma_quantum = 0.15
        beta_phase = 0.08

        synthesized_candidates = []
        for act_idx in range(N_ACTIONS):
            hx_score = plan_out["candidate_evaluations"][act_idx]["composite_score"]
            efe_score = efe_evals[act_idx]["free_energy_objective"]
            q_gain = quantum_res["branches"][act_idx]["interference_gain"]
            phase_drift = phase_audit.action_phase_drifts[act_idx]

            total_score = hx_score + (alpha_efe * efe_score) + (gamma_quantum * q_gain) - (beta_phase * phase_drift)
            synthesized_candidates.append({
                "action": act_idx,
                "action_name": ACTIONS[act_idx],
                "composite_score": total_score,
                "hx_score": hx_score,
                "efe_score": efe_score,
                "quantum_gain": q_gain,
                "phase_drift": phase_drift,
                "predicted_health": predicted_healths[act_idx],
            })

        best_cand = max(synthesized_candidates, key=lambda c: c["composite_score"])
        base_cand = synthesized_candidates[base_action]
        margin = best_cand["composite_score"] - base_cand["composite_score"]

        # Dynamic Reserve-Scaled Override
        rho_t = plan_out["reserve_index"]
        effective_threshold = self.planner.config.override_threshold * max(self.planner.config.min_risk_scale, rho_t)
        override = (best_cand["action"] != base_action) and (margin > effective_threshold)

        proposed_action = best_cand["action"] if override else base_action

        # 10. Formal Safety Guardrails (TESSERA Contracts)
        guard_audit = self.guardrails.audit_and_constrain(
            proposed_action=proposed_action,
            observation=obs_list,
            predicted_healths=predicted_healths,
        )
        final_action = guard_audit.action
        final_override = final_action != base_action

        # 11. Ghost in the Machine Introspective Monologue Synthesis
        ghost_audit = self.ghost.synthesize_monologue(
            obs=obs_list,
            phase_audit=phase_audit,
            quantum_coherence=quantum_res["branches"][final_action]["coherence"],
            quantum_gain=quantum_res["branches"][final_action]["interference_gain"],
            causal_effect=float(causeway_predictions[final_action][0] - causeway_predictions[0][0]),
            active_neurons=self.molt.current_hidden,
            morph_state=morph_state,
            chosen_action=final_action,
            base_action=base_action,
            override=final_override,
            guardrail_veto=guard_audit.vetoed,
            surprise=mem_info["surprise"],
        )

        # Construct decision
        decision = HyperDecision(
            action=final_action,
            action_name=guard_audit.action_name,
            base_action=base_action,
            base_action_name=ACTIONS[base_action],
            override=final_override,
            margin=margin,
            quantum_coherence=quantum_res["branches"][final_action]["coherence"],
            interference_gain=quantum_res["branches"][final_action]["interference_gain"],
            causal_effect=float(causeway_predictions[final_action][0] - causeway_predictions[0][0]),
            preventable_regret=False,
            free_energy_objective=efe_evals[final_action]["free_energy_objective"],
            active_neurons=self.molt.current_hidden,
            morphogenesis_state=morph_state,
            alert_level=meta_state.alert_level,
            guardrail_veto=guard_audit.vetoed,
            phase_regime=phase_audit.stability_regime,
            lyapunov_drift=phase_audit.lyapunov_drift,
            separatrix_distance=phase_audit.separatrix_distance,
            dominant_drive=ghost_audit.drives.dominant_drive,
            monologue=ghost_audit.monologue,
            sleep_cycles=self.morpheus.sleep_cycle_counter,
            candidates=synthesized_candidates,
            planning_enabled=True,
            oracle=oracle_status(self.oracle),
        )

        self.last_observation = obs_list
        return decision

    def plan(self, observation: Sequence[float]) -> dict[str, Any]:
        """Provides standard dictionary decision interface for world recorders."""
        return self.act(observation).to_dict()

    def observe_transition(
        self,
        obs: Sequence[float],
        action: int,
        reward: float,
        next_obs: Sequence[float],
        done_or_info: Any = None,
    ) -> None:
        """Audits causal attribution and updates world predictions."""
        if not oracle_status(self.oracle)["trained"]:
            self.predicted_next_obs = None
            return
        attribution = self.causeway.attribute_causality(obs, action, next_obs)
        self.last_causal_attribution = attribution

        obs_t = torch.tensor(list(obs), dtype=torch.float32).reshape(1, -1)
        a_t = torch.tensor([action], dtype=torch.long)
        with torch.no_grad():
            wm_res = self.oracle(torch.zeros(1, self.oracle.config.latent_size), obs_t, a_t)
            d = wm_res["mean_delta"].squeeze(0)
            self.predicted_next_obs = torch.clamp(obs_t.squeeze(0) + d, 0.0, 1.0).tolist()

        # Online sleep/morphology updates require a separate adaptation protocol.

    def rollout_3d(self, env: ContinuousTidePool3D, max_steps: int = 64) -> dict[str, Any]:
        """Rolls out the HyperAgent inside the Continuous 3D Genesis environment."""
        if oracle_status(self.oracle)["trained"]:
            raise ValueError("The selected ORACLE supports tidepool-v1, not the continuous 3D environment")
        self.reset()
        obs = [0.0] * OBS_SIZE
        obs[0] = 1.0
        obs[1] = 0.85
        obs[2] = 0.85
        obs[3] = 0.90

        total_reward = 0.0
        decisions: list[dict[str, Any]] = []
        steps = 0

        for _ in range(max_steps):
            decision = self.act(obs)
            decisions.append(decision.to_dict())

            next_obs, reward, done, info = env.step(decision.action)
            total_reward += reward
            steps += 1

            self.observe_transition(obs, decision.action, reward, next_obs, done)
            obs = next_obs
            if done:
                break

        overrides = sum(1 for d in decisions if d["override"])
        return {
            "schema": self.SCHEMA,
            "steps": steps,
            "total_reward": float(total_reward),
            "survived": env.health > 0.0,
            "final_health": float(env.health),
            "overrides": overrides,
            "override_rate": float(overrides / steps) if steps > 0 else 0.0,
            "decisions": decisions,
        }
