"""POSEIDON COGNITIVE AGENT: Unified v2.0 Cognitive Autonomous Agent Architecture.

Unifies all v2.0 cognitive subsystems around the immutable, frozen Tidal core:
1. ORACLE: Bootstrapped calibrated dynamics ensemble for multi-step trajectory imagination.
2. MNEMOSYNE: 3-tier memory hierarchy (working, value-gated episodic, spatial cognitive map)
   supporting 5 matched lesion assays (intact, erased, shifted, random_retrieval, memory_free).
3. HORIZON-X: Multi-horizon (H in {1, 4, 8, 16}) risk-sensitive planning with dynamic
   metabolic reserve modulation, preventing the starvation pessimism trap.
4. ADAPT: Continual residual policy adaptation with anti-forgetting anchor regularization.
5. NEURO: Neuromodulated sparse connectome routing with metabolic gain control.
6. WORLDLAB: High-fidelity partially observable POMDP environments and simulation bridge.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .active_inference import ActiveInferenceConfig, ActiveInferenceEvaluator
from .adapt import AdaptConfig, ContinualAdapter
from .core import CoreRuntime, active_core_path
from .dream import DreamConfig, DreamConsolidator
from .guardrails import FormalSafetyGuard, GuardrailAudit
from .horizon_x import HorizonXConfig, HorizonXPlanner
from .metacognition import MetacognitionConfig, MetacognitiveMonitor, MetacognitiveState
from .mnemosyne import MNEMOSYNE_MODES, MnemosyneConfig, MnemosyneMemory
from .neuro import NeuroConfig, NeuromodulatedRouter
from .oracle import OracleConfig, OracleEnsemble
from .research.oracle_adapter import ResearchOracleAdapter, oracle_status
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool
from .worldlab import POMDPConfig, POMDPTidePool, StepResult


@dataclass
class CognitiveDecision:
    """Rich audit decision output from the CognitiveAgent."""
    action: int
    action_name: str
    base_action: int
    base_action_name: str
    override: bool
    margin: float
    surprise: float
    reserve_index: float
    dynamic_lambda: float
    active_sparsity: float
    neuromodulatory_gain: float
    memory_patch: str | None
    food_lcb: float
    water_lcb: float
    mode: str
    evaluations: list[dict[str, Any]]
    free_energy_objective: float = 0.0
    pragmatic_value: float = 0.0
    epistemic_value: float = 0.0
    empowerment_value: float = 0.0
    alert_level: str = "nominal"
    epistemic_volatility: float = 0.0
    metacognitive_confidence: float = 1.0
    recommended_horizon: int = 16
    dreams_consolidated: int = 0
    guardrail_veto: bool = False
    active_invariants: list[str] = field(default_factory=list)
    guardrail_violations: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    planning_enabled: bool = False
    fallback_reason: str | None = None
    oracle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "base_action": self.base_action,
            "base_action_name": self.base_action_name,
            "override": self.override,
            "margin": round(self.margin, 6),
            "surprise": round(self.surprise, 6),
            "reserve_index": round(self.reserve_index, 6),
            "dynamic_lambda": round(self.dynamic_lambda, 6),
            "active_sparsity": round(self.active_sparsity, 6),
            "neuromodulatory_gain": round(self.neuromodulatory_gain, 6),
            "memory_patch": self.memory_patch,
            "food_lcb": round(self.food_lcb, 4),
            "water_lcb": round(self.water_lcb, 4),
            "mode": self.mode,
            "free_energy_objective": round(self.free_energy_objective, 6),
            "pragmatic_value": round(self.pragmatic_value, 6),
            "epistemic_value": round(self.epistemic_value, 6),
            "empowerment_value": round(self.empowerment_value, 6),
            "alert_level": self.alert_level,
            "epistemic_volatility": round(self.epistemic_volatility, 6),
            "metacognitive_confidence": round(self.metacognitive_confidence, 6),
            "recommended_horizon": self.recommended_horizon,
            "dreams_consolidated": self.dreams_consolidated,
            "guardrail_veto": self.guardrail_veto,
            "active_invariants": self.active_invariants,
            "guardrail_violations": self.guardrail_violations,
            "candidates": self.candidates if self.candidates else self.evaluations,
            "evaluations": self.evaluations,
            "planning_enabled": self.planning_enabled,
            "fallback_reason": self.fallback_reason,
            "oracle": self.oracle,
        }


class CognitiveAgent:
    """Supermix Poseidon v2.0 Cognitive Agent Coordinator."""

    SCHEMA = "poseidon-cognitive-agent-v2"

    def __init__(
        self,
        core: CoreRuntime | None = None,
        mode: str = "intact",
        oracle_config: OracleConfig | None = None,
        mnemosyne_config: MnemosyneConfig | None = None,
        horizon_config: HorizonXConfig | None = None,
        adapt_config: AdaptConfig | None = None,
        neuro_config: NeuroConfig | None = None,
        active_inference_config: ActiveInferenceConfig | None = None,
        oracle_checkpoint: str | Path | None = None,
        decision_value_memory: Any | None = None,
    ):
        self.core = core or CoreRuntime(active_core_path())
        self.mode = mode
        if self.mode not in MNEMOSYNE_MODES:
            raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {mode}")

        # Subsystems
        self.decision_value_memory = decision_value_memory
        self.oracle = (ResearchOracleAdapter(oracle_checkpoint) if oracle_checkpoint is not None
                       else OracleEnsemble(oracle_config or OracleConfig()))
        mnemo_cfg = mnemosyne_config or MnemosyneConfig(mode=self.mode)
        mnemo_cfg.mode = self.mode
        self.mnemosyne = MnemosyneMemory(mnemo_cfg)
        self.planner = HorizonXPlanner(core=self.core, oracle=self.oracle, config=horizon_config)
        self.adapter = ContinualAdapter(core=self.core, config=adapt_config)
        self.router = NeuromodulatedRouter(neuro_config or NeuroConfig(dim=self.core.model.config.hidden_size))
        self.active_inference = ActiveInferenceEvaluator(active_inference_config)
        self.metacognition = MetacognitiveMonitor()
        self.dream = DreamConsolidator(
            core=self.core,
            oracle=self.oracle,
            mnemosyne=self.mnemosyne,
            replay_buffer=self.adapter.replay,
        )
        self.guardrails = FormalSafetyGuard()

        # Runtime transition memory
        self.last_observation: list[float] | None = None
        self.predicted_next_obs: list[float] | None = None
        self.last_uncertainty: float = 0.0
        self.step_count: int = 0
        self.dreams_consolidated: int = 0
        self._pending_prediction: tuple[list[float], int, list[float], float] | None = None

    def reset(self, mode: str | None = None) -> None:
        """Resets the agent's internal state for a new episode."""
        if mode is not None:
            if mode not in MNEMOSYNE_MODES:
                raise ValueError(f"mode must be one of {MNEMOSYNE_MODES}, got {mode}")
            self.mode = mode

        self.mnemosyne.reset(mode=self.mode)
        if self.decision_value_memory is not None:
            self.decision_value_memory.reset(mode=self.mode if self.mode in ("intact", "similarity", "erased", "shifted", "donor", "baseline") else "intact")
        self.metacognition = MetacognitiveMonitor(self.metacognition.config)
        self.planner.override_history.clear()
        self.planner.step_counter = 0
        self.adapter.replay.buffer.clear()
        self.dream.dream_history.clear()
        self.dream.consolidated_count = 0
        self.last_observation = None
        self.predicted_next_obs = None
        self.last_uncertainty = 0.0
        self.step_count = 0
        self.dreams_consolidated = 0
        self._pending_prediction = None
        if isinstance(self.oracle, ResearchOracleAdapter):
            self.oracle.reset_counters()

    def act(self, observation: Sequence[float]) -> CognitiveDecision:
        """Computes action decision through integrated cognitive hierarchy."""
        obs_list = list(observation)
        obs_t = torch.tensor(obs_list, dtype=torch.float32).reshape(1, -1)

        # 1. Update 3-Tier Memory
        mem_info = self.mnemosyne.step_observation(
            observation=obs_list,
            predicted_observation=self.predicted_next_obs,
        )
        if self.decision_value_memory is not None:
            dv_decision = self.decision_value_memory.query(obs_list)
            mem_info["decision_value_deltas"] = dv_decision.gated_deltas
            mem_info["decision_memory"] = dv_decision.to_dict()

        # 2. Metacognitive Monitoring
        meta_state = self.metacognition.evaluate_state(obs_list)

        # 3. Neuromodulated routing
        vitals = torch.tensor([[
            obs_list[0],  # health
            obs_list[1],  # energy
            obs_list[2],  # hydration
            obs_list[5],  # threat
        ]], dtype=torch.float32)

        core_out = self.core.model(["Survive"], obs_t)
        latent_state = core_out["memory"]
        neuro_out = self.router(latent_state, vitals)

        # 4. Multi-Horizon Risk-Sensitive Planning with Causal Memory & Metacognitive Context
        plan_out = self.planner.plan(
            obs_list,
            memory_context=mem_info,
            metacognitive_state=meta_state,
        )

        if not plan_out["planning_enabled"]:
            self.last_observation = obs_list
            self._pending_prediction = None
            return CognitiveDecision(
                action=plan_out["base_action"], action_name=plan_out["base_action_name"],
                base_action=plan_out["base_action"], base_action_name=plan_out["base_action_name"],
                override=False, margin=0.0, surprise=mem_info["surprise"],
                reserve_index=plan_out["reserve_index"], dynamic_lambda=plan_out["dynamic_lambda"],
                active_sparsity=float(neuro_out["active_sparsity"].item()),
                neuromodulatory_gain=float(neuro_out["gain"].item()),
                memory_patch=mem_info["patch_id"], food_lcb=mem_info["estimated_food_lcb"],
                water_lcb=mem_info["estimated_water_lcb"], mode=self.mode,
                evaluations=plan_out["candidate_evaluations"],
                planning_enabled=False, fallback_reason=plan_out["fallback_reason"],
                oracle=oracle_status(self.oracle))

        # 5. World Model Preview & Active Inference across all candidate actions
        predicted_outcomes = []
        predicted_healths = {}
        candidate_efes = []
        with torch.no_grad():
            for act_idx in range(N_ACTIONS):
                a_tens = torch.tensor([act_idx], dtype=torch.long)
                wm_res = self.oracle(latent_state, obs_t, a_tens)
                d = wm_res["mean_delta"].squeeze(0)
                pred_s = torch.clamp(obs_t.squeeze(0) + d, 0.0, 1.0).tolist()
                predicted_outcomes.append(pred_s)
                predicted_healths[act_idx] = pred_s[0]
                unc_a = float(wm_res["calibrated_uncertainty"].item())
                candidate_efes.append((pred_s, unc_a))

        empowerment = self.active_inference.compute_empowerment(predicted_outcomes)

        efe_evals = []
        for act_idx, (pred_s, unc_a) in enumerate(candidate_efes):
            efe_eval = self.active_inference.evaluate_expected_free_energy(
                current_obs=obs_list,
                predicted_obs=pred_s,
                calibrated_uncertainty=unc_a,
                empowerment_value=empowerment,
            )
            efe_evals.append(efe_eval)

        # 6. Candidate Integration & Dynamic Reserve-Scaled Override
        alpha = self.active_inference.config.active_inference_blend
        if alpha > 0.0:
            for act_idx, c_eval in enumerate(plan_out["candidate_evaluations"]):
                efe_obj = efe_evals[act_idx]["free_energy_objective"]
                c_eval["composite_score"] += alpha * efe_obj

            base_action = plan_out["base_action"]
            best_cand = max(plan_out["candidate_evaluations"], key=lambda x: x["composite_score"])
            base_eval = next((e for e in plan_out["candidate_evaluations"] if e["action"] == base_action), plan_out["candidate_evaluations"][0])
            margin = best_cand["composite_score"] - base_eval["composite_score"]

            rho_t = plan_out["reserve_index"]
            effective_threshold = self.planner.config.override_threshold * max(self.planner.config.min_risk_scale, rho_t)
            override = (best_cand["action"] != base_action) and (margin > effective_threshold)

            chosen_action = best_cand["action"] if override else base_action
            plan_out["selected_action"] = chosen_action
            plan_out["selected_action_name"] = ACTIONS[chosen_action]
            plan_out["override"] = override
            plan_out["margin"] = margin
        else:
            chosen_action = plan_out["selected_action"]

        efe = efe_evals[chosen_action]

        # 7. Formal Safety Guardrails (TESSERA ratified invariants)
        guard_audit = self.guardrails.audit_and_constrain(
            proposed_action=chosen_action,
            observation=obs_list,
            predicted_healths=predicted_healths,
        )

        final_action = guard_audit.action
        final_action_name = guard_audit.action_name
        final_override = final_action != plan_out["base_action"]

        # Construct candidates for UI telemetry
        candidate_rows = []
        for act_idx, c_eval in enumerate(plan_out["candidate_evaluations"]):
            is_vetoed = any(v.get("recommended_action") == act_idx for v in guard_audit.violation_details if v.get("severity") == "critical_veto")
            candidate_rows.append({
                "action": act_idx,
                "action_name": ACTIONS[act_idx],
                "composite_score": round(c_eval.get("composite_score", 0.0), 4),
                "predicted_health": round(predicted_healths.get(act_idx, 0.0), 4),
                "safe": not is_vetoed,
                "executed": (act_idx == final_action),
            })

        # Create rich decision
        decision = CognitiveDecision(
            action=final_action,
            action_name=final_action_name,
            base_action=plan_out["base_action"],
            base_action_name=plan_out["base_action_name"],
            override=final_override,
            margin=plan_out["margin"],
            surprise=mem_info["surprise"],
            reserve_index=plan_out["reserve_index"],
            dynamic_lambda=plan_out["dynamic_lambda"],
            active_sparsity=float(neuro_out["active_sparsity"].item()),
            neuromodulatory_gain=float(neuro_out["gain"].item()),
            memory_patch=mem_info["patch_id"],
            food_lcb=mem_info["estimated_food_lcb"],
            water_lcb=mem_info["estimated_water_lcb"],
            mode=self.mode,
            free_energy_objective=efe["free_energy_objective"],
            pragmatic_value=efe["pragmatic_value"],
            epistemic_value=efe["epistemic_value"],
            empowerment_value=empowerment,
            alert_level=meta_state.alert_level,
            epistemic_volatility=meta_state.epistemic_volatility,
            metacognitive_confidence=meta_state.metacognitive_confidence,
            recommended_horizon=meta_state.recommended_horizon,
            dreams_consolidated=self.dreams_consolidated,
            guardrail_veto=guard_audit.vetoed,
            active_invariants=guard_audit.active_invariants,
            guardrail_violations=guard_audit.violation_details,
            candidates=candidate_rows,
            evaluations=plan_out["candidate_evaluations"],
            planning_enabled=True,
            oracle=oracle_status(self.oracle),
        )

        self.last_observation = obs_list
        # Cache the forecast for the action that will actually execute, including
        # a guardrail veto. Keep the previous forecast for the next memory update.
        prediction, uncertainty = candidate_efes[final_action]
        self._pending_prediction = (obs_list.copy(), final_action, prediction.copy(), uncertainty)
        return decision

    def plan(self, observation: Sequence[float]) -> dict[str, Any]:
        """Planner interface returning decision dictionary with action and telemetry."""
        return self.act(observation).to_dict()

    def observe_transition(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        info: Any = None,
        done: bool = False,
    ) -> None:
        """Match the executed (observation, action) forecast to its realized successor."""
        obs_list = list(observation)
        next_obs_list = list(next_observation)

        if not oracle_status(self.oracle)["trained"]:
            # Retain observed experience, without presenting random predictions
            # as model error, uncertainty, or synthetic training targets.
            self.mnemosyne.record_transition(observation=obs_list, action=action,
                                             reward=reward, next_observation=next_obs_list)
            self.adapter.replay.add(obs=obs_list, action=action, reward=reward, next_obs=next_obs_list)
            if self.decision_value_memory is not None:
                self.decision_value_memory.add_entry(obs_list, action, reward, next_obs_list)
                self.decision_value_memory.update_belief(next_obs_list, action, reward)
            self._pending_prediction = None
            self.predicted_next_obs = None
            self.step_count += 1
            return

        pending = self._pending_prediction
        if pending is not None and pending[0] == obs_list and pending[1] == action:
            prediction, uncertainty = pending[2], pending[3]
        else:
            # Compatibility for direct transition callers or externally overridden
            # actions: compute from the predecessor only, before ingesting outcome.
            with torch.no_grad():
                obs_t = torch.tensor(obs_list, dtype=torch.float32).reshape(1, -1)
                latent = self.core.model(["Survive"], obs_t)["memory"]
                wm_out = self.oracle(latent, obs_t, torch.tensor([action], dtype=torch.long))
                prediction = (obs_t + wm_out["mean_delta"]).clamp(0.0, 1.0)[0].tolist()
                uncertainty = float(wm_out["calibrated_uncertainty"].item())
        self._pending_prediction = None
        self.predicted_next_obs = list(prediction)
        self.last_uncertainty = uncertainty

        # Record in Mnemosyne episodic store
        self.mnemosyne.record_transition(
            observation=obs_list,
            action=action,
            reward=reward,
            next_observation=next_obs_list,
        )

        # Record in ContinualAdapter replay buffer
        self.adapter.replay.add(
            obs=obs_list,
            action=action,
            reward=reward,
            next_obs=next_obs_list,
        )

        if self.decision_value_memory is not None:
            res = [next_obs_list[i] - self.predicted_next_obs[i] for i in range(len(next_obs_list))]
            self.decision_value_memory.add_entry(obs_list, action, reward, next_obs_list, residual=res)
            self.decision_value_memory.update_belief(next_obs_list, action, reward)

        # Generative Dream Consolidation on Rest action (action == 0)
        if (action == 0 and self.mode not in ("erased", "memory_free")
                and getattr(self.oracle, "failure_probability_calibrated", False)):
            dream_summary = self.dream.consolidate_during_rest()
            self.dreams_consolidated += dream_summary.synthetic_rollouts_generated

        # Update Metacognitive Monitor
        self.metacognition.register_transition(
            predicted_obs=self.predicted_next_obs,
            realized_obs=next_obs_list,
            predicted_uncertainty=self.last_uncertainty,
        )

        self.step_count += 1

    def rollout(
        self,
        env: TidePool | POMDPTidePool,
        max_steps: int = 64,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Rolls out a full episode recording transitions, metrics, and overrides."""
        if type(max_steps) is not int or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be an integer in [1, 10000]")
        self.reset(mode=mode)
        obs = env.reset()
        initial = env.snapshot()
        total_reward = 0.0
        decisions: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = []

        for step in range(max_steps):
            decision = self.act(obs)
            dec_dict = decision.to_dict()
            decisions.append(dec_dict)

            act = decision.action
            if isinstance(env, POMDPTidePool):
                res = env.step(act)
                next_obs = res.observation
                reward = res.reward
                done = res.done
                info = res.info
            else:
                next_obs, reward, done, info = env.step(act)

            total_reward += reward
            self.observe_transition(obs, act, reward, next_obs, info=info, done=done)

            trajectory.append({
                "observation": obs,
                "action": act,
                "action_name": ACTIONS[act],
                "reward": reward,
                "done": bool(done),
                "next_observation": next_obs,
                "state": env.state(),
                "info": info,
                "decision": dec_dict,
            })

            obs = next_obs
            if done:
                break

        overrides = sum(1 for d in decisions if d["override"])
        return {
            "schema": self.SCHEMA,
            "mode": self.mode,
            "steps": len(decisions),
            "total_reward": float(total_reward),
            "survived": bool(env.alive),
            "death": not env.alive,
            "death_reason": env.death_reason,
            "truncated": bool(env.alive and (env.done or len(decisions) == max_steps)),
            "terminal_reason": "death" if not env.alive else "horizon" if env.done else "rollout_limit",
            "overrides": overrides,
            "override_rate": float(overrides / len(decisions)) if decisions else 0.0,
            "final_health": float(env.health),
            "initial_snapshot": initial,
            "trajectory": trajectory,
            "decisions": decisions,
        }
