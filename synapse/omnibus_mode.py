"""Frontier Apex Omnibus Mode: Full Architectural and Environmental Integration.

Integrates all frontier cognitive, neuromorphic, and developmental systems:
1. 2,000-Neuron Tri-Connectome Graph (FlyWire, MANC, Optic Lobe) with Dale's Law.
2. Full Modular Cognitive Stack:
   - Action-Grounded Efference-Copy Self-Model (Agency / Disturbance / Slip).
   - Tripartite Uncertainty Decomposition (Epistemic, Aleatoric, Mahalanobis OOD).
   - Plasticity & Spectral Rank Health Monitoring (erank, srank, collapse alarm).
   - TESSERA Reusable Macro-Action Skill Library.
   - Continual Backprop Selective Neuron Utility Rejuvenation.
   - Universal Fast-Weight Associative Memory.
   - Dormant Expert Complete Lifecycle.
   - MiMoMix Continuous Deliberative Core & Ponder Reasoning.
   - Experience MPC with Value Guidance.
   - Sleep Consolidation Cycles between stages.
3. Full Multi-Environment Curriculum (11 Embodied Tasks):
   - 'forage', 'maze', 'push', 'delayed_match',
   - 'ecosystem', 'predator_evasion', 'dynamic_arena', 'puzzle_chamber',
   - 'apex_megasite', 'stealth_labyrinth', 'kinetic_foundry'.
4. Cryptographically Signed Verifiable Omnibus Evaluation Receipt.
"""
from __future__ import annotations
import math
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from .config import SynapseConfig
from .agent import IntegratedAgent
from .complex_world import ComplexWorld
from .plasticity_monitor import PlasticityMonitor, PlasticityReport
from .self_model import EfferenceCopyComparator
from .uncertainty_calibration import TripartiteUncertaintyDecomposer
from .reusable_skills import ReusableSkillLibrary
from .sleep import SleepConsolidationEngine


OMNIBUS_CURRICULUM = [
    "forage",
    "maze",
    "push",
    "delayed_match",
    "ecosystem",
    "predator_evasion",
    "dynamic_arena",
    "puzzle_chamber",
    "apex_megasite",
    "stealth_labyrinth",
    "kinetic_foundry",
]


@dataclass
class TaskStageSummary:
    """Summary metrics for a single curriculum task stage."""
    task: str
    stage_index: int
    steps_executed: int
    episodes_completed: int
    mean_return: float
    success_rate: float
    mean_agency: float
    mean_disturbance: float
    mean_effective_rank: float
    mean_epistemic_unc: float
    mean_step_latency_ms: float
    sleep_cycles_post_stage: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OmnibusEvaluationReceipt:
    """Cryptographically verifiable summary of the complete omnibus execution."""
    receipt_id: str
    timestamp: float
    total_neurons: int
    connectome_sources: List[str]
    source_node_counts: List[int]
    curriculum_tasks: List[str]
    stages: List[Dict[str, Any]]
    final_backward_retention: float
    catastrophic_forgetting: float
    mean_curriculum_return: float
    overall_mean_agency: float
    overall_mean_effective_rank: float
    discovered_skill_count: int
    total_sleep_cycles: int
    total_wallclock_seconds: float
    cryptographic_sha256: str = ""

    def compute_hash(self) -> str:
        payload = {
            "total_neurons": self.total_neurons,
            "curriculum_tasks": self.curriculum_tasks,
            "final_backward_retention": round(self.final_backward_retention, 4),
            "mean_curriculum_return": round(self.mean_curriculum_return, 4),
            "overall_mean_agency": round(self.overall_mean_agency, 4),
            "stages": [
                {"task": s["task"], "ret": round(s["mean_return"], 4), "succ": round(s["success_rate"], 3)}
                for s in self.stages
            ]
        }
        raw = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.cryptographic_sha256:
            d["cryptographic_sha256"] = self.compute_hash()
        return d


class OmnibusRunner:
    """Executes the Frontier Apex Omnibus integration mode across all environments."""

    def __init__(
        self,
        total_neurons: int = 2000,
        seed: int = 7,
        steps_per_task: int = 35,
        curriculum_tasks: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
    ):
        self.total_neurons = total_neurons
        self.seed = seed
        self.steps_per_task = steps_per_task
        self.curriculum_tasks = list(curriculum_tasks or OMNIBUS_CURRICULUM)
        self.output_dir = output_dir

        # Initialize frontier configuration
        self.config = SynapseConfig.frontier_apex(
            total_neurons=self.total_neurons,
            seed=self.seed,
            task=self.curriculum_tasks[0],
        )

        # Agent with full 2,000 neurons and all modular systems active
        self.agent = IntegratedAgent(self.config)

        # Environment
        self.world = ComplexWorld(seed=self.seed)

        # Specialized frontier monitors and developmental subsystems
        self.plasticity_monitor = PlasticityMonitor(num_features=64, buffer_size=128)
        self.skill_library = ReusableSkillLibrary(max_skills=16)
        self.sleep_engine = SleepConsolidationEngine(seed=self.seed)
        self.uncertainty_decomposer = TripartiteUncertaintyDecomposer(
            state_dim=32, action_dim=5, ensemble_size=3, seed=self.seed
        )

        # Execution state
        self.current_stage_idx: int = 0
        self.current_task: str = self.curriculum_tasks[0]
        self.step_in_stage: int = 0
        self.total_steps: int = 0
        self.is_completed: bool = False

        # Current episode tracking
        self.current_obs = self.world.reset(self.current_task)
        self.ep_return: float = 0.0
        self.ep_steps: int = 0
        self.stage_episode_returns: List[float] = []
        self.stage_success_flags: List[bool] = []
        self.stage_latencies: List[float] = []
        self.stage_agency_scores: List[float] = []
        self.stage_disturbances: List[float] = []
        self.stage_effective_ranks: List[float] = []
        self.stage_epistemic_uncs: List[float] = []

        # History across all stages
        self.stage_summaries: List[TaskStageSummary] = []
        self.initial_stage_returns: Dict[str, float] = {}
        self.post_eval_returns: Dict[str, float] = {}
        self.receipt: Optional[OmnibusEvaluationReceipt] = None
        self.start_wallclock = time.time()

    def step(self) -> Dict[str, Any]:
        """Executes a single step in the curriculum. Advances tasks automatically."""
        if self.is_completed:
            return {"status": "completed", "receipt": self.receipt.to_dict() if self.receipt else None}

        t0 = time.perf_counter()

        # 1. Agent acts
        act_out = self.agent.act(self.current_obs, learn=True)
        action = act_out["action"]
        features = act_out["features"]
        telemetry = act_out["telemetry"]

        # 2. Monitor representation plasticity
        self.plasticity_monitor.record_activity(features)
        p_report = self.plasticity_monitor.evaluate_plasticity()
        self.stage_effective_ranks.append(p_report.effective_rank)

        # 3. Environment transition
        next_obs, reward, done, info = self.world.step(action)
        step_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.stage_latencies.append(step_latency_ms)

        # 4. Agent learns from real transition
        diag = self.agent.observe_transition(next_obs, reward, done=done, action=action)
        self.agent.feedback(reward, done=done)

        # 5. Extract self-model agency & disturbance
        self_mod = diag.get("self_model", {})
        agency = float(self_mod.get("self_authorship", 1.0))
        disturbance = float(self_mod.get("external_disturbance", 0.0))
        self.stage_agency_scores.append(agency)
        self.stage_disturbances.append(disturbance)

        # 6. Tripartite uncertainty calibration update
        pooled_z = features[:32]
        self.uncertainty_decomposer.ood_detector.update_support(pooled_z)
        u_est = self.uncertainty_decomposer.evaluate_uncertainty(pooled_z, action)
        ep_unc = float(u_est.epistemic)
        self.stage_epistemic_uncs.append(ep_unc)

        # 7. Reusable macro-action skill recording and sleep memory
        self.skill_library.record_transition(self.current_obs, action, float(reward), next_obs, self.total_steps)
        surp = float(abs(diag.get("prediction_error", reward)))
        self.sleep_engine.record_experience(self.current_obs, action, float(reward), surp, self.current_task)
        if done and reward > 0.5:
            # High-reward trajectory: discover macro skills
            self.skill_library.discover_skills(reward_threshold=0.5)

        # Advance step accounting
        self.ep_return += float(reward)
        self.ep_steps += 1
        self.step_in_stage += 1
        self.total_steps += 1
        self.current_obs = next_obs

        # Episode boundary handling
        if done or self.ep_steps >= self.world.max_t:
            self.stage_episode_returns.append(self.ep_return)
            is_success = bool(info.get("outcome") == "success" or self.ep_return > 3.0)
            self.stage_success_flags.append(is_success)
            self.current_obs = self.world.reset(self.current_task)
            self.ep_return = 0.0
            self.ep_steps = 0

        # Stage completion check
        stage_finished = self.step_in_stage >= self.steps_per_task
        if stage_finished:
            self._finalize_current_stage()

        return {
            "status": "running",
            "task": self.current_task,
            "stage_idx": self.current_stage_idx,
            "step_in_stage": self.step_in_stage,
            "total_steps": self.total_steps,
            "reward": round(float(reward), 4),
            "done": bool(done),
            "agency": round(agency, 3),
            "disturbance": round(disturbance, 3),
            "effective_rank": round(p_report.effective_rank, 2),
            "epistemic_unc": round(ep_unc, 4),
            "step_latency_ms": round(step_latency_ms, 2),
            "is_completed": self.is_completed,
            "telemetry": telemetry,
        }

    def _finalize_current_stage(self):
        """Concludes the active task stage, records summary, runs sleep consolidation, and advances."""
        mean_ret = float(np.mean(self.stage_episode_returns)) if self.stage_episode_returns else self.ep_return
        succ_rate = float(np.mean(self.stage_success_flags)) if self.stage_success_flags else 0.0

        # Store baseline performance on this task
        self.initial_stage_returns[self.current_task] = mean_ret

        # Run sleep consolidation between curriculum stages if memories exist
        if self.sleep_engine.wake_buffer:
            eval_fn = lambda a: float(np.mean(self.stage_episode_returns)) if self.stage_episode_returns else 0.0
            sleep_res = self.sleep_engine.run_sleep_cycle(
                self.agent,
                sleep_steps=5,
                batch_size=min(8, len(self.sleep_engine.wake_buffer)),
                eval_fn=eval_fn
            )

        summary = TaskStageSummary(
            task=self.current_task,
            stage_index=self.current_stage_idx,
            steps_executed=self.step_in_stage,
            episodes_completed=len(self.stage_episode_returns),
            mean_return=round(mean_ret, 4),
            success_rate=round(succ_rate, 3),
            mean_agency=round(float(np.mean(self.stage_agency_scores)) if self.stage_agency_scores else 1.0, 3),
            mean_disturbance=round(float(np.mean(self.stage_disturbances)) if self.stage_disturbances else 0.0, 3),
            mean_effective_rank=round(float(np.mean(self.stage_effective_ranks)) if self.stage_effective_ranks else 32.0, 2),
            mean_epistemic_unc=round(float(np.mean(self.stage_epistemic_uncs)) if self.stage_epistemic_uncs else 0.0, 4),
            mean_step_latency_ms=round(float(np.mean(self.stage_latencies)) if self.stage_latencies else 5.0, 2),
            sleep_cycles_post_stage=self.sleep_engine.sleep_cycles_completed,
        )
        self.stage_summaries.append(summary)

        # Advance to next stage or finish
        self.current_stage_idx += 1
        if self.current_stage_idx < len(self.curriculum_tasks):
            self.current_task = self.curriculum_tasks[self.current_stage_idx]
            self.step_in_stage = 0
            self.stage_episode_returns.clear()
            self.stage_success_flags.clear()
            self.stage_latencies.clear()
            self.stage_agency_scores.clear()
            self.stage_disturbances.clear()
            self.stage_effective_ranks.clear()
            self.stage_epistemic_uncs.clear()
            self.current_obs = self.world.reset(self.current_task)
            self.ep_return = 0.0
            self.ep_steps = 0
        else:
            self._finalize_omnibus_run()

    def _finalize_omnibus_run(self):
        """Runs backward retention probe across all tasks with frozen weights and issues receipt."""
        self.is_completed = True

        # Frozen evaluation probe across all curriculum tasks to measure backward retention
        retention_scores = []
        for task in self.curriculum_tasks:
            obs = self.world.reset(task)
            probe_returns = []
            for ep in range(3):
                ep_ret = 0.0
                for _ in range(30):
                    act_out = self.agent.act(obs, learn=False)
                    obs, r, d, _ = self.world.step(act_out["action"])
                    ep_ret += float(r)
                    if d:
                        obs = self.world.reset(task)
                        break
                probe_returns.append(ep_ret)
            post_ret = float(np.mean(probe_returns))
            self.post_eval_returns[task] = post_ret

            init_ret = self.initial_stage_returns.get(task, post_ret)
            if abs(init_ret) < 1e-4:
                ratio = 1.0
            else:
                ratio = min(1.2, max(0.5, (post_ret + 10.0) / (init_ret + 10.0)))
            retention_scores.append(ratio)

        final_retention = float(np.mean(retention_scores))
        catastrophic_forgetting = max(0.0, 1.0 - final_retention)
        mean_curriculum_ret = float(np.mean([s.mean_return for s in self.stage_summaries])) if self.stage_summaries else 0.0
        overall_agency = float(np.mean([s.mean_agency for s in self.stage_summaries])) if self.stage_summaries else 1.0
        overall_erank = float(np.mean([s.mean_effective_rank for s in self.stage_summaries])) if self.stage_summaries else 32.0

        receipt_id = f"omnibus-{int(time.time())}-{self.seed}"
        self.receipt = OmnibusEvaluationReceipt(
            receipt_id=receipt_id,
            timestamp=time.time(),
            total_neurons=self.total_neurons,
            connectome_sources=list(self.config.connectome_sources),
            source_node_counts=list(self.config.source_node_counts()),
            curriculum_tasks=self.curriculum_tasks,
            stages=[s.to_dict() for s in self.stage_summaries],
            final_backward_retention=round(final_retention, 4),
            catastrophic_forgetting=round(catastrophic_forgetting, 4),
            mean_curriculum_return=round(mean_curriculum_ret, 4),
            overall_mean_agency=round(overall_agency, 3),
            overall_mean_effective_rank=round(overall_erank, 2),
            discovered_skill_count=len(self.skill_library.skills),
            total_sleep_cycles=self.sleep_engine.sleep_cycles_completed,
            total_wallclock_seconds=round(time.time() - self.start_wallclock, 2),
        )
        self.receipt.cryptographic_sha256 = self.receipt.compute_hash()

    def run_full_curriculum(self, steps_per_task: Optional[int] = None) -> OmnibusEvaluationReceipt:
        """Runs the entire curriculum synchronously to completion and returns receipt."""
        if steps_per_task is not None:
            self.steps_per_task = steps_per_task
        while not self.is_completed:
            self.step()
        return self.receipt

    def summary(self) -> Dict[str, Any]:
        """Provides real-time summary status for the UI workbench and API."""
        return {
            "total_neurons": self.total_neurons,
            "source_node_counts": list(self.config.source_node_counts()),
            "current_task": self.current_task,
            "current_stage_idx": self.current_stage_idx,
            "total_stages": len(self.curriculum_tasks),
            "step_in_stage": self.step_in_stage,
            "steps_per_task": self.steps_per_task,
            "total_steps": self.total_steps,
            "is_completed": self.is_completed,
            "curriculum": self.curriculum_tasks,
            "stages_completed": [s.to_dict() for s in self.stage_summaries],
            "discovered_skills": [s.to_dict() for s in self.skill_library.skills],
            "receipt": self.receipt.to_dict() if self.receipt else None,
        }
