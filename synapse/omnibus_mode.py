"""Bounded composed-system curriculum with matched frozen evaluation.

The live actor is shared across stages. Monitors and macro discovery are measured
sidecars; uncertainty is trained online but is not calibrated on held-out data.
Receipt hashes bind the recorded payload, not the truth of scientific claims.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict, field
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from .config import SynapseConfig
from .agent import IntegratedAgent
from .complex_world import ComplexWorld, ALL_TASKS
from .homeostasis import HomeostaticEngine
from .plasticity_monitor import PlasticityMonitor
from .uncertainty_calibration import TripartiteUncertaintyDecomposer
from .reusable_skills import ReusableSkillLibrary
from .sleep import SleepConsolidationEngine

OMNIBUS_CURRICULUM = [
    'forage', 'maze', 'push', 'delayed_match', 'ecosystem',
    'predator_evasion', 'dynamic_arena', 'puzzle_chamber',
    'apex_megasite', 'stealth_labyrinth', 'kinetic_foundry',
    'continental_biosphere', 'chrono_tidepool_citadel',
]


def _mean(values):
    return float(np.mean(values)) if values else None


@dataclass
class TaskStageSummary:
    task: str
    stage_index: int
    steps_executed: int
    episodes_completed: int
    mean_return: float
    success_rate: Optional[float]
    mean_agency: Optional[float]
    mean_disturbance: Optional[float]
    mean_effective_rank: Optional[float]
    mean_epistemic_unc: Optional[float]
    mean_step_latency_ms: Optional[float]
    sleep_cycles_post_stage: int

    def to_dict(self):
        return asdict(self)


@dataclass
class OmnibusEvaluationReceipt:
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
    overall_mean_agency: Optional[float]
    overall_mean_effective_rank: Optional[float]
    discovered_skill_count: int
    total_sleep_cycles: int
    total_wallclock_seconds: float
    cryptographic_sha256: str = ''
    evaluation_protocol: Dict[str, Any] = field(default_factory=dict)
    task_score_matrix: List[List[Optional[float]]] = field(default_factory=list)
    stage_baseline_returns: Dict[str, float] = field(default_factory=dict)
    initial_stage_returns: Dict[str, float] = field(default_factory=dict)
    post_eval_returns: Dict[str, float] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    uncertainty_training_updates: int = 0
    sleep_screens: List[Dict[str, Any]] = field(default_factory=list)

    def compute_hash(self):
        payload = asdict(self)
        payload.pop('cryptographic_sha256')
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def to_dict(self):
        result = asdict(self)
        result['cryptographic_sha256'] = self.cryptographic_sha256 or self.compute_hash()
        return result


class OmnibusRunner:
    """Sequential learning and isolated, equally budgeted task probes."""

    def __init__(self, total_neurons=4000, seed=7, steps_per_task=35,
                 curriculum_tasks=None, output_dir: Optional[Path] = None,
                 eval_steps=30, eval_episodes=1):
        for name, value, maximum in [('steps_per_task', steps_per_task, 1000),
                                     ('eval_steps', eval_steps, 1000),
                                     ('eval_episodes', eval_episodes, 20)]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in [1,{maximum}]')
        if curriculum_tasks is None:
            curriculum_tasks = OMNIBUS_CURRICULUM
        if (not isinstance(curriculum_tasks, (list, tuple)) or not curriculum_tasks
                or any(type(t) is not str or t not in ALL_TASKS for t in curriculum_tasks)
                or len(set(curriculum_tasks)) != len(curriculum_tasks)):
            raise ValueError('curriculum_tasks must be a nonempty sequence of distinct supported tasks')
        self.total_neurons, self.seed = total_neurons, seed
        self.steps_per_task, self.eval_steps, self.eval_episodes = steps_per_task, eval_steps, eval_episodes
        self.curriculum_tasks = list(curriculum_tasks)
        self.output_dir = output_dir
        self.config = SynapseConfig.frontier_apex(total_neurons=total_neurons, seed=seed, task=self.curriculum_tasks[0])
        self.agent = IntegratedAgent(self.config)
        self.world = ComplexWorld(seed=seed)
        self.plasticity_monitor = PlasticityMonitor(num_features=64, buffer_size=128)
        self.skill_library = ReusableSkillLibrary(max_skills=16, seed=seed)
        self.sleep_engine = SleepConsolidationEngine(seed=seed)
        self.uncertainty_decomposer = TripartiteUncertaintyDecomposer(state_dim=32, action_dim=5, ensemble_size=3, seed=seed)
        self.uncertainty_training_updates = 0
        self.uncertainty_last_loss = None
        self._previous_transition = None
        self.evaluation_steps_executed = 0
        self.current_stage_idx, self.total_steps = 0, 0
        self.current_task = self.curriculum_tasks[0]
        self.current_obs = self.world.reset(self.current_task)
        self.step_in_stage = self.ep_steps = 0
        self.ep_return = 0.
        self.is_completed = False
        self.receipt = None
        self.stage_summaries = []
        self.initial_stage_returns, self.post_eval_returns = {}, {}
        self.stage_baseline_returns = {}
        self.task_score_matrix = []
        self.start_wallclock = time.time()
        self._clear_stage_metrics()

    def _clear_stage_metrics(self):
        self.stage_episode_returns, self.stage_success_flags = [], []
        self.stage_latencies, self.stage_agency_scores, self.stage_disturbances = [], [], []
        self.stage_effective_ranks, self.stage_epistemic_uncs = [], []

    @staticmethod
    def _advance_physiology(agent, action, reward, info):
        agent.homeostasis.step(applied_action=action, reward=float(reward),
            hazard_dist=info.get('min_hazard_dist', 10.), consumed_food=float(reward) > .3,
            in_friction_zone=info.get('in_friction_zone', False))

    def _evaluation_seed(self, task, episode):
        material = f'omnibus-eval-v2:{self.seed}:{task}:{episode}'.encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:4], 'big') % (2**31)

    def evaluate_agent(self, agent, tasks):
        """Evaluate independent clones, with fixed world seeds/horizons in every phase.

        Recurrent state and synthetic physiology start identically in each episode.
        No outcome is observed by a learner and no live state is changed.
        """
        with torch.random.fork_rng(devices=[]):
            try:
                candidate = deepcopy(agent)
            except Exception:
                from .agent import IntegratedAgent
                candidate = IntegratedAgent(agent.config)
                candidate.import_state(agent.export_state())
            frozen = candidate.export_state()
            scores = {}
            for task in tasks:
                returns = []
                for episode in range(self.eval_episodes):
                    candidate.import_state(frozen)
                    candidate.reset_episode()
                    eval_seed = self._evaluation_seed(task, episode)
                    candidate.homeostasis = HomeostaticEngine(seed=eval_seed)
                    world = ComplexWorld(seed=eval_seed)
                    obs = world.reset(task)
                    score = 0.
                    for _ in range(self.eval_steps):
                        action = candidate.act(obs, learn=False)['action']
                        obs, reward, done, info = world.step(action)
                        self._advance_physiology(candidate, action, reward, info)
                        candidate.pending_transition = None
                        score += float(reward)
                        self.evaluation_steps_executed += 1
                        if done:
                            break
                    returns.append(score)
                scores[task] = float(np.mean(returns))
            return scores

    def step(self):
        if self.is_completed:
            return {'status': 'completed', 'receipt': self.receipt.to_dict()}
        if self.step_in_stage == 0:
            self.stage_baseline_returns.update(self.evaluate_agent(self.agent, [self.current_task]))
        task_executed = self.current_task
        t0 = time.perf_counter()
        out = self.agent.act(self.current_obs, learn=True)
        action, features = out['action'], out['features']
        self.plasticity_monitor.record_activity(features)
        rank = self.plasticity_monitor.evaluate_plasticity()
        if rank.effective_rank is not None:
            self.stage_effective_ranks.append(rank.effective_rank)

        # The next genuine actor feature is the previous transition's target.
        # No fabricated identity transition and no speculative live actor advance.
        if self._previous_transition is not None:
            z, old_action, old_reward = self._previous_transition
            self.uncertainty_last_loss = self.uncertainty_decomposer.train_batch(
                z[None], old_action[None], np.array([old_reward], np.float32), features[None, :32])
            self.uncertainty_training_updates += 1
        pooled = features[:32]
        uncertainty = self.uncertainty_decomposer.evaluate_uncertainty(pooled, action)
        self.stage_epistemic_uncs.append(float(uncertainty.epistemic))
        next_obs, reward, done, info = self.world.step(action)
        self._advance_physiology(self.agent, action, reward, info)
        diagnostics = self.agent.observe_transition(next_obs, reward, done=done, action=action)
        self.agent.feedback(reward, done=done)
        self._previous_transition = None if done else (pooled.copy(), action.copy(), float(reward))
        agency = diagnostics.get('self_model', {}).get('self_authorship')
        disturbance = diagnostics.get('self_model', {}).get('external_disturbance')
        if agency is not None:
            self.stage_agency_scores.append(float(agency))
            self.stage_disturbances.append(float(disturbance))
        self.skill_library.record_transition(self.current_obs, action, reward, next_obs, self.total_steps)
        self.sleep_engine.record_experience(self.current_obs, action, reward,
            abs(float(diagnostics.get('prediction_error', reward))), self.current_task)
        if done and info.get('outcome') == 'success':
            self.skill_library.discover_skills()
        latency = (time.perf_counter() - t0) * 1000.
        self.stage_latencies.append(latency)
        self.ep_return += float(reward)
        self.ep_steps += 1
        self.step_in_stage += 1
        self.total_steps += 1
        self.current_obs = next_obs
        if done or self.ep_steps >= self.world.max_t:
            self.stage_episode_returns.append(self.ep_return)
            self.stage_success_flags.append(info.get('outcome') == 'success')
            self.current_obs = self.world.reset(self.current_task)
            self.agent.reset_episode()
            self._previous_transition = None
            self.ep_return, self.ep_steps = 0., 0
        if self.step_in_stage >= self.steps_per_task:
            self._finalize_current_stage()
        return {'status': 'completed' if self.is_completed else 'running',
            'task': task_executed, 'stage_idx': self.current_stage_idx,
            'step_in_stage': self.step_in_stage, 'total_steps': self.total_steps,
            'reward': float(reward), 'done': bool(done), 'agency': agency, 'disturbance': disturbance,
            'effective_rank': rank.effective_rank, 'rank_status': rank.recommended_action,
            'epistemic_unc': float(uncertainty.epistemic), 'step_latency_ms': latency,
            'is_completed': self.is_completed, 'telemetry': out['telemetry']}

    def _finalize_current_stage(self):
        seen_tasks = self.curriculum_tasks[:self.current_stage_idx + 1]
        if self.sleep_engine.wake_buffer:
            self.sleep_engine.run_sleep_cycle(self.agent, sleep_steps=5,
                batch_size=min(8, len(self.sleep_engine.wake_buffer)),
                eval_fn=lambda candidate: float(np.mean(list(self.evaluate_agent(candidate, seen_tasks).values()))))
        scores = self.evaluate_agent(self.agent, seen_tasks)
        self.initial_stage_returns[self.current_task] = scores[self.current_task]
        self.post_eval_returns = scores
        self.task_score_matrix.append([scores.get(task) for task in self.curriculum_tasks])
        mean_return = _mean(self.stage_episode_returns)
        self.stage_summaries.append(TaskStageSummary(self.current_task, self.current_stage_idx,
            self.step_in_stage, len(self.stage_episode_returns),
            float(mean_return if mean_return is not None else self.ep_return),
            _mean(self.stage_success_flags), _mean(self.stage_agency_scores), _mean(self.stage_disturbances),
            _mean(self.stage_effective_ranks), _mean(self.stage_epistemic_uncs), _mean(self.stage_latencies),
            self.sleep_engine.sleep_cycles_completed))
        self.current_stage_idx += 1
        self._previous_transition = None
        if self.current_stage_idx < len(self.curriculum_tasks):
            self.current_task = self.curriculum_tasks[self.current_stage_idx]
            self.current_obs = self.world.reset(self.current_task)
            self.agent.reset_episode()
            self.step_in_stage, self.ep_steps, self.ep_return = 0, 0, 0.
            self._clear_stage_metrics()
        else:
            self._finalize_omnibus_run()

    def _finalize_omnibus_run(self):
        self.is_completed = True
        # Signed returns require a difference metric, not a ratio with an arbitrary offset.
        old_tasks = self.curriculum_tasks[:-1]
        drops = [max(0., self.initial_stage_returns[t] - self.post_eval_returns[t]) /
                 max(1., abs(self.initial_stage_returns[t])) for t in old_tasks]
        forgetting = min(1., float(np.mean(drops))) if drops else 0.
        protocol = {'version': 2, 'episodes_per_task': self.eval_episodes,
            'steps_per_episode': self.eval_steps, 'seed': self.seed,
            'world_seeds': {t: [self._evaluation_seed(t, e) for e in range(self.eval_episodes)] for t in self.curriculum_tasks},
            'learning': False, 'isolated_clones': True, 'reset_physiology': True,
            'evaluation_steps_executed': self.evaluation_steps_executed,
            'retention_definition': '1 - clipped mean positive return drop / max(1, abs(acquisition return)), excluding final task',
            'skill_evidence': 'candidate observed trajectories; automatic skill execution disabled',
            'uncertainty': 'online adjacent-feature prediction, uncalibrated',
            'sleep_screen': 'matched actual-world average over seen tasks, engineering tolerance 0.05'}
        self.receipt = OmnibusEvaluationReceipt(
            receipt_id=f'omnibus-{time.time_ns()}-{self.seed}', timestamp=time.time(),
            total_neurons=self.total_neurons, connectome_sources=list(self.config.connectome_sources),
            source_node_counts=list(self.config.source_node_counts()), curriculum_tasks=list(self.curriculum_tasks),
            stages=[s.to_dict() for s in self.stage_summaries], final_backward_retention=1. - forgetting,
            catastrophic_forgetting=forgetting, mean_curriculum_return=_mean([s.mean_return for s in self.stage_summaries]),
            overall_mean_agency=_mean([s.mean_agency for s in self.stage_summaries if s.mean_agency is not None]),
            overall_mean_effective_rank=_mean([s.mean_effective_rank for s in self.stage_summaries if s.mean_effective_rank is not None]),
            discovered_skill_count=len(self.skill_library.skills), total_sleep_cycles=self.sleep_engine.sleep_cycles_completed,
            total_wallclock_seconds=time.time() - self.start_wallclock, evaluation_protocol=protocol,
            task_score_matrix=deepcopy(self.task_score_matrix), stage_baseline_returns=deepcopy(self.stage_baseline_returns),
            initial_stage_returns=deepcopy(self.initial_stage_returns), post_eval_returns=deepcopy(self.post_eval_returns),
            configuration=self.config.to_dict(), uncertainty_training_updates=self.uncertainty_training_updates,
            sleep_screens=deepcopy(self.sleep_engine.consolidation_history))
        self.receipt.cryptographic_sha256 = self.receipt.compute_hash()

    def run_full_curriculum(self, steps_per_task=None):
        if steps_per_task is not None:
            if type(steps_per_task) is not int or not 1 <= steps_per_task <= 1000:
                raise ValueError('steps_per_task must be an integer in [1,1000]')
            self.steps_per_task = steps_per_task
        while not self.is_completed:
            self.step()
        return self.receipt

    def summary(self):
        return {'total_neurons': self.total_neurons, 'source_node_counts': list(self.config.source_node_counts()),
            'current_task': self.current_task, 'current_stage_idx': self.current_stage_idx,
            'total_stages': len(self.curriculum_tasks), 'step_in_stage': self.step_in_stage,
            'steps_per_task': self.steps_per_task, 'total_steps': self.total_steps, 'is_completed': self.is_completed,
            'curriculum': list(self.curriculum_tasks), 'stages_completed': [s.to_dict() for s in self.stage_summaries],
            'discovered_skills': [s.to_dict() for s in self.skill_library.skills.values()],
            'skill_library': self.skill_library.summary(),
            'uncertainty': {'training_updates': self.uncertainty_training_updates,
                'calibrated': False, 'last_loss': self.uncertainty_last_loss},
            'evaluation_steps_executed': self.evaluation_steps_executed,
            'receipt': self.receipt.to_dict() if self.receipt else None}
