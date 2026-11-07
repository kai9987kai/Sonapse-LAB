"""Six-arm sequential adaptation study with matched world-transition budgets.

Grounding & Review Requirements:
Directly implements the experimental design recommended in the 6 November technical review:
Compares mechanism bundles on matched seeds; optimizer/sleep compute differs.
1. 'frozen_actor': Establish fixed baseline.
2. 'online_rl': Measure online learning capacity.
3. 'actor_replay': Measure memory and replay consolidation benefit.
4. 'continual_backprop': Measure plasticity preservation and utility replacement.
5. 'connectome_evolution': Measure structural plasticity and neurogenesis.
6. 'composed_system': Test whether accepted mechanisms compose synergistically.

Evaluates:
- Adaptation speed (steps/cumulative return per stage)
- Backward retention score (target >= 95%)
- Catastrophic forgetting index
- Inference latency per step (ms)
- Forward / negative transfer
- Real ComplexWorld environment outcomes with SHA-256 deterministic seeds.
"""
from __future__ import annotations
from copy import deepcopy
import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from .config import SynapseConfig
from .agent import IntegratedAgent
from .complex_world import ComplexWorld
from .continual_backprop import ContinualBackpropEngine
from .sleep import SleepConsolidationEngine
from .uncertainty_calibration import TripartiteUncertaintyDecomposer
from .self_model import EfferenceCopyComparator
from .homeostasis import HomeostaticEngine


def deterministic_seed(base_seed: int, label: str) -> int:
    """Generate reproducible 31-bit seed via SHA-256."""
    h = hashlib.sha256(f"{base_seed}_{label}".encode('utf-8')).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


@dataclass
class ArmResult:
    """Results and metrics for a single experimental arm."""
    arm_name: str
    purpose: str
    stage_returns: List[float]
    final_retention_rate: float
    catastrophic_forgetting: float
    adaptation_speed: float
    negative_transfer: Optional[float]
    mean_step_latency_ms: float
    task_matrix: List[List[float]]
    auxiliary_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ControlledSixArmStudy:
    """Rigorous 6-arm sequential adaptation study."""

    ARMS = [
        ('frozen_actor', 'Establish fixed baseline'),
        ('online_rl', 'Measure online learning capacity'),
        ('actor_replay', 'Measure memory benefit'),
        ('continual_backprop', 'Measure plasticity preservation'),
        ('connectome_evolution', 'Measure architectural adaptation'),
        ('composed_system', 'Test whether improvements compose')
    ]

    def __init__(
        self,
        task_sequence: Optional[List[str]] = None,
        steps_per_stage: int = 150,
        eval_episodes_per_task: int = 3,
        seed: int = 42,
        nodes_per_source: int = 16
    ):
        self.task_sequence = task_sequence or ['forage', 'maze', 'push', 'ecosystem']
        self.steps_per_stage = steps_per_stage
        self.eval_episodes_per_task = eval_episodes_per_task
        self.seed = seed
        self.nodes_per_source = nodes_per_source
        self.evaluation_steps_executed = 0

    def run_study(self) -> Dict[str, Any]:
        """Execute all 6 arms under identical conditions."""
        results: Dict[str, ArmResult] = {}

        for arm_name, purpose in self.ARMS:
            res = self._run_single_arm(arm_name, purpose)
            results[arm_name] = res

        summary = self._compile_comparative_summary(results)
        return summary

    def _run_single_arm(self, arm_name: str, purpose: str) -> ArmResult:
        """Run a single arm across the full task sequence."""
        cfg = SynapseConfig(
            nodes_per_source=self.nodes_per_source,
            seed=self.seed,
            continual_backprop=(arm_name in ('continual_backprop', 'composed_system')),
            self_model=(arm_name == 'composed_system'),
            structural_learning=(arm_name in ('connectome_evolution', 'composed_system')),
        )
        agent = IntegratedAgent(cfg)

        # Auxiliary components based on arm
        sleep_engine = SleepConsolidationEngine() if arm_name in ('actor_replay', 'composed_system') else None
        cbp_engine = agent.continual_backprop
        tripartite = TripartiteUncertaintyDecomposer(state_dim=16, action_dim=5, seed=self.seed) if arm_name == 'composed_system' else None

        num_tasks = len(self.task_sequence)
        task_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        stage_returns = []
        step_latencies = []
        training_seeds = {}
        eval_work_before = self.evaluation_steps_executed

        for stage_idx, current_task in enumerate(self.task_sequence):
            train_seed = deterministic_seed(self.seed, f"train_{current_task}")
            training_seeds[current_task] = train_seed
            world = ComplexWorld(seed=train_seed)
            obs = world.reset(task=current_task)
            agent.reset_episode()
            stage_cum_reward = 0.0

            for step in range(self.steps_per_stage):
                t0 = time.perf_counter()
                learn_flag = (arm_name != 'frozen_actor')

                # Act
                out = agent.act(obs, learn=learn_flag)
                act = out['action']

                # Step world
                next_obs, rew, done, info = world.step(act)
                stage_cum_reward += float(rew)
                self._advance_physiology(agent, act, rew, info)
                t_step = (time.perf_counter() - t0) * 1000.0
                step_latencies.append(t_step)

                # Arm-specific updates
                if arm_name != 'frozen_actor':
                    agent.observe_transition(next_obs, float(rew), done=bool(done), action=act)
                    agent.feedback(float(rew), done=bool(done))

                    if sleep_engine is not None:
                        sleep_engine.record_experience(obs, act, float(rew), task_id=current_task)

                    if tripartite is not None:
                        # Explicit observed channels, not an invented latent identity target.
                        z_curr, z_next = obs[:16].copy(), next_obs[:16].copy()
                        tripartite.train_batch(
                            z_curr[np.newaxis, :], act[np.newaxis, :], np.array([float(rew)]), z_next[np.newaxis, :]
                        )

                if done:
                    obs = world.reset(task=current_task)
                    agent.reset_episode()
                else:
                    obs = next_obs

            # Post-stage replay consolidation
            if sleep_engine is not None:
                seen_tasks = self.task_sequence[:stage_idx + 1]
                sleep_engine.run_sleep_cycle(agent, sleep_steps=10,
                    eval_fn=lambda candidate: float(np.mean([
                        self._evaluate_agent(candidate, task, 0) for task in seen_tasks])))

            # Evaluate on all tasks seen so far
            for prev_idx in range(stage_idx + 1):
                t_name = self.task_sequence[prev_idx]
                eval_score = self._evaluate_agent(agent, t_name, stage_idx)
                task_matrix[stage_idx, prev_idx] = eval_score

            stage_score = float(task_matrix[stage_idx, stage_idx])
            stage_returns.append(stage_score)

        retention, forgetting = self._calc_retention_and_forgetting(task_matrix)
        adaptation_speed = float(np.mean(stage_returns))
        # Different task returns are not a negative-transfer experiment.
        neg_transfer = None

        aux = {'training_seeds': training_seeds, 'training_transitions': self.steps_per_stage * num_tasks,
               'evaluation_world_steps': self.evaluation_steps_executed - eval_work_before,
               'structural_learning': cfg.structural_learning, 'negative_transfer_measured': False,
               'uncertainty_input': 'observed_channels_0_16' if tripartite else None}
        if cbp_engine:
            aux['cbp_replacements'] = cbp_engine.total_replacements
            aux['cbp_utility_updates'] = cbp_engine.step_count
        if sleep_engine:
            aux['sleep_consolidations'] = sleep_engine.sleep_cycles_completed
            aux['sleep_screens'] = deepcopy(sleep_engine.consolidation_history)

        return ArmResult(
            arm_name=arm_name,
            purpose=purpose,
            stage_returns=[round(float(s), 4) for s in stage_returns],
            final_retention_rate=round(float(retention), 4),
            catastrophic_forgetting=round(float(forgetting), 4),
            adaptation_speed=round(adaptation_speed, 4),
            negative_transfer=neg_transfer,
            mean_step_latency_ms=round(float(np.mean(step_latencies)), 3),
            task_matrix=[[round(float(v), 4) for v in row] for row in task_matrix.tolist()],
            auxiliary_metrics=aux
        )

    def _evaluate_agent(self, agent: IntegratedAgent, task_name: str, eval_id: int) -> float:
        """Evaluate agent in frozen evaluation mode over held-out episodes."""
        # eval_id is retained for API compatibility; phases use identical episodes.
        actor = deepcopy(agent)
        frozen = actor.export_state()
        episode_rewards = []

        for ep in range(self.eval_episodes_per_task):
            eval_seed = deterministic_seed(self.seed, f"eval_{task_name}_{ep}")
            world = ComplexWorld(seed=eval_seed)
            obs = world.reset(task=task_name)
            actor.import_state(frozen)
            actor.reset_episode()
            actor.homeostasis = HomeostaticEngine(seed=eval_seed)
            ep_rew = 0.0
            done = False
            steps = 0
            while not done and steps < 60:
                out = actor.act(obs, learn=False)
                next_obs, r, done, info = world.step(out['action'])
                self._advance_physiology(actor, out['action'], r, info)
                actor.pending_transition = None
                ep_rew += float(r)
                obs = next_obs
                steps += 1
                self.evaluation_steps_executed += 1
            episode_rewards.append(ep_rew)

        return float(np.mean(episode_rewards))

    @staticmethod
    def _advance_physiology(agent, action, reward, info):
        agent.homeostasis.step(action, float(reward), hazard_dist=info.get('min_hazard_dist', 10.),
            consumed_food=float(reward) > .3, in_friction_zone=info.get('in_friction_zone', False))

    @staticmethod
    def _calc_retention_and_forgetting(matrix: np.ndarray) -> Tuple[float, float]:
        """Compute backward retention and catastrophic forgetting."""
        N = matrix.shape[0]
        if N <= 1:
            return 1.0, 0.0

        retentions = []
        forgettings = []
        for j in range(N - 1):
            peak = float(np.max(matrix[j:, j]))
            final = float(matrix[-1, j])
            acquired = float(matrix[j, j])
            denom = max(abs(acquired), 1.)
            retentions.append(max(0.0, 1. - max(0., acquired - final) / denom))
            forgettings.append(max(0.0, peak - final))

        mean_ret = float(np.mean(retentions)) if retentions else 1.0
        mean_forg = float(np.mean(forgettings)) if forgettings else 0.0
        return mean_ret, mean_forg

    def _compile_comparative_summary(self, results: Dict[str, ArmResult]) -> Dict[str, Any]:
        """Compile comparative analysis table and empirical rankings."""
        summary = {
            'task_sequence': self.task_sequence,
            'steps_per_stage': self.steps_per_stage,
            'eval_episodes_per_task': self.eval_episodes_per_task,
            'base_seed': self.seed,
            'evidence_kind': 'measured_world_development_study',
            'protocol': {'matched_training_seeds': True, 'matched_query_seeds_across_stages': True,
                'evaluation_steps_per_episode': 60, 'query_learning': False, 'query_clones': True,
                'budget': 'equal training world transitions; optimizer and sleep work differ'},
            'limitations': ['Mechanism bundles retain common inherited learning and dormant mechanisms.',
                'online_rl is a legacy arm name: updates are reward-modulated Hebbian plasticity, not a new policy-gradient baseline.',
                'adaptation_speed stores mean post-stage return; no time-to-mastery or negative-transfer control is measured.',
                'Finite development tasks and seeds do not establish a general continual-learning advantage.'],
            'arms': {k: v.to_dict() for k, v in results.items()},
            'comparative_table': []
        }

        for arm_name, r in results.items():
            summary['comparative_table'].append({
                'arm': arm_name,
                'purpose': r.purpose,
                'mean_return': r.adaptation_speed,
                'retention_rate': r.final_retention_rate,
                'catastrophic_forgetting': r.catastrophic_forgetting,
                'step_latency_ms': r.mean_step_latency_ms
            })

        # Check if composed beats frozen baseline
        frozen_ret = results['frozen_actor'].adaptation_speed
        composed_ret = results['composed_system'].adaptation_speed
        summary['delta_composed_vs_frozen'] = round(composed_ret - frozen_ret, 4)
        summary['retention_target_met'] = bool(results['composed_system'].final_retention_rate >= 0.95)

        return summary
