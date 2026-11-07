"""Four legacy learning bundles under matched transition and query seeds.

Real-environment mode records ComplexWorld outcomes. The default historical mode
is an explicitly labelled synthetic motor-output surrogate. Neither mode equalizes
optimizer work or establishes a general continual-learning benefit.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from .config import SynapseConfig
from .agent import IntegratedAgent
from .complex_world import ComplexWorld
from .homeostasis import HomeostaticEngine
from .controlled_study import ControlledSixArmStudy
from .sleep import SleepConsolidationEngine
from .verified_curriculum import VerifiableCurriculumEngine


def deterministic_task_seed(base_seed: int, task: str) -> int:
    return int(hashlib.sha256(f'{base_seed}:{task}'.encode()).hexdigest()[:8], 16)


@dataclass
class ArmEvaluationRecord:
    arm_name: str
    stage_returns: List[float]
    final_retention_score: float
    catastrophic_forgetting: float
    total_parameter_modifications: int
    computation_steps: int


class FlagshipContinualBenchmark:
    ARMS = ('fixed_gradient', 'dynamic_plasticity', 'continual_sleep', 'full_developmental_sail')

    def __init__(self, task_sequence=None, steps_per_stage=30, eval_episodes_per_task=2,
                 seed=42, use_real_environment=False):
        self.task_sequence = list(task_sequence) if task_sequence is not None else ['forage', 'delayed_match', 'push', 'ecosystem']
        self.steps_per_stage = steps_per_stage
        self.eval_episodes_per_task = eval_episodes_per_task
        self.seed = seed
        self.use_real_environment = use_real_environment
        self.evaluation_steps_executed = 0

    def run_benchmark(self):
        results = {name: self._run_arm(name) for name in self.ARMS}
        return {'status': 'ok', 'task_sequence': self.task_sequence,
            'steps_per_stage': self.steps_per_stage, 'seed': self.seed,
            'arm_results': results, 'comparative_summary': self._compute_comparison_matrix(results),
            'evidence_kind': 'embodied_environment_outcomes' if self.use_real_environment else 'synthetic_surrogate',
            'native_world_evaluation': self.use_real_environment,
            'reproducibility': 'sha256_deterministic',
            'protocol': {'matched_training_seeds': True, 'query_clones': True,
                'evaluation_steps_per_episode': 25,
                'budget': 'equal training transitions; gradient, plasticity and sleep work differ'},
            'limitations': ['The fixed_gradient arm uses reward-scaled self-imitation, not a standard RL baseline.',
                'Learning bundles retain common inherited actor mechanisms; this is not complete causal isolation.',
                'No held-out confirmation, calibrated uncertainty or general learning advantage is established.'],
            'scope': 'four-arm finite-world development study' if self.use_real_environment else 'four-arm synthetic motor-output surrogate'}

    @staticmethod
    def _surrogate_reward(task, action):
        if task == 'forage':
            return float(action[0] - .1 * abs(action[1]))
        if task == 'delayed_match':
            return float(action[2] * 1.5 - .2 * abs(action[0]))
        if task == 'push':
            return float(action[0] * 1.2 + action[4])
        return float(action[0] * .8 - .3 * abs(action[1]))

    def _evaluate_agent_on_task(self, agent, task):
        actor = deepcopy(agent)
        frozen = actor.export_state()
        returns = []
        for episode in range(self.eval_episodes_per_task):
            seed = deterministic_task_seed(self.seed, f'eval_{task}_{episode}') % (2**31)
            actor.import_state(frozen)
            actor.reset_episode()
            actor.homeostasis = HomeostaticEngine(seed=seed)
            rng = np.random.default_rng(seed)
            world = ComplexWorld(seed=seed) if self.use_real_environment else None
            obs = world.reset(task) if world else rng.uniform(0., 1., 155).astype(np.float32)
            total = 0.
            for _ in range(25):
                action = actor.act(obs, learn=False)['action']
                if world:
                    obs, reward, done, info = world.step(action)
                    ControlledSixArmStudy._advance_physiology(actor, action, reward, info)
                else:
                    reward, done = self._surrogate_reward(task, action), False
                    obs = rng.uniform(0., 1., 155).astype(np.float32)
                actor.pending_transition = None
                total += float(reward)
                self.evaluation_steps_executed += 1
                if done:
                    break
            returns.append(total)
        return float(np.mean(returns))

    def _run_arm(self, name):
        if name not in self.ARMS:
            raise ValueError('unsupported study arm')
        fixed = name == 'fixed_gradient'
        has_sleep = name in ('continual_sleep', 'full_developmental_sail')
        full = name == 'full_developmental_sail'
        config = SynapseConfig(nodes_per_source=16, seed=self.seed,
            structural_learning=not fixed, continual_backprop=has_sleep, fast_weight_memory=full)
        agent = IntegratedAgent(config)
        optimizer = torch.optim.Adam([agent.motor.weight], lr=1e-3) if fixed else None
        sleep = SleepConsolidationEngine(seed=self.seed) if has_sleep else None
        curriculum = VerifiableCurriculumEngine(base_seed=self.seed, mastery_threshold=.1) if full else None
        matrix = np.zeros((len(self.task_sequence), len(self.task_sequence)))
        stages, training_seeds = [], {}
        eval_before = self.evaluation_steps_executed
        training_updates = 0
        for stage, task in enumerate(self.task_sequence):
            seed = deterministic_task_seed(self.seed, f'train_{task}') % (2**31)
            training_seeds[task] = seed
            rng = np.random.default_rng(seed)
            world = ComplexWorld(seed=seed) if self.use_real_environment else None
            obs = world.reset(task) if world else rng.uniform(0., 1., 155).astype(np.float32)
            agent.reset_episode()
            for step in range(self.steps_per_stage):
                out = agent.act(obs, learn=not fixed)
                if world:
                    next_obs, reward, done, info = world.step(out['action'])
                    ControlledSixArmStudy._advance_physiology(agent, out['action'], reward, info)
                else:
                    next_obs = rng.uniform(0., 1., 155).astype(np.float32)
                    reward = self._surrogate_reward(task, out['action'])
                    done, info = False, {}
                if fixed:
                    optimizer.zero_grad()
                    features = torch.as_tensor(out['features'], dtype=torch.float32)
                    prediction = agent.motor(features)
                    target = torch.as_tensor(out['action']) * float(math.tanh(reward))
                    (prediction - target).square().mean().backward()
                    optimizer.step()
                    agent.pending_transition = None
                    training_updates += 1
                else:
                    agent.observe_transition(next_obs, float(reward), done=done, action=out['action'])
                    agent.feedback(float(reward), done=done)
                if sleep:
                    sleep.record_experience(obs, out['action'], reward, task_id=task)
                if done:
                    if curriculum:
                        curriculum.complete_episode(success=info.get('outcome') == 'success')
                    obs = world.reset(task)
                    agent.reset_episode()
                else:
                    obs = next_obs
            if sleep:
                seen = self.task_sequence[:stage + 1]
                sleep.run_sleep_cycle(agent, sleep_steps=10, eval_fn=lambda candidate: float(np.mean([
                    self._evaluate_agent_on_task(candidate, task_name) for task_name in seen])))
            for previous in range(stage + 1):
                matrix[stage, previous] = self._evaluate_agent_on_task(agent, self.task_sequence[previous])
            stages.append(float(matrix[stage, stage]))
        retention, forgetting = self._calc_retention_and_forgetting(matrix)
        result = {'stage_returns': stages, 'final_retention_rate': retention,
            'catastrophic_forgetting': forgetting, 'task_matrix': matrix.tolist(),
            'training_seeds': training_seeds, 'training_transitions': self.steps_per_stage * len(self.task_sequence),
            'evaluation_steps_executed': self.evaluation_steps_executed - eval_before,
            'gradient_updates': training_updates, 'configuration': config.to_dict()}
        if sleep:
            result.update(sleep_summary=sleep.summary(),
                cbp_replacements=agent.continual_backprop.total_replacements,
                cbp_utility_updates=agent.continual_backprop.step_count)
        if full:
            result.update(fast_weight_updates=agent.fast_weight_memory.total_inner_updates,
                curriculum_progress=curriculum.summary(), verified_task_success=False)
        return result

    def _run_arm_fixed_gradient(self):
        return self._run_arm('fixed_gradient')

    def _run_arm_dynamic_plasticity(self):
        return self._run_arm('dynamic_plasticity')

    def _run_arm_continual_sleep(self):
        return self._run_arm('continual_sleep')

    def _run_arm_full_sail(self):
        return self._run_arm('full_developmental_sail')

    def _calc_retention_and_forgetting(self, matrix):
        return ControlledSixArmStudy._calc_retention_and_forgetting(matrix)

    def _compute_comparison_matrix(self, results):
        baseline = results['fixed_gradient']
        return {name: {'retention_rate': result['final_retention_rate'],
            'retention_advantage_vs_fixed': result['final_retention_rate'] - baseline['final_retention_rate'],
            'catastrophic_forgetting': result['catastrophic_forgetting'],
            'forgetting_reduction_vs_fixed': baseline['catastrophic_forgetting'] - result['catastrophic_forgetting'],
            'target_retention_met': result['final_retention_rate'] >= .95} for name, result in results.items()}
