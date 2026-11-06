"""Bounded reset-state replay with isolated, evaluator-gated sleep candidates.

This is motor-head imitation with a trainable native residual and frozen upstream
features. It does not reconstruct historical recurrent states, implement the
full research systems that motivated sleep, or establish a performance benefit.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from .agent import IntegratedAgent


@dataclass
class SalientMemory:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    surprisal: float
    importance: float
    task_id: str = 'default'


class SleepConsolidationEngine:
    """Train an inactive candidate; only a finite retention screen can commit it."""

    def __init__(self, buffer_size=128, replay_buffer_size=256,
                 shy_decay_rate=.03, lr=1e-3, lambda_task=1.,
                 lambda_distill=.5, lambda_replay=.5, lambda_reg=.05, seed=42):
        for name, value in [('buffer_size', buffer_size), ('replay_buffer_size', replay_buffer_size)]:
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError(f'{name} must be an integer in [1,10000]')
        for name, value in [('shy_decay_rate', shy_decay_rate), ('lr', lr),
                            ('lambda_task', lambda_task), ('lambda_distill', lambda_distill),
                            ('lambda_replay', lambda_replay), ('lambda_reg', lambda_reg)]:
            if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if shy_decay_rate > 1 or not 0 < lr <= 1:
            raise ValueError('decay must be at most one and learning rate in (0,1]')
        self.buffer_size = buffer_size
        self.replay_buffer_size = replay_buffer_size
        self.shy_decay_rate = float(shy_decay_rate)
        self.lr = float(lr)
        self.lambda_task = float(lambda_task)
        self.lambda_distill = float(lambda_distill)
        self.lambda_replay = float(lambda_replay)
        self.lambda_reg = float(lambda_reg)
        self.rng = np.random.default_rng(seed)
        self.wake_buffer: List[SalientMemory] = []
        self.long_term_replay: List[SalientMemory] = []
        self.current_phase = 'WAKE'
        self.sleep_cycles_completed = 0
        self.accepted_consolidations = 0
        self.rejected_consolidations = 0
        self.consolidation_history = []

    def record_experience(self, obs, action, reward, surprisal=0., task_id='default'):
        observation = np.asarray(obs, dtype=np.float32)
        executed = np.asarray(action, dtype=np.float32)
        if observation.shape != (155,) or not np.isfinite(observation).all():
            raise ValueError('sleep observation must be finite shape (155,)')
        if executed.shape != (5,) or not np.isfinite(executed).all() or np.any(np.abs(executed) > 1):
            raise ValueError('sleep action must be finite shape (5,) within [-1,1]')
        if not np.isscalar(reward) or not np.isfinite(reward) or not np.isscalar(surprisal) or not np.isfinite(surprisal) or surprisal < 0:
            raise ValueError('reward and nonnegative surprisal must be finite scalars')
        importance = abs(float(reward)) + .5 * float(surprisal)
        memory = SalientMemory(observation.copy(), executed.copy(), float(reward),
                               float(surprisal), importance, str(task_id))
        self.wake_buffer.append(memory)
        if len(self.wake_buffer) > self.buffer_size:
            self.wake_buffer.pop(int(np.argmin([m.importance for m in self.wake_buffer])))
        if len(self.long_term_replay) < self.replay_buffer_size:
            self.long_term_replay.append(memory)
        elif importance > .5 and self.rng.random() < .2:
            self.long_term_replay[int(self.rng.integers(len(self.long_term_replay)))] = memory

    @staticmethod
    def _evaluate(agent, evaluator):
        # Evaluation may advance its clone, never the candidate or live actor.
        value = evaluator(copy.deepcopy(agent))
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError('evaluator must return a finite scalar score')
        return float(value)

    @staticmethod
    def _encode_replay(agent, memories):
        """Encode each observation from the same reset-state frozen actor.

        Native inputs and downstream context come from actual execution. The
        imitation objective is tanh(motor(features)), before online action gates,
        planning and deliberation. Historical recurrent context is unavailable in
        the legacy memory format and is deliberately not inferred.
        """
        actor = copy.deepcopy(agent)
        original = actor.export_state()
        native_inputs, contexts, teachers = [], [], []
        captured = {}
        hook = actor.native.register_forward_pre_hook(
            lambda module, inputs: captured.update(native_input=inputs[0].detach().clone()))
        try:
            for memory in memories:
                actor.import_state(original)
                actor.reset_episode()
                result = actor.act(memory.observation, learn=False, planner_mode='off')
                features = torch.as_tensor(result['features'], dtype=torch.float32)
                native_inputs.append(captured['native_input'])
                contexts.append(features[32:].clone())
                with torch.no_grad():
                    teachers.append(torch.tanh(actor.motor(features)))
        finally:
            hook.remove()
        return {
            'native_input': torch.stack(native_inputs),
            'context': torch.stack(contexts),
            'teacher': torch.stack(teachers),
            'target': torch.as_tensor(np.stack([m.action for m in memories]), dtype=torch.float32),
            'weight': torch.tensor([1. + max(0., m.reward) for m in memories]),
        }

    @staticmethod
    def _head(agent, encoded, indices):
        pooled = encoded['native_input'][indices]
        pooled = pooled + .05 * torch.tanh(agent.native(pooled))
        features = torch.cat([pooled, encoded['context'][indices]], dim=-1)
        return torch.tanh(agent.motor(features))

    def run_sleep_cycle(self, agent: IntegratedAgent, sleep_steps=15, batch_size=16,
                        eval_fn: Optional[Callable[[IntegratedAgent], float]] = None,
                        replays=None) -> Dict[str, Any]:
        if replays is not None:
            sleep_steps = replays
        if type(sleep_steps) is not int or not 1 <= sleep_steps <= 1000:
            raise ValueError('sleep_steps must be an integer in [1,1000]')
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError('batch_size must be an integer in [1,256]')
        if not self.wake_buffer:
            return {'status': 'skipped', 'accepted': False, 'reason': 'empty wake buffer'}
        self.current_phase = 'SLEEP'
        before = agent.export_state()
        accepted = False
        reason = 'evaluator_required'
        pre_score = post_score = delta = None
        loss_records = []
        try:
            if eval_fn is not None:
                pre_score = self._evaluate(agent, eval_fn)
                candidate = copy.deepcopy(agent)
                wake = self._encode_replay(candidate, self.wake_buffer)
                replay = self._encode_replay(candidate, self.long_term_replay) if self.long_term_replay else None
                parameters = [candidate.native.weight, candidate.motor.weight]
                anchors = [p.detach().clone() for p in parameters]
                optimizer = torch.optim.Adam(parameters, lr=self.lr)
                for _ in range(sleep_steps):
                    optimizer.zero_grad()
                    indices = self.rng.choice(len(self.wake_buffer), min(batch_size, len(self.wake_buffer)), replace=False)
                    prediction = self._head(candidate, wake, indices)
                    task_loss = (wake['weight'][indices] * (prediction - wake['target'][indices]).square().mean(-1)).mean()
                    distill_loss = F.mse_loss(prediction, wake['teacher'][indices])
                    replay_loss = torch.tensor(0.)
                    if replay is not None:
                        replay_indices = self.rng.choice(len(self.long_term_replay), min(batch_size, len(self.long_term_replay)), replace=False)
                        replay_loss = F.mse_loss(self._head(candidate, replay, replay_indices), replay['target'][replay_indices])
                    regularizer = sum(F.mse_loss(p, anchor) for p, anchor in zip(parameters, anchors))
                    loss = self.lambda_task * task_loss + self.lambda_distill * distill_loss + self.lambda_replay * replay_loss + self.lambda_reg * regularizer
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite sleep objective')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    optimizer.step()
                    if not all(torch.isfinite(p).all() for p in parameters):
                        raise ValueError('nonfinite candidate parameters')
                    loss_records.append(float(loss.detach()))
                with torch.no_grad():
                    weights = candidate.brain.graft_w
                    weak = weights.abs() < .02
                    weights[weak] *= 1. - self.shy_decay_rate
                post_score = self._evaluate(candidate, eval_fn)
                delta = post_score - pre_score
                if delta < -.05:
                    reason = 'retention_regression'
                else:
                    # Commit permitted weights; keep live recurrence and pending
                    # transitions. Affected native-optimizer moments are stale.
                    changed_grafts = candidate.brain.graft_w.detach() != agent.brain.graft_w.detach()
                    with torch.no_grad():
                        agent.native.weight.copy_(candidate.native.weight)
                        agent.motor.weight.copy_(candidate.motor.weight)
                        agent.brain.graft_w.copy_(candidate.brain.graft_w)
                        for trace in (agent.brain.elig, agent.brain.heb):
                            trace[changed_grafts] = 0.
                        for value in agent.opt.state.get(agent.brain.graft_w, {}).values():
                            if isinstance(value, torch.Tensor) and value.shape == changed_grafts.shape:
                                value[changed_grafts] = 0.
                    for parameter in (agent.motor.weight, agent.native.weight):
                        if parameter in agent.opt.state:
                            del agent.opt.state[parameter]
                    accepted = True
                    reason = 'finite_retention_screen_passed'
        except Exception as exc:
            agent.import_state(before)
            reason = f'evaluation_or_training_error: {type(exc).__name__}: {exc}'
        finally:
            self.current_phase = 'WAKE'
        if accepted:
            self.accepted_consolidations += 1
            self.wake_buffer.clear()
        else:
            self.rejected_consolidations += 1
        self.sleep_cycles_completed += 1
        result = {
            'status': 'ok', 'cycle': self.sleep_cycles_completed, 'accepted': accepted,
            'reason': reason, 'pre_score': pre_score, 'post_score': post_score, 'delta': delta,
            'mean_loss': float(np.mean(loss_records)) if loss_records else 0.,
            'wake_buffer_size': len(self.wake_buffer), 'long_term_replay_size': len(self.long_term_replay),
            'replay_mode': 'reset_state_motor_head',
            'scope': 'bounded motor-head imitation and native residual replay; caller-provided retention screen, no demonstrated task gain',
        }
        self.consolidation_history.append(result)
        return result

    def summary(self):
        return {'phase': self.current_phase, 'cycles_completed': self.sleep_cycles_completed,
                'accepted_consolidations': self.accepted_consolidations,
                'rejected_consolidations': self.rejected_consolidations,
                'wake_buffer_len': len(self.wake_buffer), 'long_term_replay_len': len(self.long_term_replay),
                'recent_history': copy.deepcopy(self.consolidation_history[-5:])}

    def export_state(self):
        def encode(memory):
            return {'obs': memory.observation.tolist(), 'act': memory.action.tolist(),
                    'rew': memory.reward, 'surp': memory.surprisal, 'imp': memory.importance, 'task': memory.task_id}
        return copy.deepcopy({
            'cycles_completed': self.sleep_cycles_completed,
            'accepted_consolidations': self.accepted_consolidations,
            'rejected_consolidations': self.rejected_consolidations,
            'consolidation_history': self.consolidation_history,
            'rng': self.rng.bit_generator.state,
            'wake_buffer': [encode(m) for m in self.wake_buffer],
            'long_term_replay': [encode(m) for m in self.long_term_replay],
        })

    def import_state(self, state):
        def decode(memory):
            return SalientMemory(np.array(memory['obs'], dtype=np.float32), np.array(memory['act'], dtype=np.float32),
                                 memory['rew'], memory['surp'], memory['imp'], memory.get('task', 'default'))
        self.sleep_cycles_completed = int(state.get('cycles_completed', 0))
        self.accepted_consolidations = int(state.get('accepted_consolidations', 0))
        self.rejected_consolidations = int(state.get('rejected_consolidations', 0))
        self.consolidation_history = copy.deepcopy(state.get('consolidation_history', []))
        self.wake_buffer = [decode(m) for m in state.get('wake_buffer', [])]
        self.long_term_replay = [decode(m) for m in state.get('long_term_replay', [])]
        if 'rng' in state:
            self.rng.bit_generator.state = state['rng']
        self.current_phase = 'WAKE'
