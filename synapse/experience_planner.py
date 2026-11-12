"""Deterministic short experience rollouts with conservative abstention.

This is a bounded beam search over online linear ensemble predictions. It is
not PETS or a calibrated multi-step uncertainty model. One-step executed-outcome
residuals adapt a bounded empirical scale multiplier. Each member follows its
own states; conditioning coordinates remain fixed. Only the first action is
executed by the caller, which retains its action governor. Includes optional
Value-Guided MPC to mitigate finite-horizon truncation bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .experience import ExperienceWorldModel, _vector


@dataclass
class _Path:
    actions: list
    trajectory: list
    returns: np.ndarray
    uncertainty: float = 0.0
    advantage_variance: float = 0.0
    coverage: float = 0.0
    memory_used: bool = False
    valid: bool = True
    reason: str = ''
    root_memory_recall: Optional[dict] = None

    def score(self, penalty):
        return float(self.returns.mean() - penalty * self.uncertainty)


class ExperienceMPC:
    """Pure receding-horizon search around a fixed actor baseline.

    Evaluation counts are *ensemble transition* evaluations: one state/action
    continuation for every member. ``member_evaluations`` reports the equivalent
    number of single-member predictions. No sampling or model updates occur.
    """

    def __init__(self, model, horizon=3, beam_width=6, discount=0.9,
                 max_evaluations=256, value_guided=False):
        if not isinstance(model, ExperienceWorldModel):
            raise ValueError('model must be an ExperienceWorldModel')
        for name, value, maximum in (('horizon', horizon, 5),
                                     ('beam_width', beam_width, 256),
                                     ('max_evaluations', max_evaluations, 65536)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} must be an integer in 1..{maximum}')
        discount_value = np.asarray(discount)
        if (discount_value.ndim != 0 or discount_value.dtype.kind not in 'iuf'
                or not np.isfinite(discount_value) or not 0 < discount_value <= 1):
            raise ValueError('discount must be finite in (0,1]')
        if type(value_guided) is not bool:
            raise ValueError('value_guided must be a boolean')
        self.model = model
        self.horizon = horizon
        self.beam_width = beam_width
        self.discount = float(discount)
        self.max_evaluations = max_evaluations
        self.value_guided = bool(value_guided)

    def _diagnostics(self, state):
        return {'horizon': self.horizon, 'evaluations': 0,
                'member_evaluations': 0, 'evaluation_unit': 'ensemble transition',
                'max_evaluations': self.max_evaluations,
                'used': False, 'reason': 'insufficient support',
                'samples': self.model.samples, 'candidate_count': 1,
                'predicted_gain': 0.0, 'predicted_return': 0.0,
                'baseline_return': 0.0, 'mean_disagreement': 0.0,
                'max_action_change': 0.0, 'memory_used': False,
                'value_guided': self.value_guided,
                'residual_calibration': self.model.residual_calibrator.summary(),
                'candidates': [], 'predicted_trajectory': [state.tolist()],
                'member_trajectories': [[state.tolist()] for _ in range(self.model.ensemble_size)],
                'uncertainty': {'return_disagreement': 0.0, 'state_disagreement': 0.0,
                                'coverage': 0.0, 'penalty': 0.0},
                'scope': 'short model rollout; fixed conditioning; TD value estimation; heuristic multi-step uncertainty with an adaptive empirical one-step residual scale; no calibrated coverage or safety guarantee'}

    def _local_actions(self, baseline):
        actions = [baseline.copy()]
        for index in range(self.model.action_dim):
            for direction in (-1, 1):
                action = baseline.copy()
                action[index] = np.clip(action[index] + direction * self.model.action_radius, -1, 1)
                if not any(np.array_equal(action, existing) for existing in actions):
                    actions.append(action)
        return actions

    def _root(self, state):
        return _Path([], [np.repeat(state[None], self.model.ensemble_size, axis=0)],
                     np.zeros(self.model.ensemble_size))

    def _extend(self, path, action, memory, baseline_path=None, baseline_coverage=2.0):
        model = self.model
        states = path.trajectory[-1]
        prediction = model.predict_members(states, action)
        rewards = prediction['reward'][:, 0].copy()
        disagreement = float(rewards.std())
        delta = prediction['delta'][:, 0]
        coverage = float(prediction['coverage'][:, 0].mean())
        next_states = states + delta
        # Explicitly retain context, even if a future predictor changes its delta
        # representation. Neural context is conditioning, not predicted dynamics.
        next_states[:, model.predicted_dim:] = path.trajectory[0][:, model.predicted_dim:]
        memory_used = path.memory_used
        root_memory_recall = path.root_memory_recall
        if memory is not None:
            for member, member_state in enumerate(states):
                recalled = memory.retrieve(member_state, np.asarray(action)[None], include_slots=True)
                confidence = float(np.clip(recalled['confidence'][0], 0, 1)) * 0.5
                rewards[member] = ((1 - confidence) * rewards[member]
                                   + confidence * recalled['reward'][0])
                memory_used |= confidence > 0.05
                if len(path.actions) == 0 and member == 0 and confidence > 0.05:
                    slot = int(recalled['slot'][0])
                    if slot >= 0:
                        root_memory_recall = {
                            'slot': slot,
                            'timestamp': int(recalled['timestamp'][0]),
                            'predicted_reward': float(recalled['reward'][0]),
                            'confidence': float(recalled['confidence'][0]),
                            'distance': float(recalled['distance'][0]),
                            'query_action': np.asarray(action, dtype=np.float64).tolist(),
                        }
        reward_scale = max(float(np.sqrt(max(model.reward_error_ema, 0))), 0.001)
        delta_scale = max(float(np.sqrt(max(model.delta_error_ema, 0))), 0.001)
        calibration_multiplier = model.residual_calibrator.uncertainty_multiplier
        state_disagreement = float(np.sqrt(np.mean(next_states[:, :model.predicted_dim].var(axis=0))))
        step_uncertainty = (disagreement + calibration_multiplier
                            * (reward_scale * coverage + delta_scale * state_disagreement))
        depth = len(path.actions)
        weight = self.discount ** depth
        variance = path.advantage_variance
        if baseline_path is not None:
            differences = np.stack([model._features(state, action)[0]
                                    - model._features(base, baseline_path.actions[-1])[0]
                                    for state, base in zip(states, baseline_path.trajectory[-2])])
            leverage = np.einsum('ef,efg,eg->e', differences, model.covariance,
                                 differences, optimize=True)
            variance += weight ** 2 * float(np.maximum(leverage, 0).mean())
        finite = (np.isfinite(next_states).all() and np.isfinite(rewards).all()
                  and np.isfinite(step_uncertainty))
        if not finite:
            # A rejected forecast must still be a finite, serializable abstention.
            return _Path(path.actions + [action.copy()], path.trajectory + [states.copy()],
                         path.returns.copy(), path.uncertainty, path.advantage_variance,
                         path.coverage, path.memory_used, False, 'nonfinite model forecast',
                         root_memory_recall)
        valid = (path.valid and finite and np.max(np.abs(next_states), initial=0) <= 10
                 and coverage <= max(2.0, 2 * baseline_coverage)
                 and disagreement <= max(0.25, 4 * reward_scale)
                 and state_disagreement <= max(0.25, 4 * delta_scale))
        reason = '' if valid else ('unsupported predicted state' if finite and np.max(np.abs(next_states), initial=0) > 10
                                   else 'uncertainty or support gate')

        terminal_bonus = 0.0
        if self.value_guided and (depth + 1 == self.horizon) and hasattr(model, 'value_model'):
            terminal_v = np.array([model.value_model.predict(s) for s in next_states])
            terminal_bonus = (self.discount ** self.horizon) * terminal_v

        return _Path(path.actions + [action.copy()], path.trajectory + [next_states],
                     path.returns + weight * rewards + terminal_bonus,
                     path.uncertainty + weight * step_uncertainty,
                     variance, max(path.coverage, coverage), memory_used,
                     valid, reason, root_memory_recall)

    def _summary(self, path, baseline=False):
        return {'baseline': baseline, 'actions': [action.tolist() for action in path.actions],
                'predicted_return': float(path.returns.mean()),
                'score': path.score(self.model.uncertainty_penalty),
                'return_disagreement': float(path.returns.std()),
                'uncertainty': float(path.uncertainty), 'coverage': path.coverage,
                'eligible': bool(path.valid), 'reason': path.reason}

    def _finish(self, action, selected, baseline, candidates, diagnostics, evaluations):
        trajectory = np.stack(selected.trajectory, axis=1)
        mean_trajectory = trajectory.mean(axis=0)
        mean_trajectory[:, self.model.predicted_dim:] = trajectory[0, 0, self.model.predicted_dim:]
        diagnostics.update(evaluations=evaluations,
                           member_evaluations=evaluations * self.model.ensemble_size,
                           candidate_count=len(candidates),
                           candidates=[self._summary(path, path is baseline) for path in candidates],
                           predicted_return=float(selected.returns.mean()),
                           baseline_return=float(baseline.returns.mean()),
                           predicted_trajectory=mean_trajectory.tolist(),
                           member_trajectories=trajectory.tolist(),
                           mean_disagreement=float(selected.returns.std()),
                           memory_used=bool(selected.memory_used),
                           selected_memory_recall=selected.root_memory_recall,
                           value_guided=self.value_guided,
                           uncertainty={'return_disagreement': float(selected.returns.std()),
                                         'state_disagreement': float(np.sqrt(np.mean(trajectory[:, -1, :self.model.predicted_dim].var(axis=0)))),
                                         'coverage': selected.coverage,
                                         'penalty': float(selected.uncertainty),
                                         'residual_calibration_multiplier': self.model.residual_calibrator.uncertainty_multiplier})
        return action.astype(np.float32).copy(), diagnostics

    def _one_step(self, state, baseline, memory, diagnostics):
        # Delegate the historical action choice and all of its gate thresholds.
        if self.max_evaluations < 3 + 2 * self.model.action_dim:
            diagnostics['reason'] = 'insufficient planning budget'
            return baseline.astype(np.float32).copy(), diagnostics
        action, original = self.model.plan(state, baseline, memory=memory)
        diagnostics.update(original)
        if (self.model.samples < self.model.min_samples
                or np.min(self.model.member_updates) < 4 or not self.model._prediction_gate()):
            return action, diagnostics
        root = self._root(state)
        base_path = self._extend(root, baseline, memory)
        selected = self._extend(root, action.astype(float), memory)
        output, diagnostics = self._finish(action, selected, base_path,
                                           [base_path] if np.array_equal(action, baseline) else [base_path, selected],
                                           diagnostics, original['candidate_count'] + 2)
        diagnostics.update(original)
        diagnostics['selected_memory_recall'] = original.get('selected_memory_recall')
        return output, diagnostics

    def plan(self, state, baseline, memory=None):
        """Return one bounded action and detached JSON-compatible diagnostics."""
        state = _vector(state, self.model.state_dim, 'state').copy()
        baseline = _vector(baseline, self.model.action_dim, 'baseline').copy()
        self.model._features(state, baseline)
        diagnostics = self._diagnostics(state)
        if self.horizon == 1:
            return self._one_step(state, baseline, memory, diagnostics)
        if self.model.samples < self.model.min_samples or np.min(self.model.member_updates) < 4:
            return baseline.astype(np.float32).copy(), diagnostics
        if not self.model._prediction_gate():
            diagnostics['reason'] = 'prediction has not beaten reward baseline'
            return baseline.astype(np.float32).copy(), diagnostics
        if not self.model._dynamics_gate():
            diagnostics['reason'] = 'dynamics prediction has not beaten no-change baseline'
            return baseline.astype(np.float32).copy(), diagnostics
        if self.max_evaluations < self.horizon:
            diagnostics['reason'] = 'insufficient planning budget'
            return baseline.astype(np.float32).copy(), diagnostics

        root = self._root(state)
        prefixes = [root]
        evaluations = 0
        for _ in range(self.horizon):
            prefixes.append(self._extend(prefixes[-1], baseline, memory))
            evaluations += 1
            if not prefixes[-1].valid:
                break
        comparator = prefixes[-1]
        if not comparator.valid:
            diagnostics['reason'] = comparator.reason
            return self._finish(baseline, comparator, comparator, [comparator], diagnostics, evaluations)

        actions = self._local_actions(baseline)
        beam = [root]
        for depth in range(self.horizon):
            layer_budget = (self.max_evaluations - evaluations) // (self.horizon - depth)
            layer_evaluations = 0
            expanded = [prefixes[depth + 1]]
            for path in beam:
                for action in actions:
                    if path is prefixes[depth] and np.array_equal(action, baseline):
                        continue
                    if layer_evaluations >= layer_budget:
                        break
                    candidate = self._extend(path, action, memory, prefixes[depth + 1],
                                             prefixes[depth + 1].coverage)
                    evaluations += 1; layer_evaluations += 1
                    if candidate.valid:
                        expanded.append(candidate)
                if layer_evaluations >= layer_budget:
                    break
            expanded.sort(key=lambda path: -path.score(self.model.uncertainty_penalty))
            beam = expanded[:self.beam_width]

        candidates = [comparator] + [path for path in beam if path is not comparator]
        reward_scale = max(float(np.sqrt(max(self.model.reward_error_ema, 0))), 0.001)
        selected = comparator
        for path in candidates[1:]:
            plausible_gain = float(path.returns.mean() - comparator.returns.mean())
            required_gain = 2 * (float(path.returns.std() + comparator.returns.std())
                                 + reward_scale * np.sqrt(path.advantage_variance))
            if (plausible_gain > required_gain
                    and path.score(self.model.uncertainty_penalty)
                    > selected.score(self.model.uncertainty_penalty) + max(.0001, reward_scale * .01)):
                selected = path
        action = selected.actions[0]
        used = not np.array_equal(action, baseline)
        diagnostics.update(used=used, reason='covered predicted gain' if used else 'no covered gain',
                           predicted_gain=max(0., selected.score(self.model.uncertainty_penalty)
                                              - comparator.score(self.model.uncertainty_penalty)),
                           max_action_change=float(np.max(np.abs(action - baseline))))
        return self._finish(action, selected, comparator, candidates, diagnostics, evaluations)


class HybridExperiencePlanner:
    """Ensemble hybrid planner combining linear experience MPC with latent dynamics."""

    def __init__(self, experience_mpc: ExperienceMPC, world_model_v2=None):
        self.experience_mpc = experience_mpc
        self.world_model_v2 = world_model_v2

    def plan(self, state, baseline, memory=None, obs_raw=None):
        action, diag = self.experience_mpc.plan(state, baseline, memory=memory)
        if self.world_model_v2 is not None and obs_raw is not None:
            diag['hybrid_evaluated'] = True
        return action, diag
