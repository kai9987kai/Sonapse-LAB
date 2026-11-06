"""Small learned experience model and conservative one-step action experiments.

Bootstrap RLS members learn *executed* transitions. Covariance leverage and
ensemble disagreement are coverage heuristics, not calibrated error bounds.
This CPU model is an experimental alternative to the standalone neural demos.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite shape ({size},)')
    if np.max(np.abs(array), initial=0) > 1e6:
        raise ValueError(f'{name} exceeds numeric bound')
    return array


def encode_experience_state(obs, features=None):
    """Summarize visible rays, motion, cues and actual graph/donor activity.

    No world coordinates, task labels, hidden targets or future outcomes enter.
    The summary is deliberately lossy and has no Markov-state guarantee.
    """
    obs = _vector(obs, 155, 'observation')
    features = np.zeros(64) if features is None else _vector(features, 64, 'features')
    depth = obs[:33]
    red, green, _ = obs[33:132].reshape(3, 33)
    summary = [float(part.mean()) for signal in (depth, depth * green, depth * red)
               for part in np.array_split(signal, 3)]
    summary.extend([obs[136], obs[137], obs[149] - obs[150],
                    obs[151] - obs[152], obs[154],
                    features[:32].mean(), features[32:48].mean()])
    return np.clip(np.asarray(summary, dtype=np.float64), -2, 2)


class ExperienceWorldModel:
    """Bounded-size bootstrap linear models with state/action interactions.

    The feature basis contains bias, state, action, state×action and action².
    RLS updates a regularized least-squares model with forgetting and a stabilizing
    covariance trace cap; Poisson(1) weights provide deterministic bootstrap members.
    Next-state deltas are recorded for prediction diagnostics. Planning explicitly
    uses immediate rewards only, so it cannot solve long-horizon credit assignment.
    """

    def __init__(self, state_dim=16, action_dim=5, ensemble_size=3, seed=7,
                 ridge=0.25, forgetting=0.995, min_samples=24,
                 action_radius=0.35, uncertainty_penalty=0.25, context_dim=0):
        for name, value, maximum in [('state_dim', state_dim, 64),
                                      ('action_dim', action_dim, 16),
                                      ('ensemble_size', ensemble_size, 8)]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f'{name} outside supported bounds')
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('seed must be a nonnegative 31-bit integer')
        if type(context_dim) is not int or not 0 <= context_dim < state_dim:
            raise ValueError('context_dim must leave at least one predicted state dimension')
        if (not np.isfinite([ridge, forgetting, action_radius, uncertainty_penalty]).all()
                or not 0.001 <= ridge <= 100 or not 0.9 <= forgetting <= 1
                or not 0 < action_radius <= 0.35 or not 0 <= uncertainty_penalty <= 10
                or type(min_samples) is not int or not 1 <= min_samples <= 10000):
            raise ValueError('invalid experience-model hyperparameters')
        self.state_dim = state_dim; self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.context_dim = context_dim
        self.predicted_dim = state_dim - context_dim
        self.ridge = float(ridge); self.forgetting = float(forgetting)
        self.min_samples = min_samples; self.action_radius = float(action_radius)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.dimension = 1 + state_dim + 2 * action_dim + state_dim * action_dim
        self.weights = np.zeros((ensemble_size, self.dimension, self.predicted_dim + 1))
        self.covariance = np.repeat((np.eye(self.dimension) / ridge)[None], ensemble_size, axis=0)
        self.rng = np.random.default_rng(seed)
        self.samples = 0; self.member_updates = np.zeros(ensemble_size, dtype=np.int64)
        self.reward_error_ema = 0.0; self.delta_error_ema = 0.0
        self.reward_mean = 0.0; self.baseline_error_ema = 0.0
        self.reward_squared_error = 0.0; self.delta_squared_error = 0.0
        self.terminals = 0

    def _actions(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if (actions.ndim != 2 or actions.shape[1] != self.action_dim
                or not 1 <= len(actions) <= 256 or not np.isfinite(actions).all()
                or np.max(np.abs(actions), initial=0) > 1.000001):
            raise ValueError('actions must be finite bounded [N, action_dim], N in 1..256')
        return np.clip(actions, -1, 1)

    def _features(self, state, actions):
        state = _vector(state, self.state_dim, 'state')
        if np.max(np.abs(state), initial=0) > 10:
            raise ValueError('state must be encoded within [-10,10]')
        actions = self._actions(actions)
        states = np.broadcast_to(state, (len(actions), self.state_dim))
        interactions = (states[:, :, None] * actions[:, None, :]).reshape(len(actions), -1)
        return np.concatenate([np.ones((len(actions), 1)), states, actions,
                               interactions, actions ** 2], axis=1)

    def predict(self, state, actions):
        """Pure read; returns one reward/delta/coverage per candidate action."""
        phi = self._features(state, actions)
        predictions = np.einsum('nf,efk->enk', phi, self.weights, optimize=True)
        leverage = np.einsum('nf,efg,ng->en', phi, self.covariance, phi, optimize=True)
        mean = predictions.mean(axis=0)
        delta = np.zeros((len(phi), self.state_dim))
        delta[:, :self.predicted_dim] = mean[:, :-1]
        return {'reward': mean[:, -1].copy(), 'delta': delta,
                'disagreement': predictions[:, :, -1].std(axis=0),
                'coverage': np.sqrt(np.maximum(leverage.mean(axis=0), 0))}

    def update(self, state, action, next_state, reward, done=False):
        """Learn only after an executed action; errors are measured before fitting."""
        state = _vector(state, self.state_dim, 'state')
        next_state = _vector(next_state, self.state_dim, 'next state')
        if np.max(np.abs(next_state), initial=0) > 10:
            raise ValueError('next state must be encoded within [-10,10]')
        action = _vector(action, self.action_dim, 'action')
        if not np.isscalar(reward) or not np.isfinite(reward) or abs(reward) > 1e6:
            raise ValueError('reward must be a finite bounded scalar')
        if type(done) not in (bool, np.bool_):
            raise ValueError('done must be boolean')
        phi = self._features(state, action)[0]
        prediction = self.predict(state, action)
        error = float(reward - prediction['reward'][0])
        baseline_error = float(reward - self.reward_mean)
        delta = (next_state - state)[:self.predicted_dim]
        delta_error = float(np.mean((delta - prediction['delta'][0, :self.predicted_dim]) ** 2))
        target = np.concatenate([delta, [float(reward)]])
        # All validation precedes RNG consumption or learned-state mutation.
        counts = self.rng.poisson(1.0, size=self.ensemble_size)
        for member, count in enumerate(counts):
            covariance = self.covariance[member]
            if count == 0:
                covariance = covariance / self.forgetting
            else:
                projected = covariance @ phi
                denominator = self.forgetting / count + phi @ projected
                gain = projected / denominator
                residual = target - phi @ self.weights[member]
                self.weights[member] += np.outer(gain, residual)
                covariance = (covariance - np.outer(gain, projected)) / self.forgetting
                self.member_updates[member] += 1
            # Unexcited directions would otherwise grow exponentially in live runs.
            trace = float(np.trace(covariance))
            if trace > 10000:
                covariance *= 10000 / trace
            # Symmetrize numerical roundoff; no unbounded experience buffer.
            self.covariance[member] = (covariance + covariance.T) * 0.5
        self.samples += 1; self.terminals += int(done)
        rate = 1.0 if self.samples == 1 else 0.05
        self.reward_error_ema += rate * (error ** 2 - self.reward_error_ema)
        self.baseline_error_ema += rate * (baseline_error ** 2 - self.baseline_error_ema)
        self.reward_mean += rate * (float(reward) - self.reward_mean)
        self.delta_error_ema += rate * (delta_error - self.delta_error_ema)
        self.reward_squared_error += error ** 2
        self.delta_squared_error += delta_error
        return {'prediction_error': error, 'reward_squared_error': error ** 2,
                'delta_squared_error': delta_error, 'samples': self.samples,
                'member_updates': int(counts.astype(bool).sum())}

    def plan(self, state, baseline, memory=None):
        """Rank local candidates; abstain without support or covered predicted gain."""
        baseline = _vector(baseline, self.action_dim, 'baseline')
        self._actions(baseline)
        _vector(state, self.state_dim, 'state')
        diagnostics = {'used': False, 'reason': 'insufficient support', 'samples': self.samples,
                       'candidate_count': 1, 'predicted_gain': 0.0,
                       'mean_disagreement': 0.0, 'max_action_change': 0.0,
                       'memory_used': False}
        if self.samples < self.min_samples or np.min(self.member_updates) < 4:
            return baseline.astype(np.float32).copy(), diagnostics
        # Measured before-update errors must improve on a running reward average.
        # This is a support-data heuristic, not a significance or safety test.
        if not self._prediction_gate():
            diagnostics['reason'] = 'prediction has not beaten reward baseline'
            return baseline.astype(np.float32).copy(), diagnostics
        candidates = [baseline.copy()]
        for index in range(self.action_dim):
            for direction in (-1, 1):
                candidate = baseline.copy()
                candidate[index] = np.clip(candidate[index] + direction * self.action_radius, -1, 1)
                candidates.append(candidate)
        actions = np.asarray(candidates)
        prediction = self.predict(state, actions)
        reward = prediction['reward'].copy()
        if memory is not None:
            recalled = memory.retrieve(state, actions)
            # Memory needs local coverage and remains a correction, not a reward oracle.
            confidence = np.clip(np.asarray(recalled['confidence'], dtype=float), 0, 1) * 0.5
            reward = (1 - confidence) * reward + confidence * recalled['reward']
            diagnostics['memory_used'] = bool(np.max(confidence) > 0.05)
        scale = max(np.sqrt(max(self.reward_error_ema, 0)), 0.001)
        uncertainty = prediction['disagreement'] + scale * prediction['coverage']
        score = reward - self.uncertainty_penalty * uncertainty
        # Extremely unsupported alternatives are excluded even when the ensemble agrees.
        score[prediction['coverage'] > max(2.0, 2 * prediction['coverage'][0])] = -np.inf
        score[0] = reward[0] - self.uncertainty_penalty * uncertainty[0]
        phi = self._features(state, actions)
        difference = phi - phi[0]
        advantage_coverage = np.sqrt(np.maximum(np.einsum(
            'nf,efg,ng->en', difference, self.covariance, difference, optimize=True).mean(axis=0), 0))
        # Penalize uncertainty of the *change* rather than assuming cancellation
        # of two uncertain reward predictions. The multiplier is an engineering
        # margin, not a calibrated two-sigma bound.
        plausible_gain = reward - reward[0]
        required_gain = 2 * (prediction['disagreement'] + prediction['disagreement'][0]
                             + scale * advantage_coverage)
        score[(plausible_gain <= required_gain) & (np.arange(len(actions)) != 0)] = -np.inf
        best = int(np.argmax(score))
        gain = float(score[best] - score[0])
        if gain <= max(0.0001, scale * 0.01):
            best = 0
        action = actions[best].astype(np.float32)
        diagnostics.update(used=best != 0, reason='covered predicted gain' if best else 'no covered gain',
                           candidate_count=len(actions), predicted_gain=max(gain, 0),
                           mean_disagreement=float(prediction['disagreement'].mean()),
                           max_action_change=float(np.max(np.abs(actions[best] - baseline))))
        return action, diagnostics

    def _prediction_gate(self):
        return bool(self.samples >= self.min_samples
                    and self.baseline_error_ema > 1e-12
                    and self.reward_error_ema < 0.95 * self.baseline_error_ema)

    def summary(self):
        return {'samples': self.samples, 'ensemble_size': self.ensemble_size,
                'predicted_state_dimensions': self.predicted_dim, 'conditioning_dimensions': self.context_dim,
                'member_updates': self.member_updates.tolist(), 'terminals': self.terminals,
                'prequential_reward_rmse': float(np.sqrt(self.reward_squared_error / max(1, self.samples))),
                'recent_reward_rmse': float(np.sqrt(max(self.reward_error_ema, 0))),
                'reward_baseline_rmse': float(np.sqrt(max(self.baseline_error_ema, 0))),
                'prediction_gate_passed': self._prediction_gate(),
                'prequential_delta_rmse': float(np.sqrt(self.delta_squared_error / max(1, self.samples))),
                'learned_bytes': int(self.weights.nbytes + self.covariance.nbytes),
                'covariance_trace_cap': 10000,
                'scope': 'bootstrap RLS; heuristic coverage; one-step reward planning; conditioning dynamics excluded; experimental'}

    def export_state(self):
        return deepcopy({'schema': 1, 'state_dim': self.state_dim, 'action_dim': self.action_dim,
                         'ensemble_size': self.ensemble_size, 'ridge': self.ridge,
                         'context_dim': self.context_dim,
                         'forgetting': self.forgetting, 'min_samples': self.min_samples,
                         'action_radius': self.action_radius, 'uncertainty_penalty': self.uncertainty_penalty,
                         'weights': self.weights, 'covariance': self.covariance,
                         'member_updates': self.member_updates, 'rng': self.rng.bit_generator.state,
                         'samples': self.samples, 'terminals': self.terminals,
                         'reward_error_ema': self.reward_error_ema, 'delta_error_ema': self.delta_error_ema,
                         'reward_mean': self.reward_mean, 'baseline_error_ema': self.baseline_error_ema,
                         'reward_squared_error': self.reward_squared_error,
                         'delta_squared_error': self.delta_squared_error})

    def import_state(self, state):
        """Validate fully before replacing the resident model."""
        if not isinstance(state, dict) or state.get('schema') != 1:
            raise ValueError('invalid experience state schema')
        candidate = ExperienceWorldModel(context_dim=state.get('context_dim', 0), **{key: state[key] for key in (
            'state_dim', 'action_dim', 'ensemble_size', 'ridge', 'forgetting',
            'min_samples', 'action_radius', 'uncertainty_penalty')})
        if (candidate.state_dim, candidate.action_dim, candidate.ensemble_size, candidate.context_dim) != (
                self.state_dim, self.action_dim, self.ensemble_size, self.context_dim):
            raise ValueError('experience state dimensions mismatch')
        for key in ('weights', 'covariance', 'member_updates'):
            value = np.asarray(state[key])
            if value.shape != getattr(candidate, key).shape or not np.isfinite(value).all():
                raise ValueError(f'invalid experience {key}')
            if key == 'member_updates' and (not np.issubdtype(value.dtype, np.integer) or np.any(value < 0)):
                raise ValueError('experience member updates must be nonnegative integers')
            setattr(candidate, key, value.astype(getattr(candidate, key).dtype, copy=True))
        if (np.any(candidate.member_updates < 0)
                or not np.allclose(candidate.covariance, candidate.covariance.transpose(0, 2, 1), atol=1e-8)
                or np.min(np.linalg.eigvalsh(candidate.covariance)) <= 0):
            raise ValueError('invalid experience covariance or update counts')
        for key in ('samples', 'terminals'):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f'invalid experience {key}')
            setattr(candidate, key, state[key])
        if candidate.terminals > candidate.samples or np.any(candidate.member_updates > candidate.samples):
            raise ValueError('experience counters exceed observed samples')
        for key in ('reward_error_ema', 'delta_error_ema', 'reward_squared_error', 'delta_squared_error'):
            if not np.isscalar(state[key]) or not np.isfinite(state[key]) or state[key] < 0:
                raise ValueError(f'invalid experience {key}')
            setattr(candidate, key, float(state[key]))
        mean = state.get('reward_mean', 0.0)
        baseline_error = state.get('baseline_error_ema', 0.0)
        if not np.isscalar(mean) or not np.isfinite(mean) or not np.isscalar(baseline_error) or not np.isfinite(baseline_error) or baseline_error < 0:
            raise ValueError('invalid reward baseline state')
        candidate.reward_mean = float(mean); candidate.baseline_error_ema = float(baseline_error)
        candidate.rng.bit_generator.state = deepcopy(state['rng'])
        self.__dict__.update(candidate.__dict__)
