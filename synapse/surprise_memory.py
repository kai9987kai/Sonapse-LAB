"""Bounded actual-outcome memory with heuristic surprise and coverage admission.

This is a deterministic associative store, not Titans, TTT, or a calibrated
uncertainty estimator. Prediction error controls admission; stored values are
the rewards actually observed. Reads never advance time or change any state.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np


class SurpriseEpisodicMemory:
    """Small state/action outcome store with pure, distance-weighted retrieval.

    Distances use equally weighted state/action RMS differences after tanh
    scaling, so finite extreme inputs cannot overflow the metric. Confidence is
    the strongest neighbor's distance/age weight, not a probability. Time counts
    valid outcome updates, including rejected writes; reading does not age data.
    """

    _MAX_DISTANCE = math.sqrt(8.0)
    _ERROR_CAP = 1e6
    _COUNTERS = ('updates', 'writes', 'replacements', 'evictions', 'rejected')

    def __init__(self, state_dim=16, action_dim=5, capacity=128,
                 initial_coverage=8, error_decay=0.95, error_floor=0.001,
                 surprise_multiplier=1.0, match_distance=0.10,
                 novelty_distance=0.45, bandwidth=0.35,
                 recency_half_life=128.0, neighbors=4,
                 state_scale=1.0, action_scale=1.0):
        ints = {'state_dim': state_dim, 'action_dim': action_dim,
                'capacity': capacity, 'initial_coverage': initial_coverage,
                'neighbors': neighbors}
        for name, value in ints.items():
            lower = 0 if name == 'initial_coverage' else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        floats = {'error_decay': error_decay, 'error_floor': error_floor,
                  'surprise_multiplier': surprise_multiplier,
                  'match_distance': match_distance, 'novelty_distance': novelty_distance,
                  'bandwidth': bandwidth, 'recency_half_life': recency_half_life,
                  'state_scale': state_scale, 'action_scale': action_scale}
        floats = {name: self._scalar(value, name) for name, value in floats.items()}
        if not 0 <= floats['error_decay'] < 1:
            raise ValueError('error_decay must be in [0, 1)')
        for name in ('error_floor', 'surprise_multiplier', 'match_distance', 'novelty_distance'):
            if floats[name] < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('bandwidth', 'recency_half_life', 'state_scale', 'action_scale'):
            if floats[name] <= 0:
                raise ValueError(f'{name} must be positive')
        if floats['novelty_distance'] < floats['match_distance']:
            raise ValueError('novelty_distance must be >= match_distance')
        self._config = {**{name: int(value) for name, value in ints.items()}, **floats}
        for name, value in self._config.items():
            setattr(self, name, value)
        self._states = np.empty((0, self.state_dim), dtype=np.float64)
        self._actions = np.empty((0, self.action_dim), dtype=np.float64)
        self._rewards = np.empty(0, dtype=np.float64)
        self._timestamps = np.empty(0, dtype=np.int64)
        self._counters = {name: 0 for name in self._COUNTERS}
        self._error_mean = 0.0
        self._error_variance = 0.0

    @staticmethod
    def _scalar(value, name):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'{name} must be a finite scalar') from exc
        if array.ndim != 0 or not np.isfinite(array):
            raise ValueError(f'{name} must be a finite scalar')
        return float(array)

    @staticmethod
    def _array(value, shape, name):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'{name} must be finite with shape {shape}') from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
        return array.copy()

    def _distances(self, state, actions):
        with np.errstate(over='ignore'):
            state_delta = np.tanh(self._states / self.state_scale) - np.tanh(state / self.state_scale)
            action_delta = (np.tanh(self._actions / self.action_scale)[None, :, :]
                            - np.tanh(actions / self.action_scale)[:, None, :])
        return np.sqrt(np.mean(state_delta ** 2, axis=-1)[None, :]
                       + np.mean(action_delta ** 2, axis=-1))

    def _threshold(self):
        # Cap only the statistical baseline, never the actual stored reward.
        return float(min(self._ERROR_CAP,
                         max(self.error_floor, self._error_mean
                             + min(self.surprise_multiplier, self._ERROR_CAP)
                             * math.sqrt(self._error_variance))))

    def update(self, state, action, reward, prediction_error):
        """Consider one completed transition, validating everything before mutation.

        Admit the first ``initial_coverage`` outcomes, unseen state/action regions,
        or errors larger than the *pre-update* EMA mean-plus-standard-deviation
        threshold. Signed errors are accepted and compared by absolute magnitude.
        Nearby admitted keys replace their nearest outcome; other writes append
        or evict the oldest timestamp. Ties choose the first stored slot.
        """
        state = self._array(state, (self.state_dim,), 'state')
        action = self._array(action, (self.action_dim,), 'action')
        reward = self._scalar(reward, 'reward')
        error = min(abs(self._scalar(prediction_error, 'prediction_error')), self._ERROR_CAP)
        size = len(self._rewards)
        distance = self._distances(state, action[None, :])[0] if size else np.empty(0)
        nearest = int(np.argmin(distance)) if size else None
        nearest_distance = float(distance[nearest]) if size else self._MAX_DISTANCE
        threshold = self._threshold()
        initial = self._counters['updates'] < self.initial_coverage
        novel = not size or nearest_distance > self.novelty_distance
        surprising = error > threshold
        admitted = initial or novel or surprising
        self._counters['updates'] += 1
        difference = error - self._error_mean
        self._error_mean += (1.0 - self.error_decay) * difference
        self._error_variance = self.error_decay * (self._error_variance
                               + (1.0 - self.error_decay) * difference ** 2)
        replaced = evicted = False
        if admitted:
            if size and nearest_distance <= self.match_distance:
                index = nearest
                self._counters['replacements'] += 1
                replaced = True
            elif size < self.capacity:
                self._states = np.concatenate([self._states, state[None, :]])
                self._actions = np.concatenate([self._actions, action[None, :]])
                self._rewards = np.append(self._rewards, reward)
                self._timestamps = np.append(self._timestamps, self._counters['updates'])
                index = size
            else:
                index = int(np.argmin(self._timestamps))
                self._counters['evictions'] += 1
                evicted = True
            self._states[index] = state
            self._actions[index] = action
            self._rewards[index] = reward
            self._timestamps[index] = self._counters['updates']
            self._counters['writes'] += 1
        else:
            self._counters['rejected'] += 1
        return {'admitted': bool(admitted), 'replaced': replaced, 'evicted': evicted,
                'distance': nearest_distance, 'error_threshold': threshold,
                'reason': 'initial' if initial else 'coverage' if novel else 'surprise' if surprising else 'predictable'}

    def retrieve(self, state, actions, include_slots=False):
        """Return candidate reward/confidence/distance arrays without mutation."""
        if type(include_slots) is not bool:
            raise ValueError('include_slots must be boolean')
        state = self._array(state, (self.state_dim,), 'state')
        try:
            actions = np.asarray(actions, dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('actions must be a finite [N, action_dim] matrix') from exc
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or not np.isfinite(actions).all():
            raise ValueError('actions must be a finite [N, action_dim] matrix')
        count = len(actions)
        result = {'reward': np.zeros(count), 'confidence': np.zeros(count),
                  'distance': np.full(count, self._MAX_DISTANCE)}
        if include_slots:
            result['slot'] = np.full(count, -1, dtype=np.int64)
            result['timestamp'] = np.full(count, -1, dtype=np.int64)
        if not len(self._rewards) or not count:
            return result
        distances = self._distances(state, actions)
        ages = self._counters['updates'] - self._timestamps
        for row in range(count):
            indices = np.argsort(distances[row], kind='stable')[:self.neighbors]
            selected = distances[row, indices]
            if include_slots:
                result['slot'][row] = int(indices[0])
                result['timestamp'][row] = int(self._timestamps[indices[0]])
            with np.errstate(over='ignore', under='ignore'):
                weights = np.exp(-0.5 * (selected / self.bandwidth) ** 2
                                 - math.log(2.0) * (ages[indices] / self.recency_half_life))
            result['distance'][row] = float(selected[0])
            result['confidence'][row] = float(weights.max())
            total = float(weights.sum())
            scale = float(np.abs(self._rewards[indices]).max())
            if total > 0 and scale > 0:
                # Scaling the values prevents overflow when finite rewards are huge.
                average = np.sum((weights / total) * (self._rewards[indices] / scale))
                result['reward'][row] = float(np.clip(average, -1.0, 1.0) * scale)
        return result

    def get_record(self, slot, timestamp=None):
        """Return one detached record if its slot still has the requested identity."""
        if isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer)):
            raise ValueError('slot must be an integer')
        slot = int(slot)
        if not 0 <= slot < len(self._rewards):
            return None
        if timestamp is not None:
            if (isinstance(timestamp, (bool, np.bool_))
                    or not isinstance(timestamp, (int, np.integer))):
                raise ValueError('timestamp must be an integer')
            if int(timestamp) != int(self._timestamps[slot]):
                return None
        return {'slot': slot, 'timestamp': int(self._timestamps[slot]),
                'state': self._states[slot].copy(), 'action': self._actions[slot].copy(),
                'reward': float(self._rewards[slot])}

    def distance_to_record(self, slot, state, action):
        """Measure a bounded query against one slot without changing its age."""
        if isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer)):
            raise ValueError('slot must be an integer')
        slot = int(slot)
        if not 0 <= slot < len(self._rewards):
            return None
        state = self._array(state, (self.state_dim,), 'state')
        action = self._array(action, (self.action_dim,), 'action')
        return float(self._distances(state, action[None, :])[0, slot])

    def revise_rejected_record(self, slot, timestamp, state, action, reward):
        """Admit one exact observed outcome into a recalled slot after rejection.

        The caller must first submit the actual outcome through ``update``. This
        method converts that same predictable rejection into one bounded memory
        replacement. It preserves the observed state/action/reward and counter
        identities; the separate reconsolidation engine decides eligibility.
        """
        if isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer)):
            raise ValueError('slot must be an integer')
        if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer)):
            raise ValueError('timestamp must be an integer')
        slot = int(slot)
        record = self.get_record(slot, int(timestamp))
        if record is None:
            return {'revised': False, 'reason': 'slot identity changed'}
        state = self._array(state, (self.state_dim,), 'state')
        action = self._array(action, (self.action_dim,), 'action')
        reward = self._scalar(reward, 'reward')
        distance = float(self._distances(state, action[None, :])[0, slot])
        if distance > self.match_distance:
            return {'revised': False, 'reason': 'state/action outside memory match radius',
                    'distance': distance}
        if self._counters['rejected'] <= 0:
            return {'revised': False, 'reason': 'no rejected outcome to reconcile',
                    'distance': distance}
        self._states[slot] = state
        self._actions[slot] = action
        self._rewards[slot] = reward
        self._timestamps[slot] = self._counters['updates']
        self._counters['rejected'] -= 1
        self._counters['writes'] += 1
        self._counters['replacements'] += 1
        return {'revised': True, 'slot': slot,
                'timestamp': int(self._timestamps[slot]), 'distance': distance}

    def export_state(self):
        """Return a detached, JSON-compatible exact continuation checkpoint."""
        return {'schema': 1, 'config': deepcopy(self._config),
                'states': self._states.tolist(), 'actions': self._actions.tolist(),
                'rewards': self._rewards.tolist(), 'timestamps': self._timestamps.tolist(),
                'counters': dict(self._counters), 'error_mean': self._error_mean,
                'error_variance': self._error_variance}

    def import_state(self, state):
        """Restore a compatible checkpoint atomically after complete validation."""
        required = {'schema', 'config', 'states', 'actions', 'rewards', 'timestamps',
                    'counters', 'error_mean', 'error_variance'}
        if not isinstance(state, dict) or set(state) != required or state['schema'] != 1 or state['config'] != self._config:
            raise ValueError('memory state/config mismatch')
        try:
            size = len(state['rewards'])
            if size > self.capacity:
                raise ValueError('memory exceeds capacity')
            states = self._array(state['states'], (size, self.state_dim), 'states') if size else self._array(state['states'], (0,), 'states').reshape(0, self.state_dim)
            actions = self._array(state['actions'], (size, self.action_dim), 'actions') if size else self._array(state['actions'], (0,), 'actions').reshape(0, self.action_dim)
            rewards = self._array(state['rewards'], (size,), 'rewards')
            counters = state['counters']
            if not isinstance(counters, dict) or set(counters) != set(self._COUNTERS):
                raise ValueError('invalid memory counters')
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in counters.values()):
                raise ValueError('invalid memory counters')
            if counters['writes'] + counters['rejected'] != counters['updates'] or counters['writes'] - size != counters['replacements'] + counters['evictions']:
                raise ValueError('inconsistent memory counters')
            timestamps_raw = state['timestamps']
            if len(timestamps_raw) != size or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 1 <= v <= counters['updates'] for v in timestamps_raw):
                raise ValueError('invalid memory timestamps')
            if len(set(timestamps_raw)) != size:
                raise ValueError('duplicate memory timestamps')
            timestamps = np.asarray(timestamps_raw, dtype=np.int64).copy()
            error_mean = self._scalar(state['error_mean'], 'error_mean')
            error_variance = self._scalar(state['error_variance'], 'error_variance')
            if not 0 <= error_mean <= self._ERROR_CAP or not 0 <= error_variance <= self._ERROR_CAP ** 2:
                raise ValueError('invalid error statistics')
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('invalid memory checkpoint') from exc
        self._states, self._actions, self._rewards = states, actions, rewards
        self._timestamps, self._counters = timestamps, dict(counters)
        self._error_mean, self._error_variance = error_mean, error_variance

    def summary(self):
        """Report bounded occupancy and heuristic admission statistics."""
        return {'size': len(self._rewards), 'capacity': self.capacity,
                **dict(self._counters), 'error_mean': self._error_mean,
                'error_std': math.sqrt(self._error_variance),
                'error_threshold': self._threshold(),
                'scope': 'actual reward outcome memory; heuristic coverage, not calibrated probability'}
