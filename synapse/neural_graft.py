"""Bounded residual policy trained from executed transitions on the CPU.

The policy uses a clipped likelihood-ratio objective on the most recent policy
version. A separate observation-only TD critic and action-conditioned sensory
predictor can be pretrained without changing the policy. This is a local
partial-observation research controller, not a demonstrated capability upgrade.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .experience import encode_experience_state
from .stateio import digest


def _array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all() or np.max(np.abs(array), initial=0) > 1e6:
        raise ValueError(f'{name} must be finite shape {shape} within numeric bounds')
    return array.copy()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in [{low},{high}]')
    return value


class _ResidualNetwork(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.actor_encoder = nn.Sequential(nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.Tanh(),
                                           nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_head = nn.Linear(hidden, 5)
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)
        self.value = nn.Sequential(nn.Linear(155, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.dynamics = nn.Sequential(nn.Linear(160, hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 15))

    def policy(self, state):
        return self.actor_head(self.actor_encoder(state)).clamp(-3., 3.)


class NeuralResidualGraft:
    """A zero-initialized residual actor, critic and grounded transition replay.

    ``propose(..., learn=False)`` is pure. Learning proposals consume only this
    module's RNG and produce a receipt that may be attributed once. Prediction
    learning accepts manual interventions; the policy excludes those actions and
    proposals changed by a planner or final action governor.
    """
    def __init__(self, seed=7, hidden_dim=256, residual_scale=.25, capacity=4096,
                 learning_rate=3e-4, gamma=.97, exploration_std=.55):
        _integer(seed, 'seed', 0, 2**31-1)
        _integer(hidden_dim, 'hidden_dim', 32, 1024)
        _integer(capacity, 'capacity', 16, 65536)
        if (not np.isfinite([residual_scale, learning_rate, gamma, exploration_std]).all()
                or not 0 < residual_scale <= .35 or not 0 < learning_rate <= .01
                or not 0 <= gamma < 1 or not .05 <= exploration_std <= 2):
            raise ValueError('invalid neural graft hyperparameters')
        self.settings = {'seed': seed, 'hidden_dim': hidden_dim, 'residual_scale': float(residual_scale),
                         'capacity': capacity, 'learning_rate': float(learning_rate),
                         'gamma': float(gamma), 'exploration_std': float(exploration_std)}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 417)
            self.network = _ResidualNetwork(hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed + 913)
        self.replay = []
        self.pending = None
        self.proposals = self.samples = self.optimizer_updates = self.actor_updates = self.policy_version = 0
        self.last_metrics = {}
        self.connectome = None

    @property
    def residual_scale(self):
        return self.settings['residual_scale']

    def bind_connectome(self, pre, post, weights, n, provenance=None):
        """Bind a fixed sparse measured-source prior without training its edges.

        Deterministic sensory drive is propagated for two bounded rate steps.
        Regional pooling yields 32 extra conditioning coordinates. The original
        feature schema and the executing biological-proxy graph are untouched.
        """
        _integer(n, 'connectome n', 1, 20000)
        w = np.asarray(weights, dtype=np.float32)
        a, b = np.asarray(pre), np.asarray(post)
        if (w.ndim != 1 or not len(w) or len(w) > 2000000 or a.shape != w.shape or b.shape != w.shape
                or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
                or not np.isfinite(w).all() or np.max(np.abs(w), initial=0) > 1e6
                or np.any(a < 0) or np.any(b < 0) or np.any(a >= n) or np.any(b >= n)):
            raise ValueError('invalid sparse connectome arrays')
        provenance = deepcopy({} if provenance is None else provenance)
        # Check the same primitive/tensor checkpoint boundary before mutation.
        from .stateio import _encode
        _encode(provenance)
        candidate = {'pre': a.astype(np.int64), 'post': b.astype(np.int64),
                     'weights': w.copy(), 'n': n, 'provenance': provenance}
        candidate['digest'] = digest(candidate)
        self.connectome = candidate
        # Prior changes alter policy conditioning; old likelihoods are invalid.
        self.policy_version += 1
        self.pending = None
        return self.summary()['connectome']

    def _prior_activity(self, obs):
        if self.connectome is None:
            return np.zeros(32, np.float32)
        prior = self.connectome
        n = prior['n']
        sensory = np.tanh(obs[np.arange(n) % 155])
        activity = sensory.copy()
        pre, post, weights = prior['pre'], prior['post'], prior['weights']
        degree = np.bincount(post, weights=np.abs(weights), minlength=n)
        norm = np.maximum(degree, 1.)
        for _ in range(2):
            current = np.bincount(post, weights=weights * activity[pre], minlength=n) / norm
            activity = np.tanh(.5 * sensory + .5 * current).astype(np.float32)
        # Adaptive pooling handles priors smaller than 32 nodes as well.
        return F.adaptive_avg_pool1d(torch.from_numpy(activity).reshape(1, 1, n), 32).flatten().numpy()

    def _inputs(self, obs, features, baseline):
        obs = _array(obs, (155,), 'observation')
        features = _array(features, (64,), 'features')
        baseline = _array(baseline, (5,), 'baseline')
        if np.any(np.abs(baseline) > 1):
            raise ValueError('baseline action must be within [-1,1]')
        state = np.concatenate([obs, features, baseline, self._prior_activity(obs)])
        return obs, features, baseline, np.clip(state, -5, 5).astype(np.float32)

    def _log_probability(self, latent, mean):
        sigma = self.settings['exploration_std']
        return (-.5 * ((latent-mean)/sigma).square() - math.log(sigma) - .5*math.log(2*math.pi)).sum(-1)

    def propose(self, obs, features, baseline, learn=False):
        if type(learn) is not bool:
            raise ValueError('learn must be boolean')
        obs, features, baseline, state = self._inputs(obs, features, baseline)
        with torch.no_grad():
            mean = self.network.policy(torch.from_numpy(state).unsqueeze(0))[0]
            value = float(self.network.value(torch.from_numpy(np.clip(obs, -5, 5))).item())
        latent = mean.numpy().copy()
        if learn:
            latent += self.rng.normal(0, self.settings['exploration_std'], 5).astype(np.float32)
        action = np.clip(baseline + self.residual_scale * np.tanh(latent), -1, 1).astype(np.float32)
        receipt = {'schema': 1, 'learning': learn, 'id': self.proposals + 1 if learn else None,
                   'policy_version': self.policy_version, 'obs': obs, 'features': features,
                   'baseline': baseline, 'actor_input': state, 'latent': latent,
                   'log_probability': float(self._log_probability(torch.from_numpy(latent), mean)),
                   'value': value, 'proposed_action': action.copy()}
        if learn:
            self.proposals += 1
            self.pending = deepcopy(receipt)
        return action, receipt

    @staticmethod
    def _outcome(next_obs, reward, done, action):
        next_obs = _array(next_obs, (155,), 'next observation')
        action = _array(action, (5,), 'executed action')
        if np.any(np.abs(action) > 1):
            raise ValueError('executed action must be within [-1,1]')
        if (isinstance(reward, (bool, np.bool_)) or not np.isscalar(reward)
                or not np.isfinite(reward) or abs(float(reward)) > 1e6):
            raise ValueError('reward must be a finite bounded scalar')
        if type(done) not in (bool, np.bool_):
            raise ValueError('done must be boolean')
        return next_obs, float(reward), bool(done), action

    def _append(self, transition):
        self.replay.append(transition)
        if len(self.replay) > self.settings['capacity']:
            del self.replay[0]
        self.samples += 1

    def observe(self, next_obs, reward, done, executed_action, receipt=None, manual=False):
        next_obs, reward, done, action = self._outcome(next_obs, reward, done, executed_action)
        if type(manual) is not bool:
            raise ValueError('manual must be boolean')
        supplied = self.pending if receipt is None else receipt
        if (self.pending is None or not isinstance(supplied, dict)
                or digest(supplied) != digest(self.pending)):
            raise ValueError('observe requires an unused matching learning proposal')
        transition = deepcopy(supplied)
        transition.update(next_obs=next_obs, reward=reward, done=done, action=action,
                          manual=manual,
                          actor_eligible=not manual and np.allclose(action, supplied['proposed_action'], rtol=0, atol=1e-6))
        self._append(transition)
        self.pending = None
        return {'samples': self.samples, 'replay_size': len(self.replay),
                'actor_eligible': bool(transition['actor_eligible']),
                'manual_intervention': manual}

    def record_transition(self, obs, features, baseline, action, next_obs, reward, done):
        """Record an externally collected executed transition for auxiliary fitting.

        No policy likelihood is invented for demonstrations or pretraining data.
        """
        obs, features, baseline, state = self._inputs(obs, features, baseline)
        next_obs, reward, done, action = self._outcome(next_obs, reward, done, action)
        self._append({'obs': obs, 'features': features, 'baseline': baseline,
                      'actor_input': state, 'next_obs': next_obs, 'reward': reward,
                      'done': done, 'action': action, 'actor_eligible': False,
                      'manual': True, 'policy_version': -1})

    def _prediction_loss(self, records):
        obs = torch.as_tensor(np.stack([r['obs'] for r in records])).clamp(-5, 5)
        action = torch.as_tensor(np.stack([r['action'] for r in records]))
        next_obs = torch.as_tensor(np.stack([r['next_obs'] for r in records])).clamp(-5, 5)
        rewards = torch.tensor([r['reward'] for r in records], dtype=torch.float32).clamp(-20, 20)
        dones = torch.tensor([r['done'] for r in records], dtype=torch.float32)
        sensory = torch.tensor(np.stack([encode_experience_state(r['next_obs'])[:14] for r in records]), dtype=torch.float32)
        predicted = self.network.dynamics(torch.cat([obs, action], -1))
        prediction_loss = F.mse_loss(predicted[:, :14], sensory) + F.mse_loss(predicted[:, 14], rewards)
        with torch.no_grad():
            target = rewards + self.settings['gamma'] * (1-dones) * self.network.value(next_obs).flatten()
        values = self.network.value(obs).flatten()
        value_loss = F.smooth_l1_loss(values, target)
        return prediction_loss, value_loss

    def train_replay(self, updates=8, batch_size=128, actor=True):
        _integer(updates, 'updates', 1, 1000)
        _integer(batch_size, 'batch_size', 1, 2048)
        if type(actor) is not bool:
            raise ValueError('actor must be boolean')
        if not self.replay:
            return {'status': 'skipped', 'reason': 'no executed transitions', 'samples': self.samples,
                    'optimizer_updates': 0, 'actor_updates': 0, 'cpu_seconds': 0., 'wall_seconds': 0.}
        wall, cpu = time.perf_counter(), time.process_time()
        eligible = [r for r in self.replay if r['actor_eligible'] and r['policy_version'] == self.policy_version] if actor else []
        actor_state = actor_latent = old_logp = advantages = None
        if eligible:
            actor_state = torch.from_numpy(np.stack([r['actor_input'] for r in eligible]))
            actor_latent = torch.from_numpy(np.stack([r['latent'] for r in eligible]))
            old_logp = torch.tensor([r['log_probability'] for r in eligible], dtype=torch.float32)
            with torch.no_grad():
                obs = torch.from_numpy(np.stack([r['obs'] for r in eligible])).clamp(-5, 5)
                next_obs = torch.from_numpy(np.stack([r['next_obs'] for r in eligible])).clamp(-5, 5)
                rewards = torch.tensor([r['reward'] for r in eligible], dtype=torch.float32).clamp(-20, 20)
                dones = torch.tensor([r['done'] for r in eligible], dtype=torch.float32)
                advantages = rewards + self.settings['gamma']*(1-dones)*self.network.value(next_obs).flatten()-self.network.value(obs).flatten()
                if len(advantages) > 1:
                    advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp(min=.05)
                advantages = advantages.clamp(-5, 5)
        policy_steps, last_actor, last_value = 0, 0., 0.
        with torch.no_grad():
            initial = float(self._prediction_loss(self.replay[-min(batch_size, len(self.replay)):])[0])
        for _ in range(updates):
            indices = self.rng.choice(len(self.replay), min(batch_size, len(self.replay)), replace=False)
            batch = [self.replay[int(i)] for i in indices]
            prediction_loss, value_loss = self._prediction_loss(batch)
            policy_loss = torch.tensor(0.)
            if eligible:
                selected = self.rng.choice(len(eligible), min(batch_size, len(eligible)), replace=False)
                means = self.network.policy(actor_state[selected])
                logp = self._log_probability(actor_latent[selected], means)
                ratio = (logp-old_logp[selected]).clamp(-10, 10).exp()
                unclipped = ratio*advantages[selected]
                clipped = ratio.clamp(.8, 1.2)*advantages[selected]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                policy_steps += 1
            loss = prediction_loss + .5*value_loss + policy_loss
            if not torch.isfinite(loss):
                raise ValueError('nonfinite neural training objective')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()
            self.optimizer_updates += 1
            self.actor_updates += int(bool(eligible))
            last_actor, last_value = float(policy_loss.detach()), float(value_loss.detach())
        if eligible:
            self.policy_version += 1
        with torch.no_grad():
            final = float(self._prediction_loss(self.replay[-min(batch_size, len(self.replay)):])[0])
        self.last_metrics = {'initial_prediction_loss': initial, 'final_prediction_loss': final,
                             'value_loss': last_value, 'policy_loss': last_actor,
                             'eligible_policy_samples': len(eligible)}
        return {'status': 'ok', 'samples': self.samples, 'replay_size': len(self.replay),
                'optimizer_updates': updates, 'actor_updates': policy_steps,
                'total_optimizer_updates': self.optimizer_updates,
                'total_actor_updates': self.actor_updates, **self.last_metrics,
                'cpu_seconds': max(0., time.process_time()-cpu),
                'wall_seconds': max(0., time.perf_counter()-wall),
                'scope': 'executed-transition fitting; finite local training, no performance promotion'}

    def export_state(self):
        return deepcopy({'schema': 1, 'settings': self.settings, 'network': self.network.state_dict(),
                         'optimizer': self.optimizer.state_dict(), 'rng': self.rng.bit_generator.state,
                         'replay': self.replay, 'pending': self.pending, 'connectome': self.connectome,
                         'proposals': self.proposals, 'samples': self.samples,
                         'optimizer_updates': self.optimizer_updates, 'actor_updates': self.actor_updates,
                         'policy_version': self.policy_version, 'last_metrics': self.last_metrics})

    def import_state(self, state):
        """Validate a detached candidate before touching the resident module."""
        if not isinstance(state, dict) or state.get('schema') != 1:
            raise ValueError('invalid neural graft state schema')
        from .stateio import _encode
        try:
            _encode(state)
            candidate = type(self)(**state['settings'])
            if candidate.settings['hidden_dim'] != self.settings['hidden_dim']:
                raise ValueError('neural graft architecture mismatch')
            candidate.network.load_state_dict(state['network'])
            candidate.optimizer.load_state_dict(state['optimizer'])
            for parameter, moments in candidate.optimizer.state.items():
                if any(isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape != parameter.shape
                       for value in moments.values()):
                    raise ValueError('neural optimizer moment shape mismatch')
            candidate.rng.bit_generator.state = deepcopy(state['rng'])
            prior = state.get('connectome')
            if prior is not None:
                candidate.bind_connectome(prior['pre'], prior['post'], prior['weights'], prior['n'], prior['provenance'])
                if candidate.connectome['digest'] != prior['digest']:
                    raise ValueError('connectome state digest mismatch')
            records = state['replay']
            if not isinstance(records, list) or len(records) > candidate.settings['capacity']:
                raise ValueError('invalid neural replay length')
            for record in records:
                candidate._inputs(record['obs'], record['features'], record['baseline'])
                candidate._outcome(record['next_obs'], record['reward'], record['done'], record['action'])
                _array(record['actor_input'], (256,), 'actor input')
                if type(record['actor_eligible']) is not bool:
                    raise ValueError('invalid actor eligibility')
                if record['actor_eligible']:
                    _array(record['latent'], (5,), 'policy latent')
            for name in ('proposals', 'samples', 'optimizer_updates', 'actor_updates', 'policy_version'):
                _integer(state[name], name, 0, 2**63-1)
                setattr(candidate, name, state[name])
            candidate.replay = deepcopy(records)
            candidate.pending = deepcopy(state['pending'])
            candidate.last_metrics = deepcopy(state.get('last_metrics', {}))
            if candidate.samples < len(candidate.replay):
                raise ValueError('sample count smaller than replay')
        except (KeyError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError(f'invalid neural graft checkpoint: {exc}') from exc
        # The host registers this network in its editable/checkpoint module map.
        # Keep its Parameter objects and optimizer bindings stable on restore.
        self.network.load_state_dict(candidate.network.state_dict())
        self.optimizer.load_state_dict(candidate.optimizer.state_dict())
        self.__dict__.update({key: value for key, value in candidate.__dict__.items()
                              if key not in ('network', 'optimizer')})

    def summary(self):
        prior = self.connectome
        return {'enabled': True, 'parameters': sum(p.numel() for p in self.network.parameters()),
                'residual_scale': self.residual_scale, 'samples': self.samples,
                'replay_size': len(self.replay), 'replay_capacity': self.settings['capacity'],
                'actor_samples': sum(bool(record['actor_eligible']) for record in self.replay),
                'optimizer_updates': self.optimizer_updates, 'actor_updates': self.actor_updates,
                'policy_version': self.policy_version, 'last_fit': deepcopy(self.last_metrics),
                'connectome': {'bound': prior is not None, 'nodes': prior['n'] if prior else 0,
                               'edges': len(prior['weights']) if prior else 0,
                               'digest': prior['digest'] if prior else None,
                               'provenance': deepcopy(prior['provenance']) if prior else {}},
                'objective': 'clipped recent-policy gradient + TD critic + executed sensory/reward prediction',
                'scope': 'bounded experimental neural residual; no measured environment advantage'}
