"""Fixed synthetic outcome-learning probe, separate from embodied evidence.

Protocol frozen before its first run, 6 November 2026: three phases with hidden
signs +1, -1, +1; r = sign*s[0]*a - .20*a**2 + .15*sin(3*a). Each phase uses
independent uniform support and current-query inputs. A fixed independent old
stable-regime query diagnoses interference. It supplies no task ID to learners.
No query reward is used for training, choosing hyperparameters, or promotion.

This tests post-feedback adaptation of a linear ensemble and outcome store. It is
not continual backpropagation, embodied performance, or evidence of general
intelligence. The hidden contradictory regimes cannot both be inferred from the
same visible state without additional outcome history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .experience import ExperienceWorldModel
from .stateio import digest
from .surprise_memory import SurpriseEpisodicMemory


PHASES = (('stable', 1), ('reversal', -1), ('recurrence', 1))
ARMS = ('off', 'model', 'memory', 'shuffled')


def _vector(value, size, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite vector of length {size}') from exc
    if array.shape != (size,) or not np.isfinite(array).all() or np.any(np.abs(array) > 1):
        raise ValueError(f'{name} must be a finite vector of length {size} within [-1,1]')
    return array


def outcome_reward(state, action, sign):
    """Actual action-dependent reward; sign is scorer-only hidden regime state."""
    state = _vector(state, 2, 'state')
    action = _vector(action, 1, 'action')
    if isinstance(sign, (bool, np.bool_)) or not isinstance(sign, (int, np.integer)) or sign not in (-1, 1):
        raise ValueError('sign must be the integer -1 or 1')
    a = float(action[0])
    return float(int(sign) * state[0] * a - 0.20 * a ** 2 + 0.15 * np.sin(3.0 * a))


def _rng(namespace, seed, phase=''):
    # Separate deterministic streams prevent support/query sample overlap by
    # construction of their generators. Finite random draws can still coincide.
    identity = f'synapse-continual-v1:{namespace}:{seed}:{phase}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little'))


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot(model, memory, shuffled_model, shuffled_memory):
    return {'model': model.export_state(), 'memory': memory.export_state(),
            'shuffled_model': shuffled_model.export_state(),
            'shuffled_memory': shuffled_memory.export_state()}


def _predict_reward(model, memory, state, action):
    reward = float(model.predict(state, np.asarray(action)[None, :])['reward'][0])
    if memory is None:
        return reward
    recalled = memory.retrieve(state, np.asarray(action)[None, :])
    weight = 0.5 * float(np.clip(recalled['confidence'][0], 0.0, 1.0))
    return float((1.0 - weight) * reward + weight * recalled['reward'][0])


def _mean(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError('summary inputs must be a nonempty finite vector')
    return float(array.mean())


def summarize_records(support_records, query_records):
    """Summarize recorded outcomes without clipping negative paired differences."""
    prequential, frozen, paired = [], [], []
    support_groups = sorted({(r['seed'], r['phase']) for r in support_records})
    for seed, phase in support_groups:
        rows = [r for r in support_records if (r['seed'], r['phase']) == (seed, phase)]
        for arm in ARMS:
            prequential.append({'seed': seed, 'phase': phase, 'arm': arm, 'steps': len(rows),
                                'reward_mse': _mean([(r['reward'] - r['predictions'][arm]) ** 2 for r in rows])})
    groups = sorted({(r['seed'], r['phase'], r['evaluation']) for r in query_records})
    for seed, phase, evaluation in groups:
        values = {}
        for arm in ARMS:
            rows = [r for r in query_records if (r['seed'], r['phase'], r['evaluation'], r['arm'])
                    == (seed, phase, evaluation, arm)]
            if not rows:
                raise ValueError('missing paired query arm')
            values[arm] = _mean([r['chosen_reward'] for r in rows])
            frozen.append({'seed': seed, 'phase': phase, 'evaluation': evaluation,
                           'arm': arm, 'steps': len(rows), 'mean_reward': values[arm],
                           'reward_mse': _mean([(r['probe_reward'] - r['predicted_probe_reward']) ** 2 for r in rows]),
                           'local_regret': _mean([r['local_oracle_reward'] - r['chosen_reward'] for r in rows]),
                           'planner_use_fraction': _mean([float(r['planner_used']) for r in rows])})
        paired.append({'seed': seed, 'phase': phase, 'evaluation': evaluation,
                       'model_minus_off': values['model'] - values['off'],
                       'memory_minus_off': values['memory'] - values['off'],
                       'memory_minus_model': values['memory'] - values['model'],
                       'memory_minus_shuffled': values['memory'] - values['shuffled']})
    aggregate = []
    for phase, evaluation in sorted({(r['phase'], r['evaluation']) for r in frozen}):
        row = {'phase': phase, 'evaluation': evaluation, 'arms': {}, 'paired': {}}
        for arm in ARMS:
            arm_rows = [r for r in frozen if (r['phase'], r['evaluation'], r['arm']) == (phase, evaluation, arm)]
            row['arms'][arm] = {metric: _mean([r[metric] for r in arm_rows])
                                for metric in ('mean_reward', 'reward_mse', 'local_regret', 'planner_use_fraction')}
        paired_rows = [r for r in paired if (r['phase'], r['evaluation']) == (phase, evaluation)]
        for name in ('model_minus_off', 'memory_minus_off', 'memory_minus_model', 'memory_minus_shuffled'):
            row['paired'][name] = {'mean_difference': _mean([r[name] for r in paired_rows]),
                                   'per_seed_differences': [r[name] for r in paired_rows]}
        aggregate.append(row)
    return {'prequential': prequential, 'frozen': frozen, 'paired': paired,
            'aggregate': aggregate,
            'interpretation': 'synthetic mechanism development probe; no tuning or promotion from query; negative differences retained; no inferential intervals from three seeds'}


def run_continual_probe(seeds=(7, 11, 19), support_steps=64, query_steps=64):
    """Train shared actual outcomes across three regimes and score frozen arms."""
    if (not isinstance(seeds, (list, tuple)) or not 1 <= len(seeds) <= 8
            or any(type(seed) is not int or not 0 <= seed < 2 ** 31 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must contain 1..8 unique nonnegative 31-bit integers')
    for name, value in [('support_steps', support_steps), ('query_steps', query_steps)]:
        if type(value) is not int or not 8 <= value <= 512:
            raise ValueError(f'{name} must be an integer in [8,512]')
    support_records, query_records, checkpoints = [], [], []
    prototype = ExperienceWorldModel(state_dim=2, action_dim=1, seed=seeds[0])
    model_settings = {key: getattr(prototype, key) for key in (
        'state_dim', 'action_dim', 'ensemble_size', 'ridge', 'forgetting',
        'min_samples', 'action_radius', 'uncertainty_penalty', 'context_dim')}
    config = {'seeds': list(seeds), 'support_steps': support_steps, 'query_steps': query_steps,
              'phases': [{'name': phase, 'sign': sign} for phase, sign in PHASES],
              'reward': 'sign*state[0]*action - 0.20*action**2 + 0.15*sin(3*action)',
              'transition': '[0.8*state[0]+0.2*action, 0.9*state[1]]',
              'sampling': 'independent uniform[-1,1] state/action draws; phase sign is not given to learners',
              'model_settings': model_settings,
              'memory_settings': SurpriseEpisodicMemory(state_dim=2, action_dim=1).export_state()['config'],
              'arms': list(ARMS), 'baseline_action': [0.0],
              'retention': 'fixed independent old +1 regime probe after each phase; contradictory hidden task, not task-ID-conditioned retention',
              'shuffled': 'phase-local permutation of actual support rewards; state/action/next-state data unchanged; offline negative control',
              'selection': 'frozen protocol; no query-driven tuning or model promotion'}
    for seed in seeds:
        model = ExperienceWorldModel(state_dim=2, action_dim=1, seed=seed)
        memory = SurpriseEpisodicMemory(state_dim=2, action_dim=1)
        shuffled_model = ExperienceWorldModel(state_dim=2, action_dim=1, seed=seed)
        shuffled_memory = SurpriseEpisodicMemory(state_dim=2, action_dim=1)
        old_query = _rng('old-stable-query', seed)
        old_states = old_query.uniform(-1.0, 1.0, (query_steps, 2))
        old_actions = old_query.uniform(-1.0, 1.0, (query_steps, 1))
        for phase, sign in PHASES:
            support_rng = _rng('support', seed, phase)
            states = support_rng.uniform(-1.0, 1.0, (support_steps, 2))
            actions = support_rng.uniform(-1.0, 1.0, (support_steps, 1))
            actual_rewards = [outcome_reward(s, a, sign) for s, a in zip(states, actions)]
            permutation = _rng('reward-shuffle', seed, phase).permutation(support_steps)
            shuffled_rewards = np.asarray(actual_rewards)[permutation]
            for index, (state, action, reward, fake_reward) in enumerate(zip(states, actions, actual_rewards, shuffled_rewards)):
                next_state = np.array([0.8 * state[0] + 0.2 * action[0], 0.9 * state[1]])
                predictions = {'off': 0.0,
                               'model': _predict_reward(model, None, state, action),
                               'memory': _predict_reward(model, memory, state, action),
                               'shuffled': _predict_reward(shuffled_model, shuffled_memory, state, action)}
                update = model.update(state, action, next_state, reward)
                admission = memory.update(state, action, reward, update['prediction_error'])
                fake_update = shuffled_model.update(state, action, next_state, float(fake_reward))
                shuffled_memory.update(state, action, float(fake_reward), fake_update['prediction_error'])
                support_records.append({'seed': seed, 'phase': phase, 'sign': sign,
                                        'index': index, 'state': state.tolist(), 'action': action.tolist(),
                                        'next_state': next_state.tolist(), 'reward': reward,
                                        'shuffled_reward': float(fake_reward),
                                        'shuffled_source_index': int(permutation[index]),
                                        'predictions': predictions, 'memory_admitted': admission['admitted']})
            snapshot = _snapshot(model, memory, shuffled_model, shuffled_memory)
            before = digest(snapshot)
            # Check restore without running additional fitting or touching query labels.
            restored_model = ExperienceWorldModel(state_dim=2, action_dim=1, seed=seed)
            restored_memory = SurpriseEpisodicMemory(state_dim=2, action_dim=1)
            restored_model.import_state(snapshot['model'])
            restored_memory.import_state(snapshot['memory'])
            restore_exact = (digest(restored_model.export_state()) == digest(snapshot['model'])
                             and digest(restored_memory.export_state()) == digest(snapshot['memory']))
            current_rng = _rng('current-query', seed, phase)
            current_states = current_rng.uniform(-1.0, 1.0, (query_steps, 2))
            current_actions = current_rng.uniform(-1.0, 1.0, (query_steps, 1))
            for evaluation, eval_sign, query_states, probe_actions in (
                ('current', sign, current_states, current_actions),
                ('old_stable', 1, old_states, old_actions),
            ):
                for index, (state, probe_action) in enumerate(zip(query_states, probe_actions)):
                    actual_probe_reward = outcome_reward(state, probe_action, eval_sign)
                    local_candidates = [[0.0], [-model.action_radius], [model.action_radius]]
                    local_oracle = max(outcome_reward(state, a, eval_sign) for a in local_candidates)
                    for arm, arm_model, arm_memory in (
                        ('off', None, None), ('model', model, None),
                        ('memory', model, memory), ('shuffled', shuffled_model, shuffled_memory),
                    ):
                        if arm == 'off':
                            action = np.array([0.0]); predicted = 0.0
                            diagnostics = {'used': False, 'memory_used': False}
                        else:
                            predicted = _predict_reward(arm_model, arm_memory, state, probe_action)
                            action, diagnostics = arm_model.plan(state, np.array([0.0]), memory=arm_memory)
                        # Scoring occurs after action selection; its sign is never an input.
                        query_records.append({'seed': seed, 'phase': phase, 'evaluation': evaluation,
                                              'sign': eval_sign, 'index': index, 'arm': arm,
                                              'state': state.tolist(), 'probe_action': probe_action.tolist(),
                                              'probe_reward': actual_probe_reward, 'predicted_probe_reward': predicted,
                                              'chosen_action': action.astype(float).tolist(),
                                              'chosen_reward': outcome_reward(state, action, eval_sign),
                                              'local_oracle_reward': local_oracle,
                                              'planner_used': bool(diagnostics['used']),
                                              'memory_used': bool(diagnostics.get('memory_used', False))})
            after = digest(_snapshot(model, memory, shuffled_model, shuffled_memory))
            if before != after or not restore_exact:
                raise RuntimeError('frozen query or checkpoint roundtrip changed learned state')
            checkpoints.append({'seed': seed, 'phase': phase, 'learned_before': before,
                                'learned_after': after, 'freeze_ok': before == after,
                                'restore_exact': bool(restore_exact), 'model_samples': model.samples,
                                'memory_updates': memory.summary()['updates']})
    raw = {'support': support_records, 'query': query_records}
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in ('continual_probe.py', 'experience.py', 'surprise_memory.py')}
    return {'schema': 'synapse-continual-probe-v1', 'status': 'complete', 'config': config,
            'scope': 'synthetic hidden-regime outcome-learning mechanism probe; not embodied evidence or continual backpropagation',
            'raw': raw, 'summary': summarize_records(support_records, query_records),
            'checkpoints': checkpoints, 'source_hashes': sources,
            'hashes': {'config': _hash(config), 'support': _hash(support_records),
                       'query': _hash(query_records), 'records': _hash(raw)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the complete JSON receipt to this file.')
    args = parser.parse_args(argv)
    result = run_continual_probe()
    if args.output is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({'output': str(args.output.resolve()), 'hashes': result['hashes'],
                          'aggregate': result['summary']['aggregate']}, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
