"""Measured paired native-world pilot with support-only learning and frozen queries.

Replay verifies raw-record arithmetic and hashes. It does not re-execute the
environment or authenticate a receipt rewritten consistently by an adversary.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .evidence_engine import bootstrap_confidence_interval
from .stateio import digest, save_state, load_state


ARMS = ('off', 'model', 'memory', 'shuffled')
TASKS = ('forage', 'maze', 'push', 'grasp_place', 'peg_insert', 'delayed_match', 'go_nogo', 'button_sequence')
DEFAULTS = {'seeds': [7, 11, 19], 'tasks': ['forage', 'push'], 'support_steps': 64, 'query_steps': 64, 'nodes_per_source': 16}


def validate_config(config=None):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config)-set(DEFAULTS):
        raise ValueError('unknown adaptation-study configuration')
    c = deepcopy({**DEFAULTS, **config})
    if not isinstance(c['seeds'], list) or not 1 <= len(c['seeds']) <= 8 or any(type(s) is not int or not 0 <= s < 2**31 for s in c['seeds']) or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('seeds must contain 1..8 unique nonnegative 31-bit integers')
    if not isinstance(c['tasks'], list) or not 1 <= len(c['tasks']) <= len(TASKS) or any(t not in TASKS for t in c['tasks']) or len(set(c['tasks'])) != len(c['tasks']):
        raise ValueError('tasks must be unique supported native task names')
    for name, lo, hi in [('support_steps', 8, 512), ('query_steps', 8, 512), ('nodes_per_source', 16, 200)]:
        if type(c[name]) is not int or not lo <= c[name] <= hi:
            raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    if len(c['seeds'])*len(c['tasks'])*(2*c['support_steps']+4*c['query_steps']) > 16384:
        raise ValueError('total pilot inference budget exceeds 16384 steps')
    return c


def _seed(namespace, seed, task):
    return int(hashlib.sha256(f'{namespace}:{seed}:{task}'.encode()).hexdigest()[:8], 16) % (2**31)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parameters(agent):
    tensors = {f'{name}.{key}': value.detach().clone() for name, module in agent.modules.items() for key, value in module.named_parameters()}
    tensors['mb.w_mbon'] = agent.mb.w_mbon.detach().clone()
    return tensors


def _learned_hash(agent):
    return digest({'parameters': _parameters(agent), 'experience': agent.experience_model.export_state(), 'memory': agent.surprise_memory.export_state()})


def _source_fingerprints():
    from .config import ARTIFACTS
    root = Path(__file__).resolve().parent
    names = sorted(p.name for p in root.glob('*.py'))
    names.extend(f'_vendor/graftbrain/graftbrain/{name}.py' for name in ('brain', 'plasticity', 'neuro_modules', 'world', 'connectomes', 'graft'))
    names.append('_vendor/poseidon/poseidon/core.py')
    values = {name: _file_hash(root/name) for name in names}
    values['artifact:tidal_core.pt'] = _file_hash(ARTIFACTS/'tidal_core.pt')
    return values


def _environment():
    import platform
    import scipy
    import torch
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': str(np.__version__), 'scipy': str(scipy.__version__), 'torch': str(torch.__version__)}


def _summarize(records, config):
    indexed = {(r['seed'], r['task'], r['arm']): r for r in records}
    expected = {(seed, task, arm) for seed in config['seeds'] for task in config['tasks'] for arm in ARMS}
    if len(records) != len(expected) or set(indexed) != expected:
        raise ValueError('missing or duplicate paired query records')
    units = [(seed, task) for seed in config['seeds'] for task in config['tasks']]
    arms = {}
    for arm in ARMS:
        rows = [indexed[(seed, task, arm)] for seed, task in units]
        values = [r['return'] for r in rows]
        arms[arm] = {'mean_return': float(np.mean(values)), 'returns': values,
                     'mean_reward_mse': float(np.mean([r['reward_mse'] for r in rows])),
                     'planner_use_fraction': float(np.mean([r['planner_used_steps']/r['steps'] for r in rows])),
                     'memory_use_fraction': float(np.mean([r['memory_used_steps']/r['steps'] for r in rows]))}
    comparisons = {}
    for first, second in [('model', 'off'), ('memory', 'off'), ('memory', 'model'), ('memory', 'shuffled')]:
        diffs = np.asarray(arms[first]['returns'])-np.asarray(arms[second]['returns'])
        comparisons[f'{first}_minus_{second}'] = {'paired_differences': diffs.tolist(), 'mean_difference': float(diffs.mean()),
                                                'paired_bootstrap_95': list(bootstrap_confidence_interval(diffs)) if len(diffs) > 1 else None}
    return {'paired_units': [{'seed': seed, 'task': task} for seed, task in units], 'arms': arms, 'comparisons': comparisons,
            'interpretation': 'development pilot; no outcome-based model promotion; small finite native worlds and shared task types limit generalization',
            'uncertainty': 'percentile bootstrap over independent seed/task actor-world units; time steps are not independent replicates; one unit has no interval'}


def format_adaptation_report(receipt):
    lines = ['# SYNAPSE measured adaptation development pilot', '', receipt['scope'], '',
             'Support trajectories train prediction and outcome memory. All query arms share paired initial worlds and post-support actor parameters; query learning is disabled.', '',
             '| Frozen query arm | Mean return | Reward prediction MSE | Planner use | Memory use |', '|---|---:|---:|---:|---:|']
    for arm, row in receipt['summary']['arms'].items():
        lines.append(f"| {arm} | {row['mean_return']:.6f} | {row['mean_reward_mse']:.6f} | {row['planner_use_fraction']:.3f} | {row['memory_use_fraction']:.3f} |")
    lines.extend(['', '| Paired comparison | Mean return difference | Descriptive 95% interval |', '|---|---:|---|'])
    for name, row in receipt['summary']['comparisons'].items():
        ci = row['paired_bootstrap_95']
        interval = f'[{ci[0]:+.6f}, {ci[1]:+.6f}]' if ci else 'not estimated (one unit)'
        lines.append(f"| {name} | {row['mean_difference']:+.6f} | {interval} |")
    lines.extend(['', receipt['summary']['uncertainty'], '', receipt['summary']['interpretation']+'.', '',
                  'Returns are fixed-step cumulative rewards, including any episode resets. The shuffled arm permutes actual support rewards and keeps observed transitions fixed; constant rewards can make this control ineffective.', '',
                  'Freeze checks compare learned parameters, ensemble state, and memory before/after queries. Raw traces and source hashes are in receipt.json. Verification recomputes arithmetic and hashes; it does not reproduce environmental execution or prove external authenticity.', '',
                  f"Configuration: `{json.dumps(receipt['config'], sort_keys=True)}`", ''])
    return '\n'.join(lines)


def run_adaptation_study(output_dir=None, config=None):
    """Run actual action-dependent worlds; never select a policy using query scores."""
    from .agent import IntegratedAgent
    from .config import SynapseConfig
    from .experience import encode_experience_state
    from graftbrain.world import World

    c = validate_config(config)
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    supports, records, files = [], [], {}
    for seed in c['seeds']:
        for task in c['tasks']:
            agent_seed = _seed('actor', seed, task)
            support_seed, query_seed = _seed('support', seed, task), _seed('query', seed, task)
            agent = IntegratedAgent(SynapseConfig(seed=agent_seed, task=task, nodes_per_source=c['nodes_per_source'], experience_planning=False, surprise_memory=True))
            initial = agent.export_state()
            parameter_hash = digest(_parameters(agent))
            world = World(seed=support_seed, curriculum=task)
            obs = world.observe()
            rng = np.random.default_rng(_seed('exploration', seed, task))
            trace = []
            for step in range(c['support_steps']):
                result = agent.act(obs, learn=False, planner_mode='off')
                action = rng.uniform(-.8, .8, 5).astype(np.float32)
                next_obs, reward, done, _ = world.step(action)
                agent.observe_transition(next_obs, float(reward), bool(done), action=action)
                trace.append({'step': step, 'obs': obs.tolist(), 'action': action.tolist(), 'next_obs': next_obs.tolist(), 'reward': float(reward), 'done': bool(done)})
                obs = next_obs
                if done:
                    agent.reset_episode()
                    obs = world.reset()
            trained = agent.export_state()
            if digest(_parameters(agent)) != parameter_hash:
                raise RuntimeError('support changed backbone parameters in a prediction-only study')
            # Retrain only prediction/memory on identical recorded inputs and a
            # permutation of observed rewards. No fabricated reward distribution.
            permutation = rng.permutation(len(trace)).tolist()
            rewards = [trace[i]['reward'] for i in permutation]
            agent.import_state(initial)
            for row, shuffled_reward in zip(trace, rewards):
                agent.act(np.asarray(row['obs'], np.float32), learn=False, planner_mode='off')
                agent.observe_transition(np.asarray(row['next_obs'], np.float32), shuffled_reward, row['done'], action=np.asarray(row['action'], np.float32))
                if row['done']:
                    agent.reset_episode()
            shuffled_model, shuffled_memory = agent.experience_model.export_state(), agent.surprise_memory.export_state()
            shuffled_summary = {'model': agent.experience_model.summary(), 'memory': agent.surprise_memory.summary()}
            agent.import_state(trained)
            roundtrip_ok = digest(agent.export_state()) == digest(trained)
            if not roundtrip_ok:
                raise RuntimeError('post-support checkpoint did not restore exactly')
            unit = {'seed': seed, 'task': task, 'agent_seed': agent_seed, 'support_seed': support_seed, 'query_seed': query_seed,
                    'trace': trace, 'support_digest': _json_hash(trace), 'reward_permutation': permutation,
                    'reward_shuffle_changed': rewards != [r['reward'] for r in trace],
                    'post_support_parameters': parameter_hash, 'post_support_state': digest(trained), 'checkpoint_roundtrip_ok': roundtrip_ok,
                    'learned_support': {'model': agent.experience_model.summary(), 'memory': agent.surprise_memory.summary()}, 'shuffled_support': shuffled_summary}
            if out is not None:
                name = f'support_{seed}_{task}.pt'
                save_state(out/name, trained)
                agent.import_state(load_state(out/name))
                if digest(agent.export_state()) != digest(trained):
                    raise RuntimeError('persisted post-support checkpoint did not restore exactly')
                for artifact in [name, name+'.json']:
                    files[artifact] = _file_hash(out/artifact)
                unit['checkpoint'] = name
            supports.append(unit)
            for arm in ARMS:
                agent.import_state(trained)
                if arm == 'shuffled':
                    agent.experience_model.import_state(shuffled_model)
                    agent.surprise_memory.import_state(shuffled_memory)
                agent.reset_episode()
                before = _learned_hash(agent)
                world = World(seed=query_seed, curriculum=task)
                obs = world.observe()
                query_trace = []
                mode = 'memory' if arm == 'shuffled' else arm
                for step in range(c['query_steps']):
                    result = agent.act(obs, learn=False, planner_mode=mode)
                    action = np.asarray(result['action'], np.float32)
                    state = encode_experience_state(obs, result['features'])
                    prediction = float(np.asarray(agent.experience_model.predict(state, action[None])['reward']).reshape(-1)[0])
                    next_obs, reward, done, _ = world.step(action)
                    planner = result['telemetry'].get('experience_planner', {})
                    query_trace.append({'step': step, 'reward': float(reward), 'predicted_reward': prediction, 'action': action.tolist(), 'done': bool(done),
                                        'planner_used': bool(planner.get('used', False)), 'memory_used': bool(planner.get('memory_used', False)),
                                        'planner_reason': str(planner.get('reason', 'unreported'))})
                    obs = next_obs
                    if done:
                        agent.reset_episode()
                        obs = world.reset()
                after = _learned_hash(agent)
                if before != after:
                    raise RuntimeError('query execution mutated learned state')
                records.append({'seed': seed, 'task': task, 'arm': arm, 'query_seed': query_seed, 'support_digest': unit['support_digest'],
                                'steps': len(query_trace), 'return': float(sum(t['reward'] for t in query_trace)),
                                'reward_mse': float(np.mean([(t['reward']-t['predicted_reward'])**2 for t in query_trace])),
                                'planner_used_steps': sum(t['planner_used'] for t in query_trace), 'memory_used_steps': sum(t['memory_used'] for t in query_trace),
                                'learned_before': before, 'learned_after': after, 'trace': query_trace, 'trace_digest': _json_hash(query_trace)})
    n = len(supports)
    receipt = {'schema': 'synapse-adaptation-study-v1', 'status': 'complete', 'config': c,
               'scope': 'measured CPU native-world development pilot; frozen untrained actor bridges; one-step heuristic planning; no confirmed general advantage',
               'supports': supports, 'records': records, 'summary': _summarize(records, c),
               'costs': {'support_world_steps': n*c['support_steps'], 'shuffled_support_replay_steps': n*c['support_steps'],
                         'query_world_steps': n*len(ARMS)*c['query_steps'], 'model_update_calls': 2*n*c['support_steps'],
                         'query_update_calls': 0, 'actor_calls': n*(2*c['support_steps']+len(ARMS)*c['query_steps'])},
               'source_fingerprints': _source_fingerprints(), 'environment': _environment(), 'files': files,
               'wall_seconds': time.perf_counter()-started}
    if out is not None:
        (out/'report.md').write_text(format_adaptation_report(receipt), encoding='utf-8')
        receipt['files']['report.md'] = _file_hash(out/'report.md')
    receipt['integrity'] = _json_hash(receipt)
    if out is not None:
        (out/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    return receipt


def verify_adaptation_study(path_or_receipt):
    """Recompute hashes, trace metrics, paired summaries and recorded freeze checks."""
    errors, warnings, recomputed = [], [], 0
    try:
        if isinstance(path_or_receipt, dict):
            receipt, out = deepcopy(path_or_receipt), None
        else:
            path = Path(path_or_receipt)
            path = path/'receipt.json' if path.is_dir() else path
            receipt, out = json.loads(path.read_text(encoding='utf-8')), path.parent
        expected = receipt.pop('integrity', None)
        if expected != _json_hash(receipt):
            errors.append('receipt integrity')
        if receipt['schema'] != 'synapse-adaptation-study-v1' or receipt['status'] != 'complete':
            errors.append('study is not complete or schema is unsupported')
        saved_sources = receipt['source_fingerprints']
        if not isinstance(saved_sources, dict) or not saved_sources or any(not isinstance(v, str) or len(v) != 64 for v in saved_sources.values()):
            errors.append('invalid source fingerprints')
        else:
            try:
                current_sources = _source_fingerprints()
                for name in sorted(set(current_sources) | set(saved_sources)):
                    if current_sources.get(name) != saved_sources.get(name):
                        warnings.append('current source differs: '+name)
                current_environment = _environment()
                for name, version in receipt['environment'].items():
                    if current_environment.get(name) != version:
                        warnings.append('current environment differs: '+name)
            except OSError as exc:
                warnings.append('current source comparison unavailable: '+str(exc))
        c = validate_config(receipt['config'])
        if out is not None:
            for name, expected_file in receipt['files'].items():
                if Path(name).name != name or not (out/name).is_file() or _file_hash(out/name) != expected_file:
                    errors.append('file integrity:'+name)
        supports = {(s['seed'], s['task']): s for s in receipt['supports']}
        for unit in supports.values():
            if len(unit['trace']) != c['support_steps'] or unit['support_digest'] != _json_hash(unit['trace']) or not unit['checkpoint_roundtrip_ok']:
                errors.append('support integrity')
            if sorted(unit['reward_permutation']) != list(range(c['support_steps'])):
                errors.append('reward permutation')
        for row in receipt['records']:
            unit = supports[(row['seed'], row['task'])]
            trace = row['trace']
            if len(trace) != c['query_steps'] or row['steps'] != len(trace) or row['trace_digest'] != _json_hash(trace):
                errors.append('query trace integrity')
            if row['query_seed'] != unit['query_seed'] or row['query_seed'] == unit['support_seed'] or row['support_digest'] != unit['support_digest']:
                errors.append('support/query pairing')
            metrics = {'return': float(sum(t['reward'] for t in trace)), 'reward_mse': float(np.mean([(t['reward']-t['predicted_reward'])**2 for t in trace])),
                       'planner_used_steps': sum(t['planner_used'] for t in trace), 'memory_used_steps': sum(t['memory_used'] for t in trace)}
            if any(row[key] != value for key, value in metrics.items()):
                errors.append('query metric')
            if row['learned_before'] != row['learned_after']:
                errors.append('query learned-state mutation')
            recomputed += 1
        if _summarize(receipt['records'], c) != receipt['summary']:
            errors.append('summary recomputation')
        n = len(c['seeds'])*len(c['tasks'])
        if len(supports) != n or receipt['costs'] != {'support_world_steps': n*c['support_steps'], 'shuffled_support_replay_steps': n*c['support_steps'], 'query_world_steps': n*len(ARMS)*c['query_steps'],
                                                     'model_update_calls': 2*n*c['support_steps'], 'query_update_calls': 0, 'actor_calls': n*(2*c['support_steps']+len(ARMS)*c['query_steps'])}:
            errors.append('cost accounting')
    except (KeyError, TypeError, ValueError, OSError, IndexError, OverflowError) as exc:
        errors.append(f'invalid receipt: {exc}')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'recomputed': recomputed,
            'scope': 'raw-record aggregate and hash replay; recorded freeze assertions checked; environment execution is not rerun or independently authenticated'}
