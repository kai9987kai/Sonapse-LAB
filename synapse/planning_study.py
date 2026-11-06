"""Paired real-world planning pilots; receipt verification checks arithmetic/hashes.

This is a development experiment, not confirmation or receipt authentication.
Every arm receives the same supported learner and independent paired query worlds.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .adaptation_study import (_seed, _json_hash, _file_hash, _source_fingerprints,
                               _environment, _parameters)
from .stateio import digest, save_state, _atomic


DEFAULTS = {'seeds': [23, 29, 31], 'tasks': ['forage', 'push'], 'support_steps': 64,
            'query_steps': 64, 'nodes_per_source': 16, 'horizons': [1, 3]}


def validate_config(config=None):
    from .complex_world import ALL_TASKS
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config)-set(DEFAULTS):
        raise ValueError('unknown planning-study configuration')
    c = deepcopy({**DEFAULTS, **config})
    for key, lo, hi in [('seeds', 0, 2**31-1), ('horizons', 1, 5)]:
        values = c[key]
        limit = 8 if key == 'seeds' else 3
        if (not isinstance(values, list) or not 1 <= len(values) <= limit
                or any(type(v) is not int or not lo <= v <= hi for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'{key} must contain unique bounded integers')
    tasks = c['tasks']
    if (not isinstance(tasks, list) or not 1 <= len(tasks) <= 4
            or any(t not in ALL_TASKS for t in tasks) or len(set(tasks)) != len(tasks)):
        raise ValueError('tasks must contain 1..4 unique supported world tasks')
    for key, lo, hi in [('support_steps', 8, 512), ('query_steps', 8, 512), ('nodes_per_source', 16, 200)]:
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    if len(c['seeds'])*len(tasks)*(c['support_steps']+(1+len(c['horizons']))*c['query_steps']) > 16384:
        raise ValueError('planning study exceeds 16384 executed transitions')
    return c


def _learned_hash(agent):
    return digest({'parameters': _parameters(agent), 'optimizer': agent.opt.state_dict(),
                   'model': agent.experience_model.export_state(),
                   'memory': agent.surprise_memory.export_state(),
                   'structural': agent.neurogenesis.export_state(),
                   'pre': agent.graph.graft_pre, 'post': agent.graph.graft_post})


def _arms(config):
    return ['actor']+[f'horizon_{h}' for h in config['horizons']]


def _summarize(records, config):
    from .evidence_engine import bootstrap_confidence_interval
    units = [(s, t) for s in config['seeds'] for t in config['tasks']]
    index = {(r['seed'], r['task'], r['arm']): r for r in records}
    expected = {(s, t, a) for s, t in units for a in _arms(config)}
    if len(records) != len(expected) or set(index) != expected:
        raise ValueError('missing or duplicate paired records')
    if any(type(r['steps']) is not int or r['steps'] != config['query_steps'] for r in records):
        raise ValueError('invalid query step count')
    arms = {}
    for arm in _arms(config):
        rows = [index[s, t, arm] for s, t in units]
        arms[arm] = {'mean_return': float(np.mean([r['return'] for r in rows])),
                     'returns': [r['return'] for r in rows],
                     'planner_use_fraction': float(np.mean([r['planner_used_steps']/r['steps'] for r in rows])),
                     'mean_model_evaluations': float(np.mean([r['model_evaluations']/r['steps'] for r in rows]))}
    comparisons = {}
    for arm in _arms(config)[1:]:
        differences = np.asarray(arms[arm]['returns'])-arms['actor']['returns']
        comparisons[arm+'_minus_actor'] = {
            'paired_differences': differences.tolist(), 'mean_difference': float(differences.mean()),
            'descriptive_bootstrap_95': list(bootstrap_confidence_interval(differences)) if len(units) > 1 else None}
    return {'arms': arms, 'comparisons': comparisons, 'paired_units': len(units),
            'interpretation': 'finite development pilot; shared task types; no automatic promotion; no general performance claim'}


def run_planning_study(output_dir=None, config=None):
    from .config import SynapseConfig
    from .runtime import LabRuntime
    from .complex_world import ComplexWorld
    c = validate_config(config)
    started = time.perf_counter()
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    records, supports, files = [], [], {}
    for seed in c['seeds']:
        for task in c['tasks']:
            agent_seed = _seed('planning-agent', seed, task)
            support_seed = _seed('planning-support', seed, task)
            query_seed = _seed('planning-query', seed, task)
            cfg = SynapseConfig(seed=agent_seed, task=task, nodes_per_source=c['nodes_per_source'])
            support = LabRuntime(cfg, output_root=out)
            support.world = ComplexWorld(seed=support_seed, curriculum=task)
            support.obs = support.world.observe()
            rng = np.random.default_rng(_seed('planning-exploration', seed, task))
            trace = []
            # Manual actions exercise the same validated learning boundary as the UI.
            for _ in range(c['support_steps']):
                previous = support.obs.tolist()
                state = support.step(1, learn=True, action=rng.uniform(-.8, .8, 5))
                row = state['recent'][-1]
                trace.append({'observation': previous, 'action': state['action'],
                              'reward': row['reward'], 'done': row['done']})
            trained = support.agent.export_state()
            support_digest = _json_hash(trace)
            unit = {'seed': seed, 'task': task, 'agent_seed': agent_seed,
                    'support_seed': support_seed, 'query_seed': query_seed,
                    'support_digest': support_digest, 'trace': trace,
                    'learned_state_digest': _learned_hash(support.agent),
                    'model_summary': support.agent.experience_model.summary()}
            if out is not None:
                name = f'support_{seed}_{task}.pt'
                save_state(out/name, trained)
                for artifact in [name, name+'.json']:
                    files[artifact] = _file_hash(out/artifact)
                unit['checkpoint'] = name
            supports.append(unit)
            for arm in _arms(c):
                query = LabRuntime(cfg, output_root=out)
                query.agent.import_state(trained)
                query.agent.reset_episode()
                horizon = 1 if arm == 'actor' else int(arm.split('_')[1])
                query.config = replace(cfg, experience_planning=arm != 'actor', experience_horizon=horizon)
                query.agent.config = query.config
                query.world = ComplexWorld(seed=query_seed, curriculum=task)
                query.obs = query.world.observe()
                before = _learned_hash(query.agent)
                query_trace = []
                for step in range(c['query_steps']):
                    state = query.step(1, learn=False)
                    row = state['recent'][-1]
                    plan = state['experience']['planner']
                    query_trace.append({'step': step, 'reward': row['reward'], 'done': row['done'],
                                        'action': state['action'], 'planner_used': bool(plan.get('used', False)),
                                        'planner_reason': plan.get('reason', 'actor policy'),
                                        'model_evaluations': int(plan.get('evaluations', 0))})
                after = _learned_hash(query.agent)
                if before != after:
                    raise RuntimeError('frozen planning query mutated learned state')
                records.append({'seed': seed, 'task': task, 'arm': arm, 'query_seed': query_seed,
                                'support_digest': support_digest, 'steps': len(query_trace),
                                'return': float(sum(r['reward'] for r in query_trace)),
                                'planner_used_steps': sum(r['planner_used'] for r in query_trace),
                                'model_evaluations': sum(r['model_evaluations'] for r in query_trace),
                                'learned_before': before, 'learned_after': after,
                                'trace': query_trace, 'trace_digest': _json_hash(query_trace)})
    units = len(supports)
    receipt = {'schema': 'synapse-planning-study-v1', 'status': 'complete', 'config': c,
               'scope': 'actual ComplexWorld transitions; shared exploratory support; frozen paired queries; heuristic short-horizon planner',
               'supports': supports, 'records': records, 'summary': _summarize(records, c),
               'costs': {'support_world_steps': units*c['support_steps'],
                         'query_world_steps': units*len(_arms(c))*c['query_steps'],
                         'support_update_calls': units*c['support_steps'], 'query_update_calls': 0},
               'files': files, 'source_fingerprints': _source_fingerprints(), 'environment': _environment(),
               'wall_seconds': time.perf_counter()-started}
    receipt['integrity'] = _json_hash(receipt)
    if out is not None:
        _atomic(out/'receipt.json', json.dumps(receipt, indent=2, allow_nan=False).encode())
        _atomic(out/'report.md', format_planning_report(receipt).encode())
    return receipt


def verify_planning_study(source):
    errors, warnings = [], []
    root = None
    try:
        if isinstance(source, dict):
            receipt = deepcopy(source)
        else:
            path = Path(source)
            path = path/'receipt.json' if path.is_dir() else path
            root = path.parent
            receipt = json.loads(path.read_text(encoding='utf-8'))
        integrity = receipt.pop('integrity')
        if _json_hash(receipt) != integrity:
            errors.append('receipt integrity mismatch')
        if receipt['schema'] != 'synapse-planning-study-v1' or receipt['status'] != 'complete':
            raise ValueError('unsupported receipt')
        c = validate_config(receipt['config'])
        expected = {(s, t) for s in c['seeds'] for t in c['tasks']}
        supports = {(u['seed'], u['task']): u for u in receipt['supports']}
        if set(supports) != expected or len(supports) != len(receipt['supports']):
            raise ValueError('support pairing mismatch')
        for unit in supports.values():
            if len(unit['trace']) != c['support_steps'] or _json_hash(unit['trace']) != unit['support_digest']:
                errors.append('support trace mismatch')
            if unit['support_seed'] == unit['query_seed']:
                errors.append('support/query seed overlap')
        for r in receipt['records']:
            unit = supports[r['seed'], r['task']]
            trace = r['trace']
            if r['steps'] != c['query_steps'] or len(trace) != r['steps']:
                errors.append('query step count mismatch')
            if r['query_seed'] != unit['query_seed'] or r['support_digest'] != unit['support_digest']:
                errors.append('query pairing mismatch')
            if r['learned_before'] != r['learned_after'] or r['learned_before'] != unit['learned_state_digest']:
                errors.append('query learned-state freeze mismatch')
            if _json_hash(trace) != r['trace_digest']:
                errors.append('query trace digest mismatch')
            if not np.isfinite(r['return']) or abs(sum(t['reward'] for t in trace)-r['return']) > 1e-10:
                errors.append('return arithmetic mismatch')
            if r['planner_used_steps'] != sum(t['planner_used'] for t in trace):
                errors.append('planner count mismatch')
            if r['model_evaluations'] != sum(t['model_evaluations'] for t in trace):
                errors.append('model work count mismatch')
            for step, t in enumerate(trace):
                a = np.asarray(t['action'], dtype=float)
                if (t['step'] != step or a.shape != (5,) or not np.isfinite(a).all()
                        or np.any(np.abs(a) > 1.000001) or not np.isfinite(t['reward'])
                        or type(t['done']) is not bool or type(t['planner_used']) is not bool
                        or type(t['model_evaluations']) is not int or not 0 <= t['model_evaluations'] <= 256):
                    errors.append('invalid query transition')
        if _json_hash(_summarize(receipt['records'], c)) != _json_hash(receipt['summary']):
            errors.append('summary recomputation mismatch')
        units = len(expected)
        expected_costs = {'support_world_steps': units*c['support_steps'],
                          'query_world_steps': units*len(_arms(c))*c['query_steps'],
                          'support_update_calls': units*c['support_steps'], 'query_update_calls': 0}
        if receipt['costs'] != expected_costs:
            errors.append('cost recomputation mismatch')
        if root is not None:
            for name, sha in receipt['files'].items():
                if Path(name).name != name or _file_hash(root/name) != sha:
                    errors.append('artifact hash mismatch: '+name)
        current = _source_fingerprints()
        warnings.extend('current source differs: '+name for name, sha in receipt['source_fingerprints'].items() if current.get(name) != sha)
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings,
            'scope': 'hash and arithmetic replay; does not re-execute worlds or authenticate consistently rewritten receipts'}


def format_planning_report(receipt):
    lines = ['# Measured planning development pilot', '', receipt['scope'], '',
             '| Frozen arm | Mean return | Planner use | Model evaluations / step |', '|---|---:|---:|---:|']
    for name, row in receipt['summary']['arms'].items():
        lines.append(f"| {name} | {row['mean_return']:.6f} | {row['planner_use_fraction']:.3f} | {row['mean_model_evaluations']:.1f} |")
    lines += ['', 'Paired return differences from actor:']
    for name, row in receipt['summary']['comparisons'].items():
        lines.append(f"- {name}: {row['mean_difference']:.6f}; descriptive interval {row['descriptive_bootstrap_95']}")
    lines += ['', receipt['summary']['interpretation'], '',
              'Bootstrap units are seed/task runs, not individual transitions. Shared support changes graph and memory; query fitting is disabled. The planner has no terminal/value head, uses lossy sensory summaries, and may abstain. Receipts check integrity and arithmetic, not authenticity.']
    return '\n'.join(lines)+'\n'
