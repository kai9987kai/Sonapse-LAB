"""Auxiliary pretraining from recorded or cloned-world executed transitions."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import time

from .neural_graft import NeuralResidualGraft, _integer
from .stateio import digest


def _collect_worker(payload):
    """Spawn-safe private rollout worker; receives only our own detached state."""
    import os
    from .config import SynapseConfig
    from .runtime import LabRuntime
    from .complex_world import ComplexWorld
    config,state,steps,namespace=payload
    config=deepcopy(config); config['connectome_sources']=tuple(config['connectome_sources'])
    config.update(compute_threads=1,active_training=False)
    cfg=SynapseConfig(**config)
    runtime=LabRuntime(cfg)
    state=deepcopy(state); state['config']=cfg.to_dict()
    runtime.agent.import_state(state); runtime.agent.reset_episode()
    seed=(cfg.seed+100003*(namespace+1))%(2**31)
    runtime.world=ComplexWorld(seed=seed,curriculum=cfg.task); runtime.obs=runtime.world.observe()
    records=[]; wall,cpu=time.perf_counter(),time.process_time()
    for _ in range(steps):
        runtime.step(1,learn=True)
        record=runtime.agent.neural_graft.replay[-1]
        records.append({key:deepcopy(record[key]) for key in ('obs','features','baseline','action','next_obs','reward','done')})
    return {'records':records,'pid':os.getpid(),'world_seed':seed,
            'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall}


def _parallel_records(runtime,steps,workers,progress,cancelled):
    import multiprocessing as mp
    state=runtime.agent.export_state(); config=runtime.config.to_dict()
    count=min(workers,steps); base,remainder=divmod(steps,count)
    payloads=[(config,state,base+(i<remainder),i) for i in range(count)]
    outputs=[]
    # One task per child also guarantees independent workers on short pilots.
    with mp.get_context('spawn').Pool(count,maxtasksperchild=1) as pool:
        pending=[pool.apply_async(_collect_worker,(payload,)) for payload in payloads]
        while pending:
            if cancelled and cancelled():
                pool.terminate()
                return [],[],True
            for future in pending[:]:
                if future.ready():
                    outputs.append(future.get()); pending.remove(future)
                    if progress: progress({'phase':'parallel_collection','workers_completed':len(outputs),'workers':count,
                                           'completed':sum(len(o['records']) for o in outputs),'total':steps})
            if pending: time.sleep(.05)
    records=[record for output in outputs for record in output['records']]
    return records,outputs,False


def pretrain_from_transitions(graft, transitions, updates=32, batch_size=128):
    """Fit prediction/value heads transactionally; never imitate invented actions.

    Records contain ``obs``, ``features``, ``baseline``, actual ``action``,
    ``next_obs``, ``reward`` and ``done``. All records are validated on an
    isolated candidate before any resident learned state is replaced.
    """
    _integer(updates, 'updates', 1, 1000)
    _integer(batch_size, 'batch_size', 1, 2048)
    if not isinstance(transitions, (list, tuple)) or not 1 <= len(transitions) <= 10000:
        raise ValueError('transitions must contain 1..10000 executed records')
    candidate = NeuralResidualGraft(**graft.settings)
    candidate.import_state(graft.export_state())
    fields = ('obs', 'features', 'baseline', 'action', 'next_obs', 'reward', 'done')
    for record in transitions:
        if not isinstance(record, dict) or any(key not in record for key in fields):
            raise ValueError('incomplete executed transition record')
        candidate.record_transition(**{key: record[key] for key in fields})
    result = candidate.train_replay(updates=updates, batch_size=batch_size, actor=False)
    graft.import_state(candidate.export_state())
    return {**result, 'recorded_transitions': len(transitions),
            'phase': 'auxiliary_pretraining', 'policy_frozen': True}


def pretrain_runtime(runtime, steps=256, updates=32, batch_size=128, apply=True,
                     progress=None, cancelled=None,workers=1, device='cpu'):
    """Collect real transitions in a cloned world and fit only the new graft.

    With ``apply=False`` the result includes ``candidate_state`` and
    ``parent_graft_digest`` for a caller to apply under its own runtime lock.
    Publication is allowed only while the live parent digest still matches.
    No synthetic labels, hidden world targets or future observations become
    policy inputs. The clone's ordinary active auto-fit is suppressed.
    """
    _integer(steps, 'steps', 1, 10000)
    _integer(updates, 'updates', 1, 1000)
    _integer(batch_size, 'batch_size', 1, 2048)
    import os
    _integer(workers,'workers',1,min(4,os.cpu_count() or 1))
    if not isinstance(device, str) or device not in {'cpu', 'auto'}:
        raise ValueError("device must be 'cpu' or 'auto'")
    if type(apply) is not bool:
        raise ValueError('apply must be boolean')
    live_graft = getattr(runtime.agent, 'neural_graft', None)
    if live_graft is None:
        raise ValueError('neural graft must be enabled for pretraining')
    parent_digest = digest(live_graft.export_state())
    candidate_runtime = deepcopy(runtime)
    if hasattr(candidate_runtime.config, 'active_training'):
        candidate_runtime.config = replace(candidate_runtime.config, active_training=False)
        candidate_runtime.agent.config = candidate_runtime.config
    candidate = candidate_runtime.agent.neural_graft
    wall, cpu = time.perf_counter(), time.process_time()
    before = candidate.samples
    worker_outputs=[]
    if workers>1:
        records,worker_outputs,aborted=_parallel_records(candidate_runtime,steps,workers,progress,cancelled)
        if aborted:
            return {'status':'cancelled','applied':False,'recorded_transitions':0,'optimizer_updates':0,'parent_graft_digest':parent_digest}
        for record in records: candidate.record_transition(**record)
    for index in range(steps if workers==1 else 0):
        if cancelled is not None and cancelled():
            return {'status': 'cancelled', 'applied': False, 'recorded_transitions': candidate.samples-before,
                    'optimizer_updates': 0, 'actor_updates': 0,
                    'parent_graft_digest': parent_digest}
        candidate_runtime.step(1, learn=True)
        if progress is not None and (index == 0 or (index+1) % 8 == 0 or index+1 == steps):
            progress({'phase': 'collecting', 'completed': index+1, 'total': steps,
                      'recorded_transitions': candidate.samples-before})
    # These off-line trajectories may include a drifting cloned legacy actor.
    # They train prediction/value only and must never become on-policy evidence.
    for record in candidate.replay:
        record['actor_eligible'] = False
    candidate.pending = None
    if cancelled is not None and cancelled():
        return {'status': 'cancelled', 'applied': False, 'recorded_transitions': candidate.samples-before,
                'optimizer_updates': 0, 'actor_updates': 0, 'parent_graft_digest': parent_digest}
    if progress is not None:
        progress({'phase': 'fitting', 'completed': 0, 'total': updates,
                  'recorded_transitions': candidate.samples-before})
    from .compute import resolve_training_device
    selected_device, device_selection = resolve_training_device(device)
    candidate_before_fit = candidate.export_state() if selected_device != 'cpu' else None
    try:
        result = candidate.train_replay(updates=updates, batch_size=batch_size, actor=False,
                                        device=selected_device)
    except (RuntimeError, NotImplementedError) as exc:
        if selected_device == 'cpu':
            raise
        # Training is isolated on a clone. Restore its post-collection state and
        # retry on CPU so a driver/OOM/backend error does not lose the run.
        candidate.import_state(candidate_before_fit)
        selected_device = 'cpu'
        result = candidate.train_replay(updates=updates, batch_size=batch_size, actor=False,
                                        device='cpu')
        device_selection = {
            **device_selection,
            'selected': 'cpu',
            'fallback_reason': f'{type(exc).__name__}: {exc}',
        }
        result['accelerator_fallback'] = True
    else:
        result['accelerator_fallback'] = False
    result['device_selection'] = device_selection
    candidate_state = candidate.export_state()
    result.update(recorded_transitions=candidate.samples-before, collection_steps=steps,
                  phase='auxiliary_pretraining', policy_frozen=True, applied=False,
                  parent_graft_digest=parent_digest,
                  session_cpu_seconds=max(0., time.process_time()-cpu),
                  session_wall_seconds=max(0., time.perf_counter()-wall))
    result.update(workers_used=len(worker_outputs) if worker_outputs else 1,
                  worker_pids=[o['pid'] for o in worker_outputs],
                  worker_world_seeds=[o['world_seed'] for o in worker_outputs],
                  collection_worker_cpu_seconds=sum(o['cpu_seconds'] for o in worker_outputs))
    if cancelled is not None and cancelled():
        result['status'] = 'cancelled'
        return result
    if apply:
        if digest(live_graft.export_state()) != parent_digest:
            result.update(status='conflict', reason='live graft changed during pretraining')
        else:
            live_graft.import_state(candidate_state)
            result['applied'] = True
    else:
        result['candidate_state'] = candidate_state
    if progress is not None:
        progress({'phase': 'completed', 'completed': updates, 'total': updates,
                  'recorded_transitions': result['recorded_transitions']})
    return result
