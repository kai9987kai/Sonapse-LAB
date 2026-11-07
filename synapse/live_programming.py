"""Current-parent numeric weight editing with matched real-world screens."""
from copy import deepcopy
import numpy as np

from .stateio import digest
from .patches import evaluate_patch, make_patch


def _integer(value,name,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f'{name} must be integer {low}..{high}')
    return value


def screen_live_patch(agent,packet,task=None,steps=32,seeds=None,min_gain=0.):
    from .complex_world import ComplexWorld,ALL_TASKS
    steps=_integer(steps,'steps',8,128)
    seeds=[19,23] if seeds is None else seeds
    if not isinstance(seeds,list) or not 1<=len(seeds)<=3 or len(set(seeds))!=len(seeds):
        raise ValueError('seeds must contain 1..3 unique integers')
    for seed in seeds: _integer(seed,'seed',0,2**31-1)
    task=task or agent.config.task
    if task not in ALL_TASKS: raise ValueError('unsupported evaluation task')
    if not np.isfinite(min_gain) or min_gain<0: raise ValueError('min_gain must be finite and nonnegative')
    retention_task=agent.config.task if agent.config.task in ALL_TASKS else 'forage'
    scores=[]
    def rollout(candidate,current_task,seed):
        clone=deepcopy(candidate); clone.reset_episode()
        world=ComplexWorld(seed=seed,curriculum=current_task); obs=world.observe(); total=0.
        for _ in range(steps):
            action=clone.act(obs,learn=False)['action']
            obs,reward,done,info=world.step(action); total+=float(reward)
            clone.homeostasis.step(action,float(reward),hazard_dist=info.get('min_hazard_dist',10.),
                                   consumed_food=float(reward)>.3,in_friction_zone=info.get('in_friction_zone',False))
            if done: clone.reset_episode(); obs=world.reset()
        return total
    def evaluator(candidate):
        query=[rollout(candidate,task,seed) for seed in seeds]
        retention=[rollout(candidate,retention_task,seed+101) for seed in seeds]
        scores.append({'query':query,'retention':retention})
        return {'score':float(np.mean(query)),'retention':float(np.mean(retention))}
    result=evaluate_patch(agent,packet,evaluator,min_gain=float(min_gain),max_regression=0.)
    return {**result,'task':task,'retention_task':retention_task,'steps_per_world':steps,'seeds':seeds,
            'paired_returns':{'baseline':scores[0]['query'],'candidate':scores[1]['query'],
                              'retention_baseline':scores[0]['retention'],'retention_candidate':scores[1]['retention']},
            'world_steps':4*len(seeds)*steps,'evidence_kind':'measured_world_development_screen',
            'scope':'matched finite-world development screen; no statistical or global preservation guarantee'}


def run_live_rsi(agent,task=None,steps=24,candidates=3,seed=7,min_gain=0.,progress=None,cancelled=None):
    _integer(candidates,'candidates',1,4); _integer(seed,'seed',0,2**31-1)
    rng=np.random.default_rng(seed); start=digest(agent.export_state()); screens=[]
    for number in range(candidates):
        if cancelled and cancelled(): break
        tensor=agent.editable_tensors()['motor.weight']
        index=[int(rng.integers(n)) for n in tensor.shape]
        old=float(tensor[tuple(index)].detach()); value=old+float(rng.choice([-1,1]))*.025
        packet=make_patch(agent,[{'tensor':'motor.weight','indices':[index],'values':[value]}],
                          description='bounded numeric motor edit screened on actual cloned worlds')
        screen=screen_live_patch(agent,packet,task=task,steps=steps,seeds=[seed+1009],min_gain=min_gain)
        screens.append(screen)
        if progress: progress({'phase':'live_rsi','completed':number+1,'total':candidates,'accepted':sum(s['accepted'] for s in screens)})
    return {'status':'cancelled' if cancelled and cancelled() else 'complete','parent_digest':start,
            'result_digest':digest(agent.export_state()),'candidates_evaluated':len(screens),
            'accepted_count':sum(s['accepted'] for s in screens),'screens':screens,
            'evidence_kind':'measured_world_development_screen',
            'scope':'bounded stochastic numeric search, not open-ended recursive self-improvement'}
