"""Bounded CPU cue-memory development pilot and deterministic numeric replay."""
from pathlib import Path
import json,time,hashlib
import numpy as np
from .learning import adapt,predict,lesion,fresh_weights
from .rsi import Policy,propose

DEFAULTS={'seed':7,'lineages':3,'generations':3,'candidates':3,'support':16,'query':24,'nodes_per_source':16}


def validate_config(config):
    if not isinstance(config,dict) or set(config)-set(DEFAULTS): raise ValueError('unknown study configuration')
    c={**DEFAULTS,**config}
    bounds={'seed':(0,2**31-1),'lineages':(1,10),'generations':(2,8),'candidates':(1,6),'support':(8,64),'query':(8,128),'nodes_per_source':(16,200)}
    for k,(lo,hi) in bounds.items():
        if type(c[k]) is not int or not lo<=c[k]<=hi: raise ValueError(f'{k} outside [{lo},{hi}]')
    return c


def _write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def _hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def _tasks(c,cancelled):
    from .agent import IntegratedAgent
    from .config import SynapseConfig
    agent=IntegratedAgent(SynapseConfig(seed=c['seed'],nodes_per_source=c['nodes_per_source']))
    tasks=[]
    # Entirely separate seed namespaces and cue coordinate families. No query enters adaptation.
    for split,families,offset in [('development',[0,1],10000),('validation',[2,3],20000),('sealed',[4,5],30000)]:
        for family in families:
            rng=np.random.default_rng(c['seed']+offset+family)
            samples=[]; labels=[]
            for i in range(c['support']+c['query']):
                if cancelled and cancelled(): raise InterruptedError('cancelled during feature extraction')
                label=i%2; obs=np.zeros(155,dtype=np.float32)
                obs[0:4]=[1.,.5,0.,.5]
                obs[8+family*2+label]=1.
                obs[32:40]=rng.normal(0,.015,8)
                agent.reset_episode()
                # Retain cue across one occlusion step in the actual recurrent graph/Tidal actor.
                agent.act(obs,learn=False)
                delayed=obs.copy(); delayed[8:24]=0
                f=np.asarray(agent.act(delayed,learn=False)['features'],dtype=float)
                # Cue features and delayed state jointly test persistent resident readout acquisition.
                agent.reset_episode(); cue=np.asarray(agent.act(obs,learn=False)['features'],dtype=float)
                samples.append(np.concatenate([cue,f,[1.]])); labels.append(label)
            x=np.asarray(samples); x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
            tasks.append({'split':split,'family':family,'x':x,'y':np.asarray(labels,dtype=int)})
    return tasks


def _evaluate(policy,tasks,c,control='intact',collect=False):
    scores=[]; records=[]
    for task in tasks:
        n=c['support']; x=task['x']; y=task['y']
        w,cost=adapt(x[:n],y[:n],policy,control)
        query=lesion(x[n:],control)
        score=float(np.mean(predict(query,w)==y[n:])); scores.append(score)
        if collect: records.append((task,w,score,cost))
    return float(np.mean(scores)),records


def run_study(output_dir,config,progress=None,cancelled=None):
    c=validate_config(config); out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    receipt={'schema':1,'status':'running','config':c,'scope':'synthetic cue-memory development pilot; fixed local numeric operators; no frontier writer','proposals':[],'lineage':[],'evaluations':[],'archive':'shared immutable empty program archive','query_selection_exposures':0}
    def stop():
        receipt.update(status='cancelled',wall_seconds=time.perf_counter()-start); _write(out/'receipt.json',receipt); return receipt
    if cancelled and cancelled(): return stop()
    try: tasks=_tasks(c,cancelled)
    except InterruptedError: return stop()
    receipt['feature_extraction_seconds']=time.perf_counter()-start
    receipt['feature_extraction_calls']=6*(c['support']+c['query'])*3
    data={}
    for i,t in enumerate(tasks): data[f'x{i}']=t['x']; data[f'y{i}']=t['y']
    np.savez_compressed(out/'tasks.npz',**data)
    receipt['tasks']=[{'split':t['split'],'family':t['family']} for t in tasks]
    development=[t for t in tasks if t['split']=='development']; validation=[t for t in tasks if t['split']=='validation']
    frozen=[]
    for lineage in range(c['lineages']):
        for arm in ['fixed','nonrecursive','recursive','random']:
            initial=Policy(); current=initial
            for generation in range(c['generations']):
                if cancelled and cancelled(): return stop()
                parent=current if arm=='recursive' else initial
                candidates=[]
                if arm!='fixed':
                    for j in range(c['candidates']):
                        child=propose(parent,j,c['seed']+lineage*100+generation*10000)
                        dev,dev_records=_evaluate(child,development,c,collect=True); val,val_records=_evaluate(child,validation,c,collect=True)
                        proposal={'lineage':lineage,'arm':arm,'generation':generation,'generator':parent.identity,'proposal_index':j,'proposal_seed':c['seed']+lineage*100+generation*10000,'policy':child.to_dict(),'development_accuracy':dev,'validation_accuracy':val,'cost':[r[3] for r in dev_records+val_records],'accepted':False,'reason':'not selected'}
                        receipt['proposals'].append(proposal); candidates.append((child,val,proposal))
                    baseline,_=_evaluate(current,validation,c)
                    if arm=='random': chosen=candidates[(lineage+generation)%len(candidates)]
                    else:
                        # The inherited ranking program controls tie-breaking.
                        def rank(item):
                            p,v,_=item
                            return (v,-p.steps if parent.ranking=='accuracy_then_cost' else -p.rate)
                        chosen=max(candidates,key=rank)
                    if arm=='random' or chosen[1]>=baseline:
                        current=chosen[0]; chosen[2].update(accepted=True,reason='random matched change' if arm=='random' else 'validation non-regression')
                row={'lineage':lineage,'arm':arm,'generation':generation,'generator':parent.identity,'policy':current.to_dict(),'fresh_learner_digest':hashlib.sha256(fresh_weights(tasks[0]['x'].shape[1]).tobytes()).hexdigest()}
                receipt['lineage'].append(row); frozen.append(row)
                if progress: progress({'phase':'search','lineage':lineage,'arm':arm,'generation':generation})
    for lineage in range(c['lineages']):
        for method in ['gradient','direct','ridge']:
            policy=Policy(operator=method)
            frozen.append({'lineage':lineage,'arm':method+'_baseline','generation':-1,'policy':policy.to_dict()})
    # Every generation is frozen before any sealed outcome is evaluated.
    sealed=[t for t in tasks if t['split']=='sealed']
    for index,row in enumerate(frozen):
        if cancelled and cancelled(): return stop()
        policy=Policy.from_dict(row['policy'])
        for control in ['intact','graph_off','donor_off','memory_off']:
            _,records=_evaluate(policy,sealed,c,control,True)
            for task,w,score,cost in records:
                name=f'readout_{index}_{control}_{task["family"]}.npz'
                np.savez_compressed(out/name,weights=w)
                # Reload, with no fitting/operator call during query inference.
                with np.load(out/name,allow_pickle=False) as archive: loaded=archive['weights']
                n=c['support']; persisted=float(np.mean(predict(lesion(task['x'][n:],control),loaded)==task['y'][n:]))
                receipt['evaluations'].append({'lineage':row['lineage'],'arm':row['arm'],'generation':row['generation'],'policy':policy.to_dict(),'family':task['family'],'control':control,'accuracy':score,'writer_off_accuracy':persisted,'checkpoint':name,'cost':cost,'initial_accuracy':float(np.mean(predict(lesion(task['x'][n:],control),fresh_weights(w.shape[0]))==task['y'][n:]))})
    # Cross-test a co-adapted resident head, with old-family retention measured independently.
    receipt['cross_comparisons']=[]
    for row in frozen:
        if cancelled and cancelled(): return stop()
        policy=Policy.from_dict(row['policy']); oldx=np.concatenate([t['x'][:c['support']] for t in development]); oldy=np.concatenate([t['y'][:c['support']] for t in development])
        evolved,_=adapt(oldx,oldy,Policy())
        for task in sealed:
            n=c['support']; continued,cost=adapt(task['x'][:n],task['y'][:n],policy,initial=evolved)
            retention=[]
            for old in development:
                before=float(np.mean(predict(old['x'][n:],evolved)==old['y'][n:])); after=float(np.mean(predict(old['x'][n:],continued)==old['y'][n:]))
                retention.append({'family':old['family'],'before':before,'after':after,'regression':before-after})
            name=f'evolved_{row["lineage"]}_{row["arm"]}_{row["generation"]}_{task["family"]}.npz'
            np.savez_compressed(out/name,weights=continued)
            receipt['cross_comparisons'].append({'lineage':row['lineage'],'arm':row['arm'],'generation':row['generation'],'policy':policy.to_dict(),'family':task['family'],'checkpoint':name,'accuracy':float(np.mean(predict(task['x'][n:],continued)==task['y'][n:])),'retention':retention,'cost':cost})
    differences=[]
    for lineage in range(c['lineages']):
        rows=[r for r in receipt['evaluations'] if r['lineage']==lineage and r['control']=='intact']
        def mean(arm,g): return np.mean([r['accuracy'] for r in rows if r['arm']==arm and r['generation']==g])
        differences.append(float(mean('recursive',c['generations']-1)-mean('fixed',c['generations']-1)))
    uncertainty=None
    if len(differences)>1:
        rng=np.random.default_rng(c['seed']); boot=np.mean(rng.choice(differences,(2000,len(differences))),axis=1); uncertainty=np.quantile(boot,[.025,.975]).tolist()
    receipt['summary']={'paired_lineage_recursive_minus_fixed':differences,'mean_difference':float(np.mean(differences)),'lineage_bootstrap_95_interval':uncertainty,'interpretation':'development pilot only; no confirmed RSI advantage; small synthetic task-family sample','evolved_actor_comparison':'resident readout continued from paired development-task head; frozen recurrent backbone','retention':'per-old-family accuracy before/after sealed support adaptation; final metrics do not select policies'}
    receipt['cost_accounting']={'feature_calls':receipt['feature_extraction_calls'],'proposal_fits':sum(len(p['cost']) for p in receipt['proposals']),'baseline_selection_fits':c['lineages']*3*c['generations']*len(validation),'sealed_fits':len(receipt['evaluations']),'coadapted_fits':len(frozen)*3,'query_inference_is_frozen':True,'editor_external_inference_calls':0,'active_graph_nodes_per_source':c['nodes_per_source'],'archive_entries':0,'note':'fit counts include rejected proposals; coadapted fits include old-task preparation; wall time includes neural extraction, search, checkpoint writes and final assays'}
    receipt['files']={p.name:_hash(p) for p in out.iterdir() if p.suffix=='.npz'}
    receipt.update(status='complete',wall_seconds=time.perf_counter()-start)
    _write(out/'receipt.json',receipt)
    return receipt


def verify_study(path):
    out=Path(path); out=out.parent if out.is_file() else out
    receipt=json.loads((out/'receipt.json').read_text(encoding='utf-8')); errors=[]
    if receipt.get('status')!='complete': return {'ok':False,'errors':['study is not complete'],'recomputed':0}
    c=validate_config(receipt['config'])
    for name,expected in receipt['files'].items():
        if Path(name).name!=name or not (out/name).is_file() or _hash(out/name)!=expected: errors.append(f'integrity:{name}')
    if errors: return {'ok':False,'errors':errors,'recomputed':0}
    with np.load(out/'tasks.npz',allow_pickle=False) as data:
        tasks=[{**t,'x':data[f'x{i}'],'y':data[f'y{i}']} for i,t in enumerate(receipt['tasks'])]
    for row in receipt['evaluations']:
        policy=Policy.from_dict(row['policy']); task=next(t for t in tasks if t['family']==row['family'] and t['split']=='sealed')
        score,records=_evaluate(policy,[task],c,row['control'],True)
        with np.load(out/row['checkpoint'],allow_pickle=False) as saved: w=saved['weights']
        if not np.allclose(w,records[0][1],atol=1e-12,rtol=1e-12): errors.append('weights:'+row['checkpoint'])
        if score!=row['accuracy'] or score!=row['writer_off_accuracy'] or records[0][3]!=row['cost']: errors.append('metric:'+row['checkpoint'])
    development=[t for t in tasks if t['split']=='development']; n=c['support']
    oldx=np.concatenate([t['x'][:n] for t in development]); oldy=np.concatenate([t['y'][:n] for t in development]); evolved,_=adapt(oldx,oldy,Policy())
    for row in receipt.get('cross_comparisons',[]):
        task=next(t for t in tasks if t['family']==row['family'] and t['split']=='sealed')
        continued,cost=adapt(task['x'][:n],task['y'][:n],Policy.from_dict(row['policy']),initial=evolved)
        with np.load(out/row['checkpoint'],allow_pickle=False) as data: stored=data['weights']
        retention=[]
        for old in development:
            before=float(np.mean(predict(old['x'][n:],evolved)==old['y'][n:])); after=float(np.mean(predict(old['x'][n:],continued)==old['y'][n:]))
            retention.append({'family':old['family'],'before':before,'after':after,'regression':before-after})
        score=float(np.mean(predict(task['x'][n:],continued)==task['y'][n:]))
        if score!=row['accuracy'] or retention!=row['retention'] or cost!=row['cost'] or not np.allclose(continued,stored,atol=1e-12,rtol=1e-12): errors.append('cross comparison replay')
    # Proposals are regenerated from the actual inherited generator identity.
    known={Policy().identity:Policy()}
    for row in receipt['proposals']:
        p=Policy.from_dict(row['policy']); known[p.identity]=p
    for row in receipt['proposals']:
        parent=known.get(row['generator'])
        if parent is None or row['policy']['parent']!=row['generator']: errors.append('lineage parent')
        elif propose(parent,row['proposal_index'],row['proposal_seed']).to_dict()!=row['policy']: errors.append('proposal replay')
        for split,key in [('development','development_accuracy'),('validation','validation_accuracy')]:
            score,_=_evaluate(Policy.from_dict(row['policy']),[t for t in tasks if t['split']==split],c)
            if score!=row[key]: errors.append('proposal metric')
    return {'ok':not errors,'errors':errors,'recomputed':len(receipt['evaluations'])+len(receipt.get('cross_comparisons',[])),'scope':'numeric readout replay from recorded frozen integrated features; does not re-extract upstream neural features'}


def run_training(output_dir,config=None,progress=None,cancelled=None):
    return run_study(output_dir,config or {},progress,cancelled)
