"""Bounded numeric weight programs bound to the exact acting-system state."""
import copy, math, uuid
import torch
from .stateio import digest

SCHEMA='synapse-weight-program-v1'
MAX_WRITES=4096

def _validate(agent,edits,include_expected=True):
    if not isinstance(edits,list) or not 1<=len(edits)<=256: raise ValueError('edits must be a nonempty bounded list')
    surfaces=agent.editable_tensors(); seen=set(); normalized=[]; count=0; squared=0.
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)-{'tensor','indices','values','expected','shape'}: raise ValueError('invalid edit fields')
        name=edit.get('tensor')
        if name not in surfaces: raise ValueError('unknown or frozen tensor')
        tensor=surfaces[name]; indices=edit.get('indices'); values=edit.get('values')
        if not isinstance(indices,list) or not isinstance(values,list) or not indices or len(indices)!=len(values): raise ValueError('indices and values must match')
        if 'shape' in edit and edit['shape']!=list(tensor.shape): raise ValueError('tensor shape changed')
        expected=[]
        for index,value in zip(indices,values):
            if not isinstance(index,list) or len(index)!=tensor.ndim or any(type(i) is not int or not 0<=i<tensor.shape[d] for d,i in enumerate(index)):
                raise ValueError('index outside tensor bounds')
            key=(name,tuple(index))
            if key in seen: raise ValueError('duplicate edit address')
            seen.add(key); count+=1
            if count>MAX_WRITES: raise ValueError('weight-write budget exceeded')
            if type(value) not in (float,int) or not math.isfinite(value) or abs(value)>10: raise ValueError('weight must be finite and within [-10,10]')
            old=float(tensor[tuple(index)].detach())
            if not math.isfinite(old): raise ValueError('existing weight is not finite')
            expected.append(old); squared+=(float(value)-old)**2
        if include_expected and edit.get('expected')!=expected: raise ValueError('expected old weights differ')
        normalized.append({'tensor':name,'shape':list(tensor.shape),'indices':copy.deepcopy(indices),'expected':expected,'values':[float(x) for x in values]})
    if squared>100: raise ValueError('weight delta L2 budget exceeded')
    return normalized,count,math.sqrt(squared)

def make_patch(agent,edits,description=''):
    normalized,count,norm=_validate(agent,edits,False)
    return {'schema':SCHEMA,'parent_digest':digest(agent.export_state()),'description':str(description)[:2000],
            'author':'explicit-numeric-packet','edits':normalized,'writes':count,'delta_l2':norm}

def preview_patch(agent,packet):
    if not isinstance(packet,dict) or packet.get('schema')!=SCHEMA: raise ValueError('unsupported weight-program schema')
    if packet.get('parent_digest')!=digest(agent.export_state()): raise ValueError('stale parent checkpoint')
    edits,count,norm=_validate(agent,packet.get('edits'))
    return {'ok':True,'writes':count,'delta_l2':norm,'tensors':sorted({e['tensor'] for e in edits}),'parent_digest':packet['parent_digest']}

def apply_patch(agent,packet):
    preview=preview_patch(agent,packet); before=copy.deepcopy(agent.export_state())
    try:
        with torch.no_grad():
            surfaces=agent.editable_tensors()
            for e in packet['edits']:
                for index,value in zip(e['indices'],e['values']): surfaces[e['tensor']][tuple(index)]=value
        # Optimizer state touching changed tensors must not retain stale moments.
        if hasattr(agent,'reset_edit_moments'): agent.reset_edit_moments(preview['tensors'])
        identity=uuid.uuid4().hex
        undo=getattr(agent,'_synapse_patch_undo',{})
        undo[identity]=before
        # Each retained undo record can contain a full model; keep a bounded history.
        while len(undo)>8: undo.pop(next(iter(undo)))
        agent._synapse_patch_undo=undo
        return {'schema':'synapse-patch-receipt-v1','transaction_id':identity,'parent_digest':packet['parent_digest'],
                'result_digest':digest(agent.export_state()),'patch_digest':digest(packet),'writes':preview['writes'],'delta_l2':preview['delta_l2'],'applied':True}
    except BaseException:
        agent.import_state(before); raise

def rollback(agent,receipt):
    identity=receipt.get('transaction_id') if isinstance(receipt,dict) else None
    undo=getattr(agent,'_synapse_patch_undo',{})
    if identity not in undo: raise ValueError('rollback transaction unavailable')
    state=undo[identity]; agent.import_state(copy.deepcopy(state)); undo.clear()
    return {'restored':True,'state_digest':digest(agent.export_state())}

def _metrics(value):
    result={'score':float(value),'retention':1.} if isinstance(value,(int,float)) else dict(value)
    if 'score' not in result: raise ValueError('evaluation requires score')
    result.setdefault('retention',1.)
    if any(type(result[k]) not in (float,int) or not math.isfinite(result[k]) for k in ('score','retention')):
        raise ValueError('evaluation must be finite')
    return result

def evaluate_patch(agent,packet,evaluator,min_gain=0.,max_regression=.02):
    preview_patch(agent,packet)
    before=copy.deepcopy(agent.export_state()); original_undo=copy.deepcopy(getattr(agent,'_synapse_patch_undo',{}))
    try:
        base=_metrics(evaluator(agent)); agent.import_state(copy.deepcopy(before))
        apply_patch(agent,packet)
        candidate=_metrics(evaluator(agent))
        accepted=candidate['score']>base['score']+min_gain and candidate['retention']>=base['retention']-max_regression
    finally:
        agent.import_state(before); agent._synapse_patch_undo=original_undo
    receipt=apply_patch(agent,packet) if accepted else None
    return {'accepted':accepted,'baseline':base,'candidate':candidate,'gain':candidate['score']-base['score'],
            'reason':'development_screen_passed' if accepted else 'gain_or_retention_screen_failed','receipt':receipt,
            'scope':'empirical development screen; no global preservation guarantee'}
