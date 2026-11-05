import copy
import pytest
import torch
from synapse.patches import make_patch, apply_patch, rollback, evaluate_patch
from synapse.stateio import digest

class Actor:
    def __init__(self): self.w=torch.zeros(2,2); self.memory=3
    def editable_tensors(self): return {'w':self.w}
    def export_state(self): return {'w':self.w.clone(),'memory':self.memory}
    def import_state(self,state): self.w.copy_(state['w']); self.memory=state['memory']

def packet(a): return make_patch(a,[{'tensor':'w','indices':[[0,1]],'values':[0.5]}])

def test_patch_applies_and_exact_rollback():
    a=Actor(); before=digest(a.export_state()); r=apply_patch(a,packet(a))
    assert a.w[0,1]==0.5
    a.memory=99; rollback(a,r)
    assert digest(a.export_state())==before

def test_stale_and_partial_invalid_rejected():
    a=Actor(); p=packet(a); a.memory=4
    with pytest.raises(ValueError,match='parent'): apply_patch(a,p)
    p=packet(a); p['edits'].append({'tensor':'missing','indices':[[0]],'values':[1.], 'expected':[0.]})
    before=digest(a.export_state())
    with pytest.raises(ValueError): apply_patch(a,p)
    assert digest(a.export_state())==before

def test_no_nonfinite_duplicate_or_out_of_bounds_edits():
    a=Actor()
    for edit in [dict(tensor='w',indices=[[8,1]],values=[1]),dict(tensor='w',indices=[[0,1]],values=[float('nan')]),dict(tensor='w',indices=[[0,1],[0,1]],values=[1,2])]:
        with pytest.raises(ValueError): make_patch(a,[edit])

def test_rejected_evaluation_restores_all_state():
    a=Actor(); before=digest(a.export_state())
    def score(actor): actor.memory+=1; return {'score':-float(actor.w.sum()),'retention':1.0}
    result=evaluate_patch(a,packet(a),score)
    assert not result['accepted']
    assert digest(a.export_state())==before

def test_exception_during_evaluation_restores_state():
    a=Actor(); before=digest(a.export_state())
    def score(actor): actor.memory=999; raise RuntimeError('evaluation failed')
    with pytest.raises(RuntimeError): evaluate_patch(a,packet(a),score)
    assert digest(a.export_state())==before
