import pytest
from synapse.experiments import run_study, validate_config

def test_configuration_rejects_unbounded_and_unknown():
    for config in ({'generations':500},{'query_labels':[1]},{'seed':-1}):
        with pytest.raises(ValueError): validate_config(config)

def test_cancel_before_inference(tmp_path):
    result=run_study(tmp_path/'cancelled',{},cancelled=lambda:True)
    assert result['status']=='cancelled'

def test_output_exclusive(tmp_path):
    with pytest.raises(FileExistsError): run_study(tmp_path,{},cancelled=lambda:True)

def test_query_is_not_visible_to_adaptation(monkeypatch):
    import synapse.experiments as e
    import numpy as np
    seen=[]; original=e.adapt
    def spy(x,y,*args,**kwargs):
        seen.append((x.copy(),y.copy())); return original(x,y,*args,**kwargs)
    monkeypatch.setattr(e,'adapt',spy)
    x=np.eye(16); y=np.arange(16)%2
    t={'x':x,'y':y,'split':'sealed','family':4}
    e._evaluate(e.Policy(),[t],{'support':8})
    assert len(seen)==1 and np.array_equal(seen[0][0],x[:8])
    y[8:]=1-y[8:]
    e._evaluate(e.Policy(),[t],{'support':8})
    assert np.array_equal(seen[0][1],seen[1][1])


def test_lesion_masks_cue_and_delayed_pathways():
    from synapse.learning import lesion
    import numpy as np
    x=np.ones((2,129))
    masked=lesion(x,'graph_off')
    assert not masked[:,:32].any() and not masked[:,64:96].any()
    assert masked[:,32:64].all() and masked[:,96:].all()

def test_receipt_recompute_rejects_metric_tampering(tmp_path,monkeypatch):
    import synapse.experiments as e
    import numpy as np, json
    # Cheap deterministic fixture validates the evaluator; separate actual actor smoke is documented.
    rng=np.random.default_rng(7)
    tasks=[{'split':s,'family':i,'x':rng.normal(size=(16,129)),'y':np.arange(16)%2} for i,s in enumerate(['development','validation','sealed'])]
    monkeypatch.setattr(e,'_tasks',lambda c,cancelled:tasks)
    out=tmp_path/'study'; e.run_study(out,{'lineages':1,'generations':2,'candidates':1,'support':8,'query':8})
    assert e.verify_study(out)['ok']
    receipt=json.loads((out/'receipt.json').read_text()); receipt['evaluations'][0]['accuracy']=-1
    (out/'receipt.json').write_text(json.dumps(receipt))
    assert not e.verify_study(out)['ok']
