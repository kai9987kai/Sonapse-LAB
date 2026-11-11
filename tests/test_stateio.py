import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import synapse.stateio as stateio
from synapse.stateio import digest, save_state, load_state

def test_state_archive_preserves_types_and_rejects_tamper(tmp_path):
    value={'a':np.array([1,2],dtype=np.int64),'t':torch.tensor([3.25]),'rng':np.random.default_rng(2).bit_generator.state,'tuple':(1,'x')}
    path=tmp_path/'state.pt'; original=digest(value)
    save_state(path,value); restored=load_state(path)
    assert digest(restored)==original
    assert isinstance(restored['a'],np.ndarray)
    assert restored['tuple']==(1,'x')
    with path.open('ab') as handle: handle.write(b'tampered')
    with pytest.raises(ValueError,match='hash'): load_state(path)

def test_digest_changes_with_tensor_data_and_type():
    assert digest(torch.tensor([1.])) != digest(torch.tensor([2.]))
    assert digest(np.array([1],dtype=np.int32)) != digest(np.array([1],dtype=np.int64))

def test_nonfinite_state_rejected(tmp_path):
    with pytest.raises(ValueError,match='finite'): save_state(tmp_path/'bad.pt',{'t':torch.tensor([float('nan')])})


def test_legacy_archive_without_commit_id_remains_loadable(tmp_path):
    value={'legacy':np.asarray([1,2,3],dtype=np.int64)}
    stream=io.BytesIO()
    torch.save({'schema':stateio.SCHEMA,'payload':stateio._encode(value),'digest':digest(value)},stream)
    data=stream.getvalue()
    path=tmp_path/'legacy.pt'
    path.write_bytes(data)
    Path(str(path)+'.json').write_text(json.dumps({
        'schema':stateio.SCHEMA,
        'sha256':hashlib.sha256(data).hexdigest(),
        'bytes':len(data),
        'state_digest':digest(value),
    }),encoding='utf-8')
    assert digest(load_state(path))==digest(value)


def test_interrupted_manifest_commit_recovers_new_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'session.pt'
    save_state(path,{'generation':1})
    atomic=stateio._atomic
    manifest_path=Path(str(path)+'.json')

    def fail_manifest(target,data):
        if Path(target)==manifest_path:
            raise OSError('simulated manifest write failure')
        atomic(target,data)

    monkeypatch.setattr(stateio,'_atomic',fail_manifest)
    save_state(path,{'generation':2})
    monkeypatch.setattr(stateio,'_atomic',atomic)

    assert load_state(path)=={'generation':2}
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['checkpoint_id']
    assert manifest['state_digest']==digest({'generation':2})


def test_unreadable_manifest_recovers_checkpoint(tmp_path):
    path=tmp_path/'session.pt'
    save_state(path,{'generation':3})
    Path(str(path)+'.json').write_bytes(b'\xff')

    assert load_state(path)=={'generation':3}
    manifest=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
    assert manifest['state_digest']==digest({'generation':3})
