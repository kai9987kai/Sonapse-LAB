import numpy as np
import pytest
import torch
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
