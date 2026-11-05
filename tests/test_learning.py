import numpy as np
import pytest
from synapse.learning import fit_readout, predict, fresh_weights
from synapse.rsi import Policy, propose

def test_gradient_bootstraps_zero_readout():
    x=np.eye(4); y=np.array([0,1,0,1]); w=fresh_weights(4)
    fitted=fit_readout(x,y,w,method='gradient',steps=30,rate=1.)
    assert np.array_equal(predict(x,fitted),y)
    assert not w.any()

def test_inherited_policy_generates_successor():
    p=Policy(); child=propose(p,0,17); grandchild=propose(child,1,17)
    assert child.parent==p.identity
    assert grandchild.parent==child.identity
    assert grandchild.generation==2

def test_fresh_weights_equal():
    assert np.array_equal(fresh_weights(12),fresh_weights(12))
