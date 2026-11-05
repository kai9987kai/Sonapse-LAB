"""Resident linear cue-memory head; all fitting uses labeled support only."""
import numpy as np


def fresh_weights(dimension):
    return np.zeros((int(dimension),2),dtype=np.float64)


def predict(features,weights):
    return np.argmax(np.asarray(features)@weights,axis=1)


def fit_readout(features,labels,initial,method='gradient',steps=24,rate=.5,ridge=.1):
    x=np.asarray(features,dtype=np.float64); y=np.eye(2)[np.asarray(labels,dtype=int)]
    w=np.array(initial,dtype=np.float64,copy=True)
    if x.ndim!=2 or len(x)!=len(y) or w.shape!=(x.shape[1],2): raise ValueError('readout shape mismatch')
    if not np.isfinite(x).all(): raise ValueError('nonfinite features')
    if method=='ridge':
        # Dual form limits CPU cost when the support set is small.
        return x.T@np.linalg.solve(x@x.T+float(ridge)*np.eye(len(x)),y)
    if method=='direct':
        # Typed direct numeric Hebbian assignment, not gradient training or LLM writing.
        return (x.T@(2*y-1))/max(1,len(x))
    if method!='gradient': raise ValueError('unknown operator')
    for _ in range(int(steps)):
        logits=x@w; logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits); p/=p.sum(axis=1,keepdims=True)
        w-=float(rate)*x.T@(p-y)/len(x)
    return w


def adapt(features,labels,policy,control='intact',initial=None):
    x=lesion(np.asarray(features,dtype=np.float64),control)
    # Probe and replay decisions belong to the frozen improver.
    order=np.arange(len(x))
    if policy.probe=='reverse': order=order[::-1]
    elif policy.probe=='balanced': order=np.argsort(labels,kind='stable')
    chosen=order[:max(2,int(len(x)*policy.support_fraction))]
    xx=x[chosen]; yy=np.asarray(labels)[chosen]
    xx=np.tile(xx,(policy.replay,1)); yy=np.tile(yy,policy.replay)
    weights=fit_readout(xx,yy,fresh_weights(x.shape[1]) if initial is None else initial,policy.operator,policy.steps,policy.rate,policy.ridge)
    return weights,{'support_examples':len(chosen),'replayed_examples':len(xx),'update_steps':policy.steps if policy.operator=='gradient' else 1,'changed_parameters':int(np.count_nonzero(weights))}


def lesion(features,control):
    x=np.array(features,copy=True)
    slices={'graph_off':(0,32),'donor_off':(32,48),'memory_off':(48,56)}
    if control in slices:
        lo,hi=slices[control]
        for offset in range(0,x.shape[1]-1,64): x[:,offset+lo:offset+hi]=0
    elif control!='intact': raise ValueError('unknown control')
    return x
