"""Content-bound tensor/primitive checkpoints. No arbitrary pickle objects accepted."""
from pathlib import Path
import hashlib, json, math, os, tempfile
import numpy as np
import torch

SCHEMA='synapse-state-v1'

def _encode(value):
    if isinstance(value, torch.Tensor):
        tensor=value.detach().cpu().contiguous().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError('state must be finite')
        return {'kind':'tensor','value':tensor}
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject: raise ValueError('object arrays are not supported')
        if value.dtype.kind in 'fc' and not np.isfinite(value).all(): raise ValueError('state must be finite')
        return {'kind':'ndarray','dtype':value.dtype.str,'shape':list(value.shape),'bytes':value.tobytes()}
    if isinstance(value,np.generic): return _encode(value.item())
    if isinstance(value,dict):
        return {'kind':'dict','items':[(_encode(k),_encode(v)) for k,v in value.items()]}
    if isinstance(value,(list,tuple)):
        return {'kind':'tuple' if isinstance(value,tuple) else 'list','items':[_encode(v) for v in value]}
    if value is None or type(value) in (str,int,bool,bytes): return {'kind':'scalar','value':value}
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('state must be finite')
        return {'kind':'scalar','value':value}
    raise TypeError(f'unsupported checkpoint value: {type(value).__name__}')

def _decode(item):
    kind=item['kind']
    if kind in ('scalar','tensor'): return item['value']
    if kind=='ndarray':
        dtype=np.dtype(item['dtype'])
        if dtype.hasobject: raise ValueError('object arrays are not supported')
        return np.frombuffer(item['bytes'],dtype=dtype).reshape(item['shape']).copy()
    if kind=='dict': return {_decode(k):_decode(v) for k,v in item['items']}
    if kind in ('list','tuple'):
        values=[_decode(v) for v in item['items']]
        return tuple(values) if kind=='tuple' else values
    raise ValueError('unknown checkpoint encoding')

def digest(value):
    h=hashlib.sha256()
    def put(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            h.update(f'tensor:{t.dtype}:{list(t.shape)}:'.encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes()); return
        if isinstance(v,np.ndarray):
            h.update(f'array:{v.dtype.str}:{list(v.shape)}:'.encode()); h.update(v.tobytes()); return
        if isinstance(v,np.generic): put(v.item()); return
        if isinstance(v,dict):
            h.update(b'dict{')
            for k in sorted(v,key=lambda x:(type(x).__name__,repr(x))): put(k); put(v[k])
            h.update(b'}'); return
        if isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode()+b'[')
            for item in v: put(item)
            h.update(b']'); return
        if isinstance(v,bytes): h.update(b'bytes:'+str(len(v)).encode()+b':'+v); return
        h.update(type(v).__name__.encode()+b':'+json.dumps(v,allow_nan=False,ensure_ascii=False).encode()+b';')
    put(value)
    return h.hexdigest()

def _atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def save_state(path,payload):
    import io
    path=Path(path); encoded=_encode(payload)
    stream=io.BytesIO(); torch.save({'schema':SCHEMA,'payload':encoded,'digest':digest(payload)},stream)
    data=stream.getvalue(); sha=hashlib.sha256(data).hexdigest()
    _atomic(path,data)
    _atomic(Path(str(path)+'.json'),json.dumps({'schema':SCHEMA,'sha256':sha,'bytes':len(data),'state_digest':digest(payload)},indent=2).encode())
    return sha

def load_state(path):
    import io
    path=Path(path)
    if path.stat().st_size>1024**3: raise ValueError('checkpoint exceeds 1 GiB limit')
    manifest=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
    data=path.read_bytes()
    if manifest.get('schema')!=SCHEMA or hashlib.sha256(data).hexdigest()!=manifest.get('sha256'):
        raise ValueError('checkpoint hash mismatch')
    archive=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if archive.get('schema')!=SCHEMA: raise ValueError('checkpoint schema mismatch')
    value=_decode(archive['payload']); _encode(value)
    if digest(value)!=archive.get('digest') or digest(value)!=manifest.get('state_digest'):
        raise ValueError('checkpoint state hash mismatch')
    return value
