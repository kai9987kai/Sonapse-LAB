"""Content-bound tensor/primitive checkpoints. No arbitrary pickle objects accepted."""
from pathlib import Path
import hashlib, json, math, os, tempfile
import uuid
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
    checkpoint_id=uuid.uuid4().hex
    state_digest=digest(payload)
    stream=io.BytesIO(); torch.save({'schema':SCHEMA,'checkpoint_id':checkpoint_id,'payload':encoded,'digest':state_digest},stream)
    data=stream.getvalue(); sha=hashlib.sha256(data).hexdigest()
    _atomic(path,data)
    manifest={'schema':SCHEMA,'checkpoint_id':checkpoint_id,'sha256':sha,'bytes':len(data),'state_digest':state_digest}
    try:
        _atomic(Path(str(path)+'.json'),json.dumps(manifest,indent=2).encode())
    except OSError:
        # The tensor archive carries its own content digest and a unique commit
        # id. A stale sidecar can be rebuilt by load_state after an interrupted
        # metadata write, without discarding the already committed checkpoint.
        pass
    return sha

def load_state(path):
    import io
    path=Path(path)
    if path.stat().st_size>1024**3: raise ValueError('checkpoint exceeds 1 GiB limit')
    data=path.read_bytes()
    manifest_path=Path(str(path)+'.json')
    try:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        manifest=None
    archive=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    if archive.get('schema')!=SCHEMA: raise ValueError('checkpoint schema mismatch')
    value=_decode(archive['payload']); _encode(value)
    state_digest=digest(value)
    if state_digest!=archive.get('digest'):
        raise ValueError('checkpoint state hash mismatch')
    sha=hashlib.sha256(data).hexdigest()
    checkpoint_id=archive.get('checkpoint_id')
    if checkpoint_id is None:
        # Legacy v1 archives used the sidecar as their only byte-level commit
        # marker; retain that strict validation for existing saved sessions.
        if (not isinstance(manifest,dict) or manifest.get('schema')!=SCHEMA
                or manifest.get('sha256')!=sha or manifest.get('state_digest')!=state_digest):
            raise ValueError('checkpoint hash mismatch')
    else:
        if (not isinstance(checkpoint_id,str) or len(checkpoint_id)!=32
                or any(char not in '0123456789abcdef' for char in checkpoint_id)):
            raise ValueError('checkpoint commit id is invalid')
        if isinstance(manifest,dict) and manifest.get('checkpoint_id')==checkpoint_id:
            if (manifest.get('schema')!=SCHEMA or manifest.get('sha256')!=sha
                    or manifest.get('state_digest')!=state_digest or manifest.get('bytes')!=len(data)):
                raise ValueError('checkpoint hash mismatch')
        else:
            # The data file was atomically replaced but the process may have
            # stopped before replacing its sidecar. The embedded digest proves
            # the payload; complete the metadata commit before returning it.
            recovered={'schema':SCHEMA,'checkpoint_id':checkpoint_id,'sha256':sha,
                       'bytes':len(data),'state_digest':state_digest}
            _atomic(manifest_path,json.dumps(recovered,indent=2).encode())
    return value
