"""Content-bound tensor/primitive checkpoints. No arbitrary pickle objects accepted."""
from pathlib import Path
import hashlib, json, math, os, tempfile
import uuid
import zlib
import numpy as np
import torch

SCHEMA='synapse-state-v1'
_ARRAY_COMPRESSION_MIN_BYTES=1024*1024
_ARRAY_COMPRESSION_SAMPLE_BYTES=256*1024
_ARRAY_COMPRESSION_SAMPLE_RATIO=0.70
_ARRAY_COMPRESSION_CHUNK_BYTES=4*1024*1024

def _encode_ndarray(value):
    encoded={'kind':'ndarray','dtype':value.dtype.str,'shape':list(value.shape)}
    if value.nbytes >= _ARRAY_COMPRESSION_MIN_BYTES and value.flags.c_contiguous:
        view=memoryview(value).cast('B')
        sample_size=min(_ARRAY_COMPRESSION_SAMPLE_BYTES,value.nbytes)
        sample_offsets=sorted({0,max(0,(value.nbytes-sample_size)//2),value.nbytes-sample_size})
        sample_ratios=[len(zlib.compress(view[offset:offset+sample_size],level=1))/sample_size
                       for offset in sample_offsets]
        if max(sample_ratios,default=1.0) < _ARRAY_COMPRESSION_SAMPLE_RATIO:
            compressor=zlib.compressobj(level=1)
            compressed=bytearray()
            for offset in range(0,value.nbytes,_ARRAY_COMPRESSION_CHUNK_BYTES):
                compressed.extend(compressor.compress(
                    view[offset:min(value.nbytes,offset+_ARRAY_COMPRESSION_CHUNK_BYTES)]
                ))
            compressed.extend(compressor.flush())
            if len(compressed) < value.nbytes*0.90:
                encoded['bytes']=bytes(compressed)
                encoded['compression']='zlib'
                return encoded
    encoded['bytes']=value.tobytes()
    return encoded

def _encode(value, *, clone_tensors=True):
    if isinstance(value, torch.Tensor):
        tensor=value.detach().cpu().contiguous()
        if clone_tensors: tensor=tensor.clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError('state must be finite')
        return {'kind':'tensor','value':tensor}
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject: raise ValueError('object arrays are not supported')
        if value.dtype.kind in 'fc' and not np.isfinite(value).all(): raise ValueError('state must be finite')
        return _encode_ndarray(value)
    if isinstance(value,np.generic): return _encode(value.item())
    if isinstance(value,dict):
        return {'kind':'dict','items':[(_encode(k,clone_tensors=clone_tensors),_encode(v,clone_tensors=clone_tensors)) for k,v in value.items()]}
    if isinstance(value,(list,tuple)):
        return {'kind':'tuple' if isinstance(value,tuple) else 'list','items':[_encode(v,clone_tensors=clone_tensors) for v in value]}
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
        raw=item['bytes']
        compression=item.get('compression')
        if compression=='zlib': raw=zlib.decompress(raw)
        elif compression is not None: raise ValueError('unknown ndarray compression')
        expected=dtype.itemsize*math.prod(item['shape'])
        if len(raw)!=expected: raise ValueError('ndarray checkpoint byte length mismatch')
        return np.frombuffer(raw,dtype=dtype).reshape(item['shape']).copy()
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

def save_state_snapshot(path,payload):
    """Atomically persist a detached, immutable state snapshot.

    Runtime callers must capture all mutable arrays, dicts, and tensors before
    releasing their execution lock. Tensor storage is then reused rather than
    cloned a second time during serialization, reducing peak export memory.
    """
    path=Path(path)
    encoded=_encode(payload,clone_tensors=False)
    checkpoint_id=uuid.uuid4().hex
    state_digest=digest(payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save({'schema':SCHEMA,'checkpoint_id':checkpoint_id,'payload':encoded,'digest':state_digest},f)
            f.flush(); os.fsync(f.fileno())
        h=hashlib.sha256(); byte_count=0
        with open(tmp,'rb') as f:
            while True:
                chunk=f.read(1024*1024)
                if not chunk: break
                h.update(chunk); byte_count+=len(chunk)
        sha=h.hexdigest()
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    manifest={'schema':SCHEMA,'checkpoint_id':checkpoint_id,'sha256':sha,'bytes':byte_count,'state_digest':state_digest}
    try:
        _atomic(Path(str(path)+'.json'),json.dumps(manifest,indent=2).encode())
    except OSError:
        # As with save_state, the committed archive carries the authoritative
        # content digest and can recover its sidecar on the next load.
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
