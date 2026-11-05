"""Pinned source activation. Importing this module performs no upstream inference."""
import hashlib, json, sys
from .config import PACKAGE, VENDOR

def activate():
    for source in ('graftbrain','poseidon'):
        path = str(VENDOR / source)
        if path not in sys.path: sys.path.insert(0,path)

def manifest():
    return json.loads((PACKAGE / 'source_manifest.json').read_text(encoding='utf-8'))

def verify_sources():
    bad=[]; count=0
    for name, entry in manifest()['sources'].items():
        for relative, expected in entry['files'].items():
            path=VENDOR/name/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                bad.append(f'{name}/{relative}')
            count+=1
    return {'ok':not bad,'checked':count,'mismatches':bad}

activate()
