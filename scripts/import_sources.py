"""Snapshot committed upstream trees, never live mutable checkpoints or worktrees."""
from pathlib import Path
import hashlib, io, json, shutil, subprocess, zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'graftbrain': (Path(r'C:\Users\kai99\Desktop\New folder (10)'), '81d9fb8b289ada6120f5b939e67a6b0dfc504103'),
    'poseidon': (Path(r'C:\Users\kai99\Desktop\Poseidon'), 'b92654eec10ed54598bc48e21394875e055aba38'),
    'expanse': (ROOT / '.sources/expanse', '33e38a279c86bccdb1e5680f15f193d61a3f71cb'),
}

def main():
    manifest = {'schema': 'synapse-sources-v1', 'sources': {}}
    for name, (source, revision) in SOURCES.items():
        target = ROOT / 'synapse/_vendor' / name
        target.mkdir(parents=True, exist_ok=True)
        blob = subprocess.check_output(['git', '-C', str(source), 'archive', revision])
        import tarfile
        files = {}
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                rel = Path(member.name)
                # Exclude checkpoints, generated output and repository infrastructure.
                if rel.suffix in {'.pt', '.pth', '.pyc', '.gguf', '.safetensors'} or rel.parts[0] in {'outputs', 'runs', 'state', '.github'}:
                    continue
                dest = (target / rel).resolve()
                if not dest.is_relative_to(target.resolve()):
                    raise ValueError('invalid archive path')
                data = archive.extractfile(member).read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                files[rel.as_posix()] = hashlib.sha256(data).hexdigest()
        manifest['sources'][name] = {'revision': revision, 'files': files}
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    core = SOURCES['poseidon'][0] / 'runs/tidal_dagger/core.pt'
    expected = 'a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27'
    if hashlib.sha256(core.read_bytes()).hexdigest() != expected:
        raise ValueError('selected Tidal artifact changed')
    shutil.copy2(core, artifacts / 'tidal_core.pt')
    manifest['artifacts'] = {'tidal_core.pt': {'sha256': expected, 'source': 'Poseidon Tidal DAgger', 'parameters': 327230}}
    (ROOT / 'synapse/source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: len(v['files']) for k,v in manifest['sources'].items()}))

if __name__ == '__main__': main()
