"""Verified neuron-level connectome assets, bounded-memory import and sparse priors.

Anatomical synapse counts are measured data; signs, row normalization and rate
dynamics are explicit modelling assumptions. No missing asset becomes real data.
"""
from __future__ import annotations

import base64
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import uuid

import numpy as np

from .config import ARTIFACTS

ASSET_ROOT = ARTIFACTS / 'connectomes'
MALECNS_BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
MALECNS_FILES = {
    'annotations': {'filename': 'body-annotations-male-cns-v1.0-minconf-0.5.feather', 'md5_b64': 'UKdxh3DFciDxYLpPQxq4ng==', 'bytes': 14483314},
    'neurotransmitters': {'filename': 'body-neurotransmitters-male-cns-v1.0.feather', 'md5_b64': 'PYQrEv5cSe763lKNfdJKHw==', 'bytes': 43282834},
    'weights': {'filename': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather', 'md5_b64': '8w6dzKJc/QIb8eez2XVZng==', 'bytes': 1051241946},
}
FLYWIRE_BASE = 'https://zenodo.org/records/10676866/files/'
FLYWIRE_FILES = {
    'weights': {'filename': 'proofread_connections_783.feather', 'md5': 'f48f972d262323a102aed49af1396b8a'},
    'neurons': {'filename': 'proofread_root_ids_783.npy', 'md5': 'e0e6c19732fd8c7a4e39a2d170105421'},
    'annotations': {'filename': 'annotations_v2.1.0.tsv', 'sha256': '30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6'},
}
FLYWIRE_ANNOTATIONS_URL = 'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
ADDITIONAL_SOURCES = {
    'manc': {'source_page': 'https://www.janelia.org/project-team/flyem/manc-connectome', 'license': 'CC-BY-4.0', 'files': {
        'weights': {'filename': 'traced-connections.csv', 'url': 'https://storage.googleapis.com/flyem-manc-exports/v1.0/manc-traced-adjacencies-v1.0/traced-connections.csv', 'md5_b64': 'xUPcmgw2fZ98iJAdiS14oA=='},
        'neurons': {'filename': 'traced-neurons.csv', 'url': 'https://storage.googleapis.com/flyem-manc-exports/v1.0/manc-traced-adjacencies-v1.0/traced-neurons.csv', 'md5_b64': 'aaeazrfgVIhfTkc2TyOcag=='},
        'annotations': {'filename': 'manc-v1.0-neuron-properties.feather', 'url': 'https://storage.googleapis.com/flyem-manc-exports/v1.0/manc-v1.0-neuron-properties.feather', 'md5_b64': '9GcxF1WcxyfhvvywIrahQw=='},
    }},
    'hemibrain': {'source_page': 'https://www.janelia.org/project-team/flyem/hemibrain', 'license': 'CC-BY-4.0', 'files': {
        'weights': {'filename': 'exported-traced-adjacencies-v1.2.tar.gz', 'url': 'https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz', 'md5_b64': '7B4HbRVELzaPpZeLAPSmHA=='},
        'neurotransmitters': {'filename': 'hemibrain-v1.2-body-mean-neurotransmitters.feather', 'url': 'https://storage.googleapis.com/hemibrain/v1.2/hemibrain-v1.2-body-mean-neurotransmitters.feather', 'md5_b64': '711dUR0Z7ZvKXY3FyMymkA=='},
    }},
    'banc': {'source_page': 'https://doi.org/10.7910/DVN/7WTH1N', 'license': 'CC-BY-4.0', 'files': {
        'weights': {'filename': 'banc_888_edgelist_simple_v2.feather', 'url': 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_edgelist_simple_v2.feather', 'md5_b64': 'OUQG+Km98JPIlfla/09sSQ=='},
        'annotations': {'filename': 'banc_888_meta.feather', 'url': 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_meta.feather', 'md5_b64': 'jCuTpgjHFj7J2U1o4HVv9A=='},
        'neurotransmitters': {'filename': 'banc_888_neurotransmitter_prediction_v2.csv', 'url': 'https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/banc_888_neurotransmitter_prediction_v2.csv', 'md5_b64': 'TrvR1uBdQZKtDG2ydzmo4w=='},
    }},
    'optic-lobe': {'source_page': 'https://www.janelia.org/project-team/flyem/optic-lobe', 'license': 'CC-BY-4.0', 'files': {
        'weights': {'filename': 'connectome-weights-2024-09-11-a7d912-minconf-0.5-ol.feather', 'url': 'https://storage.googleapis.com/flyem-optic-lobe/v1.1/optic-lobe-v1.1-flat-connectome/connectome-weights-2024-09-11-a7d912-minconf-0.5-ol.feather', 'md5_b64': 'Ymmj4g+q2hsjeu/c4jTSTQ=='},
        'annotations': {'filename': 'Neuprint_Neurons.feather', 'url': 'https://storage.googleapis.com/flyem-optic-lobe/v1.1/optic-lobe-v1.1-neuprint-tables/Neuprint_Neurons.feather', 'md5_b64': 'lOsR+lWKtPveS4aBS4Sv2A=='},
    }},
    'larval': {'source_page': 'https://doi.org/10.1126/science.add9330', 'license': 'see Science supplement terms', 'partial': True, 'files': {
        'weights': {'filename': 'Supplementary-Data-S1.zip', 'url': 'https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/15e065f5c29f08c96ccd64ea8fe0f51510629009/Supplementary-Data-S1.zip', 'sha256': '8c1f43809ed5d527ba61b154e377cc21da26383a75eda8aab85ce05607a72a4c'},
    }},
}
SIGN_CONVENTION = ('ACh +1; GABA -1; glutamate -1 assumes inhibitory GluCl; '
                   'histamine -1; monoamines and unknown transmitters 0 (no fast signed efficacy inferred). '
                   'Counts are converted with log1p and incoming absolute row sums bounded to 0.9; '
                   'these are model parameters, not measured physiological efficacies.')
_SIGNS = {'acetylcholine': 1, 'ach': 1, 'gaba': -1, 'glutamate': -1, 'glut': -1, 'histamine': -1}
_validated_caches = set()
IMPORT_REVISIONS = {'banc': 2, 'hemibrain': 2}


def _hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 << 20), b''):
            md5.update(data)
            sha.update(data)
    return {'md5': md5.hexdigest(), 'md5_b64': base64.b64encode(md5.digest()).decode(), 'sha256': sha.hexdigest()}


def _source_dir(dataset='malecns'):
    override = os.environ.get('SYNAPSE_' + dataset.upper() + '_DIR')
    if override:
        return Path(override)
    local = ASSET_ROOT / dataset / 'raw'
    if local.exists() or dataset != 'malecns':
        return local
    # Read-only discovery of the original local source; never writes to it.
    original = Path.home() / 'Desktop' / 'New folder (9)' / 'external' / 'malecns_data'
    return original if original.exists() else local


def _manifest(cache):
    try:
        result = json.loads((Path(cache) / 'manifest.json').read_text(encoding='utf-8'))
        return result if result.get('complete') else None
    except (OSError, ValueError):
        return None


def get_source_inventory():
    """Report actual asset availability separately from catalog specimen scope."""
    routes = {
        'malecns': ('CC-BY-4.0', 'https://male-cns.janelia.org/download/', [MALECNS_BASE + x['filename'] for x in MALECNS_FILES.values()], 166700),
        'flywire': ('CC-BY-NC-4.0', 'https://zenodo.org/records/10676866', [FLYWIRE_BASE + x['filename'] + '?download=1' for k, x in FLYWIRE_FILES.items() if k != 'annotations'] + [FLYWIRE_ANNOTATIONS_URL], 139255),
        'manc': ('CC-BY-4.0', 'https://www.janelia.org/project-team/flyem/manc-connectome', [], None),
        'optic-lobe': ('CC-BY-4.0', 'https://www.janelia.org/project-team/flyem/optic-lobe', [], None),
        'banc': ('CC-BY-4.0', 'https://codex.flywire.ai/faq', [], None),
        'fanc': ('unverified', 'https://fanc.community/', [], None),
        'larval': ('see source terms', 'https://doi.org/10.1126/science.add9330', [], 3016),
        'hemibrain': ('CC-BY-4.0', 'https://www.janelia.org/project-team/flyem/hemibrain', [], None),
    }
    results = []
    for name, (license_name, source, urls, count) in routes.items():
        cache = ASSET_ROOT / name
        manifest = _manifest(cache)
        raw = _source_dir(name)
        extra = ADDITIONAL_SOURCES.get(name, {})
        if extra:
            source, license_name = extra['source_page'], extra['license']
            urls = [item['url'] for item in extra['files'].values()]
        files = MALECNS_FILES if name == 'malecns' else FLYWIRE_FILES if name == 'flywire' else extra.get('files', {})
        raw_available = bool(files) and all((raw / f['filename']).is_file() for f in files.values())
        results.append({'id': name, 'availability': 'prepared' if manifest else 'raw_available' if raw_available else 'unavailable',
                        'representation': 'neuron', 'neuron_count': manifest.get('neuron_count', count) if manifest else count,
                        'edge_count': manifest.get('edge_count') if manifest else None,
                        'synapse_count': manifest.get('synapse_count') if manifest else None,
                        'raw_path': str(raw) if raw_available else None, 'cache_path': str(cache) if manifest else None,
                        'verified': bool(manifest and manifest.get('verification') in ('official_checksums', 'pinned_repository_sha256')),
                        'verification': manifest.get('verification', 'not_yet_verified') if manifest else 'not_yet_verified',
                        'reference_neuron_count': count,
                        'source_page': source, 'license': license_name, 'download_urls': urls,
                        'full_neuron_graph_present': bool(manifest or raw_available) and not extra.get('partial', False),
                        'source_scope': 'published 2952-neuron analysis matrix; not the full 3016-neuron reconstruction' if name == 'larval' else 'full released neuron adjacency within dataset anatomical volume',
                        'selection': manifest.get('selection') if manifest else None,
                        'note': 'Independent dataset required; maleCNS homolog annotations are not this dataset.' if name == 'fanc' else 'Runtime samples and published source filters are labelled separately.'})
    return results


def _verify_files(raw, files, verify):
    evidence = {}
    for key, info in files.items():
        path = Path(raw) / info['filename']
        if not path.is_file():
            raise FileNotFoundError(f'Connectome source missing: {path}; see docs/connectome-data.md')
        digests = _hashes(path)
        if verify and any(digests[k] != info[k] for k in ('md5', 'md5_b64', 'sha256') if k in info):
            raise ValueError(f'Connectome checksum mismatch: {path}')
        evidence[key] = {'path': str(path.resolve()), 'bytes': path.stat().st_size, **digests}
    return evidence


def _role(superclass):
    value = str(superclass).lower()
    if 'motor' in value or 'efferent' in value: return 'motor'
    if 'descending' in value: return 'descending'
    if 'ascending' in value: return 'ascending'
    if 'sensory' in value: return 'sensory'
    if 'visual' in value or 'ol_' in value or value == 'optic': return 'visual'
    return 'central'


def _other_metadata(raw, dataset):
    """Read only neuron metadata columns, never materialize all segment rows."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import tarfile
    import zipfile
    files = ADDITIONAL_SOURCES[dataset]['files']
    if dataset == 'manc':
        ann = pd.read_csv(raw / files['neurons']['filename']).sort_values('bodyId')
        props = feather.read_table(raw / files['annotations']['filename'], columns=['bodyId', 'class', 'predictedNt']).to_pandas()
        ann = ann.merge(props, on='bodyId', how='left', validate='one_to_one')
        ids, types = ann['bodyId'], ann['type']
        classes, nt = ann['class'], ann['predictedNt']
    elif dataset == 'hemibrain':
        with tarfile.open(raw / files['weights']['filename']) as archive:
            name = next(item for item in archive.getnames() if item.endswith('/traced-neurons.csv'))
            ann = pd.read_csv(archive.extractfile(name)).sort_values('bodyId')
        nts = feather.read_table(raw / files['neurotransmitters']['filename'], columns=['body', 'predicted_nt']).to_pandas().reset_index()
        ann = ann.merge(nts[['body', 'predicted_nt']], left_on='bodyId', right_on='body', how='left', validate='one_to_one')
        ids, types = ann['bodyId'], ann['type']
        classes, nt = pd.Series('central', index=ann.index), ann['predicted_nt']
    elif dataset == 'banc':
        ann = feather.read_table(raw / files['annotations']['filename'], columns=['banc_888_id', 'cell_type', 'super_class', 'proofread', 'roughly_proofread', 'neurotransmitter_predicted', 'neurotransmitter_verified']).to_pandas()
        reviewed = (ann['proofread'].str.upper() == 'TRUE') | (ann['roughly_proofread'].str.upper() == 'TRUE')
        non_neurons = ann['super_class'].isin(['glia', 'not_a_neuron', 'trachea'])
        ann = ann.loc[reviewed & ~non_neurons].copy()
        ann['id'] = ann['banc_888_id'].astype(np.int64)
        ann = ann.sort_values('id').drop_duplicates('id')
        ids, types = ann['id'], ann['cell_type']
        classes = ann['super_class']
        nt = ann['neurotransmitter_verified'].fillna(ann['neurotransmitter_predicted'])
    elif dataset == 'optic-lobe':
        source = pa.memory_map(str(raw / files['annotations']['filename']), 'r')
        reader = ipc.open_file(source)
        parts = []
        for i in range(reader.num_record_batches):
            # The source contains >10M fragments. Only traced neurons enter the graph.
            frame = reader.get_batch(i).select(['bodyId:long', 'type:string', 'status:string', 'consensusNt:string']).to_pandas()
            parts.append(frame.loc[frame['status:string'] == 'Traced'])
        source.close()
        ann = pd.concat(parts, ignore_index=True).sort_values('bodyId:long')
        ids, types = ann['bodyId:long'], ann['type:string']
        classes, nt = pd.Series('visual', index=ann.index), ann['consensusNt:string']
    elif dataset == 'larval':
        with zipfile.ZipFile(raw / files['weights']['filename']) as archive:
            with archive.open('Supplementary-Data-S1/all-all_connectivity_matrix.csv') as handle:
                frame = pd.read_csv(handle, index_col=0)
        ids = pd.Series(np.sort(frame.index.to_numpy(np.int64)))
        types = pd.Series('', index=ids.index)
        classes = pd.Series('central', index=ids.index)
        nt = pd.Series('unclear', index=ids.index)
    else:
        raise ValueError(f'Unsupported connectome dataset: {dataset}')
    return (ids.to_numpy(np.int64), types.fillna('').astype(str).to_numpy(dtype=str),
            classes.fillna('unknown').astype(str).to_numpy(dtype=str), nt.fillna('unclear').str.lower().to_numpy(dtype=str))


def _edge_batches(raw, dataset, files):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import tarfile
    import zipfile
    if dataset in ('manc', 'hemibrain'):
        if dataset == 'hemibrain':
            archive = tarfile.open(raw / files['weights']['filename'])
            name = next(name for name in archive.getnames() if name.endswith('/traced-total-connections.csv'))
            handle = archive.extractfile(name)
        else:
            archive, handle = None, raw / files['weights']['filename']
        try:
            for frame in pd.read_csv(handle, chunksize=65536):
                yield (frame['bodyId_pre'].to_numpy(np.int64), frame['bodyId_post'].to_numpy(np.int64), frame['weight'].to_numpy(np.int64))
        finally:
            if archive: archive.close()
    elif dataset == 'larval':
        with zipfile.ZipFile(raw / files['weights']['filename']) as archive:
            with archive.open('Supplementary-Data-S1/all-all_connectivity_matrix.csv') as handle:
                frame = pd.read_csv(handle, index_col=0)
        matrix = frame.to_numpy()
        p, q = np.nonzero(matrix)
        yield frame.index.to_numpy(np.int64)[p], frame.columns.to_numpy(np.int64)[q], matrix[p, q].astype(np.int64)
    else:
        fields = ('pre_pt_root_id', 'post_pt_root_id', 'syn_count') if dataset == 'flywire' else ('pre', 'post', 'count') if dataset == 'banc' else ('body_pre', 'body_post', 'weight')
        source = pa.memory_map(str(raw / files['weights']['filename']), 'r')
        try:
            reader = ipc.open_file(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                yield tuple(batch.column(field).to_numpy(zero_copy_only=False).astype(np.int64) for field in fields)
        finally:
            source.close()


def _prepare(raw, output, dataset, verify=True):
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    output, raw = Path(output), Path(raw)
    existing = _manifest(output)
    if existing and existing.get('import_revision', 1) >= IMPORT_REVISIONS.get(dataset, 1) and (existing['verification'] in ('official_checksums', 'pinned_repository_sha256') or not verify):
        return existing
    files = MALECNS_FILES if dataset == 'malecns' else FLYWIRE_FILES if dataset == 'flywire' else ADDITIONAL_SOURCES[dataset]['files']
    evidence = _verify_files(raw, files, verify)
    if dataset == 'malecns':
        ann = feather.read_table(raw / files['annotations']['filename']).to_pandas()
        ann = ann.loc[ann['superclass'].notna()].sort_values('bodyId').copy()
        nts = feather.read_table(raw / files['neurotransmitters']['filename'], columns=['body', 'consensus_nt', 'celltype_predicted_nt']).to_pandas()
        ann = ann.merge(nts, left_on='bodyId', right_on='body', how='left', validate='one_to_one')
        ids = ann['bodyId'].to_numpy(np.int64)
        celltypes = ann['type'].fillna('').astype(str).to_numpy(dtype=str)
        superclasses = ann['superclass'].fillna('unknown').astype(str).to_numpy(dtype=str)
        nt = ann['consensus_nt'].fillna(ann['celltype_predicted_nt']).fillna('unclear').str.lower().to_numpy(dtype=str)
        pre_field, post_field, weight_field = 'body_pre', 'body_post', 'weight'
    elif dataset == 'flywire':
        import pandas as pd
        ids = np.sort(np.load(raw / files['neurons']['filename'], allow_pickle=False).astype(np.int64))
        ann = pd.read_csv(raw / files['annotations']['filename'], sep='\t', low_memory=False).drop_duplicates('root_id').set_index('root_id').reindex(ids)
        celltypes = ann['cell_type'].fillna('').astype(str).to_numpy(dtype=str)
        superclasses = ann['super_class'].fillna('unknown').astype(str).to_numpy(dtype=str)
        nt = ann['known_nt'].fillna(ann['top_nt']).fillna('unclear').str.lower().to_numpy(dtype=str)
        pre_field, post_field, weight_field = 'pre_pt_root_id', 'post_pt_root_id', 'syn_count'
    else:
        ids, celltypes, superclasses, nt = _other_metadata(raw, dataset)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Neuron IDs must be unique and nonempty')
    staging = output.with_name(output.name + '.staging-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    n, count, source_count, synapses, source_synapses = len(ids), 0, 0, 0, 0
    degrees = np.zeros(n, dtype=np.int64)
    try:
        with ExitStack() as stack:
            handles = [stack.enter_context((staging / filename).open('wb')) for filename in ('pre.i32', 'post.i32', 'synapses.i64')]
            for pre, post, weight in _edge_batches(raw, dataset, files):
                x = np.minimum(np.searchsorted(ids, pre), n - 1)
                y = np.minimum(np.searchsorted(ids, post), n - 1)
                valid = (ids[x] == pre) & (ids[y] == post)
                p, q, w = x[valid].astype(np.int32), y[valid].astype(np.int32), weight[valid].astype(np.int64)
                for handle, array in zip(handles, (p, q, w)): array.tofile(handle)
                source_count += len(weight); count += len(w)
                source_synapses += int(weight.sum()); synapses += int(w.sum())
                degrees += np.bincount(p, minlength=n) + np.bincount(q, minlength=n)
        np.savez_compressed(staging / 'neurons.npz', body_ids=ids, cell_types=celltypes,
                            superclasses=superclasses, neurotransmitters=nt,
                            signs=np.array([_SIGNS.get(value, 0) for value in nt], dtype=np.float32),
                            roles=np.array([_role(value) for value in superclasses]), degrees=degrees)
        payloads = {p.name: {'bytes': p.stat().st_size, 'sha256': _hashes(p)['sha256']} for p in staging.iterdir()}
        result = {'schema': 1, 'import_revision': IMPORT_REVISIONS.get(dataset, 1), 'id': dataset, 'complete': True, 'representation': 'neuron',
                  'neuron_count': n, 'untyped_neuron_count': int((celltypes == '').sum()), 'edge_count': count,
                  'source_edge_count': source_count, 'synapse_count': synapses, 'source_synapse_count': source_synapses,
                  'connected_neuron_count': int((degrees > 0).sum()), 'verification': ('pinned_repository_sha256' if dataset == 'larval' else 'official_checksums') if verify else 'unverified_fixture',
                  'source_files': evidence, 'cache_files': payloads,
                  'source_page': 'https://male-cns.janelia.org/download/' if dataset == 'malecns' else 'https://zenodo.org/records/10676866' if dataset == 'flywire' else ADDITIONAL_SOURCES[dataset]['source_page'],
                  'license': 'CC-BY-4.0' if dataset == 'malecns' else 'CC-BY-NC-4.0' if dataset == 'flywire' else ADDITIONAL_SOURCES[dataset]['license'],
                  'full_released_neuron_graph': dataset != 'larval',
                  'selection': {'malecns': 'all annotated neurons with nonnull superclass, including untyped neurons',
                                'flywire': 'all proofread v783 neuron IDs; all released pairs >=1 synapse, retained per neuropil',
                                'manc': 'all v1.0 traced neurons and traced-neuron pair counts',
                                'hemibrain': 'all non-cropped Traced neurons in official v1.2 compact adjacency export (21739); anatomical hemibrain volume only; all-traced segment archive retained separately',
                                'optic-lobe': 'all v1.1 Traced-status neurons, all induced pairs; right optic lobe volume only',
                                'banc': 'all v888 metadata proofread or roughly_proofread IDs, excluding explicit glia/trachea/not_a_neuron; v2 synapse size >=5 voxels; no pair-count threshold; original self-loops retained',
                                'larval': 'published 2952-neuron analysis matrix from Supplementary Data S1; not full 3016-neuron reconstruction'}[dataset],
                  'sign_convention': SIGN_CONVENTION}
        (staging / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            # A raw/ subfolder may live here. Move only our four generated payloads.
            for path in staging.iterdir(): path.replace(output / path.name)
            staging.rmdir()
        else:
            staging.replace(output)
        return result
    except Exception:
        # Retain incomplete staging for diagnosis; it is never accepted as an asset.
        raise


def prepare_malecns(raw_dir=None, cache_dir=None, verify=True):
    return _prepare(raw_dir or _source_dir(), cache_dir or ASSET_ROOT / 'malecns', 'malecns', verify)


def prepare_flywire(raw_dir=None, cache_dir=None, verify=True):
    return _prepare(raw_dir or _source_dir('flywire'), cache_dir or ASSET_ROOT / 'flywire', 'flywire', verify)


def prepare_source(dataset, raw_dir=None, cache_dir=None, verify=True):
    if dataset not in ('malecns', 'flywire') and dataset not in ADDITIONAL_SOURCES:
        raise FileNotFoundError(f'No independent source importer for {dataset}')
    return _prepare(raw_dir or _source_dir(dataset), cache_dir or ASSET_ROOT / dataset, dataset, verify)


def _load_cache(dataset, cache_dir=None, allow_unverified=False):
    cache = Path(cache_dir or ASSET_ROOT / dataset)
    manifest = _manifest(cache)
    if not manifest or manifest.get('import_revision', 1) < IMPORT_REVISIONS.get(dataset, 1):
        manifest = prepare_source(dataset, cache_dir=cache)
    if manifest['verification'] not in ('official_checksums', 'pinned_repository_sha256') and not allow_unverified:
        raise ValueError('Unverified fixture cannot be used as an authentic connectome')
    signatures = tuple((p, (cache / p).stat().st_size, (cache / p).stat().st_mtime_ns) for p in manifest['cache_files'])
    key = (str(cache.resolve()), signatures)
    if key not in _validated_caches:
        for filename, expected in manifest['cache_files'].items():
            if (cache / filename).stat().st_size != expected['bytes'] or _hashes(cache / filename)['sha256'] != expected['sha256']:
                raise ValueError(f'Connectome cache checksum mismatch: {filename}')
        _validated_caches.add(key)
    metadata = dict(np.load(cache / 'neurons.npz', allow_pickle=False))
    edges = [np.memmap(cache / name, mode='r', dtype=dtype, shape=(manifest['edge_count'],)) if manifest['edge_count'] else np.empty(0, dtype=dtype)
             for name, dtype in (('pre.i32', np.int32), ('post.i32', np.int32), ('synapses.i64', np.int64))]
    return manifest, metadata, edges


def load_programmed_graft(max_nodes=128, seed=7, *, dataset='malecns', cache_dir=None, allow_unverified=False, query=None):
    """Select a real induced neuron subgraph and program a bounded signed prior.

    Sampling deliberately favours connected, high-degree neurons for compact
    runtime utility. It is not an unbiased biological subsample.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')
    if dataset not in ('malecns', 'flywire') and dataset not in ADDITIONAL_SOURCES:
        raise FileNotFoundError(f'No independent neuron-level import available for {dataset}')
    manifest, meta, (pre, post, counts) = _load_cache(dataset, cache_dir, allow_unverified)
    candidates = np.arange(manifest['neuron_count'])
    if query:
        q = query.lower()
        candidates = np.array([i for i in candidates if any(q in str(meta[key][i]).lower() for key in ('cell_types', 'superclasses', 'neurotransmitters'))], dtype=np.int64)
        if not len(candidates): raise ValueError(f'No neuron matches {query!r}')
    n = min(max_nodes, len(candidates))
    rng = np.random.default_rng(seed)
    pool = candidates[np.argsort(-meta['degrees'][candidates], kind='stable')[:min(len(candidates), n * 4)]]
    chosen = []
    for role in np.unique(meta['roles'][pool]):
        available = pool[meta['roles'][pool] == role]
        chosen.append(int(rng.choice(available)))
    chosen = chosen[:n]
    rest = np.setdiff1d(pool, chosen)
    if len(chosen) < n: chosen.extend(rng.choice(rest, n-len(chosen), replace=False).tolist())
    selected = np.sort(np.array(chosen, dtype=np.int64))
    remap = np.full(manifest['neuron_count'], -1, dtype=np.int32)
    remap[selected] = np.arange(n)
    out_pre, out_post, out_counts = [], [], []
    for start in range(0, len(pre), 1 << 20):
        p, q = remap[pre[start:start+(1<<20)]], remap[post[start:start+(1<<20)]]
        valid = (p >= 0) & (q >= 0)
        out_pre.append(p[valid]); out_post.append(q[valid]); out_counts.append(counts[start:start+(1<<20)][valid])
    p, q, count = (np.concatenate(parts) if parts else np.empty(0) for parts in (out_pre, out_post, out_counts))
    p, q, count = p.astype(np.int64), q.astype(np.int64), count.astype(np.int64)
    weights = meta['signs'][selected[p]] * np.log1p(count)
    sums = np.bincount(q, weights=np.abs(weights), minlength=n)
    weights = (weights * (0.9 / np.maximum(sums[q], 0.9))).astype(np.float32)
    provenance = {**{k: manifest[k] for k in ('id', 'representation', 'source_page', 'verification', 'license', 'sign_convention')},
                  'tier': 'source_backed_analysis_subset' if dataset == 'larval' else 'authentic', 'source_dataset': dataset, 'source_neurons': manifest['neuron_count'],
                  'source_edges': manifest['edge_count'], 'sampled_neurons': n, 'edges': len(p),
                  'total_synapses': int(count.sum()), 'inhibitory_fraction': float((weights < 0).mean()) if len(weights) else 0.0,
                  'sample_is_full_connectome': n == manifest['neuron_count'] and dataset != 'larval', 'measured_synaptic_efficacy': False,
                  'source_selection': manifest.get('selection'),
                  'unknown_or_modulatory_signed_zero': int((meta['signs'][selected] == 0).sum()),
                  'sampling': 'seeded role-stratified selection from highest-degree 4N candidate pool; induced edges', 'seed': seed,
                  'source_sha256': {k: v['sha256'] for k, v in manifest['source_files'].items()}}
    return {'pre': p, 'post': q, 'weights': weights, 'synapse_counts': count, 'n': n,
            'body_ids': meta['body_ids'][selected], 'cell_types': meta['cell_types'][selected],
            'roles': meta['roles'][selected], 'superclasses': meta['superclasses'][selected],
            'neurotransmitters': meta['neurotransmitters'][selected], 'signs': meta['signs'][selected], 'provenance': provenance}


def download_source(dataset, raw_dir=None):
    """Acquire only documented public source assets, checking official digests."""
    if dataset not in ('malecns', 'flywire') and dataset not in ADDITIONAL_SOURCES:
        raise ValueError('No public source download registered for this dataset')
    target = Path(raw_dir or ASSET_ROOT / dataset / 'raw')
    target.mkdir(parents=True, exist_ok=True)
    files = MALECNS_FILES if dataset == 'malecns' else FLYWIRE_FILES if dataset == 'flywire' else ADDITIONAL_SOURCES[dataset]['files']
    for key, info in files.items():
        path = target / info['filename']
        if path.exists(): continue
        required = info.get('bytes', 900_000_000 if key == 'weights' else 30_000_000)
        if shutil.disk_usage(target).free < required + 2_000_000_000:
            raise OSError('Download would leave less than 2 GB free')
        url = info.get('url') or (MALECNS_BASE + info['filename'] if dataset == 'malecns' else FLYWIRE_ANNOTATIONS_URL if key == 'annotations' else FLYWIRE_BASE + info['filename'] + '?download=1')
        temporary = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as handle:
            shutil.copyfileobj(source, handle, 8 << 20)
        digests = _hashes(temporary)
        if any(digests[k] != info[k] for k in ('md5', 'md5_b64', 'sha256') if k in info):
            raise ValueError(f'Connectome checksum mismatch: {temporary}')
        temporary.replace(path)
    return _verify_files(target, files, True)


def build_runtime_connectomes(config, *, strict=False):
    """Return source-backed Connectomes, or explicitly labelled optional proxies."""
    from .connectome_datasets import build_runtime_connectomes as build_profiles
    return build_profiles(config, strict=strict)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'download', 'prepare'))
    parser.add_argument('--dataset', choices=('malecns', 'flywire', *ADDITIONAL_SOURCES), default='malecns')
    arguments = parser.parse_args()
    result = get_source_inventory() if arguments.action == 'inventory' else download_source(arguments.dataset) if arguments.action == 'download' else prepare_source(arguments.dataset)
    print(json.dumps(result, indent=2))
