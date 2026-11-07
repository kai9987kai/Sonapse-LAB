"""Neuron-level imports retain untyped classified neurons and source evidence."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest


def _raw(tmp_path):
    from synapse.connectome_assets import MALECNS_FILES
    raw = tmp_path / 'raw'
    raw.mkdir()
    feather.write_feather(pa.table({
        'bodyId': [10, 20, 30, 99],
        'type': ['A', None, 'C', None],
        'superclass': ['cb_sensory', 'cb_intrinsic', 'cb_motor', None],
        'class': ['olfactory', None, None, None],
        'status': ['Traced', 'Traced', 'Traced', 'Orphan'],
    }), raw / MALECNS_FILES['annotations']['filename'])
    feather.write_feather(pa.table({
        'body': [10, 20, 30],
        'consensus_nt': ['acetylcholine', 'gaba', 'unclear'],
        'celltype_predicted_nt': ['acetylcholine', 'gaba', 'unclear'],
    }), raw / MALECNS_FILES['neurotransmitters']['filename'])
    feather.write_feather(pa.table({
        'body_pre': [10, 20, 30, 99], 'body_post': [20, 30, 10, 10],
        'weight': [8, 3, 2, 11],
    }), raw / MALECNS_FILES['weights']['filename'])
    return raw


def test_neuron_import_preserves_untyped_and_excludes_unclassified_segments(tmp_path):
    from synapse.connectome_assets import prepare_malecns
    raw = _raw(tmp_path)
    manifest = prepare_malecns(raw, tmp_path / 'imported', verify=False)
    assert manifest['neuron_count'] == 3
    assert manifest['untyped_neuron_count'] == 1
    assert manifest['edge_count'] == 3
    assert manifest['source_edge_count'] == 4
    assert manifest['synapse_count'] == 13
    assert manifest['source_synapse_count'] == 24
    assert manifest['representation'] == 'neuron'
    assert manifest['verification'] == 'unverified_fixture'
    assert json.loads((tmp_path / 'imported' / 'manifest.json').read_text())['complete']


def test_programmed_sparse_graft_uses_actual_edges_and_explicit_sign_assumptions(tmp_path):
    from synapse.connectome_assets import prepare_malecns, load_programmed_graft
    raw = _raw(tmp_path)
    target = tmp_path / 'imported'
    prepare_malecns(raw, target, verify=False)
    graft = load_programmed_graft(3, 7, cache_dir=target, allow_unverified=True)
    assert graft['n'] == 3
    assert set(graft['body_ids']) == {10, 20, 30}
    assert len(graft['pre']) == 3
    signed = {int(graft['body_ids'][p]): float(w) for p, w in zip(graft['pre'], graft['weights'])}
    assert signed[10] > 0 and signed[20] < 0
    assert signed[30] == 0  # Unknown transmitter has no invented excitatory sign.
    assert graft['provenance']['representation'] == 'neuron'
    assert graft['provenance']['sample_is_full_connectome']
    assert not graft['provenance']['measured_synaptic_efficacy']
    assert 'GluCl' in graft['provenance']['sign_convention']
    assert max(np.bincount(graft['post'], weights=np.abs(graft['weights']), minlength=3)) <= 0.900001
    repeat = load_programmed_graft(3, 7, cache_dir=target, allow_unverified=True)
    assert np.array_equal(graft['weights'], repeat['weights'])


def test_absence_and_bad_checksum_never_silently_return_a_proxy(tmp_path, monkeypatch):
    from synapse.connectome_assets import prepare_malecns, load_programmed_graft
    raw = _raw(tmp_path)
    with pytest.raises(ValueError, match='checksum'):
        prepare_malecns(raw, tmp_path / 'imported')
    monkeypatch.setenv('SYNAPSE_MALECNS_DIR', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError):
        load_programmed_graft(8, cache_dir=tmp_path / 'missing-cache')


def test_catalog_requires_independent_source_files_for_homolog_datasets():
    from synapse.connectome_assets import ADDITIONAL_SOURCES
    for name in ('manc', 'optic-lobe'):
        files = ADDITIONAL_SOURCES[name]['files']
        assert files['weights']['filename'] != 'malecns_types.npz'
        assert 'male-cns' not in files['weights']['url']


def test_source_inventory_is_explicit_about_missing_datasets():
    from synapse.connectome_assets import get_source_inventory
    inventory = {item['id']: item for item in get_source_inventory()}
    assert len(inventory) == 8
    assert inventory['malecns']['representation'] == 'neuron'
    assert inventory['malecns']['neuron_count'] == 166700
    assert inventory['fanc']['availability'] == 'unavailable'
    assert not inventory['larval']['full_neuron_graph_present']
    assert inventory['malecns']['download_urls']


def test_banc_string_ids_preserve_all_64_bits(tmp_path):
    from synapse.connectome_assets import prepare_source, ADDITIONAL_SOURCES
    files = ADDITIONAL_SOURCES['banc']['files']
    raw = tmp_path / 'raw'
    raw.mkdir()
    first, second = 720575941633499884, 720575941633499885
    feather.write_feather(pa.table({'banc_888_id': [str(first), str(second)],
        'cell_type': ['A', None], 'super_class': ['sensory', 'central'],
        'proofread': ['TRUE', 'TRUE'], 'roughly_proofread': ['FALSE', 'FALSE'],
        'neurotransmitter_predicted': ['acetylcholine', 'gaba'],
        'neurotransmitter_verified': [None, None]}), raw / files['annotations']['filename'])
    feather.write_feather(pa.table({'pre': [str(first)], 'post': [str(second)], 'count': [9]}), raw / files['weights']['filename'])
    (raw / files['neurotransmitters']['filename']).write_text('root_id\n')
    manifest = prepare_source('banc', raw, tmp_path / 'cache', verify=False)
    metadata = np.load(tmp_path / 'cache' / 'neurons.npz')
    assert metadata['body_ids'].tolist() == [first, second]
    assert manifest['edge_count'] == 1
    assert manifest['synapse_count'] == 9


def test_banc_excludes_known_glia_even_when_annotated(tmp_path):
    from synapse.connectome_assets import prepare_source, ADDITIONAL_SOURCES
    files = ADDITIONAL_SOURCES['banc']['files']
    raw = tmp_path / 'raw'
    raw.mkdir()
    feather.write_feather(pa.table({'banc_888_id': ['10', '20', '30'],
        'cell_type': ['A', None, None], 'super_class': ['sensory', None, 'glia'],
        'proofread': ['TRUE', 'TRUE', 'TRUE'], 'roughly_proofread': ['FALSE'] * 3,
        'neurotransmitter_predicted': ['acetylcholine', 'gaba', None],
        'neurotransmitter_verified': [None] * 3}), raw / files['annotations']['filename'])
    feather.write_feather(pa.table({'pre': ['10', '10'], 'post': ['20', '30'], 'count': [9, 7]}), raw / files['weights']['filename'])
    (raw / files['neurotransmitters']['filename']).write_text('root_id\n')
    manifest = prepare_source('banc', raw, tmp_path / 'cache', verify=False)
    assert manifest['neuron_count'] == 2
    assert manifest['synapse_count'] == 9
