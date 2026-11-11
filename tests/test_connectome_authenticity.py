"""Tests for authentic connectome loading, provenance tiers, and connectivity validation."""
import pytest
from synapse.connectome_datasets import (
    CONNECTOME_CATALOG,
    list_connectome_datasets,
    get_dataset_info,
    get_dataset_provenance,
    load_authentic_connectome,
    validate_connectome_connectivity,
    build_multiconnectome_graph,
)


def test_connectome_dataset_tiers_and_provenance():
    datasets = list_connectome_datasets()
    assert len(datasets) == 8

    # Check that authentic datasets are correctly tagged
    authentic_ids = [d['id'] for d in datasets if d['provenance_tier'] == 'authentic']
    assert 'malecns' in authentic_ids
    assert 'flywire' in authentic_ids
    assert 'manc' in authentic_ids
    assert 'optic-lobe' in authentic_ids

    # Dataset entries describe importers; local_source reports installed files.
    proxy_ids = [d['id'] for d in datasets if d['provenance_tier'] == 'calibrated_proxy']
    assert 'fanc' in proxy_ids
    assert get_dataset_info('larval').provenance_tier == 'source_backed_analysis_subset'
    assert all('local_source' in item for item in datasets)

    prov = get_dataset_provenance('malecns')
    assert prov['provenance_tier'] == 'authentic'
    assert prov['has_authentic_data'] is True
    assert 'acetylcholine' in prov['neurotransmitter_distribution']


def test_authentic_connectome_loader_and_subgraph(require_connectome_sources):
    require_connectome_sources('flywire', 'manc')
    # Load authentic FlyWire central brain
    c_fly = load_authentic_connectome('flywire', n=48, seed=42)
    assert c_fly.n == 48
    assert c_fly.provenance['tier'] == 'authentic'
    assert c_fly.provenance['source_dataset'] == 'flywire'
    assert c_fly.e > 0
    assert c_fly.provenance['total_synapses'] > 0
    assert 0.1 <= c_fly.provenance['inhibitory_fraction'] <= 0.5

    # Load authentic MANC nerve cord
    c_manc = load_authentic_connectome('manc', n=48, seed=123)
    assert c_manc.n == 48
    assert c_manc.provenance['tier'] == 'authentic'
    assert c_manc.provenance['source_dataset'] == 'manc'


def test_connectome_connectivity_validation(require_connectome_sources):
    require_connectome_sources('flywire')
    c_fly = load_authentic_connectome('flywire', n=32, seed=42)
    val = validate_connectome_connectivity(c_fly)
    assert val['neurons'] == 32
    assert val['edges'] > 0
    assert val['dales_law_compliant'] is True
    assert 0.0 <= val['reciprocity_ratio'] <= 1.0
    assert val['provenance_tier'] == 'authentic'


def test_multiconnectome_authentic_cross_wiring(require_connectome_sources):
    require_connectome_sources('flywire', 'manc')
    # Graft authentic FlyWire brain and MANC nerve cord together
    g = build_multiconnectome_graph(['flywire', 'manc'], nodes_per_source=32, seed=42, prefer_authentic=True)
    assert g.n == 64
    assert len(g.base_w) > 0
    # Biological inter-connectome bridges constructed between descending and motor/ascending
    assert len(g.graft_w0) > 0
    # Check that summaries preserve tier
    tiers = [s['provenance'].get('tier') for s in g.summaries]
    assert all(t == 'authentic' for t in tiers)
