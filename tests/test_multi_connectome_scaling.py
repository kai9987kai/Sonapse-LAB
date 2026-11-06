"""Unit and Integration Tests for Multi-Connectome Architecture & High-Capacity Neuron Scaling.

Verifies:
1. Biological Connectome Catalog covering all 8 landmark datasets:
   - MaleCNS v1.0 (Janelia/Cambridge/MRC LMB/Google Research)
   - BANC (Harvard Lee & Wilson/Princeton Murthy/Harvard Drugowitsch)
   - FlyEM Optic Lobe (Janelia/Google)
   - FANC (Harvard Wei-Chung Allen Lee lab)
   - FlyWire FAFB v783 (Princeton Murthy & Seung/FlyWire)
   - MANC (Janelia/Cambridge/Google)
   - Larval Drosophila Brain (Cambridge Winding et al.)
   - Hemibrain (Janelia FlyEM)
2. MultiConnectomeHub cross-wiring and regional neuromodulatory susceptibility.
3. High-capacity neuron scaling and CPU throughput benchmarks.
4. IntegratedAgent multi-connectome initialization and weight-space editable surfaces.
5. 3D Neural Observatory spatial embeddings across all 8 biological regions.
6. RSI connectome graft candidate generation and multi-stage verification.
"""
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.agent import IntegratedAgent
from synapse.connectome_datasets import (
    CONNECTOME_CATALOG,
    ALL_DATASET_IDS,
    list_connectome_datasets,
    get_dataset_info,
    build_multiconnectome_graph
)
from synapse.multi_connectome import MultiConnectomeHub, benchmark_neuron_scaling
from synapse.observatory import NeuralObservatory
from synapse.rsi_verifier import BoundedRSIEngine, CandidatePatch


def test_connectome_catalog_metadata():
    """Verifies that all 8 landmark connectome datasets are cataloged with complete provenance."""
    expected_ids = {'malecns', 'banc', 'optic-lobe', 'fanc', 'flywire', 'manc', 'larval', 'hemibrain'}
    assert set(ALL_DATASET_IDS) == expected_ids

    catalog = list_connectome_datasets()
    assert len(catalog) == 8

    # Check MaleCNS
    malecns = get_dataset_info('malecns')
    assert malecns.license == 'CC-BY-4.0'
    assert 'Janelia' in malecns.organisation_authors
    assert malecns.used_by_projects >= 269

    # Check BANC
    banc = get_dataset_info('banc')
    assert banc.license == 'CC-BY-4.0'
    assert 'Harvard' in banc.organisation_authors

    # Check FlyWire
    flywire = get_dataset_info('flywire')
    assert flywire.license == 'CC-BY-NC-4.0'
    assert flywire.estimated_specimen_neurons >= 139000

    # Check Larval
    larval = get_dataset_info('larval')
    assert 'Winding' in larval.organisation_authors
    assert larval.used_by_projects >= 7


def test_multi_connectome_hub_instantiation_and_step():
    """Verifies multi-connectome hub execution, regional gating, and neuromodulation across 8 datasets."""
    hub = MultiConnectomeHub(sources=ALL_DATASET_IDS, nodes_per_source=16, microsteps=2, seed=42)
    assert hub.total_neurons == 16 * 8
    assert len(hub.region_ranges) == 8
    assert len(hub.graph.sites) > 0

    sensory_dim = len(hub.brain.sensory_idx)
    inp = torch.randn(sensory_dim)
    r = hub.forward(inp, neuromod_state={'DA': 1.2, '5-HT': 0.8, 'OA': 1.5, 'ACh': 1.0})

    assert r.shape == (hub.total_neurons,)
    assert hub.last_telemetry is not None
    assert len(hub.last_telemetry.regional_means) == 8
    assert hub.last_telemetry.inter_connectome_flux > 0.0

    summary = hub.get_summary()
    assert summary['sources_count'] == 8
    assert 'flywire' in summary['regions']
    assert 'hemibrain' in summary['regions']


def test_high_capacity_neuron_scaling():
    """Verifies that high-capacity neuron scaling executes stably on CPU with high throughput."""
    res = benchmark_neuron_scaling(target_counts=[64, 128], steps_per_benchmark=5, seed=42)
    assert res['status'] == 'ok'
    assert len(res['benchmarks']) == 2
    for bench in res['benchmarks']:
        assert bench['mean_step_ms'] < 50.0  # Real-time sub-50ms execution
        assert bench['throughput_neurons_per_sec'] > 5000.0
        assert bench['stable_recurrent_activity']


def test_agent_multi_connectome_integration():
    """Verifies that IntegratedAgent initializes with all 8 connectomes and exposes editable surfaces."""
    cfg = SynapseConfig(
        nodes_per_source=16,
        connectome_sources=('malecns', 'banc', 'optic-lobe', 'fanc', 'flywire', 'manc', 'larval', 'hemibrain'),
        seed=42
    )
    agent = IntegratedAgent(cfg)
    assert agent.brain.n == 16 * 8

    obs = np.linspace(0, 1, 155, dtype=np.float32)
    out = agent.act(obs, learn=False)
    assert 'action' in out
    assert len(out['action']) == 5
    assert out['features'].shape == (64,)

    # Verify weight-space editability of multi-connectome graft weights
    tensors = agent.editable_tensors()
    assert 'brain.graft_w' in tensors
    assert tensors['brain.graft_w'].shape == agent.brain.graft_w.shape


def test_observatory_3d_multi_connectome():
    """Verifies that NeuralObservatory embeds all 8 connectomes in 3D anatomical space."""
    cfg = SynapseConfig(
        nodes_per_source=16,
        connectome_sources=('malecns', 'banc', 'optic-lobe', 'fanc', 'flywire', 'manc', 'larval', 'hemibrain'),
        seed=42
    )
    agent = IntegratedAgent(cfg)
    obs = NeuralObservatory(agent, seed=42)
    payload = obs.get_structure_payload()

    assert payload['total_nodes'] == 16 * 8
    assert len(payload['neuropils']) == 8
    assert len(payload['nodes']) == 16 * 8
    # Verify dataset attribute in node
    datasets_found = {n['dataset'] for n in payload['nodes']}
    assert datasets_found == set(cfg.connectome_sources)


def test_rsi_connectome_graft_mutation():
    """Verifies that the bounded RSI engine can propose and evaluate connectome graft mutations."""
    engine = BoundedRSIEngine(seed=42)
    cand = engine.generate_candidate(patch_type="connectome_graft")
    assert cand.patch_type == "connectome_graft"
    assert 'inter_graft_gain' in cand.diff

    receipt = engine.verify_and_eval(cand, num_support_trials=3, num_query_trials=3)
    assert receipt.receipt_hash != ""
