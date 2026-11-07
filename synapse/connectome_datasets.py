"""Dataset catalog with independent source-backed samples and explicit proxies.

Actual full-source availability is reported by connectome_assets. Catalog entries
alone are not evidence that a dataset has been downloaded or executed. The shipped
maleCNS cell-type aggregate is distinct from every neuron-level dataset.
"""
from __future__ import annotations
import math
import zlib
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import VENDOR
from .sources import activate
activate()
from graftbrain.connectomes import Connectome, make_proxy, PROXY_PROFILES
from graftbrain.graft import build, GraftedGraph

DATA_PATH = VENDOR / 'expanse' / 'expanse' / 'data' / 'malecns_types.npz'


@dataclass(frozen=True)
class ConnectomeDatasetInfo:
    id: str
    name: str
    organisation_authors: str
    release_date: str
    license: str
    used_by_projects: int
    biological_region: str
    estimated_specimen_neurons: int
    estimated_specimen_synapses: int
    description: str
    roles_distribution: Dict[str, float]
    provenance_tier: str = 'authentic'  # 'authentic' | 'calibrated_proxy' | 'synthetic'
    has_authentic_data: bool = True
    neurotransmitter_distribution: Dict[str, float] = field(default_factory=dict)
    cell_type_count: int = 0
    synapse_count: int = 0
    biological_validation: str = "Verified anatomical connectome"


CONNECTOME_CATALOG: Dict[str, ConnectomeDatasetInfo] = {
    'malecns': ConnectomeDatasetInfo(
        id='malecns',
        name='MaleCNS v1.0',
        organisation_authors='HHMI Janelia FlyEM / University of Cambridge / MRC LMB / Google Research',
        release_date='2026-06-08',
        license='CC-BY-4.0',
        used_by_projects=269,
        biological_region='Whole Central Nervous System (Brain + VNC)',
        estimated_specimen_neurons=145000,
        estimated_specimen_synapses=60000000,
        description='Complete male Drosophila CNS connectome containing both brain and ventral nerve cord.',
        roles_distribution={'sensory': 0.013, 'central': 0.797, 'motor': 0.018, 'descending': 0.041, 'ascending': 0.068, 'visual': 0.063},
        provenance_tier='authentic',
        has_authentic_data=True,
        neurotransmitter_distribution={'acetylcholine': 0.48, 'gaba': 0.26, 'glutamate': 0.18, 'serotonin': 0.025, 'dopamine': 0.020, 'octopamine': 0.015, 'histamine': 0.020},
        cell_type_count=11751,
        synapse_count=3830931,
        biological_validation='Source-backed FIB-SEM reconstruction with full Dale\'s law signed weights'
    ),
    'flywire': ConnectomeDatasetInfo(
        id='flywire',
        name='FlyWire (FAFB v783)',
        organisation_authors='Princeton University (Murthy & Seung labs) with the FlyWire consortium',
        release_date='2024',
        license='CC-BY-NC-4.0',
        used_by_projects=180,
        biological_region='Whole Adult Female Brain',
        estimated_specimen_neurons=139255,
        estimated_specimen_synapses=54500000,
        description='First full adult female Drosophila brain connectome with 8,400+ curated cell types.',
        roles_distribution={'sensory': 0.015, 'central': 0.820, 'motor': 0.005, 'descending': 0.052, 'ascending': 0.038, 'visual': 0.070},
        provenance_tier='authentic',
        has_authentic_data=True,
        neurotransmitter_distribution={'acetylcholine': 0.52, 'gaba': 0.24, 'glutamate': 0.16, 'dopamine': 0.03, 'serotonin': 0.03, 'octopamine': 0.02},
        cell_type_count=8137,
        synapse_count=2650000,
        biological_validation='Authentic central-brain subset verified against FAFB v783 annotations'
    ),
    'manc': ConnectomeDatasetInfo(
        id='manc',
        name='MANC (Male Adult Nerve Cord)',
        organisation_authors='HHMI Janelia FlyEM, Cambridge Connectomics Group, Google Research',
        release_date='2023-06',
        license='CC-BY-4.0',
        used_by_projects=17,
        biological_region='Male Adult Ventral Nerve Cord',
        estimated_specimen_neurons=23000,
        estimated_specimen_synapses=12000000,
        description='Dense connectome of motor neurons, leg/wing neuropils, and sensory ascending fibers.',
        roles_distribution={'sensory': 0.025, 'central': 0.740, 'motor': 0.045, 'descending': 0.040, 'ascending': 0.150, 'visual': 0.000},
        provenance_tier='authentic',
        has_authentic_data=True,
        neurotransmitter_distribution={'acetylcholine': 0.44, 'gaba': 0.30, 'glutamate': 0.20, 'dopamine': 0.02, 'serotonin': 0.02, 'octopamine': 0.02},
        cell_type_count=4182,
        synapse_count=1380000,
        biological_validation='Authentic VNC motor and ascending/descending circuits'
    ),
    'optic-lobe': ConnectomeDatasetInfo(
        id='optic-lobe',
        name='FlyEM Optic Lobe Connectome',
        organisation_authors='HHMI Janelia FlyEM Project Team with Google Connectomics',
        release_date='2024',
        license='CC-BY-4.0',
        used_by_projects=5,
        biological_region='Optic Lobe (Visual Processing Neuropils)',
        estimated_specimen_neurons=65000,
        estimated_specimen_synapses=20000000,
        description='Dense reconstruction of retinotopic visual columns (medulla, lobula, lobula plate).',
        roles_distribution={'sensory': 0.15, 'central': 0.35, 'motor': 0.00, 'descending': 0.00, 'ascending': 0.00, 'visual': 0.50},
        provenance_tier='authentic',
        has_authentic_data=True,
        neurotransmitter_distribution={'acetylcholine': 0.40, 'gaba': 0.25, 'glutamate': 0.20, 'histamine': 0.15},
        cell_type_count=1824,
        synapse_count=720000,
        biological_validation='Authentic retinotopic columns and centrifugal visual projections'
    ),
    'banc': ConnectomeDatasetInfo(
        id='banc',
        name='BANC (Brain And Nerve Cord)',
        organisation_authors='Harvard Medical School (Lee & Wilson labs), Princeton University (Murthy lab), Harvard (Drugowitsch lab)',
        release_date='2026',
        license='CC-BY-4.0',
        used_by_projects=22,
        biological_region='Brain And Nerve Cord',
        estimated_specimen_neurons=140000,
        estimated_specimen_synapses=55000000,
        description='Integrated whole-fly connectome capturing ascending and descending sensorimotor coordination.',
        roles_distribution={'sensory': 0.12, 'central': 0.50, 'motor': 0.10, 'descending': 0.07, 'ascending': 0.07, 'visual': 0.14},
        provenance_tier='calibrated_proxy',
        has_authentic_data=False,
        neurotransmitter_distribution={'acetylcholine': 0.46, 'gaba': 0.28, 'glutamate': 0.19, 'biogenic_amines': 0.07},
        cell_type_count=12000,
        synapse_count=55000000,
        biological_validation='Calibrated architectural proxy with matched BANC degree and role distributions'
    ),
    'fanc': ConnectomeDatasetInfo(
        id='fanc',
        name='FANC (Female Adult Nerve Cord)',
        organisation_authors='Harvard Medical School (Wei-Chung Allen Lee lab)',
        release_date='2024',
        license='unknown',
        used_by_projects=3,
        biological_region='Female Adult Ventral Nerve Cord',
        estimated_specimen_neurons=20000,
        estimated_specimen_synapses=10000000,
        description='High-resolution serial section TEM reconstruction of female adult nerve cord motor circuits.',
        roles_distribution={'sensory': 0.20, 'central': 0.45, 'motor': 0.15, 'descending': 0.10, 'ascending': 0.10, 'visual': 0.00},
        provenance_tier='calibrated_proxy',
        has_authentic_data=False,
        neurotransmitter_distribution={'acetylcholine': 0.42, 'gaba': 0.32, 'glutamate': 0.22, 'biogenic_amines': 0.04},
        cell_type_count=3200,
        synapse_count=10000000,
        biological_validation='Calibrated architectural proxy based on FANC female VNC TEM annotations'
    ),
    'larval': ConnectomeDatasetInfo(
        id='larval',
        name='Larval Drosophila Brain Connectome (Winding et al. 2023)',
        organisation_authors='University of Cambridge (Michael Winding et al.)',
        release_date='2023-03',
        license='CC-BY-4.0',
        used_by_projects=7,
        biological_region='Larval Drosophila Brain & Cord',
        estimated_specimen_neurons=3016,
        estimated_specimen_synapses=548000,
        description='Complete synaptic connectome of an insect larva with all sensory, inter-, and motor neurons.',
        roles_distribution={'sensory': 0.20, 'central': 0.60, 'motor': 0.10, 'descending': 0.05, 'ascending': 0.05, 'visual': 0.00},
        provenance_tier='calibrated_proxy',
        has_authentic_data=False,
        neurotransmitter_distribution={'acetylcholine': 0.50, 'gaba': 0.28, 'glutamate': 0.16, 'biogenic_amines': 0.06},
        cell_type_count=3016,
        synapse_count=548000,
        biological_validation='Calibrated architectural proxy matching Winding et al. larval topologies'
    ),
    'hemibrain': ConnectomeDatasetInfo(
        id='hemibrain',
        name='Hemibrain',
        organisation_authors='HHMI Janelia Research Campus (FlyEM Project Team)',
        release_date='2020-01-22',
        license='CC-BY-4.0',
        used_by_projects=23,
        biological_region='Central Brain Hemisphere & Central Complex',
        estimated_specimen_neurons=25000,
        estimated_specimen_synapses=20000000,
        description='Dense FIB-SEM reconstruction of central brain regions (central complex, mushroom body, antennal lobe).',
        roles_distribution={'sensory': 0.06, 'central': 0.78, 'motor': 0.00, 'descending': 0.08, 'ascending': 0.04, 'visual': 0.04},
        provenance_tier='calibrated_proxy',
        has_authentic_data=False,
        neurotransmitter_distribution={'acetylcholine': 0.54, 'gaba': 0.22, 'glutamate': 0.18, 'biogenic_amines': 0.06},
        cell_type_count=4500,
        synapse_count=20000000,
        biological_validation='Calibrated architectural proxy covering central complex and mushroom body'
    )
}

# Catalog specimen scope is separate from the assets actually installed here.
# Only two independent neuron-level sources have an importer. The old maleCNS
# type-graph flags has_flywire/has_manc indicate homolog annotations, not datasets.
CONNECTOME_CATALOG['malecns'] = replace(CONNECTOME_CATALOG['malecns'],
    estimated_specimen_neurons=166700, estimated_specimen_synapses=124177617,
    synapse_count=124177617,
    biological_validation='Independent neuron-level MaleCNS v1.0 source; signed weights are modelling assumptions')
CONNECTOME_CATALOG['flywire'] = replace(CONNECTOME_CATALOG['flywire'],
    biological_validation='Independent full FlyWire v783 proofread neuron-pair source; runtime samples are induced subgraphs')
for _installed in ('manc', 'optic-lobe', 'banc', 'hemibrain'):
    CONNECTOME_CATALOG[_installed] = replace(CONNECTOME_CATALOG[_installed],
        provenance_tier='authentic', has_authentic_data=True,
        biological_validation='Independent neuron-level release importer; actual installed coverage is in local_source')
CONNECTOME_CATALOG['larval'] = replace(CONNECTOME_CATALOG['larval'],
    provenance_tier='source_backed_analysis_subset', has_authentic_data=True,
    biological_validation='Published 2952-neuron analysis matrix only; full 3016-neuron reconstruction not installed')

ALL_DATASET_IDS = list(CONNECTOME_CATALOG.keys())


def map_cell_type_to_role(cell_class: str, superclass: str) -> str:
    """Biologically maps Drosophila cell_class and superclass to standard connectome roles."""
    cc = str(cell_class).lower()
    sc = str(superclass).lower()
    if 'visual' in cc or 'ol_' in sc or 'visual' in sc:
        return 'visual'
    if 'ascending' in sc or 'an' in cc or 'ascending' in cc:
        return 'ascending'
    if 'descending' in sc or 'dn' in cc or 'descending' in cc:
        return 'descending'
    if 'motor' in sc or 'motor' in cc or 'efferent' in sc:
        return 'motor'
    if 'sensory' in sc or 'sensory' in cc or 'chemosensory' in cc or 'gustatory' in cc or 'olfactory' in cc or 'mechanosensory' in cc or 'thermosensory' in cc:
        return 'sensory'
    return 'central'


def load_authentic_connectome(dataset_id: str, n: int = 64, seed: int = 42, *, allow_proxy: bool = False) -> Connectome:
    """Load independent neuron-level source data; absence is explicit by default.

    MaleCNS homolog typing fields are never substituted for FlyWire/MANC edges.
    Setting allow_proxy explicitly permits a labelled synthetic compatibility graph.
    """
    from .connectome_assets import load_programmed_graft
    key = dataset_id.lower().strip()
    if key not in CONNECTOME_CATALOG:
        raise KeyError(f"Unknown connectome dataset {dataset_id!r}")
    try:
        asset = load_programmed_graft(n, seed, dataset=key)
    except FileNotFoundError as error:
        if not allow_proxy:
            raise
        proxy = make_proxy(key, n=n, seed=seed)
        proxy.provenance.update({'tier': 'calibrated_proxy', 'representation': 'synthetic',
                                 'source_dataset': None, 'requested_dataset': key,
                                 'note': str(error), 'sample_is_full_connectome': False})
        return proxy
    signs = asset['signs'][asset['pre']]
    provenance = dict(asset['provenance'])
    provenance['weight_conversion'] = ('Anatomical counts and assumed transmitter signs supplied to '
                                     'graftbrain.build; frozen base weights use signed log1p counts, '
                                     'incoming row normalization and its configured gain (default 1.1).')
    provenance['prior_weight_conversion'] = provenance.pop('sign_convention')
    kind = {'malecns': 'cns', 'banc': 'cns', 'manc': 'vnc', 'optic-lobe': 'optic', 'larval': 'larva'}.get(key, 'brain')
    names = np.array([str(name) or f'untyped:{key}:{body}' for name, body in zip(asset['cell_types'], asset['body_ids'])], dtype=object)
    return Connectome(name=key, kind=kind,
                      pre=asset['pre'], post=asset['post'], weight=asset['synapse_counts'].astype(np.float32),
                      sign=signs.astype(np.float32), cell_type=names, role=asset['roles'],
                      source_ids=asset['body_ids'], provenance=provenance)


def build_runtime_connectomes(config, *, strict: bool = False):
    """Build actual source-backed runtime samples with explicit profile fallbacks."""
    return [load_authentic_connectome(name, n=n, seed=config.seed + index * 101, allow_proxy=not strict)
            for index, (name, n) in enumerate(zip(config.connectome_sources, config.source_node_counts()))]


def validate_connectome_connectivity(connectome: Connectome) -> Dict[str, Any]:
    """Performs topological validation on a connectome graph, checking reciprocity, degrees, and Dale's law."""
    n = connectome.n
    e = connectome.e
    if n == 0:
        return {'status': 'empty', 'neurons': 0, 'edges': 0}

    # In-degrees and out-degrees
    in_deg = np.zeros(n, dtype=int)
    out_deg = np.zeros(n, dtype=int)
    if e > 0:
        np.add.at(in_deg, connectome.post, 1)
        np.add.at(out_deg, connectome.pre, 1)

    # Reciprocity
    edge_pairs = set(zip(connectome.pre.tolist(), connectome.post.tolist())) if e > 0 else set()
    reciprocal = sum(1 for (u, v) in edge_pairs if (v, u) in edge_pairs)

    # Dale's law validation: each neuron's outgoing synapses must be univalent (all +1 or all -1)
    dales_law_compliant = True
    if e > 0:
        for u in np.unique(connectome.pre):
            u_signs = connectome.sign[connectome.pre == u]
            if len(np.unique(u_signs)) > 1:
                dales_law_compliant = False
                break

    return {
        'neurons': n,
        'edges': e,
        'synapses': int(connectome.weight.sum()) if e > 0 else 0,
        'mean_in_degree': float(in_deg.mean()),
        'max_in_degree': int(in_deg.max()) if n > 0 else 0,
        'mean_out_degree': float(out_deg.mean()),
        'max_out_degree': int(out_deg.max()) if n > 0 else 0,
        'reciprocity_ratio': float(reciprocal / e) if e > 0 else 0.0,
        'dales_law_compliant': dales_law_compliant,
        'provenance_tier': connectome.provenance.get('tier', 'unspecified'),
    }


def list_connectome_datasets() -> List[Dict[str, Any]]:
    """Returns metadata dictionaries for all 8 landmark connectome datasets with provenance annotations."""
    from .connectome_assets import get_source_inventory
    installed = {item['id']: item for item in get_source_inventory()}
    return [{**asdict(info), 'local_source': installed[info.id]} for info in CONNECTOME_CATALOG.values()]


def get_dataset_info(dataset_id: str) -> ConnectomeDatasetInfo:
    """Returns dataset metadata for a specific connectome profile."""
    dataset_key = dataset_id.lower().strip()
    if dataset_key not in CONNECTOME_CATALOG:
        raise KeyError(f"Unknown connectome dataset '{dataset_id}'. Choose from: {ALL_DATASET_IDS}")
    return CONNECTOME_CATALOG[dataset_key]


def get_dataset_provenance(dataset_id: str) -> Dict[str, Any]:
    """Returns rich provenance report for a dataset, distinguishing authentic from proxy models."""
    info = get_dataset_info(dataset_id)
    return {
        'id': info.id,
        'name': info.name,
        'provenance_tier': info.provenance_tier,
        'has_authentic_data': info.has_authentic_data,
        'organisation_authors': info.organisation_authors,
        'license': info.license,
        'biological_region': info.biological_region,
        'estimated_neurons': info.estimated_specimen_neurons,
        'estimated_synapses': info.estimated_specimen_synapses,
        'neurotransmitter_distribution': info.neurotransmitter_distribution,
        'validation_status': info.biological_validation,
    }


def build_multiconnectome_graph(
    sources: Optional[List[str]] = None,
    nodes_per_source: int = 64,
    seed: int = 42,
    prefer_authentic: bool = True
) -> GraftedGraph:
    """Grafts multiple biological connectomes into a single unified cross-wired brain.

    Args:
        sources: list of connectome dataset IDs (defaults to all 8 datasets).
        nodes_per_source: number of sampled neurons per dataset.
        seed: random seed for reproducibility.
        prefer_authentic: when True, loads authentic biological subgraphs for datasets where physical data exists.
    """
    dataset_keys = sources if sources is not None else ALL_DATASET_IDS
    connectome_list = []

    for idx, name in enumerate(dataset_keys):
        clean_name = name.lower().strip()
        sub_seed = seed + idx * 101
        if prefer_authentic:
            c = load_authentic_connectome(clean_name, n=nodes_per_source, seed=sub_seed, allow_proxy=True)
        else:
            c = make_proxy(clean_name, n=nodes_per_source, seed=sub_seed)
            c.provenance.update({'tier': 'calibrated_proxy', 'representation': 'synthetic',
                                 'source_dataset': None, 'requested_dataset': clean_name})
        connectome_list.append(c)

    return build(connectome_list, seed=seed)
