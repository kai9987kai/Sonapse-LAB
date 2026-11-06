"""Biological Connectome Datasets Registry & Multi-Connectome Compiler.

Grounding:
Integrates the complete suite of 8 landmark Drosophila connectome datasets:
1. MaleCNS v1.0 (HHMI Janelia FlyEM / Cambridge / MRC LMB / Google Research, 2026-06-08, CC-BY-4.0, 269 projects)
2. BANC: Brain And Nerve Cord (Harvard / Princeton / Harvard, 2026, CC-BY-4.0, 22 projects)
3. FlyEM Optic Lobe Connectome (Janelia FlyEM with Google Connectomics, 2024, CC-BY-4.0, 5 projects)
4. FANC: Female Adult Nerve Cord (Harvard Wei-Chung Allen Lee lab, 2024, unknown, 3 projects)
5. FlyWire: FAFB v783 (Princeton Murthy & Seung labs / FlyWire consortium, 2024, CC-BY-NC-4.0, 180 projects)
6. MANC: Male Adult Nerve Cord (Janelia FlyEM / Cambridge / Google Research, 2023-06, CC-BY-4.0, 17 projects)
7. Larval Drosophila Brain Connectome (Cambridge Winding et al., 2023-03, CC-BY-4.0, 7 projects)
8. Hemibrain (Janelia Research Campus FlyEM, 2020-01-22, CC-BY-4.0, 23 projects)

Provides structural profiles, metadata provenance, and scalable multi-connectome synthesis.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .sources import activate
activate()
from graftbrain.connectomes import Connectome, make_proxy, PROXY_PROFILES
from graftbrain.graft import build, GraftedGraph


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
        roles_distribution={'sensory': 0.10, 'central': 0.50, 'motor': 0.08, 'descending': 0.06, 'ascending': 0.06, 'visual': 0.20}
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
        roles_distribution={'sensory': 0.12, 'central': 0.50, 'motor': 0.10, 'descending': 0.07, 'ascending': 0.07, 'visual': 0.14}
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
        roles_distribution={'sensory': 0.20, 'central': 0.30, 'motor': 0.00, 'descending': 0.00, 'ascending': 0.00, 'visual': 0.50}
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
        roles_distribution={'sensory': 0.20, 'central': 0.45, 'motor': 0.15, 'descending': 0.10, 'ascending': 0.10, 'visual': 0.00}
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
        roles_distribution={'sensory': 0.10, 'central': 0.62, 'motor': 0.00, 'descending': 0.06, 'ascending': 0.04, 'visual': 0.18}
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
        roles_distribution={'sensory': 0.20, 'central': 0.45, 'motor': 0.15, 'descending': 0.10, 'ascending': 0.10, 'visual': 0.00}
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
        roles_distribution={'sensory': 0.20, 'central': 0.60, 'motor': 0.10, 'descending': 0.05, 'ascending': 0.05, 'visual': 0.00}
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
        roles_distribution={'sensory': 0.06, 'central': 0.78, 'motor': 0.00, 'descending': 0.08, 'ascending': 0.04, 'visual': 0.04}
    )
}

ALL_DATASET_IDS = list(CONNECTOME_CATALOG.keys())


def list_connectome_datasets() -> List[Dict[str, Any]]:
    """Returns metadata dictionaries for all 8 landmark connectome datasets."""
    return [asdict(info) for info in CONNECTOME_CATALOG.values()]


def get_dataset_info(dataset_id: str) -> ConnectomeDatasetInfo:
    """Returns dataset metadata for a specific connectome profile."""
    dataset_key = dataset_id.lower().strip()
    if dataset_key not in CONNECTOME_CATALOG:
        raise KeyError(f"Unknown connectome dataset '{dataset_id}'. Choose from: {ALL_DATASET_IDS}")
    return CONNECTOME_CATALOG[dataset_key]


def build_multiconnectome_graph(
    sources: Optional[List[str]] = None,
    nodes_per_source: int = 64,
    seed: int = 42
) -> GraftedGraph:
    """Grafts multiple biological connectomes into a single unified cross-wired brain.

    Args:
        sources: list of connectome dataset IDs (defaults to all 8 datasets).
        nodes_per_source: number of sampled neurons per dataset.
        seed: random seed for reproducibility.
    """
    dataset_keys = sources if sources is not None else ALL_DATASET_IDS
    connectome_list = []

    for name in dataset_keys:
        clean_name = name.lower().strip()
        if clean_name not in PROXY_PROFILES:
            raise ValueError(f"Connectome profile '{clean_name}' not available. Available: {sorted(PROXY_PROFILES.keys())}")
        c = make_proxy(clean_name, n=nodes_per_source, seed=seed)
        connectome_list.append(c)

    return build(connectome_list, seed=seed)
