"""Neuroendocrine and Affective Connectome Subsystem.

Provides:
- Evidence taxonomy and cryptographic provenance receipts (EvidenceKind, EvidenceReceipt).
- Authentic neurophysiology dataset registry (NeurophysiologyDataRegistry).
- Multi-modal sensory event fabric (SensoryEventFabric).
- Neural spike encoding (NeuralSpikeEncoder).
- 32-dimensional continuous affective state engine (MixedAffectiveState).
- 256-neuron Weight-Programmed Affective Connectome (WeightProgrammedAffectiveCore).
- Regulatory motif programmer (AffectiveWeightProgrammer).
- Affective homeostatic equilibrium regulator (AffectiveEquilibriumRegulator).
- 12-channel synthetic gland bank (SyntheticGlandBank, GLAND_NAMES).
- Synthetic receptor affinity bank (SyntheticReceptorBank, TARGET_PROPERTIES).
- Authoritative zero-gain modulatory compositor (ModulatoryCompositor).
"""

from synapse.neuroendocrine.evidence_types import (
    EvidenceKind,
    EvidenceReceipt,
    compute_tensor_digest,
)
from synapse.neuroendocrine.data_registry import (
    NeurophysiologyDataRegistry,
    DatasetProvenanceRecord,
)
from synapse.neuroendocrine.sensory_events import (
    SensoryEvent,
    SensoryEventFabric,
)
from synapse.neuroendocrine.spike_encoder import (
    NeuralSpikeEncoder,
)
from synapse.neuroendocrine.mixed_affective_state import (
    MixedAffectiveState,
    AFFECTIVE_DIMENSIONS,
)
from synapse.neuroendocrine.affective_connectome import (
    WeightProgrammedAffectiveCore,
    POPULATION_NAMES,
    POPULATION_SLICES,
)
from synapse.neuroendocrine.affective_weight_programmer import (
    AffectiveWeightProgrammer,
)
from synapse.neuroendocrine.equilibrium_regulator import (
    AffectiveEquilibriumRegulator,
)
from synapse.neuroendocrine.synthetic_glands import (
    SyntheticGlandBank,
    GLAND_NAMES,
    GLAND_SPECS,
)
from synapse.neuroendocrine.receptor_fields import (
    SyntheticReceptorBank,
    TARGET_PROPERTIES,
)
from synapse.neuroendocrine.modulatory_compositor import (
    ModulatoryCompositor,
)

__all__ = [
    "EvidenceKind",
    "EvidenceReceipt",
    "compute_tensor_digest",
    "NeurophysiologyDataRegistry",
    "DatasetProvenanceRecord",
    "SensoryEvent",
    "SensoryEventFabric",
    "NeuralSpikeEncoder",
    "MixedAffectiveState",
    "AFFECTIVE_DIMENSIONS",
    "WeightProgrammedAffectiveCore",
    "POPULATION_NAMES",
    "POPULATION_SLICES",
    "AffectiveWeightProgrammer",
    "AffectiveEquilibriumRegulator",
    "SyntheticGlandBank",
    "GLAND_NAMES",
    "GLAND_SPECS",
    "SyntheticReceptorBank",
    "TARGET_PROPERTIES",
    "ModulatoryCompositor",
]
