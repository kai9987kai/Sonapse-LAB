"""Neurophysiology Dataset Discovery, Ingress, and Provenance Registry.

Maintains provenance for authentic neurophysiology datasets:
- Allen Neuropixels (visual cortex action potentials)
- CRCNS retina, auditory, grasshopper, tactile, olfactory, and orbitofrontal recordings
- DENS (Depression Electrophysiological Neural Signals, Nemar on003751)
- EmoEEG-MC (Multi-channel Emotion EEG, Nemar on005540)
- Emo-FilM (Film-elicited multimodal physiological signals)
- SEED & SEED-OLF (SJTU Emotion EEG Dataset & Olfactory)
- DEAP & DREAMER (Multimodal emotional physiological datasets)
- MODMA (Multi-modal Open Dataset for Mental-disorder Analysis)
- IBL Brain-Wide Map (Neuropixels action potential spike trains)
"""
from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Dict, List, Optional
from .evidence_types import EvidenceKind, EvidenceReceipt


@dataclasses.dataclass
class DatasetManifestEntry:
    dataset_id: str
    title: str
    modality: str  # e.g., 'spike_trains', 'eeg', 'local_field_potentials', 'patch_clamp'
    species: str
    evidence_kind: EvidenceKind
    sampling_rate_hz: float
    channel_count: int
    license: str
    official_uri: str
    description: str
    access_requirements: str
    is_available_locally: bool = False
    local_path: Optional[str] = None
    checksum_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "title": self.title,
            "modality": self.modality,
            "species": self.species,
            "evidence_kind": str(self.evidence_kind.value if isinstance(self.evidence_kind, EvidenceKind) else self.evidence_kind),
            "sampling_rate_hz": self.sampling_rate_hz,
            "channel_count": self.channel_count,
            "license": self.license,
            "official_uri": self.official_uri,
            "description": self.description,
            "access_requirements": self.access_requirements,
            "is_available_locally": self.is_available_locally,
            "local_path": self.local_path,
            "checksum_sha256": self.checksum_sha256,
        }


class NeurophysiologyDataRegistry:
    """Registry managing dataset catalog, ingress verification, and provenance."""

    def __init__(self):
        self._registry: Dict[str, DatasetManifestEntry] = {}
        self._init_standard_catalog()

    def _init_standard_catalog(self):
        entries = [
            DatasetManifestEntry(
                dataset_id="allen_neuropixels_visual_cortex",
                title="Allen Institute Neuropixels Visual Coding",
                modality="spike_trains",
                species="Mus musculus",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=30000.0,
                channel_count=384,
                license="Allen Institute Terms of Use (Open Access)",
                official_uri="https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels",
                description="High-density extracellular action potential recordings across visual cortical and subcortical areas.",
                access_requirements="Public REST API / AllenSDK",
            ),
            DatasetManifestEntry(
                dataset_id="crcns_auditory_locust",
                title="CRCNS Auditory Receptor Spike Trains in Locust",
                modality="spike_trains",
                species="Locusta migratoria",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=20000.0,
                channel_count=16,
                license="CRCNS Open Science Commons",
                official_uri="https://crcns.org/data-sets/auditory",
                description="Action potential recordings in auditory receptors under acoustic pulse stimuli.",
                access_requirements="Free registration",
            ),
            DatasetManifestEntry(
                dataset_id="crcns_ret_1",
                title="CRCNS Retinal Ganglion Cell Spike Responses",
                modality="spike_trains",
                species="Ambystoma tigrinum",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=10000.0,
                channel_count=30,
                license="CRCNS Open Science Commons",
                official_uri="https://crcns.org/data-sets/retina",
                description="Multi-electrode array recordings of retinal ganglion cell spiking to natural movie stimuli.",
                access_requirements="Free registration",
            ),
            DatasetManifestEntry(
                dataset_id="crcns_olfactory_antennal",
                title="CRCNS Projection Neuron Spikes in Olfactory Antennal Lobe",
                modality="spike_trains",
                species="Manduca sexta",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=25000.0,
                channel_count=8,
                license="CRCNS Open Science Commons",
                official_uri="https://crcns.org/data-sets/olfaction",
                description="Intracellular action potential traces during multi-odor floral plume presentations.",
                access_requirements="Free registration",
            ),
            DatasetManifestEntry(
                dataset_id="nemar_dens_on003751",
                title="DENS: Depression Electrophysiological Neural Signals",
                modality="eeg",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=500.0,
                channel_count=64,
                license="NEMAR Open Science (CC-BY 4.0)",
                official_uri="https://nemar.org/dataset/on003751",
                description="Resting-state and task-evoked 64-channel EEG recordings characterizing affective neurodynamics.",
                access_requirements="Open download via NEMAR / OpenNeuro",
            ),
            DatasetManifestEntry(
                dataset_id="nemar_emoeeg_mc_on005540",
                title="EmoEEG-MC: Multi-channel Emotion EEG Dataset",
                modality="eeg",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=1000.0,
                channel_count=128,
                license="NEMAR Open Science (CC-BY 4.0)",
                official_uri="https://www.nemar.org/dataset/on005540",
                description="High-density 128-channel EEG acquired during affective video and acoustic stimulation.",
                access_requirements="Open download via NEMAR",
            ),
            DatasetManifestEntry(
                dataset_id="deap_affective_multimodal",
                title="DEAP: A Database for Emotion Analysis using Physiological Signals",
                modality="multimodal_physiological",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=512.0,
                channel_count=32,
                license="Research Use Agreement",
                official_uri="https://eecs.qmul.ac.uk/mmv/datasets/deap/",
                description="32-channel EEG and peripheral physiological signals during 40 affective music video trials.",
                access_requirements="EULA submission",
            ),
            DatasetManifestEntry(
                dataset_id="dreamer_eeg_ecg",
                title="DREAMER: Database for Emotion Recognition through EEG and ECG",
                modality="eeg_ecg",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=128.0,
                channel_count=14,
                license="Zenodo Open Access",
                official_uri="https://zenodo.org/records/546113",
                description="Wireless 14-channel EEG and ECG during affective film clip sessions with valence/arousal ground-truth.",
                access_requirements="Open download",
            ),
            DatasetManifestEntry(
                dataset_id="seed_emotion_eeg",
                title="SEED: SJTU Emotion EEG Dataset",
                modality="eeg",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=200.0,
                channel_count=62,
                license="Research Use Only",
                official_uri="https://bcmi.sjtu.edu.cn/home/seed/",
                description="62-channel EEG data across positive, neutral, and negative emotion-inducing film clips.",
                access_requirements="Online application",
            ),
            DatasetManifestEntry(
                dataset_id="modma_depression_multimodal",
                title="MODMA: Multi-modal Open Dataset for Mental-disorder Analysis",
                modality="eeg_audio",
                species="Homo sapiens",
                evidence_kind=EvidenceKind.MEASURED_BIOLOGICAL,
                sampling_rate_hz=250.0,
                channel_count=128,
                license="MODMA Research Agreement",
                official_uri="https://modma.lzu.edu.cn/data/index/",
                description="128-channel EEG and interview speech audio for clinical and control subjects.",
                access_requirements="EULA submission",
            ),
        ]
        for e in entries:
            self._registry[e.dataset_id] = e
        # Compatibility aliases
        if "allen_neuropixels_visual_cortex" in self._registry:
            self._registry["allen_neuropixels_visual"] = self._registry["allen_neuropixels_visual_cortex"]
        if "crcns_ret_1" in self._registry:
            self._registry["crcns_retina_salamander"] = self._registry["crcns_ret_1"]

    def get_dataset(self, dataset_id: str) -> Optional[DatasetManifestEntry]:
        return self._registry.get(dataset_id)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._registry.values()]

    def register_dataset(self, entry: DatasetManifestEntry) -> None:
        self._registry[entry.dataset_id] = entry

    def create_provenance_receipt(self, dataset_id: str, action: str, samples_processed: int) -> EvidenceReceipt:
        ds = self.get_dataset(dataset_id)
        if ds is None:
            kind = EvidenceKind.SYNTHETIC_SIMULATION
            src = dataset_id
        else:
            kind = ds.evidence_kind
            src = f"{ds.dataset_id}:{ds.species}:{ds.modality}"

        return EvidenceReceipt(
            kind=kind,
            source_identifier=src,
            description=f"Action: {action} on {dataset_id}",
            metrics={
                "dataset_id": dataset_id,
                "samples_processed": samples_processed,
                "official_uri": ds.official_uri if ds else "none",
            },
        )


DatasetProvenanceRecord = DatasetManifestEntry

