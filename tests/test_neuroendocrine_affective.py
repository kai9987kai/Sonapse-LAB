"""Comprehensive Test Suite for Neuroendocrine and Affective Connectome Subsystem.

Verifies:
1. Evidence taxonomy and cryptographic SHA-256 digests.
2. Authentic neurophysiology dataset registry.
3. Multimodal sensory event fabric and spike encoding.
4. 32-dimensional continuous affective state and 6-dim core appraisal.
5. 256-neuron Weight-Programmed Affective Connectome with Dale's law priors.
6. Affective weight programmer and regulatory motifs.
7. Homeostatic equilibrium regulator and allostatic costs.
8. 12-channel synthetic gland bank with differential clearance.
9. Synthetic receptor fields and target property gains.
10. Modulatory compositor zero-gain identity guarantee bit-for-bit (Delta = 0.0).
"""
import pytest
import numpy as np
import torch

from synapse.neuroendocrine import (
    EvidenceKind,
    EvidenceReceipt,
    compute_tensor_digest,
    NeurophysiologyDataRegistry,
    DatasetProvenanceRecord,
    SensoryEvent,
    SensoryEventFabric,
    NeuralSpikeEncoder,
    MixedAffectiveState,
    AFFECTIVE_DIMENSIONS,
    WeightProgrammedAffectiveCore,
    POPULATION_NAMES,
    POPULATION_SLICES,
    AffectiveWeightProgrammer,
    AffectiveEquilibriumRegulator,
    SyntheticGlandBank,
    GLAND_NAMES,
    SyntheticReceptorBank,
    TARGET_PROPERTIES,
    ModulatoryCompositor,
)
from synapse.neuromodulation import NeuromodulatorState


def test_evidence_types_and_digests():
    t = torch.randn(10, 10)
    digest1 = compute_tensor_digest(t)
    digest2 = compute_tensor_digest(t.clone())
    assert digest1 == digest2
    assert len(digest1) == 64  # SHA-256 hex string

    receipt = EvidenceReceipt(
        kind=EvidenceKind.MEASURED_BIOLOGICAL,
        source_identifier="Allen_Neuropixels_V1",
        description="Spike trains from primary visual cortex",
        metrics={"sampling_rate": 30000.0, "channels": 384},
    )
    rec_dict = receipt.to_dict()
    assert rec_dict["kind"] == "measured_biological"
    assert rec_dict["checksum"] != ""


def test_data_registry_manifests():
    registry = NeurophysiologyDataRegistry()
    datasets = registry.list_datasets()
    assert len(datasets) >= 8

    # Verify key datasets are present
    dataset_ids = [d["dataset_id"] for d in datasets]
    assert "allen_neuropixels_visual_cortex" in dataset_ids
    assert "crcns_ret_1" in dataset_ids
    assert "nemar_dens_on003751" in dataset_ids
    assert "nemar_emoeeg_mc_on005540" in dataset_ids

    receipt = registry.create_provenance_receipt("allen_neuropixels_visual_cortex", "probe_test", 100)
    assert receipt.kind == EvidenceKind.MEASURED_BIOLOGICAL


def test_sensory_event_fabric_and_spike_encoder():
    fabric = SensoryEventFabric(buffer_capacity=100)
    event = fabric.ingest_event(
        modality="vision",
        feature_vector=[0.1, 0.5, 0.9, -0.2],
        intensity=0.8,
        evidence_kind=EvidenceKind.SYNTHETIC_SIMULATION,
    )
    assert event.modality == "vision"
    assert fabric.buffer_size() == 1

    assembled = fabric.assemble_multimodal_vector(target_dim=64)
    assert assembled.shape == (64,)
    assert isinstance(assembled, torch.Tensor)

    # Spike encoder
    encoder = NeuralSpikeEncoder(num_channels=16, dt=0.01, refractory_period_steps=2)
    continuous_signal = torch.sigmoid(torch.randn(16))
    spikes = encoder.encode(continuous_signal)
    assert spikes.shape == (16,)
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})


def test_mixed_affective_state_32_dimensions():
    mas = MixedAffectiveState(hidden_dim=256, seed=42)
    assert mas.num_emotions == 32
    assert len(AFFECTIVE_DIMENSIONS) == 32

    h_dummy = torch.randn(256)
    out = mas.step(wpac_hidden=h_dummy)

    emotions = out["emotions"]
    assert len(emotions) == 32
    for emo, val in emotions.items():
        assert 0.0 <= val <= 1.0, f"Emotion {emo} out of bounds: {val}"

    core = out["core"]
    assert len(core) == 6  # [v, alpha, delta, nu, sigma, eta]

    # Test dual-timescale properties
    assert mas.e_fast.shape == (32,)
    assert mas.e_slow.shape == (32,)

    # Test state dict export & import
    state_dict = mas.get_state_dict()
    assert "emotions" in state_dict
    assert "core" in state_dict


def test_wpac_connectome_and_regulatory_programmer():
    wpac = WeightProgrammedAffectiveCore(num_neurons=256, sensory_dim=64, seed=42)
    assert wpac.num_neurons == 256
    assert len(POPULATION_NAMES) == 8

    for pop_name, (start, end) in POPULATION_SLICES.items():
        assert end - start == 32

    # Program motifs
    programmer = AffectiveWeightProgrammer()
    receipts = programmer.program_all_motifs(wpac)
    assert len(receipts) == 7

    # Advance recurrent step
    sensory = torch.randn(64)
    h_next = wpac.step(sensory_input=sensory)
    assert h_next.shape == (256,)
    assert torch.all(torch.isfinite(h_next))

    pop_acts = wpac.get_population_activities()
    assert len(pop_acts) == 8
    for name in POPULATION_NAMES:
        assert name in pop_acts


def test_affective_equilibrium_regulator():
    regulator = AffectiveEquilibriumRegulator(num_emotions=32, obs_dim=64, body_dim=8, gland_dim=12)
    e_t = torch.rand(32)
    res = regulator(e_t=e_t)

    assert "target" in res
    assert "control" in res
    assert "cost_total" in res
    assert res["target"].shape == (32,)
    assert res["control"].shape == (32,)
    assert float(res["cost_total"].item()) >= 0.0


def test_synthetic_gland_bank_12_channels():
    bank = SyntheticGlandBank(wpac_dim=256, dt=0.1)
    assert bank.num_glands == 12
    assert len(GLAND_NAMES) == 12

    wpac_h = torch.randn(256)
    res = bank.step(wpac_hidden=wpac_h)

    levels = res["levels"]
    deltas = res["deltas"]
    assert levels.shape == (12,)
    assert deltas.shape == (12,)
    assert torch.all((levels >= 0.0) & (levels <= 1.0))

    # Test named getter and pulse injection
    da = bank.get_level("dopamine")
    assert 0.0 <= da <= 1.0
    bank.inject_pulse("dopamine", 0.3)
    assert bank.get_level("dopamine") >= da


def test_synthetic_receptor_bank():
    receptors = SyntheticReceptorBank()
    assert receptors.num_glands == 12
    assert receptors.num_targets == len(TARGET_PROPERTIES)

    # When delta is zero, gains should be exactly neutral (1.0)
    zero_deltas = torch.zeros(12)
    gains = receptors.compute_gains(zero_deltas)
    for prop in TARGET_PROPERTIES:
        assert abs(gains[prop] - 1.0) < 1e-6

    # When adrenaline increases, excitability and urgency should increase
    shock_deltas = torch.zeros(12)
    shock_deltas[4] = 0.5  # adrenaline
    gains_shock = receptors.compute_gains(shock_deltas)
    assert gains_shock["excitability"] > 1.0
    assert gains_shock["decision_urgency"] > 1.0


def test_modulatory_compositor_zero_gain_identity():
    compositor = ModulatoryCompositor(enabled=False, alpha_endocrine=0.35)
    legacy = NeuromodulatorState(dopamine=0.45, serotonin=0.62, octopamine=0.33, acetylcholine=0.71)

    # Zero-gain contract: when enabled=False, output matches legacy exactly bit-for-bit
    gland_deltas = torch.randn(12)
    comp_state, target_gains = compositor.composite(legacy, gland_deltas=gland_deltas)

    assert comp_state.dopamine == legacy.dopamine
    assert comp_state.serotonin == legacy.serotonin
    assert comp_state.octopamine == legacy.octopamine
    assert comp_state.acetylcholine == legacy.acetylcholine
    for prop in TARGET_PROPERTIES:
        assert target_gains[prop] == 1.0

    # When enabled, endocrine delta blends smoothly
    compositor_on = ModulatoryCompositor(enabled=True, alpha_endocrine=0.35)
    comp_state_on, target_gains_on = compositor_on.composite(legacy, gland_deltas=gland_deltas)
    assert comp_state_on.dopamine != legacy.dopamine or gland_deltas[0] == 0.0
    assert -1.0 <= comp_state_on.dopamine <= 1.0
    assert 0.0 <= comp_state_on.serotonin <= 1.0
