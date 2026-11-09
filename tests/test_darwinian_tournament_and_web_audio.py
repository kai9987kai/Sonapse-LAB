"""Tests for Darwinian Ecology Tournament Engine, Speciation ANOVA F-Divergence, and Web Audio Streaming Ingress."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.darwinian_tournament import (
    DarwinianEcologyTournamentEngine,
    DarwinianTournamentReceipt,
    LineageNode,
    SpeciationMetrics,
)
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.virtual_ecology import VirtualEcologyBiome


def test_darwinian_tournament_initialization():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=1,
        seed=101,
        evolution_enabled=True,
    )
    engine = DarwinianEcologyTournamentEngine(biome=biome, seed=101)

    assert len(engine.lineage_tree) == 9
    assert len(engine.known_agent_ids) == 9

    for aid, node in engine.lineage_tree.items():
        assert isinstance(node, LineageNode)
        assert node.agent_id == aid
        assert node.parent_id is None
        assert node.generation == 0
        assert node.birth_step == 0
        assert "speed" in node.genome
        assert "sense" in node.genome


def test_darwinian_tournament_step_and_lotka_volterra_series():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=3,
        num_sentinels=1,
        num_artisans=1,
        num_predators=1,
        seed=202,
        evolution_enabled=True,
    )
    engine = DarwinianEcologyTournamentEngine(biome=biome, seed=202)

    # Step tournament 25 steps
    snap = engine.step_tournament(steps=25)
    assert snap.step_count == 25
    assert len(engine.population_series) == 25

    last_pt = engine.population_series[-1]
    assert last_pt["step"] == 25
    assert "counts" in last_pt
    assert last_pt["counts"]["forager"] >= 0
    assert last_pt["counts"]["predator"] >= 0
    assert last_pt["biomass"] > 0.0


def test_speciation_divergence_anova_f_stat():
    biome = VirtualEcologyBiome(
        biome_half=10.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=2,
        seed=303,
        evolution_enabled=True,
    )
    engine = DarwinianEcologyTournamentEngine(biome=biome, seed=303)
    engine.step_tournament(steps=15)

    metrics = engine.compute_speciation_divergence()
    assert isinstance(metrics, SpeciationMetrics)
    assert metrics.divergence_index >= 0.0
    assert "speed" in metrics.trait_f_ratios
    assert "sense" in metrics.trait_f_ratios
    assert metrics.shannon_entropy >= 0.0
    assert metrics.active_lineages > 0

    m_dict = metrics.to_dict()
    assert "divergence_index" in m_dict
    assert "niche_mean_traits" in m_dict


def test_cataclysm_bottleneck_and_migrant_recovery():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=1,
        seed=404,
        evolution_enabled=True,
    )
    engine = DarwinianEcologyTournamentEngine(biome=biome, seed=404)

    # 1. Drought
    drought_res = engine.inject_cataclysm("drought")
    assert drought_res["status"] == "injected"
    assert drought_res["cataclysm"]["type"] == "drought"

    # 2. Arctic freeze
    freeze_res = engine.inject_cataclysm("arctic_freeze")
    assert freeze_res["status"] == "injected"
    assert freeze_res["cataclysm"]["type"] == "arctic_freeze"
    assert biome.temperature_c == 4.0

    # 3. Predator surge
    pred_res = engine.inject_cataclysm("predator_bloom")
    assert pred_res["status"] == "injected"
    assert pred_res["cataclysm"]["type"] == "predator_bloom"

    # Advance steps after shock to verify recovery
    snap = engine.step_tournament(steps=10)
    assert snap.population_count > 0


def test_multi_epoch_tournament_receipt_and_evidence():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=3,
        num_sentinels=1,
        num_artisans=1,
        num_predators=1,
        seed=505,
        evolution_enabled=True,
    )
    engine = DarwinianEcologyTournamentEngine(biome=biome, seed=505)

    receipt = engine.run_tournament_epochs(epochs=3, steps_per_epoch=15)
    assert isinstance(receipt, DarwinianTournamentReceipt)
    assert receipt.receipt_id.startswith("darw_rcpt_")
    assert receipt.total_steps == 45
    assert receipt.evidence.item_id == receipt.receipt_id
    assert "divergence_index" in receipt.evidence.metrics

    status = engine.get_status()
    assert status["status"] == "active"
    assert status["step_count"] == 45
    assert status["history_receipts_count"] == 1
    assert status["last_receipt"] is not None


def test_server_darwinian_and_audio_stream_endpoints(tmp_path):
    cfg = SynapseConfig(total_neurons=8000, seed=606)
    rt = LabRuntime(cfg, output_root=tmp_path)
    wb = Workbench(rt, output_root=tmp_path)

    # 1. GET /api/ecology/tournament/status
    t_status = wb.get('/api/ecology/tournament/status')
    assert t_status['status'] == 'active'
    assert 'speciation_metrics' in t_status
    assert 'recent_series' in t_status

    # 2. POST /api/ecology/tournament/cataclysm
    cat_res = wb.post('/api/ecology/tournament/cataclysm', {'type': 'drought'})
    assert cat_res['status'] == 'injected'
    assert cat_res['cataclysm']['type'] == 'drought'

    # 3. POST /api/ecology/tournament/run
    run_res = wb.post('/api/ecology/tournament/run', {'epochs': 2, 'steps_per_epoch': 10})
    assert run_res['status'] == 'completed'
    assert 'receipt_id' in run_res
    assert 'metrics' in run_res

    # 4. POST /api/audio/push with multi-tone audio buffer and check rates array
    pcm = [0.4 * np.sin(2 * np.pi * 500 * (i / 16000)) for i in range(512)]
    audio_res = wb.post('/api/audio/push', {'samples': pcm})
    assert audio_res['status'] == 'ingested'
    assert 'features' in audio_res
    assert 'rates' in audio_res
    assert len(audio_res['rates']) == 16
    assert audio_res['mean_rate_hz'] > 0.0
