"""Tests for Measured Circuit Evolution Sandbox, Live Audio Ingress, SWR Live Taps, and Heritable Virtual Ecology."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.evo_compat.circuit_evolution import CausalCircuitEvolutionEngine
from synapse.evo_compat.sandbox_evaluator import RateCircuitSandbox
from synapse.multi_connectome_fusion import MultiConnectomeFusionEngine
from synapse.multimodal_live import AuditorySpikeEncoder, LiveAudioGateway
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.sleep_replay.consolidation_engine import (
    LiveEpisodeRecorder,
    MemoryConsolidationAndSHYEngine,
)
from synapse.virtual_ecology import VirtualEcologyBiome


def test_measured_circuit_evolution_and_sandbox():
    sb = RateCircuitSandbox(num_nodes=32, seed=42)
    assert sb.num_nodes == 32
    assert sb.dales_law_ok() is True
    assert sb.status()["dales_law_ok"] is True

    eng = CausalCircuitEvolutionEngine(seed=42)
    # Evaluate a cycle with 3 trials using paired-seed measurement
    res = eng.run_measured_evolution_cycle(sb, trials=3, num_seeds=4, require_positive_lcb=True)

    assert res["status"] == "completed"
    assert res["mode"] == "measured"
    assert res["evaluated_candidates"] == 3
    assert "sandbox" in res
    assert res["sandbox"]["dales_law_ok"] is True

    for r in res["results"]:
        assert "delta_reward" in r
        assert "delta_reward_se" in r
        assert "net_utility" in r
        assert "promoted" in r
        assert "measurement" in r


def test_runtime_transition_taps_and_live_episode_recorder(tmp_path):
    cfg = SynapseConfig(total_neurons=8000, seed=123)
    rt = LabRuntime(cfg, output_root=tmp_path)
    engine = MemoryConsolidationAndSHYEngine(seed=123)
    recorder = LiveEpisodeRecorder(engine, max_len=8)

    rt.transition_taps = [recorder]

    # Step the runtime with learning to generate transitions
    for _ in range(12):
        rt.step(steps=1, learn=True)

    assert recorder.transitions_seen == 12
    # At least one episode was flushed because max_len=8
    assert recorder.episodes_flushed >= 1
    assert len(engine.episode_buffer) >= 1

    # Now verify SWR replay replays genuine recorded steps
    replayed = engine.execute_swr_replay(max_episodes=2)
    assert replayed > 0

    # Test empirical reward model fitting on recorded transitions
    model = engine.fit_empirical_reward_model(min_transitions=4)
    assert model is not None
    assert model["n_transitions"] >= 4
    assert "holdout_r2" in model

    # Predict empirical reward on an observation
    ref_obs = engine.episode_buffer[0].observations[0]
    ref_act = engine.episode_buffer[0].actions[0]
    pred = engine.predict_empirical_reward(ref_obs, ref_act)
    assert pred is not None
    assert np.isfinite(pred)


def test_live_audio_gateway_and_fusion_connectome_ingress():
    sr = 16000
    gw = LiveAudioGateway(sample_rate_hz=sr, enabled=True)
    enc = AuditorySpikeEncoder(num_channels=16, seed=42)
    fusion = MultiConnectomeFusionEngine(seed=42)

    # Synthesize 440 Hz sinusoidal audio burst
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sine_pcm = 0.6 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    gw.ingest_samples(sine_pcm)
    features = gw.process_latest_frame()
    assert features is not None
    assert features.rms_energy > 0.1
    assert len(features.mel_energies) == 16

    spikes, rates = enc.encode(features)
    assert len(spikes) == 16
    assert len(rates) == 16
    assert np.sum(spikes) >= 0

    # Ingress feature vector into 12,082-neuron fusion connectome
    feat_vec = torch.from_numpy(enc.to_feature_vector()).float()
    new_r, tel = fusion.step(sensory_input=feat_vec)
    assert new_r.shape[0] == 12082
    assert tel.total_neurons == 12082
    assert tel.dales_law_compliance == 1.0


def test_virtual_ecology_heritable_evolution_and_stats():
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=3,
        num_sentinels=2,
        num_artisans=1,
        num_predators=1,
        seed=77,
        evolution_enabled=True,
    )
    assert biome.evolution_enabled is True
    # All initial agents must receive heritable genomes
    for ag in biome.agents:
        assert ag.genome is not None
        assert "speed" in ag.genome
        assert "sense" in ag.genome

    # Advance biome 30 steps
    for _ in range(30):
        biome.step()

    stats = biome.evolution_stats()
    assert stats["evolution_enabled"] is True
    assert stats["step_count"] == 30
    assert "niches" in stats
    assert stats["niches"]["forager"]["count"] > 0

    # Non-mutating snapshot check
    snap1 = biome.snapshot()
    snap2 = biome.snapshot()
    assert snap1["step_count"] == snap2["step_count"]
    assert snap1["population_count"] == snap2["population_count"]


def test_server_new_endpoints_live(tmp_path):
    cfg = SynapseConfig(total_neurons=8000, seed=55)
    rt = LabRuntime(cfg, output_root=tmp_path)
    wb = Workbench(rt, output_root=tmp_path)

    # 1. GET /api/ecology/stats
    stats = wb.get('/api/ecology/stats')
    assert 'evolution_enabled' in stats
    assert 'niches' in stats

    # 2. POST /api/ecology/evolution
    res_evo = wb.post('/api/ecology/evolution', {'action': 'enable'})
    assert res_evo['status'] == 'enabled'
    assert res_evo['stats']['evolution_enabled'] is True

    # 3. POST /api/audio/push
    pcm_samples = (0.3 * np.sin(np.linspace(0, 10, 256))).tolist()
    res_audio = wb.post('/api/audio/push', {'samples': pcm_samples})
    assert res_audio['status'] == 'ingested'
    assert 'features' in res_audio

    # 4. POST /api/evo/transact (measured mode)
    res_tx = wb.post('/api/evo/transact', {'mode': 'measured', 'trials': 2, 'num_seeds': 3})
    assert res_tx['status'] == 'completed'
    assert res_tx['mode'] == 'measured'
    assert len(res_tx['results']) == 2

    # 5. POST /api/sleep/dream_cycle
    res_sleep = wb.post('/api/sleep/dream_cycle', {})
    assert res_sleep['status'] == 'completed'
    assert 'percent_downscaled' in res_sleep
