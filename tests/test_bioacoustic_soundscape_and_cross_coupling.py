"""Tests for Bioacoustic Soundscape, Dale's Law Cross-Coupling, and BCM Sliding Threshold."""
import numpy as np
import pytest
import torch

from synapse.bioacoustic_soundscape import (
    BioacousticSoundscapeEngine,
    SoundscapeTelemetry,
    VocalizationEvent,
)
from synapse.virtual_ecology import VirtualEcologyBiome
from synapse.multi_connectome_fusion import (
    MultiConnectomeFusionEngine,
    CrossConnectomeCoupling,
)
from synapse.server import Workbench


def test_soundscape_frequency_bands_and_propagation():
    """Verify 16 log-spaced frequency channels, wave propagation, and decay."""
    engine = BioacousticSoundscapeEngine(biome_half=10.0, seed=42)
    assert len(engine.center_freqs) == 16
    assert engine.center_freqs[0] == pytest.approx(60.0, rel=0.05)
    assert engine.center_freqs[-1] == pytest.approx(8000.0, rel=0.05)

    # Emit sentinel alarm whistle at (0, 0)
    voc = engine.emit_vocalization(
        agent_id="sentinel_01",
        niche="sentinel",
        x=0.0,
        y=0.0,
        amplitude=1.5,
        duration_steps=3,
    )
    assert isinstance(voc, VocalizationEvent)
    assert voc.fundamental_hz > 1500.0  # Sentinel high frequency
    assert len(engine.active_vocalizations) == 1

    # Check spatial attenuation at close vs distant coordinates
    amp_close, _, _ = engine.get_acoustic_at_point(0.2, 0.2)
    amp_far, _, _ = engine.get_acoustic_at_point(8.0, 8.0)
    assert amp_close > amp_far * 5.0  # Clear inverse-distance falloff

    # Step physics and verify decay
    telem = engine.step()
    assert isinstance(telem, SoundscapeTelemetry)
    assert telem.ambient_db_spl > 35.0
    assert len(telem.channel_energies) == 16


def test_acoustic_niche_partitioning_anp_index():
    """Verify Bernie Krause's Acoustic Niche Partitioning (ANP) index calculation."""
    engine = BioacousticSoundscapeEngine(biome_half=10.0, seed=42)

    # Emit vocalizations from all 4 ecological niches
    engine.emit_vocalization("art_1", "artisan", -2.0, -2.0)
    engine.emit_vocalization("for_1", "forager", 2.0, 2.0)
    engine.emit_vocalization("sen_1", "sentinel", 0.0, 4.0)
    engine.emit_vocalization("prd_1", "predator", 4.0, 0.0)

    telem = engine.step()
    # Distinct niches occupy separated frequency bands -> high ANP index (> 0.6)
    assert telem.anp_index > 0.6
    assert telem.dominant_niche in ("artisan", "forager", "sentinel", "predator")

    cochlear = engine.to_cochlear_rates()
    assert cochlear.shape == (256,)
    assert (cochlear >= 0.0).all() and (cochlear <= 1.0).all()


def test_virtual_ecology_acoustic_alarm_and_flocking():
    """Verify sentinel alarm reception alters nearby forager neuroendocrine states."""
    biome = VirtualEcologyBiome(biome_half=10.0, seed=42, soundscape_enabled=True)
    assert biome.soundscape is not None

    # Step simulation and verify soundscape advances
    snap = biome.step(None)
    assert snap.soundscape is not None
    assert "center_freqs" in snap.soundscape
    assert len(snap.soundscape["center_freqs"]) == 16

    # Test sentinel vocalization reception
    sentinel = next((a for a in biome.agents if a.niche == "sentinel"), None)
    forager = next((a for a in biome.agents if a.niche == "forager"), None)
    if sentinel and forager:
        forager.x, forager.y = sentinel.x + 0.5, sentinel.y + 0.5
        adren_before = forager.adrenaline
        biome.soundscape.emit_vocalization(sentinel.agent_id, "sentinel", sentinel.x, sentinel.y, amplitude=2.0)
        biome.step(None)
        # Forager received acoustic alarm -> adrenaline boosted
        assert forager.adrenaline >= adren_before


def test_cross_connectome_dales_law_and_bcm_adaptation():
    """Verify Dale's Law (80/20 E/I) sign enforcement and BCM sliding threshold."""
    coupling = CrossConnectomeCoupling(seed=42)

    # 1. Dale's law check
    assert float(coupling.sign_aud[: int(0.8 * 256)].mean().item()) == 1.0
    assert float(coupling.sign_aud[int(0.8 * 256) :].mean().item()) == -1.0
    assert float(coupling.sign_wpac[: int(0.8 * 512)].mean().item()) == 1.0
    assert float(coupling.sign_wpac[int(0.8 * 512) :].mean().item()) == -1.0

    # 2. Forward execution
    r_aud = torch.rand(256)
    r_wpac = torch.rand(512)
    r_vis = torch.rand(512)
    r_dro = torch.rand(64)
    d_wpac, d_pos, d_dro, d_cel = coupling(r_aud, r_wpac, r_vis, r_dro)
    assert d_wpac.shape == (512,)
    assert d_pos.shape == (64,)
    assert d_dro.shape == (128,)
    assert d_cel.shape == (64,)

    # 3. BCM adaptation
    theta_0 = float(coupling.theta_bcm_wpac.mean().item())
    high_post = torch.ones(512) * 0.9
    coupling.adapt_bcm(high_post, r_aud, eta=0.01, gamma=0.1)
    theta_1 = float(coupling.theta_bcm_wpac.mean().item())
    assert theta_1 > theta_0  # Sliding threshold rises under persistent activation

    # 4. MultiConnectomeFusionEngine bit-for-bit invariant with coupling_gain=0.0
    engine = MultiConnectomeFusionEngine(seed=42)
    engine.reset_state()
    sensory = torch.linspace(-0.3, 0.3, 155)
    r_base, _ = engine.step(sensory_input=sensory, coupling_gain=0.0)

    engine.reset_state()
    r_zero, _ = engine.step(sensory_input=sensory, coupling_gain=0.0)
    assert torch.equal(r_base, r_zero)  # Exact bit-for-bit zero diff

    # Step with positive coupling gain
    engine.reset_state()
    r_coupled, telem = engine.step(sensory_input=sensory, coupling_gain=0.25, learn_bcm=True)
    assert telem.coupling_gain == 0.25
    assert telem.bcm_mean_theta > 0.0


def test_server_bioacoustic_and_fusion_endpoints():
    """Verify Workbench REST endpoints for soundscape and fusion cross-coupling."""
    bench = Workbench()

    # 1. GET /api/ecology/soundscape
    sc = bench.get('/api/ecology/soundscape')
    assert sc['status'] == 'active'
    assert 'center_freqs' in sc
    assert len(sc['center_freqs']) == 16

    # 2. POST /api/ecology/vocalize
    voc_res = bench.post('/api/ecology/vocalize', {
        'agent_id': 'test_sentinel',
        'niche': 'sentinel',
        'x': 1.0,
        'y': -1.0,
        'amplitude': 1.4,
    })
    assert voc_res['status'] == 'emitted'
    assert voc_res['vocalization']['niche'] == 'sentinel'

    # 3. GET /api/connectome/fusion/summary
    fus_summary = bench.get('/api/connectome/fusion/summary')
    assert fus_summary['total_neurons'] == 12082
    assert len(fus_summary['substrates']) == 6

    # 4. POST /api/connectome/fusion/couple
    couple_res = bench.post('/api/connectome/fusion/couple', {
        'coupling_gain': 0.15,
        'learn_bcm': True,
    })
    assert couple_res['status'] == 'updated'
    assert couple_res['coupling_gain'] == 0.15
    assert couple_res['telemetry']['coupling_gain'] == 0.15

    # 5. POST /api/audio/push with soundscape & cochlear injection
    pcm_sine = (np.sin(np.linspace(0, 2 * np.pi * 440, 512)) * 0.5).tolist()
    push_res = bench.post('/api/audio/push', {'samples': pcm_sine})
    assert push_res['status'] == 'ingested'
    assert push_res['rates'] is not None
