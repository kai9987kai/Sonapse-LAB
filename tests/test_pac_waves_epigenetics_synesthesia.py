"""Tests for PAC (Phase-Amplitude Coupling), Epigenetics, and Synesthetic Resonance.

Validates:
1. PhaseAmplitudeCouplingEngine: Tort Modulation Index, Kuramoto wave coherence, primary invariant.
2. EpigeneticRemodelingEngine: stress-induced chromatin remodeling, transgenerational inheritance, zero-regression invariant.
3. SynestheticResonanceEngine: pairwise PLV calculation, Cross-Modal Binding Index (CMBI), and resonance pulses.
4. MultiConnectomeFusionEngine: PAC-gated 12k connectome ODE integration.
5. Workbench Server: REST endpoints for PAC, Epigenetics, and Synesthesia.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from synapse.pac_waves import PhaseAmplitudeCouplingEngine, PACTelemetry
from synapse.epigenetics import EpigeneticRemodelingEngine, EpigeneticProfile
from synapse.synesthetic_resonance import SynestheticResonanceEngine, SynestheticTelemetry
from synapse.multi_connectome_fusion import MultiConnectomeFusionEngine
from synapse.virtual_ecology import VirtualEcologyBiome
from synapse.server import Workbench


def test_pac_engine_tort_modulation_and_invariants():
    engine = PhaseAmplitudeCouplingEngine(num_substrates=6, theta_freq_hz=6.0, seed=42)

    # 1. Primary Invariant: when pac_gain == 0.0, gating must be exactly 1.0 for all substrates
    phases = engine.advance_phase()
    assert phases.shape == (6,)
    for p in phases:
        assert -math.pi <= float(p.item()) <= math.pi

    gating_zero = engine.calculate_gating(phases, pac_gain=0.0)
    assert torch.allclose(gating_zero, torch.ones_like(phases))

    # 2. When pac_gain > 0.0, gating oscillates around 1.0 with amplitude <= pac_gain
    gating_active = engine.calculate_gating(phases, pac_gain=0.3)
    assert not torch.allclose(gating_active, torch.ones_like(phases))
    assert (gating_active >= 0.7).all() and (gating_active <= 1.3).all()

    # 3. Simulate phase-locked activity and verify Tort Modulation Index > 0
    engine.reset_state()
    # Feed modulated samples: high gamma when theta phase near 0 (cos(phi) > 0.5)
    for step in range(120):
        sub_phases = engine.advance_phase()
        phi = float(sub_phases[3].item())
        # Amplitude strongly concentrated near phase 0
        gamma_amp = 0.1 + 0.9 * max(0.0, math.cos(phi))
        engine.record_step(phi, gamma_amp)

    mi, pref_phase, p_dist = engine.compute_tort_modulation_index()
    assert 0.0 < mi <= 1.0
    assert len(p_dist) == 18
    assert math.isclose(sum(p_dist), 1.0, rel_tol=1e-3)
    assert -math.pi <= pref_phase <= math.pi

    # 4. Kuramoto wave coherence
    coh = engine.compute_wave_coherence(sub_phases)
    assert 0.0 <= coh <= 1.0


def test_epigenetic_remodeling_and_transgenerational_inheritance():
    engine = EpigeneticRemodelingEngine(heritability=0.7, remodeling_rate=0.1, modulation_scale=0.25)
    genome = {"speed": 1.2, "sense": 1.0, "metabolism": 0.8, "vocal_freq": 200.0, "vocal_amp": 1.0}

    # 1. Primary Invariant: when profile is None or disabled, expressed equals genome exactly
    expr_disabled = engine.express_phenotype(genome, None, enabled=False)
    assert expr_disabled == genome

    # 2. Baseline profile produces zero net modification
    default_prof = engine.default_profile()
    expr_baseline = engine.express_phenotype(genome, default_prof, enabled=True)
    for k in genome:
        assert math.isclose(expr_baseline[k], genome[k], rel_tol=1e-5)

    # 3. Acute predation stress acetylates speed & sense, methylates metabolism
    engine.update_from_stress(default_prof, cortisol=0.8, adrenaline=0.9, starvation_stress=0.0)
    assert default_prof.acetylation["speed"] > 0.1
    assert default_prof.acetylation["sense"] > 0.1
    assert default_prof.methylation["metabolism"] > 0.1
    assert default_prof.plasticity_index > 0.0

    # Expressed phenotype under stress: speed boosted, metabolism conserved
    expr_stressed = engine.express_phenotype(genome, default_prof, enabled=True)
    assert expr_stressed["speed"] > genome["speed"]
    assert expr_stressed["sense"] > genome["sense"]
    assert expr_stressed["metabolism"] < genome["metabolism"]

    # 4. Transgenerational Inheritance across generations
    rng = np.random.default_rng(123)
    child_prof = engine.inherit_epigenetics(default_prof, rng)
    assert child_prof.epigenetic_generation == 1
    # Child should retain a high fraction (heritability=0.7) of parent's marks
    assert child_prof.acetylation["speed"] > 0.1
    assert child_prof.acetylation["sense"] > 0.1


def test_synesthetic_resonance_and_cross_modal_binding():
    engine = SynestheticResonanceEngine(window_size=32, binding_threshold=0.5)

    # 1. Zero state produces zero binding
    summary = engine.get_summary()
    assert summary["cross_modal_binding_index"] == 0.0

    # 2. Feed synchronized multimodal signals (correlated phase)
    t = torch.linspace(0, 4 * math.pi, 40)
    for val in t:
        sin_val = math.sin(float(val))
        cos_val = math.cos(float(val))
        # High phase coherence across vision, audio, and endocrine
        vis = torch.full((64,), sin_val)
        aud = torch.full((64,), sin_val)
        endo = torch.full((12,), sin_val)
        pulse, telem = engine.step(vis, aud, endo, resonance_gain=0.5)

    assert 0.0 <= telem.cross_modal_binding_index <= 1.0
    assert 0.0 <= telem.visual_auditory_plv <= 1.0
    assert 0.0 <= telem.visual_endocrine_plv <= 1.0
    assert 0.0 <= telem.auditory_endocrine_plv <= 1.0


def test_multi_connectome_fusion_with_pac():
    fusion = MultiConnectomeFusionEngine(seed=42)
    assert fusion.pac_gain == 0.0

    # Invariant: step with pac_gain == 0.0
    r_base, telem_base = fusion.step(pac_gain=0.0)
    assert telem_base.pac_modulation_index >= 0.0
    assert telem_base.pac_wave_coherence >= 0.0

    # Enable PAC
    fusion.enable_pac(pac_gain=0.3)
    assert fusion.pac_gain == 0.3
    r_pac, telem_pac = fusion.step()
    assert r_pac.shape == (12082,)
    assert (r_pac >= 0.0).all() and (r_pac <= 1.0).all()

    # Disable PAC restores pac_gain to 0.0
    fusion.disable_pac()
    assert fusion.pac_gain == 0.0


def test_virtual_ecology_epigenetics_integration():
    biome = VirtualEcologyBiome(biome_half=8.0, seed=101, evolution_enabled=True, epigenetics_enabled=True)
    assert biome.epigenetics_enabled is True

    # Check all agents received an epigenetic profile
    for ag in biome.agents:
        assert ag.epigenetic_profile is not None

    stats_pre = biome.epigenetics_stats()
    assert stats_pre["count"] == len(biome.agents)
    assert "speed" in stats_pre["traits"]

    # Step simulation under predator panic to trigger chromatin remodeling
    biome.step({"type": "trigger_predator_wave"})
    for _ in range(10):
        biome.step(None)

    stats_post = biome.epigenetics_stats()
    assert stats_post["count"] > 0
    # Snapshot contains epigenetic profile
    snap = biome.snapshot()
    assert any("epigenetic_profile" in a for a in snap["agents"])


def test_server_pac_epigenetics_synesthesia_endpoints():
    wb = Workbench()

    # 1. GET /api/connectome/pac
    res_pac = wb.get("/api/connectome/pac")
    assert "modulation_index" in res_pac
    assert "num_phase_bins" in res_pac

    # 2. POST /api/connectome/pac/enable
    res_pac_en = wb.post("/api/connectome/pac/enable", {"gain": 0.35})
    assert res_pac_en["status"] == "pac_configured"
    assert math.isclose(res_pac_en["pac_gain"], 0.35, rel_tol=1e-3)

    # 3. GET /api/ecology/epigenetics
    res_epi = wb.get("/api/ecology/epigenetics")
    assert "epigenetics_enabled" in res_epi

    # 4. POST /api/ecology/epigenetics/enable
    res_epi_en = wb.post("/api/ecology/epigenetics/enable", {"heritability": 0.75})
    assert res_epi_en["status"] == "enabled"
    assert math.isclose(res_epi_en["heritability"], 0.75, rel_tol=1e-3)

    # 5. GET /api/multimodal/synesthesia
    res_syn = wb.get("/api/multimodal/synesthesia")
    assert "cross_modal_binding_index" in res_syn

    # 6. POST /api/multimodal/synesthesia/burst
    res_syn_burst = wb.post("/api/multimodal/synesthesia/burst", {})
    assert res_syn_burst["status"] == "synesthetic_burst_triggered"
    assert "pulse" in res_syn_burst
    assert "cross_modal_binding_index" in res_syn_burst
