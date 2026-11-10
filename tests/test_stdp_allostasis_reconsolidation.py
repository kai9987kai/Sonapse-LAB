"""Tests for Spike-Timing-Dependent Plasticity (STDP), Allostatic Load Index (ALI),
and Memory Reconsolidation & Labile State Engine.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from synapse.stdp import (
    SpikeTimingPlasticityEngine,
    STDPConfig,
    STDPTelemetry,
)
from synapse.allostatic_load import (
    AllostaticLoadEngine,
    AllostaticProfile,
    AllostaticPopulationStats,
)
from synapse.reconsolidation import (
    MemoryReconsolidationEngine,
    LabileMemoryTrace,
    ReconsolidationTelemetry,
)
from synapse.multi_connectome_fusion import (
    MultiConnectomeFusionEngine,
    CrossConnectomeCoupling,
)
from synapse.virtual_ecology import (
    VirtualEcologyBiome,
    EcologyAgentState,
)
from synapse.server import Workbench


def test_stdp_engine_properties_and_primary_invariant():
    """Verifies Bi & Poo LTP/LTD dynamics, Dale's Law compliance, and legacy invariants."""
    engine = SpikeTimingPlasticityEngine(n_pre=8, n_post=8, seed=42)

    # 1. Primary Invariant: When learn=False, weights remain bit-for-bit identical
    w_init = torch.full((8, 8), 0.25, dtype=torch.float32)
    w_orig = w_init.clone()
    s_pre = torch.ones(8)
    s_post = torch.ones(8)
    w_out, telem = engine.step(s_pre, s_post, w_init, learn=False, stdp_gain=0.0)

    assert torch.equal(w_out, w_orig), "Weights must not be altered when learn=False"
    assert telem.mean_dw == 0.0
    assert telem.weight_stability == 1.0

    # 2. Pre-before-post timing causes LTP (Delta t > 0)
    engine.reset_state()
    # Step 1: Pre-synaptic spike arrives first
    engine.step(s_pre=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                s_post=torch.zeros(8),
                weights=w_orig,
                learn=False)
    # Step 2: Post-synaptic spike arrives while pre trace is high
    w_plastic, telem_ltp = engine.step(
        s_pre=torch.zeros(8),
        s_post=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        weights=w_orig,
        learn=True,
        stdp_gain=1.0,
    )
    assert telem_ltp.dw_ltp_sum > 0.0
    # Synapse (0, 0) should be potentiated
    assert w_plastic[0, 0] > w_orig[0, 0]

    # 3. Dale's Law projection enforcement
    signs = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    w_proj, telem_dale = engine.step(
        s_pre=torch.ones(8),
        s_post=torch.ones(8),
        weights=w_plastic,
        presynaptic_signs=signs,
        learn=True,
        stdp_gain=1.0,
    )
    # Excitatory columns (0..3) must be >= 0
    assert (w_proj[:, :4] >= 0.0).all()
    # Inhibitory columns (4..7) must be <= 0
    assert (w_proj[:, 4:] <= 0.0).all()
    assert telem_dale.compliance_rate == 1.0


def test_allostatic_load_engine_accumulation_and_overload():
    """Verifies McEwen allostatic wear-and-tear and allostatic overload thresholds."""
    engine = AllostaticLoadEngine(overload_threshold=0.60, seed=42)

    # Initial profile
    prof = engine.get_or_create_profile("test_agent_1", generation=0)
    assert prof.allostatic_load_index == 0.0
    assert not prof.is_overloaded

    # Acute brief stress
    engine.update_agent(prof, cortisol=0.2, adrenaline=0.1, energy=80.0, speed_magnitude=0.5, dt=0.1)
    assert prof.allostatic_load_index > 0.0
    assert not prof.is_overloaded

    # Chronic sustained high stress (predator pursuit, starvation, panic)
    for _ in range(80):
        engine.update_agent(
            prof,
            cortisol=0.9,
            adrenaline=0.8,
            energy=15.0,  # Starvation
            speed_magnitude=2.5,  # Sprinting
            dt=0.1,
        )

    assert prof.allostatic_load_index >= 0.60, "Severe chronic stress must elevate ALI"
    assert prof.is_overloaded, "Overload threshold must be triggered"
    assert prof.hpa_exhaustion > 0.3, "Sustained cortisol must cause HPA axis exhaustion"

    # Generational transmission
    child = engine.inherit_allostasis(prof, "test_child_1", heritability=0.30)
    assert child.generation == 1
    assert child.cortisol_auc > 0.0, "Child must inherit baseline parental stress burden"
    assert child.cortisol_auc < prof.cortisol_auc

    # Population stats
    stats = engine.compute_population_stats()
    assert stats.total_tracked_agents == 2
    assert stats.overloaded_count == 1
    assert stats.overloaded_fraction == 0.5


def test_memory_reconsolidation_destabilization_and_update():
    """Verifies Nader & Sara memory reconsolidation boundary, updating, and decay."""
    engine = MemoryReconsolidationEngine(
        destabilization_threshold=0.25,
        extinction_threshold=0.85,
        labile_window_steps=20,
        reconsolidation_lr=0.20,
        seed=42,
    )

    dummy_obs = np.ones(10)
    dummy_act = np.zeros(3)

    # 1. Prediction error below threshold does not destabilize
    t1 = engine.reactivate_trace(
        trace_id="mem_stable",
        obs=dummy_obs,
        action=dummy_act,
        reward=1.0,
        importance=0.9,
        prediction_error=0.10,
        enabled=True,
    )
    assert t1 is None, "Sub-threshold retrieval must leave memory trace consolidated"

    # 2. Moderate prediction error (surprisal in [0.25, 0.85)) triggers labile state
    t2 = engine.reactivate_trace(
        trace_id="mem_labile",
        obs=dummy_obs,
        action=dummy_act,
        reward=1.0,
        importance=0.9,
        prediction_error=0.45,
        enabled=True,
    )
    assert t2 is not None, "Surprisal mismatch must destabilize trace into labile state"
    assert t2.lability == 1.0
    assert not t2.is_restabilized

    # 3. Massive prediction error triggers extinction, not destabilization
    t3 = engine.reactivate_trace(
        trace_id="mem_extinct",
        obs=dummy_obs,
        action=dummy_act,
        reward=1.0,
        importance=0.9,
        prediction_error=0.95,
        enabled=True,
    )
    assert t3 is None, "Extreme prediction error must trigger extinction pathway"
    assert engine.total_extinctions == 1

    # 4. Updating during labile window modifies memory trace
    novel_obs = np.zeros(10)
    novel_act = np.ones(3)
    updated = engine.update_labile_trace(
        trace_id="mem_labile",
        new_obs=novel_obs,
        new_action=novel_act,
        new_reward=0.0,
        enabled=True,
    )
    assert updated is not None
    updated_obs, updated_act, updated_rew = updated
    assert not np.array_equal(updated_obs, dummy_obs), "Observation should be updated with new evidence"
    assert updated_rew < 1.0, "Reward expectation should be updated downward"

    # 5. Step decay toward restabilization
    for _ in range(25):
        engine.step_decay(enabled=True)

    summary = engine.get_summary()
    assert summary["active_labile_count"] == 0, "Trace must re-stabilize after labile window expires"


def test_multi_connectome_fusion_with_stdp():
    """Verifies STDP plastic adaptation across 12,082 connectome bridge pathways."""
    fusion = MultiConnectomeFusionEngine(seed=42)
    assert fusion.total_dim >= 12000

    # 1. Step with coupling and STDP active
    fusion.enable_stdp(stdp_gain=1.0)
    r_sens = torch.randn(155)
    r_aud = torch.randn(256)

    rates, telem = fusion.step(
        sensory_input=r_sens,
        cochlear_input=r_aud,
        coupling_gain=0.20,
        learn_stdp=True,
    )
    assert rates.shape[0] == fusion.total_dim
    assert telem.stdp_dw >= 0.0
    assert telem.dales_law_compliance == 1.0

    # 2. Disable STDP: invariant check
    fusion.disable_stdp()
    _, telem_disabled = fusion.step(
        sensory_input=r_sens,
        cochlear_input=r_aud,
        coupling_gain=0.20,
        learn_stdp=False,
    )
    assert telem_disabled.stdp_dw == 0.0

    summary = fusion.get_summary()
    assert "stdp_gain" in summary
    assert "stdp_summary" in summary


def test_virtual_ecology_allostasis_integration():
    """Verifies allostatic load engine integration within VirtualEcologyBiome."""
    biome = VirtualEcologyBiome(
        biome_half=8.0,
        num_foragers=4,
        num_sentinels=2,
        num_artisans=2,
        num_predators=1,
        seed=42,
        allostatic_load_enabled=True,
    )
    assert biome.allostatic_load_enabled

    # Step biome multiple times to accumulate physiological wear
    snap = None
    for _ in range(15):
        snap = biome.step(None)

    assert snap is not None
    assert snap.allostasis is not None, "Snapshot must include allostasis summary when enabled"
    assert snap.allostasis["total_tracked_agents"] > 0
    assert "mean_ali" in snap.allostasis
    assert snap.allostasis["evidence_kind"] == "synthetic_simulation"

    stats = biome.allostasis_stats()
    assert stats["total_tracked_agents"] == len(biome.agents)


def test_server_stdp_allostasis_reconsolidation_endpoints():
    """Verifies all 6 REST endpoints for STDP, allostasis, and memory reconsolidation."""
    bench = Workbench()

    # GET /api/connectome/stdp
    res_stdp = bench.get('/api/connectome/stdp')
    assert 'mean_dw' in res_stdp
    assert 'compliance_rate' in res_stdp

    # POST /api/connectome/stdp/enable
    post_stdp = bench.post('/api/connectome/stdp/enable', {'gain': 0.75})
    assert post_stdp['status'] == 'stdp_configured'
    assert post_stdp['stdp_gain'] == 0.75

    # GET /api/ecology/allostasis
    res_allo = bench.get('/api/ecology/allostasis')
    assert 'mean_ali' in res_allo

    # POST /api/ecology/allostasis/enable
    post_allo = bench.post('/api/ecology/allostasis/enable', {})
    assert post_allo['status'] == 'enabled'
    assert bench.virtual_biome.allostatic_load_enabled

    # GET /api/sleep/reconsolidation
    res_recon = bench.get('/api/sleep/reconsolidation')
    assert 'active_labile_count' in res_recon

    # POST /api/sleep/reconsolidation/reactivate
    post_recon = bench.post('/api/sleep/reconsolidation/reactivate', {
        'trace_id': 'bench_test_trace_1',
        'prediction_error': 0.50,
    })
    assert post_recon['status'] == 'reactivated'
    assert post_recon['trace'] is not None
    assert post_recon['trace']['lability'] == 1.0
