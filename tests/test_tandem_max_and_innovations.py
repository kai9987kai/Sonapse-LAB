"""Tests for Dendritic LTC, Oscillatory Coherence, Frontier Worlds, and 5-Section Tandem Max Mode."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.dendritic_ltc import DendriticLiquidSubGraft
from synapse.oscillatory_coherence import OscillatoryCoherenceEngine
from synapse.dvgem_tether import DecisionValueEpisodicMemory, TetherSuccessorCoordinator, DVGEM_MODES
from synapse.complex_world import ComplexWorld, ALL_TASKS
from synapse.runtime import LabRuntime
from synapse.server import Workbench
from synapse.stateio import digest


def test_dendritic_ltc_subgraft_dynamics_and_frozen_invariance():
    graft = DendriticLiquidSubGraft(basal_dim=32, apical_dim=26, hidden_dim=16, seed=19)
    basal = torch.linspace(-0.5, 0.5, 32)
    apical = torch.linspace(0.5, -0.5, 26)

    # Zero-initialized readout produces zero residual initially
    res0, diag0 = graft.step(basal, apical, ach_precision=0.7, astrocyte_ca=0.4, learn=False)
    assert res0.shape == (32,)
    assert torch.allclose(res0, torch.zeros(32))
    assert 0.15 <= diag0['mean_tau'] <= 4.0

    # Frozen inference does not mutate internal state voltage or eligibility
    state_before = digest(graft.export_state())
    graft.step(basal, apical, ach_precision=0.9, astrocyte_ca=0.8, learn=False)
    assert digest(graft.export_state()) == state_before

    # Online learning advances state and applies reward modulation
    graft.step(basal, apical, ach_precision=0.8, astrocyte_ca=0.5, learn=True)
    upd = graft.update_from_outcome(reward=0.6, prediction_error=0.25)
    assert upd['updates'] == 1
    assert graft.updates == 1

    res1, _ = graft.step(basal, apical, ach_precision=0.8, astrocyte_ca=0.5, learn=False)
    assert float(torch.norm(res1).item()) > 0.0

    # Round-trip state export/import
    exported = graft.export_state()
    fresh = DendriticLiquidSubGraft(basal_dim=32, apical_dim=26, hidden_dim=16, seed=99)
    fresh.import_state(exported)
    assert digest(fresh.export_state()) == digest(exported)


def test_oscillatory_coherence_kuramoto_and_ctc_gain():
    osc = OscillatoryCoherenceEngine(source_names=('flywire', 'manc', 'optic-lobe'))
    rates = np.array([0.45, 0.62, 0.51, 0.30, 0.80, 0.25], dtype=np.float32)
    origins = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    res = osc.step(rates, origins, surprisal=0.3, ach=0.7, dopamine=0.8, learn=True)
    assert res['enabled'] is True
    assert 0.0 <= res['kuramoto_order_r'] <= 1.0
    assert 0.85 <= res['ctc_routing_gate'] <= 1.25
    assert len(res['coherence_matrix']) == 3

    # Frozen learn=False does not mutate oscillator phases or ticks
    before = digest(osc.export_state())
    osc.step(rates * 1.5, origins, surprisal=0.8, ach=0.9, dopamine=1.2, learn=False)
    assert digest(osc.export_state()) == before


def test_new_frontier_tasks_quantum_bifurcation_and_cybernetic_sanctuary():
    assert 'quantum_bifurcation' in ALL_TASKS
    assert 'cybernetic_sanctuary' in ALL_TASKS

    w_qb = ComplexWorld(seed=7, curriculum='quantum_bifurcation')
    obs_qb = w_qb.reset('quantum_bifurcation')
    assert obs_qb.shape == (155,)
    assert len(w_qb.dynamic_gates) == 2
    snap_qb = w_qb.snapshot()
    assert snap_qb['task'] == 'quantum_bifurcation'

    w_cs = ComplexWorld(seed=7, curriculum='cybernetic_sanctuary')
    obs_cs = w_cs.reset('cybernetic_sanctuary')
    assert obs_cs.shape == (155,)
    assert len(w_cs.helper_drones) >= 1
    snap_cs = w_cs.snapshot()
    assert len(snap_cs['helper_drones']) >= 1

    # Place agent inside helper drone shield radius and verify shield bonus activates
    drone = w_cs.helper_drones[0]
    w_cs.a['x'], w_cs.a['y'] = float(drone['x']), float(drone['y'])
    _, _, _, info = w_cs.step(np.zeros(5, dtype=np.float32))
    assert info.get('helper_shield_active') is True


def test_max_tandem_config_hybrid_planner_and_checkpoint_replay(tmp_path):
    cfg = SynapseConfig.max_tandem(total_neurons=96, seed=11, task='cybernetic_sanctuary', compute_threads=2)
    assert cfg.dendritic_ltc is True
    assert cfg.oscillatory_coherence is True
    assert cfg.verifiable_dpo is True
    assert cfg.planner_algorithm == 'hybrid'

    lab = LabRuntime(cfg, output_root=tmp_path)
    lab.step(2, learn=False)
    # Dendritic & DPO weights remain untouched when learn=False
    assert lab.agent.dendritic_ltc.updates == 0
    assert len(lab.agent.dpo_engine.preference_buffer) == 0

    # Learning steps + mid-episode save/load exact replay
    lab.step(3, learn=True)
    assert 'dendritic_ltc' in lab.last
    assert lab.last['dendritic_ltc']['enabled'] is True
    assert 'oscillatory_coherence' in lab.last
    assert lab.last['oscillatory_coherence']['enabled'] is True
    assert 'uncertainty' in lab.last
    assert 'plasticity_monitor' in lab.last

    lab.save('max_ckpt')
    expected = lab.step(2, learn=True)
    fresh = LabRuntime(cfg, output_root=tmp_path)
    fresh.load('max_ckpt')
    actual = fresh.step(2, learn=True)
    assert actual == expected


def test_workbench_tandem_max_and_new_endpoints(tmp_path):
    cfg = SynapseConfig(nodes_per_source=16, seed=7)
    wb = Workbench(config=cfg, output_root=tmp_path)

    unc = wb.get('/api/uncertainty/status')
    assert unc['status'] == 'active'
    assert unc['tripartite_decomposer'] is True
    assert 'uncertainty' in unc
    assert 'plasticity_monitor' in unc

    diff_dpo = wb.post('/api/diffusion_dpo', {'horizon': 4, 'seed': 17})
    assert diff_dpo['status'] == 'completed'
    assert 'diffusion_repair' in diff_dpo
    assert 'verifiable_dpo' in diff_dpo

    causal_9 = wb.post('/api/causal_9arm', {'steps_per_arm': 6, 'episodes_per_arm': 1, 'seed': 17})
    assert causal_9['status'] == 'completed'
    assert 'causal_9arm_assay' in causal_9
    assert 'unified_scorecard' in causal_9

    tandem = wb.post('/api/tandem/max', {
        'total_neurons': 96,
        'task': 'forage',
        'pretrain_steps': 16,
        'pretrain_updates': 4,
        'batch_size': 32,
        'live_steps': 4,
        'workers': 1,
        'omnibus_steps_per_task': 2,
        'omnibus_eval_steps': 3,
        'omnibus_tasks': ['forage', 'cybernetic_sanctuary'],
        'six_arm_tasks': ['forage', 'ecosystem'],
        'six_arm_steps': 5,
    })
    assert tandem['status'] == 'completed'
    assert len(tandem['sections_completed']) == 5
    for sec in ['section_01', 'section_02', 'section_03', 'section_04', 'section_05']:
        assert sec in tandem
    status = wb.get('/api/tandem/status')
    assert status['status'] == 'completed'
    assert status['tandem_sha256'] == tandem['tandem_sha256']


def test_poseidon_dvgem_6arm_lesion_and_tether_coordinator():
    dvgem = DecisionValueEpisodicMemory(seed=42)
    tether = TetherSuccessorCoordinator(seed=42)
    rng = np.random.default_rng(42)
    obs = rng.normal(0.0, 0.2, size=(155,)).astype(np.float32)
    base_act = np.array([0.4, -0.1, 0.2, 0.0, 0.0], dtype=np.float32)

    for k in range(6):
        o_k = np.clip(obs + rng.normal(0.0, 0.02, size=(155,)).astype(np.float32), -1.0, 1.0)
        dvgem.update_belief(o_k)
        a_k = np.clip(base_act + np.array([0.25, 0.1, 0.0, 0.0, 0.0], dtype=np.float32), -1.0, 1.0)
        dvgem.record_transition(o_k, a_k, reward=0.65, next_obs=obs, viability_reserve=0.9)

    # Viability reserve guard blocks episodic override when reserve < floor
    _, guard_diag = dvgem.query_and_gate(obs, base_act, viability_reserve=0.10)
    assert guard_diag['admitted'] is False
    assert guard_diag['reserve_guard_triggered'] is True

    # 6-Arm Lesion Assay runs all 6 matched arms without mutating live belief state
    before_state = digest(dvgem.export_state())
    assay = dvgem.run_six_arm_lesion_assay(obs, base_act, viability_reserve=0.90)
    assert digest(dvgem.export_state()) == before_state
    assert set(assay['arms'].keys()) == set(DVGEM_MODES)
    assert assay['arms']['baseline']['admitted'] is False
    assert assay['arms']['erased']['admitted'] is False

    # TETHER specialist coordination + TD Successor Features + Poincaré/Lyapunov
    features = rng.normal(0.0, 0.5, size=(64,)).astype(np.float32)
    next_features = rng.normal(0.0, 0.5, size=(64,)).astype(np.float32)
    rates = rng.uniform(0.0, 1.0, size=(120,)).astype(np.float32)
    origins = np.array([i % 3 for i in range(120)], dtype=np.int64)
    td_res = tether.update_successor_td(features, base_act, 0.5, next_features, base_act, done=False)
    assert 'successor_td_norm' in td_res
    proposals = {
        'base_actor': base_act,
        'neural_graft': base_act + 0.05,
        'planner': base_act + 0.08,
        'dvgem': np.asarray(assay['arms']['intact']['action'], dtype=np.float32),
    }
    coord_act, summary = tether.coordinate_specialists(proposals, features, rates, origins)
    assert coord_act.shape == (5,)
    assert 0.0 <= summary['epistemic_trust'] <= 1.0
    assert 0.0 <= summary['poincare_mean_radius'] < 1.0
    assert summary['dynamical_regime'] in ('attractor', 'edge_of_chaos', 'chaotic')


def test_expanded_mega_arenas_continental_biosphere_and_chrono_tidepool(tmp_path):
    assert 'continental_biosphere' in ALL_TASKS
    assert 'chrono_tidepool_citadel' in ALL_TASKS

    w_cb = ComplexWorld(seed=11, curriculum='continental_biosphere')
    obs_cb = w_cb.reset('continental_biosphere')
    assert obs_cb.shape == (155,)
    assert w_cb.arena_half == 8.5
    assert len(w_cb.wormholes) >= 2
    assert len(w_cb.weather_fronts) >= 2
    snap_cb = w_cb.snapshot()
    assert snap_cb['half'] == 8.5
    assert len(snap_cb['wormholes']) >= 2
    assert len(snap_cb['weather_fronts']) >= 2

    # Verify wormhole teleportation triggers when agent enters Mouth A
    wh = w_cb.wormholes[0]
    w_cb.a['x'], w_cb.a['y'] = float(wh['x1']), float(wh['y1'])
    w_cb.wormhole_cooldown = 0
    _, _, _, info = w_cb.step(np.zeros(5, dtype=np.float32))
    assert info.get('wormhole_teleported') is True

    w_ct = ComplexWorld(seed=11, curriculum='chrono_tidepool_citadel')
    obs_ct = w_ct.reset('chrono_tidepool_citadel')
    assert obs_ct.shape == (155,)
    assert w_ct.arena_half == 7.5

    wb = Workbench(config=SynapseConfig(nodes_per_source=16, seed=7), output_root=tmp_path)
    res = wb.post('/api/dvgem_tether', {'seed': 19})
    assert res['status'] == 'completed'
    assert 'dvgem_6arm_lesion' in res
    assert 'tether_coordinator' in res


def test_uncapped_speed_and_criticality_vsa_working_memory(tmp_path):
    from pathlib import Path
    from synapse.criticality_vsa import CriticalityAvalancheEngine, HolographicVSAWorkingMemory

    # 1. Verify Uncapped Steps Per Second option in index.html & Max Mode wiring in app.js
    web_root = Path(__file__).resolve().parent.parent / 'synapse' / 'web'
    html_text = (web_root / 'index.html').read_text(encoding='utf-8')
    js_text = (web_root / 'app.js').read_text(encoding='utf-8')
    assert '<option value="0">MAX · Uncapped (∞ SPS)</option>' in html_text
    assert 'scheduleUncappedTick' in js_text
    assert "$('simspeed').value = '0'" in js_text

    # 2. Verify CriticalityAvalancheEngine (frozen invariance + reservoir criticality assay)
    crit = CriticalityAvalancheEngine(seed=42)
    rng = np.random.default_rng(42)
    rates = rng.normal(0.0, 0.35, size=(120,)).astype(np.float32)
    pooled = torch.from_numpy(rng.normal(0.0, 0.25, size=(32,)).astype(np.float32))
    before_crit = digest(crit.export_state())
    mod_pooled, summary_frozen = crit.step(rates, pooled, learn=False)
    assert digest(crit.export_state()) == before_crit
    assert mod_pooled.shape == (32,)
    assert summary_frozen['enabled'] is True

    _, summary_learn = crit.step(rates, pooled, learn=True)
    assert digest(crit.export_state()) != before_crit
    assert 0.80 <= summary_learn['critical_gain'] <= 1.25
    res_assay = crit.run_reservoir_criticality_assay(seed=42)
    assert res_assay['schema'] == 'connectome-reservoir-criticality-v1'
    assert res_assay['total_memory_capacity_mc'] > 0.5
    assert 0.0 <= res_assay['chaotic_timeseries_nrmse'] < 2.0

    # 3. Verify HolographicVSAWorkingMemory (FFT circular convolution + 6-pair binding assay)
    vsa = HolographicVSAWorkingMemory(dim=64, seed=42)
    obs = rng.normal(0.0, 0.3, size=(155,)).astype(np.float32)
    act = np.array([0.8, -0.2, 0.1, 0.0, 0.0], dtype=np.float32)
    vsa.record_transition(pooled.numpy(), obs, act, reward=0.75)
    before_vsa = digest(vsa.export_state())
    res_32, vsa_summary = vsa.step(pooled, obs, learn=False)
    assert digest(vsa.export_state()) == before_vsa
    assert res_32.shape == (32,)
    assert vsa_summary['bound_episodes'] == 1
    assert vsa_summary['trace_norm'] > 0.0

    vsa_assay = vsa.run_vsa_binding_assay()
    assert vsa_assay['schema'] == 'holographic-vsa-6pair-assay-v1'
    assert vsa_assay['exact_selective_recovery'] is True
    assert vsa_assay['separation_margin'] > 0.12

    # 4. Verify live agent + Workbench endpoint integration
    cfg = SynapseConfig(
        nodes_per_source=16,
        seed=7,
        criticality_homeostat=True,
        holographic_vsa=True,
    )
    wb = Workbench(config=cfg, output_root=tmp_path)
    step_snap = wb.post('/api/step', {'steps': 2, 'learn': True})
    assert step_snap['telemetry']['criticality_homeostat']['enabled'] is True
    assert step_snap['telemetry']['holographic_vsa']['enabled'] is True
    api_res = wb.post('/api/criticality_vsa', {'seed': 42})
    assert api_res['status'] == 'completed'
    assert 'reservoir_criticality_assay' in api_res
    assert 'vsa_binding_assay' in api_res
