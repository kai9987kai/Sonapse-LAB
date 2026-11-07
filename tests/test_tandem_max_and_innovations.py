"""Tests for Dendritic LTC, Oscillatory Coherence, Frontier Worlds, and 5-Section Tandem Max Mode."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.dendritic_ltc import DendriticLiquidSubGraft
from synapse.oscillatory_coherence import OscillatoryCoherenceEngine
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
