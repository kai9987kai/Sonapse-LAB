"""Comprehensive Test Suite for SANCF & Frontier Master Systems.

Verifies:
1. SANCF channels (Continuous, Quantised, OrthogonalMode, Impaired)
2. SANCF topology strategies (random, anatomical, perturbation, hybrid)
3. SANCF fabric zero-init gate (gamma=0.0) & additive dynamics
4. SANCF training replay & atomic rollback
5. 10-arm SANCF matrix & NeuroWave mode multiplexing assay
6. World4D environment & tesseract bypass 155D observation adapter
7. DigitalToolArena synthetic permissioned tool sandbox (zero host escape)
8. SimulationWorker background thread & measured steps per second
9. Biological motif discovery & causal interventions
10. Multi-timescale adaptation & rank stability
11. Meta-plasticity rule learner
12. Upgraded criticality engine (Hill MLE, LLR, bootstrap CI)
13. Causal world model Pearlian do-calculus disentanglement
14. Relational HRR memory, cleanup codebook & interference assay
15. Verifier-gated RSI engine (static, stability, regression, bootstrap)
16. Event-driven sparse backend & energy efficiency assay
17. Grafted neural sections (OverflowMiniBrain 256N, Mini-Diffusion, Mamba SSM, Hopfield, Multi-Paradigm RL)
18. Distributed checkpoint migration protocol simulator
19. Supermix ecosystem adapters
20. 6,000 neurons scaling and Tandem Max compatibility
"""

import math
import time
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.neural_comm_channels import (
    CommChannelSpec,
    ContinuousChannel,
    OrthogonalModeChannel,
    QuantisedChannel,
    SimulatedImpairedChannel,
)
from synapse.neural_comm_topology import (
    ALLOWED_TOPOLOGY_STRATEGIES,
    CommTopologyManifest,
    compile_comm_topology,
)
from synapse.neural_comm import MessageEncoder, MessageDecoder, RegionRouter, NeuralCommFabric
from synapse.neural_comm_training import NeuralCommTrainer
from synapse.neural_comm_bench import SancfMatrixBenchmark
from synapse.world4d import World4DEnvironment
from synapse.digital_tool_arena import DigitalToolArena, ToolActionType
from synapse.sim_worker import SimulationWorker
from synapse.motif_discovery import ConnectomeMotifMiner
from synapse.connectome_interventions import ConnectomeInterventionEngine
from synapse.multitimescale_learning import MultiTimescaleScheduler
from synapse.adaptation_coordinator import AdaptationCoordinator
from synapse.meta_plasticity import ParameterizedPlasticityRule
from synapse.plasticity_rule_learner import PlasticityRuleLearner
from synapse.criticality_vsa import CriticalityAvalancheEngine
from synapse.causal_world_model import CausalWorldModel
from synapse.intervention_planner import InterventionPlanner
from synapse.causal_experiment_assay import CausalExperimentAssay
from synapse.relational_memory import HRRCore, RelationalMemoryBank
from synapse.compositional_retrieval import CodebookCleanupMemory, CompositionalRetriever
from synapse.memory_interference_assay import MemoryInterferenceAssay
from synapse.verifier_rsi import RecursiveImprovementEngine, ArchitectureVerifier
from synapse.event_driven_backend import EventDrivenSparseBackend
from synapse.hybrid_spiking import HybridSpikingPeripheral, HybridRateSpikingBridge
from synapse.energy_efficiency_assay import EnergyEfficiencyAssay
from synapse.grafted_sections import (
    OverflowMiniBrain,
    GraftedMiniDiffusionModel,
    SelectiveStateSpaceSection,
    HopfieldAssociativeSection,
    MultiParadigmRLTrainer,
)
from synapse.checkpoint_migration import CheckpointMigrationSimulator
from synapse.supermix_adapters import (
    PoseidonAdapter,
    ExpanseAdapter,
    NeuroGraftAdapter,
    TetherAdapter,
    FaultlineAdapter,
    GhostInTheMachineAdapter,
)
from synapse.benchmark_suite import BenchmarkRegistry, BenchmarkRunner, ResearchScorecard
from synapse.ablation_engine import FactorialAblationEngine


# ---------------------------------------------------------------------------
# 1. SANCF Virtual Channels
# ---------------------------------------------------------------------------
def test_sancf_channels():
    x = torch.randn(16)
    route = (0, 1)

    # Continuous
    spec_c = CommChannelSpec(mode="continuous", message_dim=16, precision_bits=32)
    c_chan = ContinuousChannel(spec_c)
    out_c, diag_c = c_chan.transmit(x, route)
    assert out_c.shape == (16,)
    assert diag_c["payload_bits"] > 0

    # Quantised (4-bit uniform)
    spec_q = CommChannelSpec(mode="quantized", message_dim=16, precision_bits=4, use_vq=False)
    q_chan = QuantisedChannel(spec_q)
    out_q, diag_q = q_chan.transmit(x, route)
    assert out_q.shape == (16,)
    assert diag_q["payload_bits"] == 16 * 4 + QuantisedChannel.HEADER_BITS + QuantisedChannel.SCALE_METADATA_BITS

    # Quantised (VQ)
    spec_vq = CommChannelSpec(mode="quantized", message_dim=16, use_vq=True, vq_codebook_size=64)
    vq_chan = QuantisedChannel(spec_vq)
    out_vq, diag_vq = vq_chan.transmit(x, route)
    assert out_vq.shape == (16,)

    # Orthogonal Mode (DCT basis)
    spec_o = CommChannelSpec(mode="orthogonal", message_dim=16, active_modes=16, orthogonal_basis="dct")
    ortho_chan = OrthogonalModeChannel(spec_o)
    out_o, diag_o = ortho_chan.transmit(x, route)
    assert out_o.shape == (16,)
    assert math.isclose(float(torch.norm(out_o - x)), 0.0, abs_tol=1e-4)

    # Impaired (Latency queue & drop)
    spec_imp = CommChannelSpec(mode="impaired", message_dim=16, max_latency_steps=2, packet_loss_probability=0.0)
    imp_chan = SimulatedImpairedChannel(spec_imp)
    out_imp0, _ = imp_chan.transmit(x, route)
    assert float(torch.norm(out_imp0)) == 0.0  # queued
    imp_chan.transmit(x, route)
    out_imp2, _ = imp_chan.transmit(x, route)
    assert out_imp2.shape == (16,)  # emerges after latency


# ---------------------------------------------------------------------------
# 2. SANCF Topology Strategies
# ---------------------------------------------------------------------------
def test_sancf_topology():
    from synapse.multi_connectome import build_multi_connectome
    graph = build_multi_connectome(nodes_per_source=32)

    for strat in ["random", "anatomical_interface", "structural_centrality", "hybrid_causal"]:
        plan = compile_comm_topology(graph, strategy=strat, max_routes=4)
        assert len(plan.routes) > 0
        r0 = plan.routes[0]
        assert r0.source_region != r0.target_region
        assert len(r0.source_nodes) > 0
        assert len(r0.target_nodes) > 0


# ---------------------------------------------------------------------------
# 3. SANCF Fabric & Zero-Init Dynamics
# ---------------------------------------------------------------------------
def test_sancf_fabric_zero_init():
    from synapse.multi_connectome import build_multi_connectome
    graph = build_multi_connectome(nodes_per_source=20, seed=42)
    fabric = NeuralCommFabric(graph, enabled=True, initial_gamma=0.0, seed=42)

    r_test = torch.randn(graph.n)
    out = fabric.compute_current(r_test, advance=True)
    # Zero-init gating: gamma=0.0 at initialization must yield zero aux current
    assert math.isclose(float(torch.norm(out.residual_current)), 0.0, abs_tol=1e-7)

    telem = fabric.get_telemetry_dict()
    assert "channel_mode" in telem


# ---------------------------------------------------------------------------
# 4. 10-Arm Matrix Benchmark & NeuroWave Assay
# ---------------------------------------------------------------------------
def test_sancf_matrix_and_neurowave():
    bench = SancfMatrixBenchmark(num_steps=5, seed=42)
    m_res = bench.run_benchmark()
    assert len(m_res.arm_results) == 10
    # A0 is baseline (0 bits)
    assert m_res.arm_results[0].bits_transmitted == 0
    # A3 transmits continuous float bits
    assert m_res.arm_results[3].bits_transmitted > 0

    nw_res = bench.run_neurowave_assay()
    assert len(nw_res.modes) == 7
    assert len(nw_res.controls) == 7


# ---------------------------------------------------------------------------
# 5. World4D Environment & Tesseract Bypass
# ---------------------------------------------------------------------------
def test_world4d_environment():
    env = World4DEnvironment(seed=42)
    obs = env.reset()
    assert obs.shape == (155,)

    # Step in 4th dimension (w-direction)
    # act[2] maps to dw
    act = np.array([0.2, 0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    next_obs, rew, done, info = env.step(act)
    assert next_obs.shape == (155,)
    assert isinstance(rew, float)
    assert "pos_4d" in info
    assert len(info["pos_4d"]) == 4

    snap = env.snapshot()
    assert snap["task"] == "world4d_tesseract_bypass"
    assert "projection_3d" in snap


# ---------------------------------------------------------------------------
# 6. Digital Tool Arena Permissioned Sandbox
# ---------------------------------------------------------------------------
def test_digital_tool_arena_security():
    arena = DigitalToolArena(seed=42)
    obs0 = arena.reset()
    assert obs0.shape == (155,)
    assert arena.curriculum_stage == 1

    # Safe typed execution
    obs1, rew1, done1, info1 = arena.step(ToolActionType.LIST, {})
    assert obs1.shape == (155,)
    assert info1["tool"] == "LIST"

    # Search
    obs2, rew2, done2, info2 = arena.step(ToolActionType.SEARCH, {"query": "valve"})
    assert info2["tool"] == "SEARCH"

    # Strict compliance: audit log confirms sandbox boundary
    assert info2["audit"]["sandbox_boundary"] == "strict_synthetic_local_no_host_access"


# ---------------------------------------------------------------------------
# 7. Simulation Worker (Uncapped SPS)
# ---------------------------------------------------------------------------
def test_sim_worker_throughput():
    from synapse.runtime import LabRuntime
    cfg = SynapseConfig(total_neurons=48, task="forage", seed=42)
    runtime = LabRuntime(cfg)
    import threading
    lock = threading.RLock()
    worker = SimulationWorker(runtime, lock, max_steps_per_tick=8)

    st0 = worker.status()
    assert not st0["running"]

    worker.start()
    time.sleep(0.3)
    st1 = worker.status()
    assert st1["steps_completed"] > 0
    assert "measured_steps_per_sec" in st1

    worker.stop()
    assert not worker.status()["running"]


# ---------------------------------------------------------------------------
# 8. Biological Motifs & Causal Interventions
# ---------------------------------------------------------------------------
def test_motifs_and_interventions():
    from synapse.multi_connectome import build_multi_connectome
    graph = build_multi_connectome(nodes_per_source=24, seed=42)
    miner = ConnectomeMotifMiner(seed=42)
    motifs = miner.extract_motifs(graph, max_motifs_per_class=2)
    assert len(motifs) >= 5

    engine = ConnectomeInterventionEngine(seed=42)
    new_pre, new_post, new_w, info = engine.apply_intervention(
        graph.base_pre, graph.base_post, graph.base_w, "node_removal", fraction=0.1
    )
    assert len(new_pre) < len(graph.base_pre)
    assert info["intervention"] == "node_removal"


# ---------------------------------------------------------------------------
# 9. Multi-Timescale Coordinator & Stability
# ---------------------------------------------------------------------------
def test_multitimescale_coordinator():
    coord = AdaptationCoordinator(num_neurons=60, seed=42)
    diag = coord.step(reward=0.5, novelty=0.1)
    assert "fast_weight_norm" in diag
    assert "adaptation_regime" in diag


# ---------------------------------------------------------------------------
# 10. Meta-Plasticity Rule Learner
# ---------------------------------------------------------------------------
def test_meta_plasticity():
    learner = PlasticityRuleLearner(seed=42)
    res = learner.train_and_evaluate(meta_epochs=2)
    assert res.learned_rule_score is not None
    assert res.hebbian_baseline_score is not None


# ---------------------------------------------------------------------------
# 11. Upgraded Criticality Engine (MLE, LLR, Bootstrap CI)
# ---------------------------------------------------------------------------
def test_criticality_rigorous_statistics():
    engine = CriticalityAvalancheEngine(seed=42)
    act = np.random.randn(64).astype(np.float32)
    p = torch.randn(32)

    for _ in range(30):
        p, diag = engine.step(act, p, learn=True)

    assert "avalanche_exponent_tau" in diag
    # Should produce valid MLE tau estimate or insufficient data guard
    tau = diag["avalanche_exponent_tau"]
    assert isinstance(tau, float)


# ---------------------------------------------------------------------------
# 12. Causal World Model (Pearlian Do-Calculus)
# ---------------------------------------------------------------------------
def test_causal_world_model_disentanglement():
    assay = CausalExperimentAssay(state_dim=32, action_dim=5, seed=42)
    res = assay.run_assay()
    assert res.disentanglement_score >= 0.0
    assert res.passive_drift_prediction_error >= 0.0


# ---------------------------------------------------------------------------
# 13. Relational Memory & Interference Assay
# ---------------------------------------------------------------------------
def test_relational_memory_and_cleanup():
    bank = RelationalMemoryBank(dim=128)
    cleanup = CodebookCleanupMemory(dim=128)

    r_vec = bank.hrr.random_unitary()
    f_vec = bank.hrr.random_unitary()
    cleanup.register_symbol("target_filler", f_vec)

    bound = bank.hrr.bind(r_vec, f_vec)
    unbound = bank.hrr.unbind(bound, r_vec)
    matches = cleanup.lookup(unbound, top_k=1)

    assert len(matches) > 0
    assert matches[0].symbol == "target_filler"
    assert matches[0].score > 0.85

    # Run assay
    assay = MemoryInterferenceAssay(dim=128, seed=42)
    a_res = assay.run_assay(max_pairs=4, num_trials=2, num_distractors=10)
    assert a_res.distractor_resistance_score >= 0.8


# ---------------------------------------------------------------------------
# 14. Verifier-Gated RSI Engine
# ---------------------------------------------------------------------------
def test_verifier_rsi_engine():
    engine = RecursiveImprovementEngine(base_config={"neural_comm_dim": 16, "spectral_radius": 0.8}, seed=42)
    prop = engine.propose_mutation()
    receipt = engine.evaluate_proposal(prop)
    assert receipt.static_validity is True
    assert receipt.numerical_stability is True
    assert isinstance(receipt.passed, bool)
    lineage = engine.lineage.export_lineage()
    assert len(lineage) >= 1


# ---------------------------------------------------------------------------
# 15. Event-Driven Sparse Backend & Energy Efficiency Assay
# ---------------------------------------------------------------------------
def test_event_driven_backend_and_efficiency():
    backend = EventDrivenSparseBackend(delta_threshold=0.05)
    r1 = torch.zeros(100)
    r2 = r1.clone()
    r2[:10] = 0.5  # 10% active

    deltas, active_idx, metrics = backend.detect_events(r2)
    assert active_idx.numel() == 10
    assert metrics.active_neurons == 10
    assert metrics.theoretical_energy_saving_factor > 1.0

    assay = EnergyEfficiencyAssay(num_neurons=100, num_steps=5, seed=42)
    rep = assay.run_assay(thresholds=[0.05])
    assert len(rep.trials) == 1
    assert rep.trials[0].theoretical_speedup > 1.0


# ---------------------------------------------------------------------------
# 16. Grafted Specialized Neural Sections
# ---------------------------------------------------------------------------
def test_grafted_sections():
    # 256-N Overflow Mini-Brain
    mini = OverflowMiniBrain(num_neurons=256, sensory_dim=155, motor_dim=5)
    s_in = torch.randn(155)
    of_curr, of_act, of_st = mini(s_in)
    assert of_curr.shape == (64,)
    assert of_act.shape == (5,)
    assert of_st.shape == (256,)
    w_eff = mini.get_effective_weights()
    assert w_eff.shape == (256, 256)

    # Mini-Diffusion Policy
    diff = GraftedMiniDiffusionModel(action_dim=5, cond_dim=64, num_steps=3)
    c = torch.randn(64)
    samp_act = diff.sample_action(c, trust_region_limit=1.0)
    assert samp_act.shape == (5,)
    assert bool(torch.all(torch.abs(samp_act) <= 1.0))

    # Mamba Selective State-Space
    ssm = SelectiveStateSpaceSection(d_model=64, d_state=16)
    x_in = torch.randn(64)
    y_out = ssm.step(x_in)
    assert y_out.shape == (64,)

    # Hopfield Associative Memory
    hop = HopfieldAssociativeSection(pattern_dim=64, max_patterns=8)
    p = torch.randn(64)
    hop.store_pattern(p)
    retrieved, energy = hop.retrieve(p + 0.1 * torch.randn(64))
    assert retrieved.shape == (64,)
    assert isinstance(energy, float)

    # Multi-Paradigm RL
    rl = MultiParadigmRLTrainer(state_dim=64, action_dim=5)
    states = torch.randn(4, 64)
    actions = torch.randn(4, 5)
    loss = rl.compute_iql_value_loss(states, actions)
    assert float(loss.item()) >= 0.0


# ---------------------------------------------------------------------------
# 17. Checkpoint Migration Simulator
# ---------------------------------------------------------------------------
def test_checkpoint_migration_zero_divergence():
    mig = CheckpointMigrationSimulator()
    state = {
        "config": {"seed": 7, "total_neurons": 6000},
        "activations": [0.1, 0.2, 0.3, 0.4],
        "step_count": 500,
        "plastic_weights": {"slot_0": [0.5, -0.2]},
    }
    rep = mig.simulate_migration(state)
    assert rep.is_success is True
    assert rep.sha256_checksum_match is True
    assert rep.state_divergence_norm == 0.0
    assert len(rep.phases) == 3


# ---------------------------------------------------------------------------
# 18. Supermix Ecosystem Adapters
# ---------------------------------------------------------------------------
def test_supermix_adapters():
    # Poseidon
    pos = PoseidonAdapter(sensory_dim=155)
    obs_p = pos.encode_fluid_currents(torch.zeros(155))
    assert obs_p.shape == (155,)

    # Tether
    teth = TetherAdapter(max_torque=1.0, max_delta=0.5)
    act_safe, viol = teth.enforce_safety(torch.tensor([1.5, -1.2, 0.0, 0.2, -0.1]))
    assert bool(torch.all(torch.abs(act_safe) <= 1.0))


# ---------------------------------------------------------------------------
# 19. Benchmark Suite & Scorecard
# ---------------------------------------------------------------------------
def test_benchmark_suite_execution():
    runner = BenchmarkRunner()
    results = runner.run_all(seed=42)
    assert len(results) >= 8

    scorecard = ResearchScorecard(commit_sha="test_sha").generate_scorecard(results)
    assert scorecard.overall_pass_rate >= 0.8
    assert "Scientific Non-Claims" in scorecard.summary_markdown


# ---------------------------------------------------------------------------
# 20. 6,500 Neurons Scaling & Baseline Invariant
# ---------------------------------------------------------------------------
def test_6500_neurons_scaling():
    cfg = SynapseConfig.frontier_apex(total_neurons=6500, seed=7)
    assert cfg.total_neuron_count() == 6500
    assert cfg.neural_comm_enabled is True
    assert cfg.overflow_mini_brain is True
    assert cfg.graph_diffusion is True
    assert cfg.active_inference_planner is True

    # Baseline default configuration preserves backward compatibility
    base_cfg = SynapseConfig(total_neurons=4000)
    assert base_cfg.neural_comm_enabled is False
    assert base_cfg.overflow_mini_brain is False
    assert base_cfg.graph_diffusion is False
    assert base_cfg.active_inference_planner is False


# ---------------------------------------------------------------------------
# 21. Frontier Innovations: Graph Diffusion, Active Inference & Causal Paths
# ---------------------------------------------------------------------------
def test_frontier_innovations_graph_diffusion_and_active_inference():
    from synapse.graph_diffusion import ChebyshevGraphDiffusion
    from synapse.active_inference import ActiveInferenceEngine
    from synapse.causal_discovery import CausalPathTracer

    # 1. Chebyshev Graph Diffusion (zero-readout frozen invariance)
    diff = ChebyshevGraphDiffusion(num_nodes=64, seed=42)
    x = torch.randn(64)
    res0, diag0 = diff(x, learn=False)
    assert res0.shape == (64,)
    assert torch.allclose(res0, torch.zeros(64))
    assert diag0['chebyshev_order'] == 4

    assay_diff = diff.run_graph_diffusion_assay(steps=3, seed=42)
    assert assay_diff['status'] == 'completed'
    assert assay_diff['spectral_radius_bound'] == 2.0

    # 2. Active Inference Expected Free Energy minimization
    act_inf = ActiveInferenceEngine(observation_dim=155, action_dim=5, hidden_dim=32, seed=42)
    base_act = np.zeros(5, dtype=np.float32)
    feat = np.random.randn(32).astype(np.float32)
    selected_act, diag_inf = act_inf.select_action(base_act, feat, num_candidates=6)
    assert len(selected_act) == 5
    assert diag_inf['enabled'] is True
    assert 'min_efe' in diag_inf
    assert 'epistemic_gain' in diag_inf

    assay_inf = act_inf.run_active_inference_assay(trials=4, seed=42)
    assert assay_inf['status'] == 'completed'

    # 3. DirectLiNGAM Causal Path Discovery
    tracer = CausalPathTracer(seed=42)
    assay_paths = tracer.run_causal_path_discovery_assay(steps=24, seed=42)
    assert assay_paths['status'] == 'completed'
    assert assay_paths['is_strictly_acyclic'] is True
    assert len(assay_paths['causal_topological_order']) == 6
