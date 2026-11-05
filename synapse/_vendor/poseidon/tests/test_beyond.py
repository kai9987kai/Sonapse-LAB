"""Comprehensive test suite for Supermix Beyond modules.

Verifies:
1. Multi-model ensemble dynamics and epistemic uncertainty disagreement metric.
2. Risk-sensitive MPC lookahead planner with hazard/failure penalties.
3. Connectome biological graph generation and Maslov-Sneppen degree-preserving shuffle.
4. Persistent cross-step recurrent memory and episodic retrieval.
5. Sparse mixture of experts with top-k gating and confidence early-exit.
6. Surprise-prioritized experience replay buffer and promotion auditor.
7. Object-centric 3D scene graphs with physics evolution and Wavefront OBJ export.
"""
import tempfile
from pathlib import Path
import pytest
import torch

from poseidon.core import CoreRuntime
from poseidon.ensemble import (
    DynamicsHead,
    RiskConfig,
    UncertaintyAwarePlanner,
    WorldModelEnsemble,
    evaluate_survival_state,
)
from poseidon.connectome import (
    CircuitConfig,
    ConnectomeRecurrentCell,
    compare_circuit_topologies,
    generate_authentic_connectome_topology,
    maslov_sneppen_edge_swap,
)
from poseidon.persistent_memory import (
    CrossStepRecurrentMemory,
    DelayedRecallTask,
    EpisodicMemoryStore,
    EpisodicRecord,
    run_delayed_recall_benchmark,
)
from poseidon.sparse_moe import (
    AdaptiveTidalMoE,
    SparseMoEConfig,
    SparseMoERouter,
    benchmark_sparse_vs_dense,
)
from poseidon.adaptation import (
    PromotionAuditor,
    PromotionCriteria,
    SurprisePrioritizedBuffer,
)
from poseidon.scene_graph import (
    ObjectCentricSceneGraph,
    PhysicsProperties,
    SceneObject,
    create_procedural_beyond_world,
)
from poseidon.world import TidePool


def test_ensemble_disagreement_and_variance():
    """Verify that WorldModelEnsemble computes K predictions, variance, and non-negative disagreement."""
    ens = WorldModelEnsemble(hidden_size=64, k=3, base_seed=123)
    state = torch.randn(4, 64)
    obs = torch.rand(4, 16)
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    out = ens(state, obs, actions)
    assert out["deltas"].shape == (3, 4, 16)
    assert out["mean_delta"].shape == (4, 16)
    assert out["variance"].shape == (4, 16)
    assert out["disagreement"].shape == (4,)
    assert (out["disagreement"] >= 0.0).all()
    assert torch.isfinite(out["mean_delta"]).all()


def test_survival_state_evaluation():
    """Verify calibrated vitalities and failure probabilities."""
    # Healthy state
    safe_obs = [1.0, 0.9, 0.9, 0.9, 0.1, 0.0, 0.8, 0.8, 0.8, 0.1, 0.5, 0.5, 0.1, 0.5, 0.0, 0.0]
    vitality, fail_prob = evaluate_survival_state(safe_obs, action=0)
    assert vitality > 5.0
    assert fail_prob == 0.0

    # Lethal starvation/dehydration state
    dying_obs = [0.15, 0.02, 0.01, 0.02, 0.90, 0.9, 0.0, 0.0, 0.0, 0.8, 0.9, 0.5, 0.9, 0.0, 0.0, 0.5]
    vitality_dying, fail_prob_dying = evaluate_survival_state(dying_obs, action=1)
    assert fail_prob_dying > 0.8
    assert vitality_dying < vitality


def test_uncertainty_aware_planner_execution():
    """Verify that UncertaintyAwarePlanner produces valid risk-adjusted planning outputs."""
    runtime = CoreRuntime("runs/tidal_dagger/core.pt")
    planner = UncertaintyAwarePlanner(runtime, config=RiskConfig(horizon=2, uncertainty_penalty=1.5))

    env = TidePool(seed=42)
    obs = env.reset(42)
    plan_result = planner.plan(obs)

    assert 0 <= plan_result["action"] < 6
    assert plan_result["action_name"] in ("rest", "forage", "drink", "shelter", "explore", "flee")
    assert "epistemic_uncertainty" in plan_result
    assert "failure_probability" in plan_result
    assert isinstance(plan_result["is_high_surprise"], bool)
    assert len(plan_result["combined_scores"]) == 6
    assert len(plan_result["predicted_futures"]) == 6


def test_connectome_topologies_and_maslov_sneppen():
    """Verify authentic modular topology generation and exact degree-preserving shuffle."""
    adj = generate_authentic_connectome_topology(n_nodes=128, density=0.08, seed=42)
    assert adj.shape == (128, 128)
    assert adj.sum() > 0

    in_degrees_before = adj.sum(dim=0)
    out_degrees_before = adj.sum(dim=1)

    # Apply Maslov-Sneppen edge swaps
    shuffled = maslov_sneppen_edge_swap(adj, n_swaps=500, seed=42)
    in_degrees_after = shuffled.sum(dim=0)
    out_degrees_after = shuffled.sum(dim=1)

    # Maslov-Sneppen MUST strictly preserve exact in-degrees and out-degrees
    torch.testing.assert_close(in_degrees_before, in_degrees_after)
    torch.testing.assert_close(out_degrees_before, out_degrees_after)


def test_connectome_recurrent_cell():
    """Verify ConnectomeRecurrentCell forward pass and temporal update."""
    cfg = CircuitConfig(nodes=64, input_dim=16, output_dim=6, density=0.1, graph_type="authentic")
    cell = ConnectomeRecurrentCell(cfg, seed=42)

    x = torch.randn(2, 16)
    logits1, h1 = cell(x)
    assert logits1.shape == (2, 6)
    assert h1.shape == (2, 64)

    logits2, h2 = cell(x, h1)
    assert logits2.shape == (2, 6)
    assert h2.shape == (2, 64)
    # State should evolve under leaky dynamics
    assert not torch.allclose(h1, h2)


def test_persistent_memory_and_delayed_recall():
    """Verify cross-step recurrent memory and episodic store."""
    mem = CrossStepRecurrentMemory(obs_dim=16, action_dim=6, hidden_dim=32)
    h0 = mem.init_state(1)
    obs = torch.rand(1, 16)
    act = torch.tensor([2], dtype=torch.long)
    h1 = mem.step(obs, act, h0)
    assert h1.shape == (1, 32)
    assert not torch.allclose(h0, h1)

    # Episodic store tests
    store = EpisodicMemoryStore(capacity=10)
    store.append(EpisodicRecord(0, 1, 2, [0.0]*16, 1, 0.5, 0.05, "food"))
    store.append(EpisodicRecord(1, 4, 4, [0.0]*16, 2, 0.8, 0.12, "water"))

    assert len(store.retrieve_by_resource("food")) == 1
    assert len(store.retrieve_by_location(1, 2, radius=0)) == 1
    assert len(store.retrieve_high_surprise(top_k=1)) == 1
    assert store.retrieve_high_surprise(top_k=1)[0].uncertainty == 0.12


def test_sparse_moe_routing_and_adaptive_depth():
    """Verify Sparse MoE routing and early exit logic."""
    cfg = SparseMoEConfig(hidden_size=64, num_experts=4, top_k=2, max_recurrent_steps=3, confidence_exit_threshold=0.30)
    moe = AdaptiveTidalMoE(cfg)

    context = torch.randn(4, 64)
    out = moe(context)

    assert out["state"].shape == (4, 64)
    assert 1 <= out["steps_taken"] <= 3
    assert out["aux_loss"] >= 0.0


def test_surprise_buffer_and_auditor():
    """Verify surprise-weighted sampling and promotion auditor gates."""
    buf = SurprisePrioritizedBuffer(capacity=50)
    buf.add([0.1]*16, action=1, reward=0.0, next_observation=[0.1]*16, disagreement=0.01, is_failure=False)
    buf.add([0.5]*16, action=5, reward=-1.0, next_observation=[0.2]*16, disagreement=0.25, is_failure=True)

    assert len(buf) == 2
    samples = buf.sample(batch_size=2)
    assert len(samples) == 2

    # Auditor check with high threshold criteria
    auditor = PromotionAuditor(PromotionCriteria(min_survival_rate=0.99, max_dynamics_mse=0.001))
    runtime = CoreRuntime("runs/tidal_dagger/core.pt")
    ens = WorldModelEnsemble(hidden_size=runtime.model.config.hidden_size, k=2)
    report = auditor.audit_candidate(runtime.model, ens, test_seeds=[101, 102], scarcity=2.0)

    assert "promoted" in report
    assert "metrics" in report
    assert "survival_rate" in report["metrics"]


def test_object_centric_scene_graph():
    """Verify 3D scene graph, interactable physics updates, and Wavefront OBJ output."""
    world = create_procedural_beyond_world()
    assert len(world.objects) == 5
    assert len(world.relations) == 1

    # Step simulation
    pos_before = list(world.objects["sentinel"].position)
    world.step_simulation(dt=0.5)
    pos_after = list(world.objects["sentinel"].position)
    # Orbital sentinel must have moved
    assert pos_before != pos_after

    # Apply impulse
    agent = world.objects["agent"]
    world.apply_agent_impulse("agent", "explore")
    assert agent.physics.velocity[0] > 0.0

    # Export OBJ
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = Path(tmpdir) / "test_world.obj"
        exported = world.export_wavefront_obj(obj_path)
        assert Path(exported).exists()
        content = Path(exported).read_text(encoding="utf-8")
        assert "o agent_sphere" in content
        assert "v " in content
        assert "f " in content


def test_unity_bridge_session():
    """Verify Unity socket bridge server and client communication loop."""
    from poseidon.unity_bridge import simulate_unity_session
    res = simulate_unity_session(port=8099, steps=3)
    assert res["handshake"]["status"] == "connected"
    assert res["scene_objects_count"] >= 5
    assert res["steps_completed"] == 3
    assert res["final_action"] in ("rest", "forage", "drink", "shelter", "explore", "flee")


def test_adaptation_cycle_fast():
    """Verify surprise collection, fine-tuning, and audit execution."""
    from poseidon.train_adaptation import run_adaptation_cycle
    with tempfile.TemporaryDirectory() as tmpdir:
        receipt = run_adaptation_cycle(
            output_dir=tmpdir,
            collect_episodes=3,
            adaptation_steps=4,
            batch_size=8,
            base_seed=9000,
        )
        assert receipt["status"] == "complete"
        assert receipt["steps_collected"] > 0
        assert "audit" in receipt
        assert Path(receipt["artifact_path"]).exists()

