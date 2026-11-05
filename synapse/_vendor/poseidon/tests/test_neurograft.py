"""Unit tests for Project NEUROGRAFT: Multi-Source Connectomes, Liquid Time-Constant (LTC) Dynamics,
Tri-Monoaminergic Neuromodulation, Differentiable Neural Grafting & Virtual Lesion Causal Attribution.
"""
import pytest
from poseidon.neurograft import (
    HeterogeneousConnectomeGraph,
    LiquidTimeConstantEngine,
    LTCConfig,
    TriMonoaminergicEngine,
    NeuromodulatorState,
    CrossSourceGraftRouter,
    VirtualConnectomeLesionAuditor,
    NeurograftEngine,
    _kl_divergence,
    _softmax,
    _sigmoid,
)


def test_heterogeneous_connectome_graphs():
    cg = HeterogeneousConnectomeGraph(seed=42)
    assert set(cg.subgraphs.keys()) == {"worm", "fly_female", "fly_male", "human"}

    # Test worm sub-graph motifs
    worm_motifs = cg.compute_motifs("worm")
    assert worm_motifs["total_nodes"] == 12
    assert worm_motifs["chemical_edges"] >= 10
    assert worm_motifs["electrical_edges"] >= 3
    assert worm_motifs["reciprocal_dyads"] >= 2
    assert 0.0 < worm_motifs["excitatory_ratio"] <= 1.0
    assert len(worm_motifs["rich_club_nodes"]) >= 2
    assert 0.0 < worm_motifs["rich_club_coefficient"] <= 1.0

    # Test fly female (CX-ring) motifs
    fly_f_motifs = cg.compute_motifs("fly_female")
    assert fly_f_motifs["total_nodes"] == 10
    assert fly_f_motifs["reciprocal_dyads"] >= 1

    # Test human executive motifs
    human_motifs = cg.compute_motifs("human")
    assert human_motifs["total_nodes"] == 5
    assert human_motifs["macro_edges"] >= 5


def test_liquid_time_constant_engine():
    ltc = LiquidTimeConstantEngine(LTCConfig(beta_0=0.35, beta_1=0.65, beta_2=0.30))
    tau = ltc.compute_tau(mean_act=0.5, graft_gate=0.8)
    assert 0.1 <= tau <= 5.0

    # Test continuous step
    h_prev = [0.2, 0.4, -0.1, 0.7]
    drive = [0.8, -0.5, 0.3, 0.1]
    h_next, tau_s, alpha_s = ltc.step_continuous(h_prev, drive, graft_gate=0.7, dt=0.1)
    assert len(h_next) == len(h_prev)
    assert 0.0 < alpha_s < 1.0

    # Test Lipschitz contraction bound
    lipschitz = ltc.compute_lipschitz_bound(alpha_s, w_rec_norm=0.75)
    assert lipschitz["is_contractive"] is True
    assert lipschitz["stability_margin"] > 0.0


def test_tri_monoaminergic_engine():
    engine = TriMonoaminergicEngine()

    # Case 1: High reward and accuracy -> goal-exploitation
    st1 = engine.evaluate(
        chemoattractant=0.9,
        recent_reward=0.8,
        policy_confidence=0.9,
        accuracy=0.95,
        goals_reached=10,
        wall_proximity=0.1,
    )
    assert st1.dopamine >= 0.60
    assert st1.serotonin >= 0.60
    assert st1.regime == "goal-exploitation"
    assert 0.1 <= st1.plasticity_gain <= 2.5

    # Case 2: High wall proximity and choke point -> choke-point-maneuver
    st2 = engine.evaluate(
        wall_proximity=0.8,
        is_choke_point=True,
        detour_index=1.4,
    )
    assert st2.octopamine >= 0.35
    assert st2.regime == "choke-point-maneuver"

    # Case 3: High octopamine without choke point -> arousal-exploration
    st3 = engine.evaluate(
        wall_proximity=0.7,
        detour_index=1.8,
        route_entropy=0.8,
        is_choke_point=False,
    )
    assert st3.octopamine >= 0.50
    assert st3.regime == "arousal-exploration"


def test_cross_source_graft_router():
    router = CrossSourceGraftRouter(num_sources=4, hidden_dim=8, top_k=2, seed=42)
    pooled_states = [
        [0.1 * (i + 1) * (d + 1) for d in range(8)]
        for i in range(4)
    ]

    res = router.route(pooled_states)
    route_matrix = res["route_matrix"]
    assert len(route_matrix) == 4
    for i, row in enumerate(route_matrix):
        assert len(row) == 5  # 4 sources + null route
        assert abs(sum(row) - 1.0) < 1e-6
        # Self-route must be masked to near-zero
        assert row[i] < 1e-6

    assert 0.0 <= res["route_entropy"] <= 1.0
    assert len(res["grafted_states"]) == 4


def test_virtual_connectome_lesion_auditor():
    router = CrossSourceGraftRouter(num_sources=4, hidden_dim=8, top_k=2, seed=42)
    auditor = VirtualConnectomeLesionAuditor(router)

    pooled_states = [
        [0.2 * (i + 1) * (d + 1) for d in range(8)]
        for i in range(4)
    ]

    audit_res = auditor.audit(pooled_states)
    assert "intact_policy_probs" in audit_res
    assert len(audit_res["intact_policy_probs"]) == 4
    assert abs(sum(audit_res["intact_policy_probs"]) - 1.0) < 1e-6

    lesions = audit_res["lesion_evaluations"]
    assert set(lesions.keys()) == {"worm", "fly_female", "fly_male", "human"}
    for src, data in lesions.items():
        assert data["kl_divergence"] >= 0.0
        assert data["l2_logit_shift"] >= 0.0
        assert data["criticality_classification"] in (
            "CRITICAL_BOTTLENECK", "ROBUST_COGNITIVE_CORE", "PERIPHERAL_MODULATOR", "REDUNDANT_PATHWAY"
        )

    shapley = audit_res["shapley_causal_attribution"]
    assert len(shapley) == 4
    assert abs(sum(shapley.values()) - 1.0) < 1e-3
    assert audit_res["most_critical_source"] in lesions


def test_neurograft_engine_end_to_end():
    engine = NeurograftEngine(seed=42)
    synth = engine.synthesize_neurograft_state(
        seed=42,
        chemoattractant=0.8,
        wall_proximity=0.3,
        recent_reward=0.7,
        accuracy=0.9,
    )

    assert synth["status"] == "CERTIFIED_VALID"
    assert synth["stage"] == "v3.4.0-NEUROGRAFT-55"
    assert "certificate_sha256" in synth
    assert len(synth["certificate_sha256"]) == 64
    assert len(synth["liquid_time_constants"]) == 4
    assert "neuromodulation" in synth
    assert "graft_routing" in synth
    assert "virtual_lesion_audit" in synth
