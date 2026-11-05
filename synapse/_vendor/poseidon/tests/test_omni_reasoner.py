"""Comprehensive tests for Stage 14 Omni-Reasoner & Cognitive Coprocessor."""
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest

from poseidon.omni_reasoner import OmniReasoner, OmniReasoningResult
from poseidon.runtime import Poseidon
from poseidon.server import make_handler
from http.server import ThreadingHTTPServer
import threading


@pytest.fixture
def test_server(tmp_path):
    runtime = Poseidon(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, 8787))
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}", tmp_path
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def post_json(url, data):
    headers = {"Content-Type": "application/json"}
    return urlopen(Request(url, json.dumps(data).encode(), headers))


def test_intent_classification():
    reasoner = OmniReasoner(seed=42)
    assert reasoner.classify_intent("How do we cross a river when stepping stones sink?") == "physical_discovery"
    assert reasoner.classify_intent("Synthesize an enzyme for polyester cleavage") == "biocatalytic_synthesis"
    assert reasoner.classify_intent("Verify formal safety contracts under LTL model checking") == "formal_verification"
    assert reasoner.classify_intent("Calculate Clifford Cl(3,0) multivector rotor rotation") == "clifford_geometric"
    assert reasoner.classify_intent("Evaluate Causeway quantum superposition wavefunction collapse") == "causal_superposition"
    assert reasoner.classify_intent("What is the capital of France?") == "general_hyper_reasoning"


def test_cognitive_drives():
    reasoner = OmniReasoner(seed=42)
    drives = reasoner.compute_cognitive_drives("Cross wide river with tools", "physical_discovery")
    assert 0.0 <= drives["curiosity"] <= 1.0
    assert 0.0 <= drives["epistemic_uncertainty"] <= 1.0
    assert 0.0 <= drives["rigor_pressure"] <= 1.0
    assert 0.0 <= drives["causal_drive"] <= 1.0
    assert 0.0 <= drives["entropy"] <= 1.0


def test_physical_discovery_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_physical_discovery(seed=42)
    assert sim["narrow_channel"]["stepping_stones_viable"] is True
    assert sim["wide_channel"]["stepping_stones_viable"] is False
    assert sim["granite_physics"]["floats"] is False
    assert sim["granite_physics"]["net_buoyant_force_n"] < 0
    assert sim["pine_log_physics"]["net_buoyant_force_n"] > 0
    assert sim["pine_log_physics"]["single_log_payload_kg"] == 36.0
    assert sim["pine_log_physics"]["raft_3log_payload_kg"] == 108.0
    assert sim["pine_log_physics"]["raft_supports_agent"] is True
    assert sim["biocatalytic_binding"]["curing_ticks"] <= 5


def test_biocatalytic_enzyme_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_biocatalytic_synthesis(seed=42)
    assert "enzyme_name" in sim
    assert len(sim["active_site"]["residues"]) == 3
    assert sim["michaelis_menten_kinetics"]["delta_delta_g_barrier_kcal"] <= -8.0
    assert sim["michaelis_menten_kinetics"]["v_max_umol_min_mg"] > 0.0
    assert sim["curing_properties"]["bond_tensile_strength"] >= 0.95


def test_clifford_geometric_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_clifford_geometric()
    assert sim["isometry_invariant_preserved"] is True
    assert abs(sim["initial_velocity"]["norm"] - sim["rotated_velocity"]["norm"]) < 1e-5
    assert "vortex_lorentz_force" in sim


def test_causal_superposition_simulation():
    reasoner = OmniReasoner(seed=42)
    sim = reasoner._simulate_causal_superposition()
    assert sim["num_superposed_states"] == 6
    assert sim["evolved_entropy_nats"] <= sim["initial_entropy_nats"]
    assert 0.0 <= sim["optimal_action_probability"] <= 1.0
    assert sim["optimal_collapsed_action"] in ("REST", "FORAGE", "DRINK", "SHELTER", "EXPLORE", "FLEE")


def test_formal_ltl_safety_verification():
    reasoner = OmniReasoner(seed=42)
    proof = reasoner._verify_formal_safety("physical_discovery")
    assert proof["verdict"] == "PROVEN_SAFE"
    assert proof["contract_satisfied"] is True
    assert proof["shield_intervened"] is False
    assert len(proof["certificate_sha256"]) == 64


def test_full_reasoning_pipeline(tmp_path):
    runtime = Poseidon(tmp_path)
    res = runtime.reason("How do I cross a 10m wide river when stepping stones submerge?")
    assert res["intent"] == "physical_discovery"
    assert res["verified"] is False
    assert res["verification"]["answer"]["scope"] == "synthetic_demonstration"
    assert len(res["thought_steps"]) == 5
    assert len(res["certificate_sha256"]) == 64
    assert res["formal_proof"]["verdict"] == "PROVEN_SAFE"
    assert "Reasoning and Evidence Summary" in res["text"]

    # Test via respond with mode="reason"
    resp = runtime.respond("Synthesize an enzyme for wood crosslinking", mode="reason")
    assert resp["intent"] == "biocatalytic_synthesis"
    assert resp["verified"] is False


def test_server_reasoning_endpoints(test_server):
    url, _ = test_server

    # 1. Test GET /api/reasoning-status
    status_res = json.load(urlopen(url + "/api/reasoning-status"))
    assert status_res["ready"] is True
    assert "GhostInTheMachine" in status_res["deliberation"]
    assert "physical_discovery" in status_res["intents"]

    # 2. Test POST /api/reason
    payload = {
        "prompt": "Construct a buoyant raft across a wide river channel",
        "seed": 42
    }
    reason_res = json.load(post_json(url + "/api/reason", payload))
    assert reason_res["verified"] is False
    assert reason_res["formal_proof"]["query_bound"] is False
    assert reason_res["intent"] == "physical_discovery"
    assert len(reason_res["thought_steps"]) == 5
    assert len(reason_res["certificate_sha256"]) == 64


def test_biomechanical_gait_and_naval_stability():
    from poseidon.discovery_lab.biomechanics import BiomechanicalGaitModel, NavalStabilityModel

    gait = BiomechanicalGaitModel(agent_mass=70.0)
    # Safe step on earthen bank (1.0m)
    stride_safe = gait.assess_stride((1.0, 5.0, 0.0), (2.0, 5.0, 0.0), target_material="earth", is_wet=False)
    assert stride_safe.stride_feasible is True
    assert stride_safe.verdict == "SAFE_STEP"
    assert stride_safe.dynamic_balance_margin > 0.0
    assert stride_safe.ground_friction_coeff > 0.70

    # Wet granite rock leap (1.7m)
    stride_wet = gait.assess_stride((2.0, 5.0, 0.0), (3.7, 5.0, 0.0), target_material="granite_stone", is_wet=True)
    assert stride_wet.ground_friction_coeff == 0.28
    assert stride_wet.slip_probability > 0.50
    assert stride_wet.verdict in ("SLIP_SUBMERSION", "RISKY_LEAP")

    # Impossible jump (3.0m gap)
    stride_imp = gait.assess_stride((2.0, 5.0, 0.0), (5.0, 5.0, 0.0), target_material="granite_stone", is_wet=True)
    assert stride_imp.stride_feasible is False
    assert stride_imp.verdict == "STRIDE_EXCEEDED"

    # Naval stability: 1 log vs 3 logs
    naval = NavalStabilityModel()
    single_log = naval.assess_assemblage(num_logs=1, agent_mass=70.0, agent_onboard=True)
    assert single_log.metacentric_height_gm_m < 0.0
    assert single_log.is_stably_righting is False
    assert single_log.capsize_risk_rating == "CRITICAL_CAPSIZE"

    raft_3log = naval.assess_assemblage(num_logs=3, agent_mass=70.0, agent_onboard=True)
    assert raft_3log.metacentric_height_gm_m > 0.15
    assert raft_3log.is_stably_righting is True
    assert raft_3log.capsize_risk_rating == "STABLE"
    assert raft_3log.righting_moment_at_10deg_nm > 0.0
    assert raft_3log.net_buoyant_reserve_kg > 100.0


def test_omni_reasoner_extended_domains():
    reasoner = OmniReasoner(seed=42)

    # Biomechanical query
    res_bio = reasoner.reason("Evaluate humanoid articulated gait, stride reach limits, and wet rock slip hazard")
    assert res_bio.intent == "biomechanical_kinematics"
    assert res_bio.verified is False
    assert "gait" in res_bio.empirical_simulation["simulation_type"].lower()

    # Naval stability query
    res_naval = reasoner.reason("Analyze transverse metacentric height GM and righting moment for log raft hull stability")
    assert res_naval.intent == "naval_stability"
    assert res_naval.verified is False
    assert res_naval.empirical_simulation["catamaran_raft_3log"]["is_stably_righting"] is True

    # Multi-substrate enzyme generation
    gen = reasoner.enzyme_gen
    enz_chitin = gen.generate_catalyst_for_substrate("chitin_polymer", target_shear_force=400.0)
    assert enz_chitin.target_class == "chitinase_transpeptidase"
    assert enz_chitin.catalytic_efficiency > 0.0

    enz_pectin = gen.generate_catalyst_for_substrate("pectin_mucilage", ambient_ph=7.2)
    assert enz_pectin.target_class == "pectin_mucilage_lyase"

    enz_pet = gen.generate_catalyst_for_substrate("synthetic_bioplastic_ester", ambient_temperature_c=50.0)
    assert enz_pet.target_class == "pet_depolymerase_cutinase"


def test_omni_reasoner_scientific_hypothesis_falsification():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Formulate Popperian falsifiable hypothesis and design crucial experiment with Lakatos research programme analysis")
    assert res.intent == "scientific_hypothesis_falsification"
    assert res.verified is False
    assert "OLYMPUS" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["receipt_verified"] is True
    assert "paradigm_synthesis" in res.empirical_simulation


def test_omni_reasoner_aether_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Execute Project AETHER lattice gauge field holonomy, Jarzynski non-equilibrium thermodynamics, and open game optics")
    assert res.intent == "aether_gauge_thermo"
    assert res.verified is False
    assert "AETHER" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["symplectic_conserved"] is True
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_chronos_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Execute Project CHRONOS Causal Dynamical Triangulations, Connes spectral triple noncommutative geometry, and Stiefel optimization")
    assert res.intent == "chronos_quantum_geometry"
    assert res.verified is False
    assert "CHRONOS" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["euler_characteristic"] == 0
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_cartan_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Execute Project CARTAN exterior differential systems, PT-symmetric exceptional points, Albert algebra, and Turing morphogenesis")
    assert res.intent == "cartan_differential_systems"
    assert res.verified is False
    assert "CARTAN" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["pt_unbroken"] is True
    assert res.empirical_simulation["turing_active"] is True
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_tether_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Execute Project TETHER continuous epistemic reliability gating, counterfactual covariation, causal attribution, and fallback gate")
    assert res.intent == "tether_reliability_gating"
    assert res.verified is False
    assert "TETHER" in res.empirical_simulation["simulation_type"]
    assert 0.0 <= res.empirical_simulation["epistemic_trust"] <= 1.0
    assert 4 in res.empirical_simulation["corruption_detected"]
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_faultline_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Run Project FAULTLINE joint failure selection, condition-stratified risk matrix, and continuous covariance audit")
    assert res.intent == "faultline_joint_failure"
    assert res.verified is False
    assert "FAULTLINE" in res.empirical_simulation["simulation_type"]
    assert len(res.empirical_simulation["selected_panel"]) == 3
    assert res.empirical_simulation["joint_advantage"] >= 0.0
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_crosswire_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Synthesize Project CROSSWIRE unfamiliar protocol interoperability, active diagnostic probing, and closed-loop runtime shielding")
    assert res.intent == "crosswire_interoperability"
    assert res.verified is False
    assert "CROSSWIRE" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["interoperability_certified"] is True
    assert res.empirical_simulation["probing_steps"] > 0
    assert res.empirical_simulation["zero_hazard_violations"] is True
    assert len(res.empirical_simulation["receipt_sha256"]) == 64


def test_omni_reasoner_neurograft_synthesis():
    reasoner = OmniReasoner(seed=42)
    res = reasoner.reason("Execute Project NEUROGRAFT successor representation, sharp-wave ripple replay, and graph Laplacian spatial synthesis")
    assert res.intent == "neurograft_connectome"
    assert res.verified is False
    assert "NEUROGRAFT" in res.empirical_simulation["simulation_type"]
    assert res.empirical_simulation["swr_sweeps_executed"] == 6
    assert res.empirical_simulation["swr_total_updates"] > 0
    assert res.empirical_simulation["fiedler_value"] > 0.0
    assert res.empirical_simulation["mean_diffusion_distance"] > 0.0
    assert res.empirical_simulation["algebraic_connectivity_positive"] is True
    assert res.empirical_simulation["routing_entropy"] > 0.0
    assert len(res.empirical_simulation["receipt_sha256"]) == 64







