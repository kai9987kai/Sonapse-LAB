"""Answer verification must follow the query's solver, not a safety demo."""
import pytest

from poseidon.omni_reasoner import OmniReasoner


def test_failed_solver_cannot_be_certified_by_unrelated_safe_trajectory():
    result = OmniReasoner().reason("Solve Sudoku: " + "1" * 81)

    assert result.empirical_simulation["verified"] is False
    assert "repeats" in result.empirical_simulation["error"]
    assert result.verified is False
    assert result.verification["answer"]["status"] == "failed"
    assert result.formal_proof["contract_satisfied"] is True
    assert result.formal_proof["query_bound"] is False
    assert result.formal_proof["scope"] == "fixed_synthetic_trajectory"
    assert "provably robust" not in result.text
    assert result.thought_steps[-1]["telemetry"]["status"] == "ANSWER_UNVERIFIED"


def test_unknown_query_has_no_answer_verifier():
    result = OmniReasoner().reason("What is the capital of France?")

    assert result.intent == "general_hyper_reasoning"
    assert result.verified is False
    assert result.verification["answer"]["scope"] == "unsupported_query"
    assert result.verification["answer"]["query_bound"] is False
    assert "no answer-specific verifier" in result.text.lower()
    assert "eliminating hallucinations" not in str(result.to_dict())


def test_fixed_physics_demo_does_not_verify_the_users_proposed_raft():
    result = OmniReasoner().reason("Can a single log raft carry 500 kg across a river?")

    assert result.intent == "physical_discovery"
    assert result.verified is False
    assert result.verification["answer"]["scope"] == "synthetic_demonstration"
    assert result.verification["answer"]["query_bound"] is False
    assert result.verification["safety_demo"]["query_bound"] is False
    assert result.formal_proof["initial_observation"] == [0.85, 0.70, 0.65, 0.90, 0.10, 0.05, 1.0, 1.0]
    assert result.formal_proof["actions"] == [1, 0, 3, 1, 2]


@pytest.mark.parametrize("prompt, domain", [
    ("Make 24 using the numbers 3, 3, 8, 8.", "number_game"),
    ("Water jugs: I have a 3-liter jug and a 5-liter jug. How do I measure out exactly 4 liters?", "water_jugs"),
])
def test_solver_verification_is_scoped_to_the_parsed_problem(prompt, domain):
    result = OmniReasoner().reason(prompt)

    assert result.verified is True
    answer = result.verification["answer"]
    assert answer["verified"] is True
    assert answer["scope"] == "parsed_problem_model"
    assert answer["domain"] == domain
    assert answer["parsed_problem"] == result.empirical_simulation["full_result"]["parsed"]
    assert answer["input_interpretation_verified"] is False
    assert result.verification["certificate"]["scope"] == "session_data_fingerprint"
    assert result.to_dict()["verification"] == result.verification
    assert "provably robust" not in result.text


def test_generated_prose_is_explicitly_unverified_even_with_a_checked_solver_answer():
    class LanguageStub:
        def generate(self, prompt, **kwargs):
            return {"text": "The answer is 999."}

    result = OmniReasoner().reason("Make 24 using the numbers 3, 3, 8, 8.", language_runtime=LanguageStub())

    assert result.verification["answer"]["verified"] is True
    assert result.verification["language_synthesis"] == {"present": True, "verified": False}
    assert "Unverified Generative Synthesis" in result.text
    assert "The answer is 999." in result.text


def test_direct_noether_reasoning_reads_the_current_engine_schema():
    result = OmniReasoner().reason("Evaluate Noether symmetry conserved charge and Poisson bracket.")

    engine_result = result.empirical_simulation["noether_receipt"]
    assert result.intent == "noether_invariant_symmetries"
    assert result.empirical_simulation["conserved_charge"] == engine_result["noether_symmetry"]["noether_conserved_charge"]
    assert result.empirical_simulation["poisson_zero"] == engine_result["noether_symmetry"]["poisson_bracket_bracket_zero"]
    assert result.verified is False
    assert result.verification["answer"]["scope"] == "synthetic_demonstration"


@pytest.mark.parametrize("prompt, intent", [
    ("Cross a wide river with a raft.", "physical_discovery"),
    ("Synthesize an enzyme for polyester cleavage.", "biocatalytic_synthesis"),
    ("Evaluate articulated gait and stride reach.", "biomechanical_kinematics"),
    ("Analyze metacentric height and righting moment.", "naval_stability"),
    ("Calculate Clifford multivector rotor rotation.", "clifford_geometric"),
    ("Evaluate Causeway superposition wavefunction collapse.", "causal_superposition"),
    ("Run Pantheon 24-pillar synthesis.", "pantheon_24_pillar_synthesis"),
    ("Run creative bisociation synthesis.", "creative_bisociation_synthesis"),
    ("Design a Popper falsification hypothesis.", "scientific_hypothesis_falsification"),
    ("Evaluate Neurograft connectome routing.", "neurograft_connectome"),
    ("Evaluate Aether gauge field thermodynamics.", "aether_gauge_thermo"),
    ("Evaluate Chronos causal dynamical triangulation.", "chronos_quantum_geometry"),
    ("Verify formal LTL safety contracts.", "formal_verification"),
])
def test_dedicated_demonstration_routes_preserve_their_answer_evidence_scope(prompt, intent):
    result = OmniReasoner().reason(prompt)

    assert result.intent == intent
    assert result.empirical_simulation["simulation_type"]
    assert result.verified is False
    assert result.verification["answer"]["query_bound"] is False
    assert result.verification["answer"]["scope"] == "synthetic_demonstration"
