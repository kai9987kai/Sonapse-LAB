from fractions import Fraction
from poseidon.hephaestus import (
    WakeSleepLibraryLearner,
    GFlowNetSampler,
    KoopmanEDMDLinearizer,
    BuchbergerGroebnerProver,
    HephaestusEngine,
)


def test_wake_sleep_library_learner_stitch_compression():
    learner = WakeSleepLibraryLearner()
    corpus = [
        "(2 * x + 1) * (2 * x - 1)",
        "(2 * y + 1) * (2 * y - 1)",
        "(2 * z + 1) * (2 * z - 1)",
    ]
    res = learner.compress_corpus(corpus, max_abstractions=2)
    assert res["verified"] is True
    assert res["beta_reduction_verified"] is True
    assert res["num_abstractions"] >= 1
    assert res["final_mdl_cost"] < res["initial_mdl_cost"]
    assert res["compression_ratio"] > 1.2


def test_gflownet_trajectory_and_detailed_balance():
    sampler = GFlowNetSampler()
    res = sampler.calibrate_and_verify({
        "H1": 5.0,
        "H2": 15.0,
        "H3": 10.0,
        "H4": 20.0,
    })
    assert res["verified"] is True
    assert abs(res["partition_function_Z"] - 50.0) < 1e-6
    assert res["max_trajectory_balance_loss"] < 1e-18
    assert res["max_detailed_balance_residual"] < 1e-12
    assert res["kl_divergence_to_reward_posterior"] < 1e-12
    assert res["highest_probability_mode"] == "H4"
    assert abs(res["terminal_sampling_probabilities"]["H4"] - 0.4) < 1e-6


def test_koopman_edmd_linearizer_and_eigenfunctions():
    koopman = KoopmanEDMDLinearizer()
    res = koopman.analyze_system(lam=0.8, mu=0.5, gamma=-0.6, steps=8)
    assert res["verified"] is True
    assert res["asymptotically_stable"] is True
    assert res["eigenfunction_linearity_residual"] < 1e-9
    assert res["multi_step_prediction_error"] < 1e-5
    assert 0.8 in res["koopman_eigenvalues"]
    assert 0.64 in res["koopman_eigenvalues"]
    assert 0.5 in res["koopman_eigenvalues"]


def test_buchberger_groebner_basis_and_ideal_membership():
    prover = BuchbergerGroebnerProver()
    res = prover.compute_groebner_and_verify()
    assert res["verified"] is True
    assert res["buchberger_criterion_satisfied"] is True
    assert res["ideal_membership_certified"] is True
    assert res["target_remainder"] == "0"
    assert "x - 2*y" in res["reduced_groebner_basis"]
    assert "y^2 - 1" in res["reduced_groebner_basis"]

    # Non-member polynomial p = x^2 - 3 should have non-zero remainder mod <x^2+y^2-5, x-2y>
    res_non = prover.compute_groebner_and_verify(
        target_poly={(2, 0): Fraction(1), (0, 0): Fraction(-3)}
    )
    assert res_non["ideal_membership_certified"] is False
    assert res_non["target_remainder"] != "0"


def test_hephaestus_unified_orchestrator():
    engine = HephaestusEngine()
    stat = engine.status()
    assert stat["ready"] is True
    assert stat["wake_sleep_library_learning"] is True

    syn = engine.synthesize()
    assert syn["project"] == "HEPHAESTUS"
    assert syn["verified"] is True
    assert len(syn["receipt_sha256"]) == 64
