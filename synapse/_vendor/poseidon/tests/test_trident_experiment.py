from poseidon.trident_experiment import run_trident_benchmark, verify_trident_receipt


def test_trident_experiment_execution_and_verification():
    receipt = run_trident_benchmark(episodes=2, max_steps=8, scarcity=2.5, seed_base=88100000)

    assert receipt["schema"] == "poseidon-trident-experiment-v1"
    assert receipt["promotion"] is False
    assert len(receipt["seeds"]) == 2
    assert set(receipt["arms"]) == {"core", "trident", "erased", "shifted", "no_topology", "ungated"}
    assert receipt["modes"]["trident"] == "intact"
    assert receipt["modes"]["core"] == "policy"

    verification = verify_trident_receipt(receipt)
    assert verification["verified"] is True
    assert verification["verification_scope"] == "receipt integrity, controller replay, and TidePool transition replay"
    assert receipt["trident_telemetry"]["decision_count"] == sum(
        len(episode["trajectory"]) for episode in receipt["arms"]["trident"]["episodes_data"]
    )
