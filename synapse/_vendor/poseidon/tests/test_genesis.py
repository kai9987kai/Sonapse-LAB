"""Tests for Genesis Planetary Ecology & Evolutionary Genome Engine."""
from poseidon.genesis import GenesisEcosystemEngine


def test_genesis_initialization_and_chromosome():
    engine = GenesisEcosystemEngine(seed=42)
    engine.reset()

    assert len(engine.chromosome) == 16
    assert all(0.0 <= g <= 1.0 for g in engine.chromosome)
    assert engine.generation == 1
    assert len(engine.lineage_digest) == 16


def test_genesis_trophic_rk4_dynamics():
    engine = GenesisEcosystemEngine(seed=42)
    engine.reset()

    initial_p = engine.producers
    # Step forward with abundant sunlight and calm weather
    info = engine.step(daylight=1.0, severity=0.0, step_reward=1.5)

    assert "producers_biomass" in info
    assert "grazers_biomass" in info
    assert "predators_biomass" in info
    assert info["trophic_richness"] > 0.0
    assert info["cumulative_fitness"] == 1.5


def test_genesis_mutation_and_lineage_digest():
    engine = GenesisEcosystemEngine(seed=42)
    digest_before = engine.lineage_digest

    engine.mutate_chromosome(mutation_rate=0.1)
    digest_after = engine.lineage_digest

    assert engine.generation == 2
    assert digest_before != digest_after
    assert len(engine.chromosome) == 16
