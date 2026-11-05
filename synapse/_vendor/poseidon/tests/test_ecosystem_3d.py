"""Tests for Poseidon Ecosystem-3D Multi-Agent Dynamics & Bio-Pulse Signaling."""
import pytest
from poseidon.ecosystem_3d import (
    ContinuousEcosystem3D,
    Ecosystem3DConfig,
    PeriodicPulseBeacon,
    TROPHIC_GRAZER,
    TROPHIC_PREDATOR,
    TROPHIC_SYMBIONT,
)
from poseidon.genesis_3d import PhysicsVector3


def test_beacon_periodic_transmission():
    beacon = PeriodicPulseBeacon(
        emitter_id="test_node",
        trophic_role=TROPHIC_GRAZER,
        duty_cycle=0.25,
        period_ticks=8,
    )
    # Active window is max(1, int(8 * 0.25)) = 2 ticks (ticks 0 and 1)
    assert beacon.is_transmitting(0) is True
    assert beacon.is_transmitting(1) is True
    assert beacon.is_transmitting(2) is False
    assert beacon.is_transmitting(7) is False
    assert beacon.is_transmitting(8) is True


def test_ecosystem_3d_multi_agent_step_and_pulses():
    eco = ContinuousEcosystem3D(seed=123)
    eco.spawn_default_colony(n_grazers=2, n_predators=1, n_symbionts=1)
    assert len(eco.agents) == 4

    # Run 10 steps
    for _ in range(10):
        actions = {
            "grazer_0": 4,  # explore
            "grazer_1": 1,  # forage
            "predator_0": 4,  # explore
            "symbiont_0": 0,  # rest
        }
        obs_dict, rewards, dones, telemetry = eco.step(actions)
        assert len(obs_dict) == 4
        assert len(rewards) == 4
        assert len(dones) == 4
        assert "alive_counts" in telemetry
        assert len(obs_dict["grazer_0"]) == 16


def test_ecosystem_trophic_predation_and_symbiosis():
    eco = ContinuousEcosystem3D(seed=42)
    # Spawn predator right next to grazer to trigger strike
    pos_grz = PhysicsVector3(50.0, 50.0, 3.0)
    pos_pred = PhysicsVector3(51.0, 50.0, 3.0)
    pos_sym = PhysicsVector3(51.5, 50.0, 3.0)

    eco.spawn_agent("grz", TROPHIC_GRAZER, pos=pos_grz)
    eco.spawn_agent("pred", TROPHIC_PREDATOR, pos=pos_pred)
    eco.spawn_agent("sym", TROPHIC_SYMBIONT, pos=pos_sym)

    initial_grz_health = eco.agents["grz"].health
    initial_sym_services = eco.agents["sym"].symbiotic_services

    actions = {"grz": 0, "pred": 0, "sym": 0}
    eco.step(actions)

    # Predator should have executed strike on nearby grazer
    assert eco.agents["grz"].health < initial_grz_health
    assert eco.agents["pred"].strikes_delivered >= 1
    # Symbiont should have cleaned nearby predator
    assert eco.agents["sym"].symbiotic_services >= 1
    assert eco.total_trophic_transfers >= 1
