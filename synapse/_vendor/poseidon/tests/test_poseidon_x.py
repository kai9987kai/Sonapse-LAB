"""Tests for Poseidon-X Unified Apex Cognitive Operating System."""
import pytest
from poseidon.poseidon_x import (
    PoseidonXDecision,
    PoseidonXKernel,
)


def test_poseidon_x_kernel_decision_cycle():
    kernel = PoseidonXKernel()
    obs_nominal = [1.0, 0.8, 0.8, 0.9, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]

    dec = kernel.decide(obs_nominal)
    assert isinstance(dec, PoseidonXDecision)
    assert 0 <= dec.action < 6
    assert isinstance(dec.action_name, str)
    assert isinstance(dec.affect.mood_label, str) and len(dec.affect.mood_label) > 0
    assert len(dec.certificate_sha256) == 64

    # Dict export
    d_dict = dec.to_dict()
    assert "curiosity_drive" in d_dict
    assert "rotor_orientation" in d_dict
    assert "affect" in d_dict


def test_poseidon_x_kernel_shielding_under_lethal_threat():
    kernel = PoseidonXKernel()
    # Starving and dehydrated
    obs_lethal = [0.4, 0.05, 0.06, 0.2, 0.1, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]

    dec = kernel.decide(obs_lethal)
    assert dec.shield_intervened is True
    # Action must be guided toward survival (drink or forage or flee)
    assert dec.action in (1, 2, 5)
