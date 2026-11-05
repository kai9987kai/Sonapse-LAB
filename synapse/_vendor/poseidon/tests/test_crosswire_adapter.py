"""Tests for Project CROSSWIRE Adapter."""

import pytest

from poseidon.crosswire_engine import CandidateMapping
from poseidon.world import OBSERVATION_NAMES
from poseidon.research.crosswire_adapter import (
    CrosswireAdapterVerifier,
    discretize_tidepool_obs_16d,
    STATE_GOAL,
    STATE_HAZARD,
    STATE_NOMINAL,
    STATE_READY,
)


def _tidepool_observation(**changes):
    values = dict(zip(OBSERVATION_NAMES, [0.7, 0.5, 0.5, 0.5, 0.1, 0.1,
                                       0.1, 0.1, 0.5, 0.1, 0.5, 0.5,
                                       0.5, 0.5, 0.0, 0.5]))
    values.update(changes)
    return [values[name] for name in OBSERVATION_NAMES]


@pytest.mark.parametrize("changes", [
    {"health": 0.10}, {"energy": 0.05}, {"hydration": 0.05},
    {"stamina": 0.02}, {"exposure": 0.90}, {"threat": 0.90},
])
def test_discretize_tidepool_hazards_use_actual_survival_fields(changes):
    assert discretize_tidepool_obs_16d(_tidepool_observation(**changes)) == STATE_HAZARD


@pytest.mark.parametrize("field", ["energy", "hydration"])
def test_discretize_tidepool_full_reserves_are_not_depth_or_salinity_hazards(field):
    assert discretize_tidepool_obs_16d(_tidepool_observation(**{field: 0.95})) == STATE_NOMINAL


def test_discretize_tidepool_initial_and_resourced_categories():
    assert discretize_tidepool_obs_16d(_tidepool_observation(episode_progress=0.0)) == STATE_READY
    assert discretize_tidepool_obs_16d(_tidepool_observation(
        energy=0.8, hydration=0.8, stamina=0.8, food=0.5, water=0.5,
    )) == STATE_GOAL
    # Low stamina is not a distance-to-goal sensor.
    assert discretize_tidepool_obs_16d(_tidepool_observation(stamina=0.15)) == STATE_NOMINAL


@pytest.mark.parametrize("observation", [[0.5] * 15, [float("nan")] * 16, [1.1] * 16])
def test_discretize_tidepool_rejects_observations_outside_world_contract(observation):
    with pytest.raises(ValueError, match="observation"):
        discretize_tidepool_obs_16d(observation)


def test_crosswire_adapter_verifier_audit():
    verifier = CrosswireAdapterVerifier(num_actions=3, seed=42)
    receipt = verifier.run_interoperability_audit(
        true_permutation=[1, 2, 0],
        probing_budget=8,
        eval_ticks=16,
    )

    assert receipt["exact_permutation_match"] is True
    assert receipt["recovered_permutation"] == [1, 2, 0]
    assert receipt["zero_hazard_violations"] is True
    assert receipt["interoperability_verified"] is True
    assert "audit_sha256" in receipt
    assert verifier.verified_audit_receipt is not None


def test_crosswire_adapter_shields_canonical_hazard_not_canonical_goal():
    verifier = CrosswireAdapterVerifier(num_actions=4)
    identity = CandidateMapping("known_identity", (0, 1, 2, 3))
    # The owning canonical model sends state 1 + action 2 to hazard 4,
    # whereas state 1 + action 1 safely reaches goal 3.
    rejected = verifier.shield.evaluate_and_filter(1, 2, [identity])
    assert rejected["is_fallback"] is True
    assert rejected["dispatched_action_ext"] == 0
    admitted = verifier.shield.evaluate_and_filter(1, 1, [identity])
    assert admitted["decision"] == "certified_admit"
    assert admitted["dispatched_action_ext"] == 1
