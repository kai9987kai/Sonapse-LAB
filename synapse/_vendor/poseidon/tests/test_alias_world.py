import pytest

from poseidon.research.alias_world import (
    DEFAULT_CUE_HORIZON,
    ObservationAliasTidePool,
)
from poseidon.research.contracts import alias_tidepool_contract, tidepool_contract
from poseidon.world import ACTIONS, OBS_SIZE


def test_alias_contract_compliance():
    contract = alias_tidepool_contract()
    contract.validate()
    assert contract.world_version == "alias-tidepool-v1"
    assert len(contract.observation_names) == OBS_SIZE
    assert len(contract.actions) == len(ACTIONS)


def test_alias_hand_constructed_ambiguity_and_divergence():
    """Verify that during aliased window, observations are identical across regimes,

    yet optimal actions differ causally.
    """
    env0 = ObservationAliasTidePool(seed=100, max_steps=20, cue_horizon=3, forced_regime=0)
    env1 = ObservationAliasTidePool(seed=100, max_steps=20, cue_horizon=3, forced_regime=1)

    # Initial step: cue channel 13 is distinct
    obs0 = env0.reset()
    obs1 = env1.reset()
    assert obs0[13] != obs1[13]
    assert obs0[13] == 0.85
    assert obs1[13] == 0.15

    # Step both environments forward to the aliased window using neutral actions (rest)
    for _ in range(3):
        obs0, r0, d0, info0 = env0.step(0)
        obs1, r1, d1, info1 = env1.step(0)
        assert not d0 and not d1

    assert env0.is_aliased
    assert env1.is_aliased

    # In aliased window, current observations must be bit-for-bit identical!
    assert obs0 == obs1
    assert obs0[13] == 0.50

    # Test causal divergence:
    # 1. Action 1 (Forage): high reward in Regime 0, penalty in Regime 1
    snap0 = env0.snapshot()
    snap1 = env1.snapshot()

    next_obs0, r_forage0, _, _ = env0.step(1)
    next_obs1, r_forage1, _, _ = env1.step(1)
    assert r_forage0 == 0.80
    assert r_forage1 == -0.05

    # 2. Restore and test Action 2 (Drink): high reward in Regime 1, penalty in Regime 0
    env0.restore(snap0)
    env1.restore(snap1)

    next_obs0_d, r_drink0, _, _ = env0.step(2)
    next_obs1_d, r_drink1, _, _ = env1.step(2)
    assert r_drink0 == -0.05
    assert r_drink1 == 0.80


def test_alias_snapshot_restore_exactness():
    env = ObservationAliasTidePool(seed=42, max_steps=30, cue_horizon=4)
    env.reset()

    # Step a few actions
    env.step(0)
    env.step(1)

    snap = env.snapshot()

    # Branch A: step 5 actions
    branch_a_obs = []
    for a in [0, 1, 2, 3, 4]:
        obs, r, d, _ = env.step(a)
        branch_a_obs.append((obs, r, d))

    # Restore to snap
    env.restore(snap)

    # Branch B: step the same 5 actions
    branch_b_obs = []
    for a in [0, 1, 2, 3, 4]:
        obs, r, d, _ = env.step(a)
        branch_b_obs.append((obs, r, d))

    # Must be bit-for-bit identical
    assert branch_a_obs == branch_b_obs
