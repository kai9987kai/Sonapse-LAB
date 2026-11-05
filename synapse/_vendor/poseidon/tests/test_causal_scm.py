import pytest

from poseidon.causal_scm import StructuralCausalModel, CausalAttribution
from poseidon.world import N_ACTIONS, OBS_SIZE


def test_causal_scm_interventional_prediction():
    scm = StructuralCausalModel()
    obs = [0.5] * OBS_SIZE
    obs[0] = 0.90  # health
    obs[1] = 0.40  # energy
    obs[2] = 0.40  # hydration
    obs[6] = 0.80  # food_local
    obs[7] = 0.80  # water_local

    # Intervene with do(forage): action=1
    pred_forage = scm.interventional_prediction(obs, do_action=1)
    assert pred_forage[1] > obs[1]  # Energy should increase from foraging

    # Intervene with do(drink): action=2
    pred_drink = scm.interventional_prediction(obs, do_action=2)
    assert pred_drink[2] > obs[2]  # Hydration should increase from drinking


def test_causal_scm_attribution():
    scm = StructuralCausalModel()
    obs = [0.5] * OBS_SIZE
    obs[0] = 1.0
    obs[1] = 0.5
    obs[2] = 0.5

    # Realized drop in hydration
    next_obs = list(obs)
    next_obs[2] = 0.45
    next_obs[1] = 0.46

    attribution = scm.attribute_causality(obs, action_taken=4, realized_next_obs=next_obs)
    assert isinstance(attribution, CausalAttribution)
    assert attribution.action == 4
    assert len(attribution.counterfactual_regrets) == N_ACTIONS
    assert isinstance(attribution.primary_causal_driver, str)
