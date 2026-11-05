import pytest

from poseidon.active_inference import (
    ActiveInferenceConfig,
    ActiveInferenceEvaluator,
    HomeostaticTarget,
)
from poseidon.world import OBS_SIZE


def test_active_inference_pragmatic_value():
    evaluator = ActiveInferenceEvaluator()

    # Optimal homeostatic state
    optimal_obs = [0.0] * OBS_SIZE
    optimal_obs[0] = 1.0  # health
    optimal_obs[1] = 0.90 # energy
    optimal_obs[2] = 0.90 # hydration
    optimal_obs[3] = 0.90 # stamina
    optimal_obs[4] = 0.05 # exposure
    optimal_obs[5] = 0.05 # threat

    pragmatic_opt = evaluator.compute_pragmatic_value(optimal_obs)
    assert abs(pragmatic_opt) < 1e-4  # Zero divergence from ideal

    # Critical starving state
    starving_obs = list(optimal_obs)
    starving_obs[1] = 0.02
    starving_obs[2] = 0.02
    pragmatic_starving = evaluator.compute_pragmatic_value(starving_obs)
    assert pragmatic_starving < -1.0
    assert pragmatic_starving < pragmatic_opt


def test_active_inference_epistemic_curiosity_drive():
    evaluator = ActiveInferenceEvaluator(ActiveInferenceConfig(base_epistemic_weight=0.5, min_epistemic_weight=0.05))

    # Satiated state: curiosity should be high
    satiated = [1.0, 0.95, 0.95] + [0.0] * (OBS_SIZE - 3)
    epistemic_high = evaluator.compute_epistemic_drive(satiated)
    assert epistemic_high > 0.40

    # Critical state: curiosity should throttle to minimum
    depleted = [0.2, 0.05, 0.05] + [0.0] * (OBS_SIZE - 3)
    epistemic_low = evaluator.compute_epistemic_drive(depleted)
    assert epistemic_low == 0.05
    assert epistemic_low < epistemic_high


def test_expected_free_energy_evaluation():
    evaluator = ActiveInferenceEvaluator()
    curr_obs = [0.8, 0.7, 0.7] + [0.0] * (OBS_SIZE - 3)
    pred_obs = [0.85, 0.75, 0.75] + [0.0] * (OBS_SIZE - 3)

    efe = evaluator.evaluate_expected_free_energy(
        current_obs=curr_obs,
        predicted_obs=pred_obs,
        calibrated_uncertainty=0.15,
        step_cost=0.01,
    )

    assert "free_energy_objective" in efe
    assert "pragmatic_value" in efe
    assert "epistemic_value" in efe
    assert efe["epistemic_value"] == 0.15


def test_active_inference_empowerment():
    evaluator = ActiveInferenceEvaluator()

    # Trapped / zero survival outcomes
    dead_outcomes = [[0.0] * OBS_SIZE, [0.0] * OBS_SIZE, [0.0] * OBS_SIZE]
    emp_zero = evaluator.compute_empowerment(dead_outcomes)
    assert emp_zero == 0.0

    # Diverse surviving outcomes (high freedom of action)
    diverse_outcomes = [
        [1.0, 0.9, 0.5, 0.8] + [0.0] * (OBS_SIZE - 4),  # fed
        [1.0, 0.5, 0.9, 0.8] + [0.0] * (OBS_SIZE - 4),  # drank
        [0.9, 0.7, 0.7, 0.9] + [0.0] * (OBS_SIZE - 4),  # rested
        [0.8, 0.6, 0.6, 0.5] + [0.0] * (OBS_SIZE - 4),  # explored
    ]
    emp_high = evaluator.compute_empowerment(diverse_outcomes)
    assert emp_high > 0.10
