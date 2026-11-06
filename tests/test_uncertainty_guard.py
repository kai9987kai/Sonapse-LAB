"""Guard screening must not manufacture task-performance measurements."""
import numpy as np
import pytest
import torch

from synapse.uncertainty_guard import UncertaintyGuard, ModificationProposal, apply_safe_modification


def proposals(delta, count=3):
    return [ModificationProposal('weight', delta.clone(), np.zeros(2), 999) for _ in range(count)]


def test_no_evaluator_reports_unknown_performance_not_an_invented_gain():
    layer = torch.nn.Linear(2, 1, bias=False)
    assessment = UncertaintyGuard().assess_ensemble_proposals(proposals(torch.zeros(1, 2)), layer)
    assert assessment.validation_performance_delta is None
    assert assessment.to_dict()['validation_performance_delta'] is None
    assert assessment.to_dict()['scope'] == 'structural screening; task performance unmeasured'


def test_single_member_and_nonfinite_proposals_do_not_bypass_screening():
    layer = torch.nn.Linear(2, 1, bias=False)
    guard = UncertaintyGuard()
    assert guard.assess_ensemble_proposals(proposals(torch.zeros(1, 2), 1), layer).approved is False
    assert guard.assess_ensemble_proposals(proposals(torch.full((1, 2), float('nan'))), layer).approved is False


def test_measured_validation_uses_inactive_candidate_and_rejects_regression():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad(): layer.weight.zero_()
    before = layer.weight.detach().clone()
    assessment = UncertaintyGuard().assess_ensemble_proposals(
        proposals(torch.ones(1, 2)), layer,
        evaluator=lambda model: -float(model(torch.ones(1, 2)).square().item()))
    assert assessment.validation_performance_delta == -4.0
    assert assessment.approved is False
    assert torch.equal(layer.weight, before)


def test_invalid_application_is_atomic():
    layer = torch.nn.Linear(2, 1, bias=False)
    assessment = UncertaintyGuard().assess_ensemble_proposals(proposals(torch.zeros(1, 2)), layer)
    before = layer.weight.detach().clone()
    with pytest.raises(ValueError):
        apply_safe_modification(layer.weight, assessment, torch.tensor([[float('inf'), 0.0]]))
    assert torch.equal(layer.weight, before)


def test_assessment_cannot_authorize_a_different_delta_or_stale_target():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad(): layer.weight.zero_()
    delta = torch.ones(1, 2)
    assessment = UncertaintyGuard().assess_ensemble_proposals(
        proposals(delta), layer, evaluator=lambda model: float(model.weight.sum()))
    with pytest.raises(ValueError, match='assessed'):
        apply_safe_modification(layer.weight, assessment, -100 * delta)
    with torch.no_grad(): layer.weight.add_(0.1)
    with pytest.raises(ValueError, match='changed'):
        apply_safe_modification(layer.weight, assessment, delta)
