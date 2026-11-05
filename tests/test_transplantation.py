import numpy as np
import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.transplantation import (
    DonorSubcircuit,
    extract_donor_subcircuit,
    fit_representation_alignment,
    transplant_subcircuit,
    run_transplantation_experiment,
)

def test_extract_donor_subcircuit():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    agent = IntegratedAgent(config)
    donor = extract_donor_subcircuit(agent, source_task='forage')
    assert isinstance(donor, DonorSubcircuit)
    assert donor.source_task == 'forage'
    assert len(donor.graft_weights) > 0
    d = donor.to_dict()
    assert 'identity' in d
    assert d['graft_shape'] == list(donor.graft_weights.shape)

def test_fit_representation_alignment():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(50, 16))
    true_M = rng.normal(0, 1, size=(16, 8))
    Y = X @ true_M + rng.normal(0, 0.01, size=(50, 8))
    M = fit_representation_alignment(X, Y, ridge=0.01)
    assert M.shape == (16, 8)
    assert np.all(np.isfinite(M))

def test_transplant_subcircuit_arms():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    donor_agent = IntegratedAgent(config)
    donor = extract_donor_subcircuit(donor_agent, source_task='forage')

    recipient = IntegratedAgent(SynapseConfig(seed=99, nodes_per_source=20))
    initial_w = recipient.brain.graft_w.clone().detach()

    arms = ['correct', 'random', 'shuffled', 'irrelevant', 'disabled', 'permuted_host']
    for arm in arms:
        res = transplant_subcircuit(recipient, donor, mode=arm, seed=42)
        assert res['status'] == 'ok'
        assert res['mode'] == arm
        current_w = recipient.brain.graft_w
        if arm == 'disabled':
            assert float(current_w.detach().abs().sum()) == 0.0
        else:
            assert float(current_w.detach().abs().sum()) > 0.0

def test_run_transplantation_experiment():
    config = SynapseConfig(seed=42, nodes_per_source=20)
    donor_agent = IntegratedAgent(config)
    donor = extract_donor_subcircuit(donor_agent, source_task='forage')

    recipient = IntegratedAgent(SynapseConfig(seed=99, nodes_per_source=20))
    report = run_transplantation_experiment(
        recipient,
        donor=donor,
        test_task='forage',
        episodes=1,
        seed=42,
    )
    assert report['status'] == 'ok'
    assert 'arms' in report
    for arm in ['correct', 'random', 'shuffled', 'irrelevant', 'disabled', 'permuted_host']:
        assert arm in report['arms']
    assert 'comparisons' in report
