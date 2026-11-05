import pytest
import torch
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.auto_programmer import synthesize_weight_patch
from synapse.dales_law import audit_dales_law, project_dales_law, make_dales_compliant_patch

def test_audit_dales_law():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    audit = audit_dales_law(agent)
    assert audit['status'] == 'ok'
    assert 'audited_neurons' in audit
    assert 'compliance_rate' in audit
    assert 0.0 <= audit['compliance_rate'] <= 1.0

def test_project_dales_law():
    weights = torch.tensor([1.5, -2.0, 3.0, -0.5])
    signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    projected = project_dales_law(weights, signs)
    # Excitatory (signs > 0) must be non-negative
    assert projected[0] == 1.5
    assert projected[1] == 0.0
    # Inhibitory (signs < 0) must be non-positive
    assert projected[2] == 0.0
    assert projected[3] == -0.5

def test_make_dales_compliant_patch():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    diag = {'primary_surface': 'brain.graft_w', 'suggested_fix': 'boost_grafts', 'mean_return': 0.0}
    raw_patch = synthesize_weight_patch(agent, diag)
    compliant = make_dales_compliant_patch(agent, raw_patch)
    assert compliant['schema'] == 'synapse-weight-program-v1'
    assert compliant['writes'] == raw_patch['writes']
