import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.auto_programmer import synthesize_weight_patch
from synapse.causal_mediation import run_causal_mediation_assay

def test_causal_mediation_assay():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)
    diag = {
        'primary_surface': 'mb.w_mbon',
        'suggested_fix': 'reinforce_approach',
        'mean_return': 0.0,
    }
    packet = synthesize_weight_patch(agent, diag)
    res = run_causal_mediation_assay(agent, packet, task='forage', num_episodes=1, max_steps=10, seed=42)
    assert res['status'] == 'ok'
    assert 'total_effect' in res
    assert 'natural_direct_effect' in res
    assert 'natural_indirect_effect' in res
    assert 'mediation_ratio' in res
    assert 0.0 <= res['mediation_ratio'] <= 1.0
    assert 'connectome_rate_shift_l2' in res
