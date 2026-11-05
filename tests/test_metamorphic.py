import pytest
from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.metamorphic import (
    verify_mr1_rotational_invariance,
    verify_mr2_valence_monotonicity,
    verify_mr3_sensor_fault_tolerance,
    verify_mr5_lipschitz_stability,
    run_metamorphic_verification_suite,
)

def test_metamorphic_relations():
    config = SynapseConfig(seed=42, nodes_per_source=16)
    agent = IntegratedAgent(config)

    res1 = verify_mr1_rotational_invariance(agent)
    assert 'passed' in res1

    res2 = verify_mr2_valence_monotonicity(agent)
    assert res2['passed'] is True

    res3 = verify_mr3_sensor_fault_tolerance(agent)
    assert res3['passed'] is True

    res5 = verify_mr5_lipschitz_stability(agent)
    assert res5['passed'] is True

    full_suite = run_metamorphic_verification_suite(agent)
    assert full_suite['status'] == 'ok'
    assert full_suite['passed_count'] >= 3
    assert len(full_suite['relations']) == 4
