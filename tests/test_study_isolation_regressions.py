"""Matched-world and candidate-screen regressions for inherited study runners."""
from copy import deepcopy
import pytest

from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.controlled_study import ControlledSixArmStudy
from synapse.flagship_benchmark import FlagshipContinualBenchmark
from synapse.rsi_verifier import BoundedRSIEngine
from synapse.stateio import digest


def test_six_arm_queries_are_phase_matched_and_leave_live_state_untouched():
    study = ControlledSixArmStudy(task_sequence=['forage'], steps_per_stage=2, eval_episodes_per_task=1)
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16))
    before = digest(agent.export_state())
    first = study._evaluate_agent(agent, 'forage', 0)
    second = study._evaluate_agent(agent, 'forage', 9)
    assert digest(agent.export_state()) == before
    assert first == second


def test_six_arm_training_seeds_and_sleep_screens_are_recorded():
    study = ControlledSixArmStudy(task_sequence=['forage'], steps_per_stage=2, eval_episodes_per_task=1)
    result = study.run_study()
    arms = result['arms']
    seeds = [a['auxiliary_metrics'].get('training_seeds') for a in arms.values()]
    assert seeds[0] is not None and all(value == seeds[0] for value in seeds)
    assert arms['continual_backprop']['auxiliary_metrics']['cbp_utility_updates'] == 2
    assert arms['composed_system']['auxiliary_metrics']['cbp_utility_updates'] == 2
    for name in ('actor_replay', 'composed_system'):
        screen = arms[name]['auxiliary_metrics']['sleep_screens'][0]
        assert screen['pre_score'] is not None
        assert screen['post_score'] is not None


def test_flagship_queries_leave_live_state_untouched():
    benchmark = FlagshipContinualBenchmark(task_sequence=['forage'], steps_per_stage=2,
        eval_episodes_per_task=1, use_real_environment=True)
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16))
    before = digest(agent.export_state())
    benchmark._evaluate_agent_on_task(agent, 'forage')
    assert digest(agent.export_state()) == before


def test_flagship_arms_record_identical_world_seeds_and_actual_sleep_screens():
    benchmark = FlagshipContinualBenchmark(task_sequence=['forage'], steps_per_stage=2,
        eval_episodes_per_task=1, use_real_environment=True)
    result = benchmark.run_benchmark()
    seeds = [a.get('training_seeds') for a in result['arm_results'].values()]
    assert seeds[0] is not None and all(value == seeds[0] for value in seeds)
    for name in ('continual_sleep', 'full_developmental_sail'):
        screen = result['arm_results'][name]['sleep_summary']['recent_history'][0]
        assert screen['pre_score'] is not None
        assert screen['post_score'] is not None


def test_legacy_rsi_receipts_are_explicitly_simulated_and_content_bound():
    engine = BoundedRSIEngine(seed=9)
    receipt = engine.verify_and_eval(engine.generate_candidate())
    assert getattr(receipt, 'simulated', None) is True
    assert getattr(receipt, 'evidence_kind', None) == 'illustrative'
    assert engine.get_summary()['measured'] is False
    changed = deepcopy(receipt)
    changed.stage_results['forged'] = {'status': 'pass'}
    changed.finalize()
    assert changed.receipt_hash != receipt.receipt_hash


def test_legacy_rsi_stale_parent_is_rejected_without_promotion():
    engine = BoundedRSIEngine(seed=9)
    candidate = engine.generate_candidate()
    candidate.parent_hash = 'stale-parent'
    before = deepcopy(engine.current_config)
    receipt = engine.verify_and_eval(candidate)
    assert receipt.passed_all is False
    assert engine.current_config == before
    assert engine.failure_archive[-1]['failed_stage'] == 'schema_check'


@pytest.mark.parametrize('name', ['num_query_trials', 'num_support_trials'])
def test_legacy_rsi_requires_positive_bounded_trial_counts(name):
    engine = BoundedRSIEngine(seed=9)
    with pytest.raises(ValueError, match='trials'):
        engine.verify_and_eval(engine.generate_candidate(), **{name: 0})
