"""Behavioral regressions for evidence provenance and isolated sleep candidates."""
import numpy as np
import pytest
import torch

from synapse.agent import IntegratedAgent
from synapse.capabilities import catalog, invoke
from synapse.config import SynapseConfig
from synapse.sleep import SleepConsolidationEngine
from synapse.stateio import digest


@pytest.mark.parametrize('capability', [
    'rsi.meta_evaluate', 'reasoning.causal', 'planning.trident',
    'nexus.causal_efe', 'coordination.pantheon',
])
def test_illustrations_cannot_present_literal_values_as_measurements(capability):
    result = invoke(capability, {'generations': 0})
    assert result['status'] == 'ok'  # preserve the exploration API
    assert result.get('evidence_kind') == 'illustrative'
    assert result.get('measured') is False
    assert result.get('limitations')
    entry = next(x for x in catalog()['capabilities'] if x['id'] == capability)
    assert entry.get('evidence_kind') == 'illustrative'
    assert entry.get('measured') is False


def test_rsi_illustration_has_no_claimed_evaluations_or_top_level_measured_gain():
    result = invoke('rsi.meta_evaluate', {'generations': 0})
    assert result.get('generations_evaluated') == 0
    assert result.get('meta_retention_rate') is None
    assert result.get('fresh_learner_adaptation_delta') is None


def test_capability_tool_memory_is_runtime_owned_bounded_and_checkpointed(tmp_path):
    from synapse.runtime import LabRuntime
    config = SynapseConfig(nodes_per_source=16, seed=17)
    first = LabRuntime(config, output_root=tmp_path / 'first')
    second = LabRuntime(config, output_root=tmp_path / 'second')
    stored = invoke('reasoning.solve', {'expression': '17 + 25'}, runtime=first)
    assert stored['consumer']['stored'] is True
    assert invoke('memory.retrieve', {'query': '17 + 25'}, runtime=first)['matches'] == 1
    assert invoke('memory.retrieve', {'query': '17 + 25'}, runtime=second)['matches'] == 0
    first.save('tools')
    invoke('memory.store', {'key': 'later-entry', 'value': 'unsaved'}, runtime=first)
    first.load('tools')
    assert invoke('memory.retrieve', {'query': '17 + 25'}, runtime=first)['matches'] == 1
    assert invoke('memory.retrieve', {'query': 'later-entry'}, runtime=first)['matches'] == 0
    for index in range(140):
        invoke('memory.store', {'key': f'key-{index}', 'value': index}, runtime=first)
    assert len(first.agent.tool_memory) == 128
    assert invoke('memory.retrieve', {'query': '17 + 25'}, runtime=first)['matches'] == 0


def test_retrieved_tool_results_cannot_mutate_the_runtime_memory(tmp_path):
    from synapse.runtime import LabRuntime
    runtime = LabRuntime(SynapseConfig(nodes_per_source=16), output_root=tmp_path)
    invoke('memory.store', {'key': 'immutable-result', 'value': {'answer': 42}}, runtime=runtime)
    result = invoke('memory.retrieve', {'query': 'immutable-result'}, runtime=runtime)
    result['result'][0]['value']['answer'] = -1
    again = invoke('memory.retrieve', {'query': 'immutable-result'}, runtime=runtime)
    assert again['result'][0]['value']['answer'] == 42


def _agent_and_sleep():
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16, seed=13))
    agent.act(np.linspace(0, 1, 155, dtype=np.float32), learn=False)
    engine = SleepConsolidationEngine(lr=.01, shy_decay_rate=.03)
    for index in range(4):
        observation = np.roll(np.linspace(0, 1, 155, dtype=np.float32), index * 19)
        engine.record_experience(observation, np.full(5, .7 - index * .3, np.float32), 1.)
    return agent, engine


@pytest.mark.parametrize('failure', ['missing', 'baseline_error', 'candidate_error', 'nan', 'regression'])
def test_sleep_rejects_unverifiable_candidates_without_mutating_live_state(failure):
    agent, engine = _agent_and_sleep()
    before = digest(agent.export_state())
    calls = []

    def evaluator(candidate):
        # Evaluation is allowed to advance its own state, never the live actor.
        candidate.act(np.ones(155, np.float32), learn=False)
        calls.append(candidate)
        if failure == 'baseline_error' or (failure == 'candidate_error' and len(calls) == 2):
            raise RuntimeError('evaluator unavailable')
        if failure == 'nan':
            return float('nan')
        return 1. if len(calls) == 1 else -1.

    result = engine.run_sleep_cycle(agent, sleep_steps=2, eval_fn=None if failure == 'missing' else evaluator)
    assert result['accepted'] is False
    assert result.get('reason')
    assert digest(agent.export_state()) == before
    assert len(engine.wake_buffer) == 4
    assert engine.current_phase == 'WAKE'
    assert all(candidate is not agent for candidate in calls)


def test_sleep_trains_observation_conditioned_native_path_and_preserves_live_context():
    agent, engine = _agent_and_sleep()
    before = agent.export_state()
    motor_inputs = []
    hook = agent.motor.register_forward_pre_hook(lambda module, inputs: motor_inputs.append(inputs[0].detach().clone()))
    try:
        result = engine.run_sleep_cycle(agent, sleep_steps=3, eval_fn=lambda candidate: 1.)
    finally:
        hook.remove()
    assert result['accepted'] is True
    assert not torch.equal(before['modules']['native']['weight'], agent.native.weight)
    assert not torch.equal(before['modules']['motor']['weight'], agent.motor.weight)
    # Copies retain this observer; actual replay must encode distinct observations.
    assert len(motor_inputs) >= 2
    assert any(not torch.equal(motor_inputs[0], item) for item in motor_inputs[1:])
    assert digest(before['transient']) == digest(agent.export_state()['transient'])
    assert result.get('replay_mode') == 'reset_state_motor_head'


def test_sleep_round_trip_preserves_full_replay_buffer_and_sampling_state():
    engine = SleepConsolidationEngine(replay_buffer_size=96)
    for index in range(80):
        engine.record_experience(np.full(155, index / 100, np.float32), np.zeros(5, np.float32), 1.)
    restored = SleepConsolidationEngine(replay_buffer_size=96)
    restored.import_state(engine.export_state())
    assert len(restored.long_term_replay) == 80
    assert digest(restored.export_state()) == digest(engine.export_state())


def test_accepted_sleep_resets_only_changed_graft_moments_and_eligibility():
    agent, engine = _agent_and_sleep()
    agent.opt.zero_grad()
    (agent.motor.weight.square().sum() + agent.brain.graft_w.square().sum()).backward()
    agent.opt.step()
    with torch.no_grad():
        agent.brain.graft_w[0] = .01
        agent.brain.elig.fill_(.4)
        agent.brain.heb.fill_(.5)
    old_moment = agent.opt.state[agent.brain.graft_w]['exp_avg'].clone()
    result = engine.run_sleep_cycle(agent, sleep_steps=1, eval_fn=lambda candidate: 1.)
    assert result['accepted'] is True
    moment = agent.opt.state[agent.brain.graft_w]['exp_avg']
    assert moment[0] == 0
    assert torch.equal(moment[1:], old_moment[1:])
    assert agent.brain.elig[0] == 0
    assert agent.brain.heb[0] == 0
    assert torch.all(agent.brain.elig[1:] == .4)
    assert torch.all(agent.brain.heb[1:] == .5)


def test_flagship_declares_surrogate_evidence_and_does_not_claim_curriculum_mastery():
    from synapse.flagship_benchmark import FlagshipContinualBenchmark
    result = FlagshipContinualBenchmark(task_sequence=['forage'], steps_per_stage=1, eval_episodes_per_task=1).run_benchmark()
    assert result.get('evidence_kind') == 'synthetic_surrogate'
    assert result.get('native_world_evaluation') is False
    assert result['arm_results']['full_developmental_sail'].get('verified_task_success') is False
