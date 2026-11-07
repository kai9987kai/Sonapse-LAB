"""Behavioral regressions for the inherited Antigravity integration contracts."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from synapse.agent import IntegratedAgent
from synapse.config import SynapseConfig
from synapse.self_model import EfferenceCopyComparator, ControlMode
from synapse.stateio import digest
from synapse.omnibus_mode import OmnibusRunner
from synapse.reusable_skills import ReusableSkillLibrary, MacroSkill
from synapse.research_fabric import EvaluationReceipt
from synapse.plasticity_monitor import PlasticityMonitor
from synapse.runtime import LabRuntime
from synapse.dormant_lifecycle import DormantLifecycleManager


def test_frozen_actor_cannot_recruit_or_age_dormant_experts():
    agent = IntegratedAgent(SynapseConfig(nodes_per_source=16))
    before = digest(agent.dormant_lifecycle.export_state())
    for _ in range(3):
        agent.act(np.ones(155, np.float32) * .1, learn=False)
    assert digest(agent.dormant_lifecycle.export_state()) == before


def test_dormant_recycling_continues_private_random_state_after_restore():
    first = DormantLifecycleManager(seed=19)
    second = DormantLifecycleManager(seed=2)
    second.import_state(first.export_state())
    first.recycle_slot(0)
    second.recycle_slot(0)
    assert torch.equal(first.modules[0].weight, second.modules[0].weight)


def test_self_model_restores_learning_controls_and_telemetry_exactly():
    model = EfferenceCopyComparator(sensory_dim=8, action_dim=2, seed=19)
    s = np.arange(8, dtype=np.float32) * .01
    action = np.array([.4, -.1], np.float32)
    for _ in range(5):
        model.process_transition(s, action, s + .03)
    saved = deepcopy(model.export_state())
    clone = EfferenceCopyComparator(sensory_dim=8, action_dim=2, seed=2)
    clone.import_state(saved)
    assert clone.summary() == model.summary()
    for candidate in (model, clone):
        candidate.process_transition(s, action, s + .04)
        candidate.set_control_mode(ControlMode.SHUFFLED_ACTION)
        candidate.process_transition(s, action, s + .05, learn=False)
    assert digest(clone.export_state()) == digest(model.export_state())


def test_self_model_construction_preserves_global_random_state():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    EfferenceCopyComparator(sensory_dim=8, action_dim=2)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_all_inherited_advanced_features_replay_learned_checkpoint_exactly(tmp_path):
    config = SynapseConfig(nodes_per_source=16, self_model=True, continual_backprop=True,
        fast_weight_memory=True, latent_reasoning=True, value_guided_planning=True,
        experience_planning=True, experience_horizon=2)
    original = LabRuntime(config, tmp_path / 'first')
    original.step(4, learn=True)
    original.save('advanced')
    restored = LabRuntime(config, tmp_path / 'first')
    restored.load('advanced')
    assert digest(restored.agent.export_state()) == digest(original.agent.export_state())
    original.step(3, learn=True)
    restored.step(3, learn=True)
    assert digest(restored.agent.export_state()) == digest(original.agent.export_state())
    assert digest(restored._world_state()) == digest(original._world_state())


def test_omnibus_advances_physiology_once_per_executed_transition():
    runner = OmnibusRunner(total_neurons=48, curriculum_tasks=['forage'], steps_per_task=100)
    before = runner.agent.homeostasis.export_state()
    runner.step()
    assert runner.agent.homeostasis.export_state() != before


def test_omnibus_status_handles_discovered_macro_skills():
    runner = OmnibusRunner(total_neurons=48, curriculum_tasks=['forage'])
    runner.skill_library.skills['x'] = MacroSkill('x', 'Example', [0.] * 32, 1., [[0.] * 5] * 3, 3, 1.)
    assert runner.summary()['discovered_skills'][0]['skill_id'] == 'x'


def test_macro_replay_checkpoint_preserves_discovery_window():
    library = ReusableSkillLibrary(seed=4)
    for i in range(6):
        library.record_transition(np.ones(155) * i, np.zeros(5), float(i == 5), np.ones(155) * (i + 1), i)
    restored = ReusableSkillLibrary(seed=99)
    restored.import_state(library.export_state())
    assert [s.to_dict() for s in restored.discover_skills()] == [s.to_dict() for s in library.discover_skills()]


def test_rank_warmup_does_not_report_measured_full_rank():
    monitor = PlasticityMonitor(num_features=64)
    monitor.record_activity(np.zeros(64))
    report = monitor.evaluate_plasticity()
    assert report.effective_rank is None
    assert report.normalized_erank is None


@pytest.mark.parametrize('field,value', [('adaptation_speed', 999.), ('fabric_version', 'forged')])
def test_fabric_receipt_binds_all_scientific_metrics(field, value):
    receipt = EvaluationReceipt.create('example', ['forage'], [[1.]], 1., 0., .5, 7)
    setattr(receipt, field, value)
    assert not receipt.verify()


def test_omnibus_evaluation_is_matched_and_never_mutates_live_actor():
    runner = OmnibusRunner(total_neurons=48, curriculum_tasks=['forage', 'maze'], steps_per_task=2)
    assert callable(getattr(runner, 'evaluate_agent', None))
    before = digest(runner.agent.export_state())
    first = runner.evaluate_agent(runner.agent, ['forage'])
    second = runner.evaluate_agent(runner.agent, ['forage'])
    assert first == second
    assert digest(runner.agent.export_state()) == before


def test_omnibus_trains_uncertainty_on_successive_real_neural_features():
    runner = OmnibusRunner(total_neurons=48, curriculum_tasks=['forage'], steps_per_task=100)
    before = {k: v.detach().clone() for k, v in runner.uncertainty_decomposer.ensemble.state_dict().items()}
    runner.step()
    runner.step()
    assert any(not torch.equal(v, runner.uncertainty_decomposer.ensemble.state_dict()[k]) for k, v in before.items())
    assert runner.summary()['uncertainty']['training_updates'] == 1
    assert runner.summary()['uncertainty']['calibrated'] is False


def test_omnibus_sleep_evaluates_candidate_in_the_world():
    runner = OmnibusRunner(total_neurons=48, curriculum_tasks=['forage'], steps_per_task=1)
    assert callable(getattr(runner, 'evaluate_agent', None))
    seen_weights = []
    evaluator = runner.evaluate_agent

    def tracking_evaluator(candidate, tasks):
        seen_weights.append(candidate.motor.weight.detach().clone())
        return evaluator(candidate, tasks)

    runner.evaluate_agent = tracking_evaluator
    runner.run_full_curriculum()
    assert len(seen_weights) >= 4  # baseline, pre/candidate sleep screens, post stage
    assert any(not torch.equal(seen_weights[0], w) for w in seen_weights[1:])
    receipt = runner.receipt
    assert receipt.evaluation_protocol['steps_per_episode'] == runner.eval_steps
    assert receipt.compute_hash() == receipt.cryptographic_sha256
    receipt.total_sleep_cycles += 1
    assert receipt.compute_hash() != receipt.cryptographic_sha256
