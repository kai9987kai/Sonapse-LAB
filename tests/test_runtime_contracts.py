"""Executed-transition, motor coordinate and learned-topology regressions."""
import numpy as np
import pytest
import torch

from synapse.config import SynapseConfig
from synapse.runtime import LabRuntime
from synapse.stateio import digest
from synapse.homeostasis import HomeostaticEngine
from synapse.complex_world import ComplexWorld


def runtime(tmp_path, **kwargs):
    return LabRuntime(SynapseConfig(nodes_per_source=16, **kwargs), tmp_path)


def test_manual_learning_attributes_the_executed_action(tmp_path):
    lab = runtime(tmp_path)
    action = np.array([.6, -.3, .2, -.1, .4], np.float32)
    lab.step(1, learn=True, action=action)
    assert lab.world.t == lab.total_steps == 1
    assert lab.agent.experience_model.samples == 1
    assert np.array_equal(lab.last_action, action)
    assert lab.last['manual_teleop']
    assert lab.last['action_source'] == 'manual'


@pytest.mark.parametrize('action', [[1, 2, 3, 4, 5], [np.nan, 0, 0, 0, 0], [0, 0]])
def test_invalid_manual_action_is_rejected_without_mutation(tmp_path, action):
    lab = runtime(tmp_path)
    before = digest({'agent': lab.agent.export_state(), 'world': lab._world_state()})
    with pytest.raises(ValueError):
        lab.step(1, learn=True, action=action)
    assert lab.total_steps == 0
    assert digest({'agent': lab.agent.export_state(), 'world': lab._world_state()}) == before


def test_learning_advances_physiology_once_for_real_action(tmp_path):
    lab = runtime(tmp_path)
    expected = HomeostaticEngine()
    lab.step(1, learn=True)
    expected.step(lab.last_action, lab.recent[-1]['reward'])
    assert lab.agent.homeostasis.state.to_dict() == expected.state.to_dict()


def test_homeostasis_counts_drive_and_does_not_count_turn_as_locomotion():
    drive, turn = HomeostaticEngine(), HomeostaticEngine()
    drive.step(np.array([1, 0, 0, 0, 0]), 0)
    turn.step(np.array([0, 1, 0, 0, 0]), 0)
    assert drive.state.stamina < turn.state.stamina
    assert drive.state.energy < turn.state.energy


def test_terrain_friction_slows_drive_without_reducing_turn():
    normal = ComplexWorld(seed=7, curriculum='forage')
    friction = ComplexWorld(seed=7, curriculum='forage')
    friction.terrain_zones = [{'x': friction.a['x'], 'y': friction.a['y'], 'r': 100, 'friction': 2.0}]
    normal.terrain_zones = []
    action = np.array([.8, .4, 0, 0, 0])
    normal.step(action); friction.step(action)
    assert friction.a['v'] == pytest.approx(.56)
    assert friction.a['w'] == normal.a['w']


def test_world_returns_post_environment_observation():
    world = ComplexWorld(seed=7, curriculum='dynamic_arena')
    observation, _, _, _ = world.step(np.zeros(5))
    assert np.array_equal(observation, world.observe())


def test_frontier_reset_does_not_award_base_task_potential_windfall():
    world = ComplexWorld(seed=7, curriculum='dynamic_arena')
    _, reward, _, _ = world.step(np.zeros(5))
    assert reward == pytest.approx(-.01)


def test_predator_penalties_and_geometry_match_recorded_transition():
    world = ComplexWorld(seed=7, curriculum='forage')
    world.objects = []
    world.a.update(x=0., y=0., th=0.)
    world.predators = [{'id': 100, 'x': 1., 'y': 0., 'th': np.pi, 'speed': .4,
                        'kind': 'pursuer', 'r': .4, 'color': 'red'}]
    world._prev = world._progress()
    before = world.ep_return
    obs, reward, _, info = world.step(np.array([1., 0, 0, 0, 0]))
    distance = np.hypot(world.predators[0]['x']-world.a['x'], world.predators[0]['y']-world.a['y'])
    assert info.get('predator_impact') is True
    assert info['min_hazard_dist'] == pytest.approx(distance)
    assert world.ep_return-before == pytest.approx(reward)
    assert world.last_reward == reward
    assert obs[153] == pytest.approx(reward)


def test_frontier_timeout_records_episode_and_final_penalty():
    world = ComplexWorld(seed=7, curriculum='predator_evasion')
    world.max_t = 1
    world.predators[0].update(x=world.a['x'], y=world.a['y'], speed=0.)
    _, reward, done, info = world.step(np.zeros(5))
    assert done
    assert len(world.history['predator_evasion']) == 1
    assert world.ep_return == pytest.approx(reward)
    assert info['ret'] == pytest.approx(reward)


def test_fatigue_governs_drive_after_planner_and_preserves_turn(tmp_path):
    lab = runtime(tmp_path)
    lab.agent.homeostasis.state.stamina = .05
    # Deliberately aggressive planner tests the real final acting boundary.
    lab.agent.experience_model.plan = lambda state, baseline, **kwargs: (np.ones(5), {})
    result = lab.agent.act(lab.obs, planner_mode='model')
    assert abs(result['action'][0]) <= .350001
    assert np.array_equal(lab.agent.pending_transition['action'], result['action'])
    manual = np.array([1, .8, 0, 0, 0], np.float32)
    lab.step(1, action=manual)
    assert lab.last_action[0] == pytest.approx(.35)
    assert lab.last_action[1] == pytest.approx(.8)


def test_query_does_not_update_structural_weights_or_statistics(tmp_path):
    lab = runtime(tmp_path)
    with torch.no_grad():
        lab.agent.brain.graft_w[0] = .001
    before = digest({'weights': lab.agent.brain.graft_w, 'growth': lab.agent.neurogenesis.export_state(),
                     'pre': lab.agent.graph.graft_pre, 'post': lab.agent.graph.graft_post})
    lab.step(17, learn=False)
    assert digest({'weights': lab.agent.brain.graft_w, 'growth': lab.agent.neurogenesis.export_state(),
                   'pre': lab.agent.graph.graft_pre, 'post': lab.agent.graph.graft_post}) == before


def sprout_fixture(agent):
    graph = agent.graph
    pairs = set(zip(graph.graft_pre.tolist(), graph.graft_post.tolist()))
    for slot, (pre, post) in enumerate(zip(graph.graft_pre, graph.graft_post)):
        if not bool(agent.plasticity.active_mask[slot]):
            continue
        for new_pre in np.flatnonzero(graph.origin == graph.origin[pre]):
            for new_post in np.flatnonzero(graph.origin == graph.origin[post]):
                if (new_pre, new_post) not in pairs:
                    with torch.no_grad():
                        agent.brain.graft_w[slot] = .001
                    agent.neurogenesis.rate_cov.fill(0)
                    agent.neurogenesis.rate_cov[new_pre, new_post] = 1
                    agent.neurogenesis.activity_steps = 20
                    return slot, int(new_pre), int(new_post)
    raise AssertionError('fixture needs one unused directional candidate')


def test_neurogenesis_updates_executing_topology_and_resets_credit(tmp_path):
    lab = runtime(tmp_path)
    agent = lab.agent
    slot, pre, post = sprout_fixture(agent)
    parameter = agent.brain.graft_w
    agent.opt.state[parameter] = {'step': torch.tensor(1.), 'exp_avg': torch.ones_like(parameter), 'exp_avg_sq': torch.ones_like(parameter)}
    agent.brain.elig.fill_(1); agent.brain.heb.fill_(1)
    agent.plasticity.coactivity_ema.fill_(1)
    event = agent.neurogenesis.evaluate_structural_remodeling(agent, dopamine=1.5, astrocyte_calcium=1.2)
    assert event['sprouted'] == 1
    assert agent.brain.graft_pre[slot] == pre and agent.brain.graft_post[slot] == post
    assert agent.plasticity.candidate_pre[slot] == pre and agent.plasticity.candidate_post[slot] == post
    assert agent.graph.graft_pre[slot] == pre and agent.graph.graft_post[slot] == post
    assert agent.plasticity.active_mask[slot]
    assert abs(float(agent.plasticity.effective_weights(parameter)[slot].detach())) > 0
    assert agent.brain.graft_w is parameter
    assert agent.opt.state[parameter]['exp_avg'][slot] == 0
    assert agent.opt.state[parameter]['exp_avg_sq'][slot] == 0
    assert any(parameter is live for group in agent.opt.param_groups for live in group['params'])
    assert agent.brain.elig[slot] == agent.brain.heb[slot] == 0
    assert agent.plasticity.coactivity_ema[slot] == 0


def test_changed_endpoint_checkpoint_replays_in_fresh_runtime(tmp_path):
    lab = runtime(tmp_path)
    sprout_fixture(lab.agent)
    lab.agent.neurogenesis.evaluate_structural_remodeling(lab.agent, dopamine=1.5, astrocyte_calcium=1.2)
    saved_pre = lab.agent.graph.graft_pre.copy()
    saved_post = lab.agent.graph.graft_post.copy()
    lab.save('changed')
    expected = lab.step(2, learn=True)
    fresh = runtime(tmp_path)
    fresh.load('changed')
    assert np.array_equal(fresh.agent.graph.graft_pre, saved_pre)
    assert np.array_equal(fresh.agent.graph.graft_post, saved_post)
    assert np.array_equal(fresh.agent.graph.graft_pre, fresh.agent.brain.graft_pre.numpy())
    assert fresh.step(2, learn=True) == expected


def test_invalid_agent_restore_preserves_resident_state(tmp_path):
    lab = runtime(tmp_path)
    before = digest(lab.agent.export_state())
    state = lab.agent.export_state()
    state['modules']['motor']['weight'].add_(.1)
    state['experience_model']['covariance'][0, 0, 0] = -10
    with pytest.raises(ValueError):
        lab.agent.import_state(state)
    assert digest(lab.agent.export_state()) == before


@pytest.mark.parametrize('value', [True, 0, 6, 1.5])
def test_planning_horizon_rejects_unsupported_values(value):
    with pytest.raises(ValueError):
        SynapseConfig(experience_horizon=value)


def test_legacy_config_uses_unambiguous_new_defaults():
    cfg = SynapseConfig()
    old = cfg.to_dict()
    old.pop('experience_horizon'); old.pop('experimental_cognition')
    assert cfg.matches_saved(old)


def test_untrained_corrections_have_no_default_action_influence(tmp_path):
    lab = runtime(tmp_path)
    agent = lab.agent
    saved = agent.export_state()
    baseline = agent.act(lab.obs)['action']
    agent.import_state(saved)
    with torch.no_grad():
        for parameter in agent.predictive.parameters():
            parameter.fill_(10)
        for parameter in agent.mimomix.parameters():
            parameter.fill_(10)
    assert np.array_equal(agent.act(lab.obs)['action'], baseline)


def test_live_actor_dispatches_short_horizon_without_learning(tmp_path):
    lab = runtime(tmp_path, experience_horizon=3)
    before = digest(lab.agent.experience_model.export_state())
    result = lab.agent.act(lab.obs, planner_mode='model')
    planner = result['telemetry']['experience_planner']
    assert planner['horizon'] == 3
    assert planner['reason'] == 'insufficient support'
    assert planner['evaluations'] == 0
    assert digest(lab.agent.experience_model.export_state()) == before


def test_legacy_checkpoint_without_topology_loads_default_config(tmp_path):
    lab = runtime(tmp_path)
    lab.step(2, learn=True)
    legacy = lab.agent.export_state()
    legacy.pop('topology')
    legacy['config'].pop('experience_horizon')
    legacy['config'].pop('experimental_cognition')
    expected = lab.agent.act(lab.obs)['action']
    fresh = runtime(tmp_path / 'fresh')
    fresh.agent.import_state(legacy)
    assert np.array_equal(fresh.agent.act(lab.obs)['action'], expected)


def test_neurogenesis_does_not_report_dormant_slot_as_executing_growth(tmp_path):
    agent = runtime(tmp_path).agent
    graph = agent.graph
    slot = int(torch.nonzero(~agent.plasticity.active_mask, as_tuple=True)[0][0])
    pre, post = int(graph.graft_pre[slot]), int(graph.graft_post[slot])
    pairs = set(zip(graph.graft_pre.tolist(), graph.graft_post.tolist()))
    candidate = next((int(a), int(b)) for a in np.flatnonzero(graph.origin == graph.origin[pre])
                     for b in np.flatnonzero(graph.origin == graph.origin[post]) if (a, b) not in pairs)
    with torch.no_grad():
        agent.brain.graft_w.fill_(.2)
        agent.brain.graft_w[slot] = .001
    agent.neurogenesis.rate_cov.fill(0)
    agent.neurogenesis.rate_cov[candidate] = 1
    agent.neurogenesis.activity_steps = 20
    event = agent.neurogenesis.evaluate_structural_remodeling(agent, dopamine=1.5, astrocyte_calcium=1.2)
    assert event['sprouted'] == 0
    assert (graph.graft_pre[slot], graph.graft_post[slot]) == (pre, post)
