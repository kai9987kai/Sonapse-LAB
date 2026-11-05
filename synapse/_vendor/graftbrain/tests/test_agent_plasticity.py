import copy
import json

import numpy as np
import pytest
import torch

from graftbrain.brain import Agent, NeuralInterface
from graftbrain.connectomes import make_proxy
from graftbrain.graft import build
from graftbrain.world import World


def graph():
    return build([make_proxy("flywire", n=300), make_proxy("manc", n=300)])


def inputs():
    return World(seed=12).observe(), np.zeros((2, 48, 64), np.float32)


def test_rollout_accounts_every_transition_and_terminal_reward():
    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1)
    obs, cam = inputs()
    for i in range(6):
        agent.act(obs, cam)
        agent.feedback(5.0, i == 5)
    assert agent.stats["transitions"] == 6
    assert agent.stats["updates"] == 3
    assert agent.stats["td"] > 0
    assert not agent.buf


def test_terminal_reward_modulates_the_last_action():
    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    obs, cam = inputs()
    agent.act(obs, cam)
    agent.feedback(5.0, True)
    assert agent.stats["td"] > 0.1


def test_frozen_action_leaves_plastic_state_unchanged():
    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    obs, cam = inputs()
    before = {k: v.clone() for k, v in agent.brain.state_dict().items()}
    mb_before = copy.deepcopy(agent.mb.state_dict())
    action, value = agent.act(obs, cam, training=False)
    assert np.isfinite(action).all() and np.isfinite(value)
    assert not agent.buf
    for key, value in before.items():
        if key != "r":
            assert torch.equal(value, agent.brain.state_dict()[key]), key
    for key, value in mb_before.items():
        assert torch.equal(value, agent.mb.state_dict()[key]), key


def test_legacy_state_loads_without_new_optimizer_groups():
    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    state = agent.state_dict()
    # Simulate the previous checkpoint shape.
    for key in ("plasticity", "plasticity_config", "router_opt"):
        state.pop(key, None)
    resumed = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    resumed.load_state_dict(state)
    assert resumed.plasticity.config.mode == "legacy"
    assert resumed.plasticity.active_mask.all()


def test_tiny_agent_uses_unique_available_ports():
    tiny = build([make_proxy("flywire", n=20), make_proxy("manc", n=20)])
    agent = Agent(tiny, World.obs_dim, (2, 48, 64), 5, microsteps=1)
    assert len(agent.brain.sensory_idx) == tiny.n
    assert len(torch.unique(agent.brain.sensory_idx)) == tiny.n


def test_runtime_restores_adaptive_mode_and_effective_flow(tmp_path):
    from graftbrain.plasticity import PlasticityConfig
    from graftbrain.runtime import Simulation

    path = str(tmp_path / "adaptive.pt")
    cfg = PlasticityConfig(mode="adaptive", interval=1, warmup=0)
    sim = Simulation(["proxy:flywire", "proxy:manc"], path, max_neurons=40,
                     microsteps=1, fresh=True, plasticity_config=cfg)
    sim.run_steps(2)
    sim.agent.flush(sim.obs, sim.cam.read()[0])
    sim._make_snapshot(0.0)
    snap = sim.get_snapshot()
    assert snap["plasticity"]["active"] < snap["plasticity"]["candidates"]
    assert sum(s["active"] for s in snap["sites"]) == snap["plasticity"]["active"]
    sim.save("test")
    sim.cam.close()
    resumed = Simulation(["proxy:flywire", "proxy:manc"], path, max_neurons=40, microsteps=1)
    assert resumed.agent.plasticity.config == cfg
    assert torch.equal(sim.agent.plasticity.active_mask, resumed.agent.plasticity.active_mask)
    resumed.cam.close()


def test_router_learns_and_frozen_adaptive_evaluation_keeps_support():
    from graftbrain.plasticity import PlasticityConfig

    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1,
                  plasticity_config=PlasticityConfig(mode="adaptive", interval=1, warmup=0))
    obs, cam = inputs()
    weights = agent.plasticity.router.weight.detach().clone()
    base = agent.brain.base_w.clone()
    for i in range(2):
        agent.act(obs, cam)
        agent.feedback(1.0, i == 1, next_obs=obs, next_cam=cam)
    assert not torch.equal(weights, agent.plasticity.router.weight)
    assert torch.equal(base, agent.brain.base_w)
    before = copy.deepcopy(agent.plasticity.state_dict())
    agent.act(obs, cam, training=False)
    after = agent.plasticity.state_dict()
    for k in before:
        if k == "last_gates":
            continue  # current-observation telemetry is allowed to change
        if isinstance(before[k], torch.Tensor):
            assert torch.equal(before[k], after[k]), k
        else:
            assert before[k] == after[k]


def test_adaptive_agent_state_roundtrip_has_identical_next_update():
    from graftbrain.plasticity import PlasticityConfig

    cfg = PlasticityConfig(mode="adaptive", interval=1, warmup=0)
    first = Agent(graph(), World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1, plasticity_config=cfg)
    obs, cam = inputs()
    for i in range(2):
        first.act(obs, cam)
        first.feedback(1.0, i == 1)
    second = Agent(graph(), World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1, plasticity_config=cfg)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    for agent in (first, second):
        torch.manual_seed(400)
        for i in range(2):
            agent.act(obs, cam)
            agent.feedback(1.0, i == 1)
    assert torch.equal(first.plasticity.active_mask, second.plasticity.active_mask)
    assert torch.equal(first.brain.graft_w, second.brain.graft_w)


def test_legacy_checkpoint_with_smaller_saved_port_sets_loads():
    source = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    source.brain.sensory_idx = source.brain.sensory_idx[:64]
    source.brain.motor_idx = source.brain.motor_idx[:50]
    source.iface = NeuralInterface(World.obs_dim, (2, 48, 64), 64, 50, len(source.brain.probe_idx), 5)
    source.opt = torch.optim.Adam([{"params": source.iface.parameters()},
                                   {"params": [source.brain.graft_w]}], lr=3e-4)
    state = copy.deepcopy(source.state_dict())
    for key in ("plasticity", "plasticity_config", "router_opt"):
        state.pop(key, None)
    resumed = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=1)
    resumed.load_state_dict(state)
    assert torch.equal(source.brain.sensory_idx, resumed.brain.sensory_idx)
    assert torch.equal(source.iface.enc[-1].weight, resumed.iface.enc[-1].weight)


def test_resume_compares_only_explicit_cli_graft_options(tmp_path):
    from graftbrain.__main__ import _sim, parse_args
    from graftbrain.plasticity import PlasticityConfig
    from graftbrain.runtime import Simulation

    path = str(tmp_path / "saved.pt")
    cfg = PlasticityConfig(mode="adaptive", density=.5, interval=3)
    sim = Simulation(["proxy:flywire", "proxy:manc"], path, max_neurons=40,
                     microsteps=1, fresh=True, plasticity_config=cfg)
    sim.save("test")
    sim.cam.close()
    for flags in (["--graft-mode", "adaptive"], ["--graft-density", ".5"]):
        resumed = _sim(parse_args(["train", "--state", path, "--steps", "1", *flags]))
        assert resumed.agent.plasticity.config == cfg
        resumed.cam.close()
    with pytest.raises(ValueError, match="different graft"):
        _sim(parse_args(["train", "--state", path, "--graft-density", ".7"]))


def test_single_connectome_without_grafts_emits_strict_json(tmp_path):
    from graftbrain.plasticity import PlasticityConfig
    from graftbrain.runtime import Simulation

    sim = Simulation(["proxy:flywire"], str(tmp_path / "single.pt"), max_neurons=40,
                     microsteps=1, fresh=True, plasticity_config=PlasticityConfig(mode="adaptive"))
    sim.step_once()
    sim.agent.flush(sim.obs, sim.cam.read()[0])
    sim._make_snapshot(0.0)
    assert sim.agent.stats["graft_w_abs"] == 0
    assert sim.get_snapshot()["plasticity"]["active"] == 0
    json.dumps(sim.get_snapshot(), allow_nan=False)
    sim.cam.close()


def test_new_state_directory_supports_episode_journal_before_first_save(tmp_path):
    from graftbrain.runtime import Simulation

    state = tmp_path / "new-run" / "brain.pt"
    sim = Simulation(["proxy:flywire"], str(state), max_neurons=40, microsteps=1, fresh=True)
    sim._journal({"outcome": "timeout"})
    assert (state.parent / "brain_journal.jsonl").is_file()
    sim.cam.close()


def test_legacy_forward_and_gradients_match_unrouted_zero_noise_dynamics():
    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, microsteps=2)
    first = agent.brain
    first.noise = 0
    first.r.fill_(.2)
    second = copy.deepcopy(first)
    drive = torch.linspace(-.5, .5, len(first.sensory_idx))
    direct = first(drive)
    gates = agent.plasticity.gates(torch.zeros(World.obs_dim))
    routed = second(drive, agent.plasticity.effective_weights(second.graft_w, gates))
    assert torch.equal(direct, routed)
    direct_grad = torch.autograd.grad(direct.sum(), first.graft_w)[0]
    routed_grad = torch.autograd.grad(routed.sum(), second.graft_w)[0]
    assert torch.equal(direct_grad, routed_grad)


def test_dormant_edges_have_zero_current_and_no_gradient_or_hebbian_drift():
    from graftbrain.plasticity import PlasticityConfig

    agent = Agent(graph(), World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1,
                  plasticity_config=PlasticityConfig(mode="static"))
    dormant = ~agent.plasticity.active_mask
    before = agent.brain.graft_w.detach().clone()
    isolated = agent.brain.graft_w.detach().clone()
    isolated[~dormant] = 0
    effective = agent.plasticity.effective_weights(isolated)
    current = agent.brain.graft_current(torch.ones(agent.brain.n), effective)
    assert torch.count_nonzero(current) == 0
    obs, cam = inputs()
    for i in range(8):
        agent.act(obs, cam)
        agent.feedback(5.0, i == 7, next_obs=obs, next_cam=cam)
    assert torch.equal(before[dormant], agent.brain.graft_w.detach()[dormant])
    assert torch.count_nonzero(agent.brain.graft_w.grad[dormant]) == 0


def test_headless_cli_flushes_partial_rollout_before_saving(tmp_path):
    from graftbrain.__main__ import main

    path = tmp_path / "cli" / "brain.pt"
    main(["train", "--fresh", "--steps", "3", "--state", str(path),
          "-c", "proxy:flywire", "-c", "proxy:manc", "--max-neurons", "40",
          "--microsteps", "1", "--graft-mode", "adaptive"])
    saved = torch.load(path, weights_only=False)
    assert saved["agent"]["stats"]["transitions"] == 3
    assert saved["agent"]["stats"]["updates"] == 1
