import copy

import numpy as np
import pytest
import torch

from graftbrain.__main__ import main, parse_args
from graftbrain.brain import Agent
from graftbrain.connectomes import make_proxy
from graftbrain.graft import build
from graftbrain.plasticity import PlasticityConfig
from graftbrain.world import World


def make_agent(**settings):
    graph = build([make_proxy("flywire", n=40), make_proxy("manc", n=40)])
    cfg = PlasticityConfig(mode="adaptive", interval=1, warmup=0, turnover=.5, **settings)
    agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, rollout=2, microsteps=1, plasticity_config=cfg)
    return agent


def train_update(agent):
    obs = World(seed=14).observe()
    cam = np.zeros((2, 48, 64), np.float32)
    for i in range(2):
        agent.act(obs, cam)
        agent.feedback(1.0, i == 1, next_obs=obs, next_cam=cam)


def test_agent_collects_unmasked_dormant_credit_during_backward():
    agent = make_agent(growth_rule="gradient", growth_init="zero", min_age=2)
    initially_dormant = ~agent.plasticity.active_mask.clone()
    train_update(agent)
    assert int(agent.plasticity.gradient_update_count) == 1
    assert torch.count_nonzero(agent.plasticity.gradient_ema[initially_dormant]) > 0
    assert torch.count_nonzero(agent.brain.graft_w.grad[initially_dormant]) == 0


def test_no_rewire_control_still_learns_and_roundtrips():
    agent = make_agent(growth_rule="gradient", growth_init="zero", min_age=2)
    agent.rewiring_enabled = False
    initial = agent.plasticity.active_mask.clone()
    initial_weights = agent.brain.graft_w.detach().clone()
    for _ in range(4):
        train_update(agent)
    assert torch.equal(initial, agent.plasticity.active_mask)
    assert not torch.equal(initial_weights, agent.brain.graft_w)
    assert int(agent.plasticity.gradient_update_count) == 4
    restored = make_agent(growth_rule="gradient", growth_init="zero", min_age=2)
    restored.load_state_dict(copy.deepcopy(agent.state_dict()))
    assert restored.rewiring_enabled is False
    assert torch.equal(restored.plasticity.edge_age, agent.plasticity.edge_age)


def test_frozen_evaluation_does_not_collect_gradient_or_age_evidence():
    agent = make_agent(growth_rule="gradient", growth_init="zero", min_age=2)
    train_update(agent)
    before = copy.deepcopy(agent.plasticity.state_dict())
    agent.act(World(seed=4).observe(), np.zeros((2, 48, 64), np.float32), training=False)
    after = agent.plasticity.state_dict()
    for key, value in before.items():
        if key == "last_gates":
            continue
        if torch.is_tensor(value):
            assert torch.equal(value, after[key]), key
        else:
            assert value == after[key], key


def test_cli_accepts_credit_settings_and_only_records_explicit_options():
    args = parse_args(["train", "--steps", "1", "--graft-growth", "gradient",
                       "--graft-growth-init", "zero", "--graft-min-age", "20"])
    assert args.plasticity_config.growth_rule == "gradient"
    assert args.plasticity_config.growth_init == "zero"
    assert args.plasticity_config.min_age == 20
    assert args.plasticity_overrides == {"growth_rule": "gradient", "growth_init": "zero", "min_age": 20}


def test_cli_exposes_separate_research_benchmark_and_probe():
    benchmark = parse_args(["benchmark-graft-credit", "--train-steps", "1", "--tasks", "go_nogo"])
    assert benchmark.cmd == "benchmark-graft-credit"
    assert benchmark.train_steps == 1
    probe = parse_args(["probe-graft-credit", "--train-steps", "4", "--recovery-steps", "4"])
    assert probe.cmd == "probe-graft-credit"
    assert probe.recovery_steps == 4


def test_probe_rejects_seed_overflow_before_claiming_output(tmp_path):
    from graftbrain.graft_probe import run_graft_probe

    out = tmp_path / "overflow"
    with pytest.raises(ValueError, match="seed"):
        run_graft_probe(out, seeds=[2**64], train_steps=1, recovery_steps=1)
    assert not out.exists()
    with pytest.raises(SystemExit) as error:
        parse_args(["probe-graft-credit", "--seeds", str(2**64)])
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [("--graft-growth", "bad"), ("--graft-growth-init", "bad"),
                                       ("--graft-min-age", "-1")])
def test_cli_rejects_invalid_credit_controls(flag, value):
    with pytest.raises(SystemExit) as error:
        parse_args(["train", flag, value])
    assert error.value.code == 2


def test_headless_credit_training_saves_new_evidence(tmp_path):
    state = tmp_path / "new.pt"
    main(["train", "--fresh", "--state", str(state), "--steps", "3", "--max-neurons", "40",
          "--microsteps", "1", "-c", "proxy:flywire", "-c", "proxy:manc",
          "--graft-mode", "adaptive", "--graft-growth", "gradient", "--graft-growth-init", "zero",
          "--graft-min-age", "20"])
    saved = torch.load(state, weights_only=False)
    assert saved["config"]["plasticity"]["growth_rule"] == "gradient"
    assert saved["agent"]["rewiring_enabled"] is True
    assert int(saved["agent"]["plasticity"]["gradient_update_count"]) == 1


def test_full_agent_migrates_previous_plasticity_schema_and_resume_flags(tmp_path):
    from graftbrain.__main__ import _sim
    from graftbrain.runtime import Simulation

    path = tmp_path / "previous.pt"
    sim = Simulation(["proxy:flywire", "proxy:manc"], str(path), max_neurons=40,
                     microsteps=1, fresh=True, plasticity_config=PlasticityConfig(mode="adaptive"))
    sim.save("migration fixture")
    sim.cam.close()
    saved = torch.load(path, weights_only=False)
    old_keys = {"candidate_pre", "candidate_post", "candidate_site", "original_weights", "bucket_ids",
                "bucket_keys", "bucket_budgets", "active_mask", "coactivity_ema", "last_gates", "site_swaps",
                "update_count", "last_rewire_update", "rewire_count", "swap_count", "rng_state", "_extra_state"}
    for config in (saved["config"]["plasticity"], saved["agent"]["plasticity_config"],
                   saved["agent"]["plasticity"]["_extra_state"]["config"]):
        for key in ("growth_rule", "growth_init", "min_age"):
            config.pop(key, None)
    saved["agent"]["plasticity"] = {key: value for key, value in saved["agent"]["plasticity"].items()
                                        if key in old_keys or key.startswith("router.")}
    saved["agent"]["plasticity"]["_extra_state"]["version"] = 1
    saved["agent"].pop("rewiring_enabled", None)
    torch.save(saved, path)
    args = parse_args(["train", "--state", str(path), "--steps", "1", "--graft-growth", "coactivity"])
    resumed = _sim(args)
    assert resumed.agent.plasticity.config.growth_rule == "coactivity"
    assert resumed.agent.plasticity.config.min_age == 0
    assert resumed.agent.rewiring_enabled is True
    resumed.cam.close()
    with pytest.raises(ValueError, match="different graft"):
        _sim(parse_args(["train", "--state", str(path), "--graft-growth", "gradient"]))
