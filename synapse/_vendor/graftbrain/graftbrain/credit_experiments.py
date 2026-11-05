"""Matched credit-growth pilots with predeclared conditions and frozen policies.

These artifacts report bounded synthetic/proxy observations. Edge turnover and
gradient magnitude are mechanism diagnostics, not evidence of a task benefit.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from . import __version__, connectomes as cx
from .brain import Agent
from .experiments import (_digest_arrays, _graph_digest,
                          _outcome_rates, _support_digest, _write_json,
                          validate_benchmark)
from .graft import build
from .plasticity import PlasticityConfig
from .senses import Webcam
from .world import World


CREDIT_TASKS = ("forage", "go_nogo")
BEHAVIOR_METRICS = ("initial_target_distance", "min_target_distance", "final_target_distance",
                    "path_length", "mean_abs_drive", "mean_abs_turn", "mean_squared_action_norm",
                    "action_saturation_fraction")
CREDIT_ARMS = tuple(
    {"arm": f"{rule}_age{age}", "growth_rule": rule, "min_age": age, "rewiring_enabled": True}
    for rule in ("coactivity", "gradient", "random") for age in (0, 20)
) + ({"arm": "routing_only", "growth_rule": "coactivity", "min_age": 0, "rewiring_enabled": False},)


class _BehaviorTrace:
    """Reporter-only measurements from actual world states and executed commands.

    Target distance is descriptive: approaching the orb is not universally
    desirable on no-go episodes. Squared action norm is dimensionless command
    magnitude, not mechanical work. This trace never supplies inputs to Agent.
    """

    def __init__(self, world):
        self.target_key = "goal" if world.task == "forage" else "orb"
        self.previous_position = (world.a["x"], world.a["y"])
        self.initial_distance = self._distance(world)
        self.min_distance = self.final_distance = self.initial_distance
        self.path_length = self.abs_drive = self.abs_turn = self.squared_action_norm = 0.0
        self.saturated_coordinates = self.action_coordinates = self.steps = 0

    def _distance(self, world):
        target = world.st[self.target_key]
        return float(np.hypot(target["x"] - world.a["x"], target["y"] - world.a["y"]))

    def observe(self, world, action):
        """Record once after each real World.step, using its clipping rule."""
        executed = np.clip(np.asarray(action, dtype=float), -1, 1)
        position = (world.a["x"], world.a["y"])
        self.path_length += float(np.hypot(position[0] - self.previous_position[0],
                                           position[1] - self.previous_position[1]))
        self.previous_position = position
        self.final_distance = self._distance(world)
        self.min_distance = min(self.min_distance, self.final_distance)
        self.abs_drive += float(abs(executed[0]))
        self.abs_turn += float(abs(executed[1]))
        self.squared_action_norm += float(np.square(executed).sum())
        self.saturated_coordinates += int((np.abs(executed) >= 1.0).sum())
        self.action_coordinates += executed.size
        self.steps += 1

    def summary(self):
        denominator = max(1, self.steps)
        return {"initial_target_distance": self.initial_distance, "min_target_distance": self.min_distance,
                "final_target_distance": self.final_distance, "path_length": self.path_length,
                "mean_abs_drive": self.abs_drive / denominator, "mean_abs_turn": self.abs_turn / denominator,
                "mean_squared_action_norm": self.squared_action_norm / denominator,
                "action_saturation_fraction": self.saturated_coordinates / max(1, self.action_coordinates)}


def validate_credit_benchmark(specs=None, max_neurons=400, train_steps=384, tasks=None,
                              seeds=None, eval_seeds=None, microsteps=1):
    """Return normalized arguments without creating files or changing RNG state."""
    normalized = []
    for name, supplied, default in (("training", seeds, [0]), ("evaluation", eval_seeds, [1000, 1001])):
        try:
            values = list(default if supplied is None else supplied)
        except TypeError as exc:
            raise ValueError(f"{name} seeds must be an iterable of integers") from exc
        if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64 for seed in values):
            raise ValueError(f"{name} seeds must be integers in [0, 2**64 - 1]")
        normalized.append(values)
    try:
        values = validate_benchmark(specs, max_neurons, train_steps, tasks, *normalized, microsteps)
    except TypeError as exc:
        raise ValueError("benchmark specifications and tasks must be valid iterables") from exc
    if any(task not in CREDIT_TASKS for task in values[1]):
        raise ValueError("credit benchmark tasks must be forage or go_nogo")
    return values


def build_credit_scenarios(tasks, eval_seeds, training_seeds=()):
    """Choose actual world seeds using initial task conditions, never rewards.

    Each requested go/no-go evaluation seed contributes one go and one no-go
    scenario. An ascending deterministic scan starts at that seed; selected
    seeds cannot be reused within the task or coincide with any training seed.
    """
    scenarios = []
    excluded = set(training_seeds)
    for task in tasks:
        used = set()
        for requested in eval_seeds:
            conditions = ("go", "nogo") if task == "go_nogo" else ("forage",)
            for condition in conditions:
                candidate = requested
                for _ in range(100000):
                    if candidate not in excluded and candidate not in used:
                        world = World(seed=candidate, curriculum=task)
                        actual = ("go" if world.st["go"] else "nogo") if task == "go_nogo" else "forage"
                        if actual == condition:
                            break
                    candidate = (candidate + 1) % 2**64
                else:
                    raise ValueError("could not construct balanced evaluation scenarios")
                used.add(candidate)
                scenarios.append({"scenario_id": f"{task}:{requested}:{condition}", "task": task,
                                  "requested_seed": int(requested), "world_seed": int(candidate),
                                  "condition": condition})
    return scenarios


@contextlib.contextmanager
def _preserve_rng():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _row(scenario, train_seed, phase, episode, ret, steps, outcome, terminal, mode=None, behavior=None):
    row = {**scenario, "seed": scenario["world_seed"], "training_seed": train_seed,
           "phase": phase, "stage": phase, "episode": int(episode), "return": float(ret),
           "steps": int(steps), "outcome": outcome, "terminal": bool(terminal), "mode": mode}
    if behavior is not None:
        row["behavior"] = behavior
    return row


def evaluate_credit(agent, graph, scenarios, train_seed=0, microsteps=1, phase="pre"):
    """Evaluate fresh saved-state clones with policy means and no feedback.

    The complete source state, pending rollout, and global random streams remain
    untouched. Each scenario starts with fresh neural, compass and camera state;
    its independent recurrent-noise stream is shared across pre/post and arms.
    """
    saved = copy.deepcopy(agent.state_dict())
    rows = []
    with _preserve_rng():
        for episode, scenario in enumerate(scenarios):
            clone = Agent(graph, World.obs_dim, (2, 48, 64), 5, seed=train_seed,
                          rollout=agent.rollout, gamma=agent.gamma, gae_lambda=agent.gae_lambda,
                          ppo_clip=agent.ppo_clip, microsteps=microsteps,
                          plasticity_config=agent.plasticity.config)
            clone.load_state_dict(copy.deepcopy(saved))
            clone.reset_episode()
            torch.manual_seed(scenario["world_seed"])
            world = World(seed=scenario["world_seed"], curriculum=scenario["task"])
            behavior = _BehaviorTrace(world)
            cam = Webcam(enabled=False)
            obs, ret, steps, done, info = world.observe(), 0.0, 0, False, {}
            try:
                with torch.no_grad():
                    while not done:
                        frame, world.webcam_target = cam.read()
                        action, _ = clone.act(obs, frame, training=False)
                        obs, reward, done, info = world.step(action)
                        behavior.observe(world, action)
                        ret += reward
                        steps += 1
                rows.append(_row(scenario, train_seed, phase, episode, ret, steps,
                                 info["outcome"], True, agent.plasticity.config.mode, behavior.summary()))
            finally:
                cam.close()
    return rows


def zero_action_references(scenarios):
    """Full episodes with zero motor actions expose passive no-go success."""
    rows = []
    for episode, scenario in enumerate(scenarios):
        world = World(seed=scenario["world_seed"], curriculum=scenario["task"])
        behavior = _BehaviorTrace(world)
        ret, steps, done, info = 0.0, 0, False, {}
        while not done:
            action = np.zeros(5, dtype=np.float32)
            _, reward, done, info = world.step(action)
            behavior.observe(world, action)
            ret += reward
            steps += 1
        rows.append({**_row(scenario, None, "reference", episode, ret, steps, info["outcome"], True,
                           behavior=behavior.summary()),
                     "policy": "zero_action"})
    return rows


def _returns(rows):
    return {"episodes": len(rows), "mean_return": float(np.mean([row["return"] for row in rows])) if rows else None,
            "successes": sum(row["outcome"] == "success" for row in rows),
            "outcome_rates": _outcome_rates(rows)}


def paired_credit_changes(before, after):
    """Describe paired scenario changes for one training seed, without tests/CIs."""
    pre = {row["scenario_id"]: row for row in before}
    post = {row["scenario_id"]: row for row in after}
    if len(pre) != len(before) or len(post) != len(after) or pre.keys() != post.keys() or not pre:
        raise ValueError("pre/post evaluation must contain the same unique nonempty scenarios")
    pairs = []
    for key, old in pre.items():
        new = post[key]
        if any(old.get(field) != new.get(field) for field in ("task", "world_seed", "condition", "training_seed")):
            raise ValueError("pre/post scenario identity or training seed differs")
        pair = {"scenario_id": key, "condition": old["condition"], "world_seed": old["world_seed"],
                "pre_return": old["return"], "post_return": new["return"],
                "return_delta": new["return"] - old["return"],
                "pre_outcome": old["outcome"], "post_outcome": new["outcome"]}
        old_behavior, new_behavior = old.get("behavior", {}), new.get("behavior", {})
        delta = {metric: new_behavior[metric] - old_behavior[metric] for metric in BEHAVIOR_METRICS
                 if metric in old_behavior and metric in new_behavior}
        if delta:
            pair.update(pre_behavior=old_behavior, post_behavior=new_behavior, behavior_delta=delta)
        pairs.append(pair)

    def mean_behavior_delta(rows):
        # Additive compatibility: historical rows lacking behavior remain valid.
        common = [metric for metric in BEHAVIOR_METRICS
                  if all(metric in row.get("behavior_delta", {}) for row in rows)]
        return ({"mean_paired_behavior_delta": {
            metric: float(np.mean([row["behavior_delta"][metric] for row in rows])) for metric in common}}
                if common else {})
    conditions = {}
    for condition in sorted({row["condition"] for row in pairs}):
        subset = [row for row in pairs if row["condition"] == condition]
        conditions[condition] = {"episodes": len(subset),
                                 "mean_paired_return_delta": float(np.mean([row["return_delta"] for row in subset])),
                                 "pre_successes": sum(row["pre_outcome"] == "success" for row in subset),
                                 "post_successes": sum(row["post_outcome"] == "success" for row in subset),
                                 **mean_behavior_delta(subset)}
    return {"mean_paired_return_delta": float(np.mean([row["return_delta"] for row in pairs])),
            "scenarios": pairs, "conditions": conditions, **mean_behavior_delta(pairs)}


def _train_credit(agent, task, seed, train_steps):
    world, cam = World(seed=seed, curriculum=task), Webcam(enabled=False)
    obs = world.observe()
    behavior = _BehaviorTrace(world)
    agent.reset_episode()
    ret, steps, episode = 0.0, 0, 0
    rows, events, updates = [], [], []
    frame = np.zeros((2, 48, 64), dtype=np.float32)
    last_update, last_event_update = int(agent.stats["updates"]), -1

    def scenario():
        condition = ("go" if world.st["go"] else "nogo") if task == "go_nogo" else "forage"
        return {"scenario_id": f"training:{task}:{seed}:{episode}", "task": task, "world_seed": seed,
                "condition": condition, "world_episode": episode}

    def record_update(global_step):
        nonlocal last_update, last_event_update
        if int(agent.stats["updates"]) == last_update:
            return
        last_update = int(agent.stats["updates"])
        stats = copy.deepcopy(agent.plasticity.summary())
        updates.append({"step": global_step, "update": last_update, "support_digest": _support_digest(agent),
                        "plasticity": stats})
        event = stats.get("last_event")
        if event is not None and event["update"] != last_event_update:
            last_event_update = event["update"]
            events.append({"step": global_step, **event})

    try:
        for global_step in range(1, train_steps + 1):
            frame, world.webcam_target = cam.read()
            action, _ = agent.act(obs, frame, training=True)
            obs, reward, done, info = world.step(action)
            behavior.observe(world, action)
            agent.feedback(reward, done, next_obs=obs, next_cam=frame)
            record_update(global_step)
            ret += reward
            steps += 1
            if done:
                rows.append(_row(scenario(), seed, "train", episode, ret, steps, info["outcome"], True,
                                 agent.plasticity.config.mode, behavior.summary()))
                episode += 1
                ret, steps = 0.0, 0
                obs = world.reset()
                behavior = _BehaviorTrace(world)
                agent.reset_episode()
        agent.flush(next_obs=obs, next_cam=frame)
        record_update(train_steps)
        if steps:
            rows.append(_row(scenario(), seed, "train", episode, ret, steps, "budget_exhausted", False,
                             agent.plasticity.config.mode, behavior.summary()))
    finally:
        cam.close()
    return rows, events, updates


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "graftbrain").rglob("*.py"))}


def _initial_model_digest(agent):
    arrays = []
    for name, module in (("brain", agent.brain), ("iface", agent.iface), ("mb", agent.mb),
                         ("curiosity", agent.curiosity), ("router", agent.plasticity.router)):
        if module is not None:
            for key, value in sorted(module.state_dict().items()):
                if torch.is_tensor(value):
                    arrays.extend((np.frombuffer(f"{name}:{key}".encode(), dtype=np.uint8), value.detach().cpu().numpy()))
    return _digest_arrays(arrays)


def _credit_graph_digest(graph):
    # Ports depend on annotations as well as edge geometry.
    metadata = json.dumps({"names": graph.names, "sites": graph.sites,
                           "role": graph.role.tolist(), "cell_type": graph.cell_type.tolist()}, sort_keys=True)
    return hashlib.sha256((_graph_digest(graph) + metadata).encode()).hexdigest()


def benchmark_graft_credit(out, specs=None, max_neurons=400, train_steps=384, tasks=None,
                            seeds=None, eval_seeds=None, microsteps=1):
    """Write seven paired arms to a new directory, retaining every outcome.

    Evaluation requests are scenario anchors: each creates a balanced go/no-go
    pair, or one forage episode. All scenarios are selected before any policy
    rewards. Training return summaries separate finished and unfinished episodes.
    """
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}; choose a new --out")
    specs, tasks, seeds, eval_seeds = validate_credit_benchmark(
        specs, max_neurons, train_steps, tasks, seeds, eval_seeds, microsteps)
    scenarios = build_credit_scenarios(tasks, eval_seeds, seeds)
    graphs = {seed: build([cx.load(spec, max_neurons=max_neurons, seed=seed) for spec in specs], seed=seed)
              for seed in seeds}
    source_sha256 = _source_hashes()
    manifest = {
        "schema_version": 1, "status": "running", "source_sha256": source_sha256,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "graftbrain": __version__},
        "platform": platform.platform(), "training_seeds": seeds, "evaluation_seeds": eval_seeds,
        "config": {"specs": specs, "max_neurons": max_neurons, "train_steps": train_steps,
                   "tasks": tasks, "microsteps": microsteps, "rollout": 16, "torch_threads": 1,
                   "arms": list(CREDIT_ARMS)},
        "scenarios": scenarios,
        "behavior_metrics": {
            "reporter_only": True, "fields": list(BEHAVIOR_METRICS),
            "target": "forage: world.st.goal; go_nogo: world.st.orb; never supplied to the agent",
            "distance_and_path_units": "world units",
            "path_length": "sum of actual planar displacement after each World.step, including collisions",
            "mean_squared_action_norm": "mean per-step sum of squared commands after World action clipping; dimensionless, not mechanical work",
            "action_saturation_fraction": "fraction of executed command coordinates with absolute value 1",
            "interpretation": "Signed post-minus-pre descriptive changes; closer is not universally better, especially for no-go",
        },
        "evaluation": {"actions": "policy_mean", "learning": False,
                       "state": "independent saved-state clone, reset episode per scenario",
                       "noise": "private per-scenario torch stream seeded with actual world_seed",
                       "camera": "fresh deterministic synthetic stream per scenario",
                       "selection": "ascending uint64 world-seed scan (wrap at maximum) by initial go/nogo condition; excludes training seeds; never reward-based",
                       "zero_action_reference": True},
        "evidence_limit": "Bounded synthetic/proxy CPU pilot; no general task benefit, biological fidelity, novelty, or physical reliability claim.",
        "arms": [],
    }
    summary = {"arms": [], "across_training_seeds": [], "evidence_limit": manifest["evidence_limit"],
               "inference": {"independent_training_seeds": len(seeds), "inferential_statistics": False,
                             "description": "Descriptive paired deltas only. Scenarios are repeated measurements, not independent training seeds."}}
    episodes, structural_events = [], []
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "manifest.json", manifest)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with _preserve_rng():
            references = zero_action_references(scenarios)
            episodes.extend(references)
            summary["zero_action_reference"] = {
                task: {condition: _returns([row for row in references if row["task"] == task and row["condition"] == condition])
                       for condition in sorted({row["condition"] for row in references if row["task"] == task})}
                for task in tasks}
            for seed, graph in graphs.items():
                for task in tasks:
                    task_scenarios = [row for row in scenarios if row["task"] == task]
                    for definition in CREDIT_ARMS:
                        started = time.perf_counter()
                        print(f"credit seed={seed} task={task} arm={definition['arm']}: pre/train/post", flush=True)
                        config = PlasticityConfig(mode="adaptive", growth_rule=definition["growth_rule"],
                                                  growth_init="zero", min_age=definition["min_age"])
                        agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, rollout=16, microsteps=microsteps,
                                      seed=seed, plasticity_config=config)
                        agent.rewiring_enabled = definition["rewiring_enabled"]
                        arm = {**definition, "task": task, "seed": seed, "plasticity_config": dataclasses.asdict(config),
                               "agent_config": {"rollout": agent.rollout, "gamma": agent.gamma, "gae_lambda": agent.gae_lambda,
                                                "ppo_clip": agent.ppo_clip, "microsteps": microsteps,
                                                "optimizer_groups": [{key: value for key, value in group.items() if key != "params"}
                                                                     for group in agent.opt.param_groups]},
                               "graph_digest": _credit_graph_digest(graph), "initial_support_digest": _support_digest(agent),
                               "initial_model_digest": _initial_model_digest(agent),
                               "bucket_budgets": agent.plasticity.bucket_budgets.tolist()}
                        manifest["arms"].append(arm)
                        pre = evaluate_credit(agent, graph, task_scenarios, seed, microsteps, "pre")
                        train_started = time.perf_counter()
                        train, events, updates = _train_credit(agent, task, seed, train_steps)
                        train_seconds = time.perf_counter() - train_started
                        post = evaluate_credit(agent, graph, task_scenarios, seed, microsteps, "post")
                        for row in pre + train + post:
                            row["arm"] = definition["arm"]
                        episodes.extend(pre + train + post)
                        for event in events:
                            structural_events.append({"arm": definition["arm"], "task": task, "training_seed": seed, **event})
                        stats = copy.deepcopy(agent.plasticity.summary())
                        paired = paired_credit_changes(pre, post)
                        summary["arms"].append({
                            **arm, "train_steps": train_steps, "updates": int(agent.stats["updates"]),
                            "candidate_records": len(graph.graft_pre), "active_records": int(agent.plasticity.active_mask.sum()),
                            "final_support_digest": _support_digest(agent), "structural_diagnostics": stats,
                            "structural_events": events, "update_trace": updates,
                            "training_complete": _returns([row for row in train if row["terminal"]]),
                            "training_incomplete": _returns([row for row in train if not row["terminal"]]),
                            "pre_evaluation": _returns(pre), "post_evaluation": _returns(post), "paired": paired,
                            "train_seconds": train_seconds, "wall_seconds": time.perf_counter() - started,
                        })
                        print(f"credit seed={seed} task={task} arm={definition['arm']}: delta={paired['mean_paired_return_delta']:.6g}, swaps={stats.get('swaps', 0)}", flush=True)
                        _write_json(out / "manifest.json", manifest)
                        _write_json(out / "episodes.json", episodes)
                        _write_json(out / "structural_events.json", structural_events)
                        _write_json(out / "summary.json", summary)
            for task in tasks:
                for definition in CREDIT_ARMS:
                    results = [row for row in summary["arms"] if row["task"] == task and row["arm"] == definition["arm"]]
                    summary["across_training_seeds"].append({
                        "task": task, "arm": definition["arm"], "training_seeds": [row["seed"] for row in results],
                        "per_seed_mean_return_delta": [row["paired"]["mean_paired_return_delta"] for row in results],
                        "mean_of_seed_deltas": float(np.mean([row["paired"]["mean_paired_return_delta"] for row in results])),
                    })
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        torch.set_num_threads(previous_threads)
        manifest["source_sha256_end"] = _source_hashes()
        manifest["source_changed_during_run"] = manifest["source_sha256_end"] != source_sha256
        _write_json(out / "manifest.json", manifest)
        _write_json(out / "episodes.json", episodes)
        _write_json(out / "structural_events.json", structural_events)
        _write_json(out / "summary.json", summary)
    return out
