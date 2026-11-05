"""Bounded, paired CPU experiments; artifacts describe observations, not superiority."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from . import connectomes as cx
from .brain import Agent
from .graft import build
from .plasticity import PlasticityConfig
from .senses import Webcam
from .world import TASKS, World

MODES = ("legacy", "static", "routing", "structural", "adaptive")
BENCHMARK_SPECS = ("proxy:flywire", "proxy:manc")


def validate_benchmark(specs=None, max_neurons=400, train_steps=512, tasks=None,
                       seeds=None, eval_seeds=None, microsteps=2):
    """Validate the experiment before creating its output directory."""
    specs = list(BENCHMARK_SPECS if specs is None else specs)
    tasks = list(["forage", "go_nogo"] if tasks is None else tasks)
    seeds = list([0] if seeds is None else seeds)
    eval_seeds = list([1000, 1001] if eval_seeds is None else eval_seeds)
    for name, value in (("max_neurons", max_neurons), ("train_steps", train_steps), ("microsteps", microsteps)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not specs or any(not isinstance(spec, str) or not spec for spec in specs):
        raise ValueError("at least one nonempty connectome spec is required")
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError(f"tasks must be unique names from {', '.join(TASKS)}")
    for name, values in (("training", seeds), ("evaluation", eval_seeds)):
        if not values or len(set(values)) != len(values) or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in values):
            raise ValueError(f"{name} seeds must be unique nonnegative integers")
    if set(seeds) & set(eval_seeds):
        raise ValueError("training and held-out evaluation seeds must be disjoint")
    return specs, tasks, seeds, eval_seeds


def _code_status():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _digest_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _graph_digest(graph):
    return _digest_arrays([graph.offsets, graph.origin, graph.base_pre, graph.base_post, graph.base_w,
                           graph.graft_pre, graph.graft_post, graph.graft_w0, graph.graft_site])


def _support_digest(agent):
    return _digest_arrays([agent.plasticity.active_mask.detach().cpu().numpy()])


def _episode_row(mode, task, stage, seed, train_seed, episode, ret, steps, outcome, terminal):
    return {"mode": mode, "task": task, "stage": stage, "seed": int(seed), "training_seed": int(train_seed),
            "episode": int(episode), "return": float(ret), "steps": int(steps), "outcome": outcome,
            "terminal": bool(terminal)}


def evaluate_frozen(agent, graph, task, seeds, train_seed=0, microsteps=2):
    """Evaluate independent saved-state clones with policy means and no learning.

    The supplied training agent is untouched. Every evaluation episode starts
    from the same trained state and fresh episodic neural/compass/camera state.
    Repeated world seeds therefore produce identical outcomes, independent of
    which arm or episode ran previously.
    """
    saved = copy.deepcopy(agent.state_dict())
    config = agent.plasticity.config
    rows = []
    with torch.random.fork_rng(devices=[]):
        for episode, seed in enumerate(seeds):
            clone = Agent(graph, World.obs_dim, (2, 48, 64), 5, seed=train_seed,
                          microsteps=microsteps, plasticity_config=config)
            clone.load_state_dict(copy.deepcopy(saved))
            clone.reset_episode()
            torch.manual_seed(seed)
            world = World(seed=seed, curriculum=task)
            cam = Webcam(enabled=False)
            obs = world.observe()
            ret, steps, done, info = 0.0, 0, False, {}
            try:
                with torch.no_grad():
                    while not done:
                        frame, centroid = cam.read()
                        world.webcam_target = centroid
                        action, _ = clone.act(obs, frame, training=False)
                        obs, reward, done, info = world.step(action)
                        ret += reward
                        steps += 1
                rows.append(_episode_row(config.mode, task, "eval", seed, train_seed, episode,
                                         ret, steps, info["outcome"], True))
            finally:
                cam.close()
    return rows


def _train(agent, task, seed, train_steps):
    world = World(seed=seed, curriculum=task)
    cam = Webcam(enabled=False)
    obs = world.observe()
    agent.reset_episode()
    ret, steps, episode = 0.0, 0, 0
    rows = []
    frame = np.zeros((2, 48, 64), dtype=np.float32)
    try:
        for _ in range(train_steps):
            frame, centroid = cam.read()
            world.webcam_target = centroid
            action, _ = agent.act(obs, frame, training=True)
            obs, reward, done, info = world.step(action)
            agent.feedback(reward, done, next_obs=obs, next_cam=frame)
            ret += reward
            steps += 1
            if done:
                rows.append(_episode_row(agent.plasticity.config.mode, task, "train", seed, seed,
                                         episode, ret, steps, info["outcome"], True))
                episode += 1
                ret, steps = 0.0, 0
                obs = world.reset()
                agent.reset_episode()
        agent.flush(next_obs=obs, next_cam=frame)
        if steps:
            rows.append(_episode_row(agent.plasticity.config.mode, task, "train", seed, seed,
                                     episode, ret, steps, "budget_exhausted", False))
    finally:
        cam.close()
    return rows


def _outcome_rates(rows):
    if not rows:
        return {}
    return {outcome: sum(row["outcome"] == outcome for row in rows) / len(rows)
            for outcome in sorted({row["outcome"] for row in rows})}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def benchmark_plasticity(out, specs=None, max_neurons=400, train_steps=512, tasks=None,
                         seeds=None, eval_seeds=None, microsteps=2):
    """Write a reproducible five-mode pilot to a new directory and return its path.

    Training budgets apply separately to every mode, task, and training seed.
    Evaluation runs one complete environment episode for each held-out seed.
    Candidate graphs, model/action seeds, and initial sparse masks are paired.
    """
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}; choose a new --out")
    specs, tasks, seeds, eval_seeds = validate_benchmark(specs, max_neurons, train_steps, tasks,
                                                       seeds, eval_seeds, microsteps)
    code = _code_status()
    graphs = {seed: build([cx.load(spec, max_neurons=max_neurons, seed=seed) for spec in specs], seed=seed)
              for seed in seeds}
    if any(graph.n < 600 for graph in graphs.values()):
        raise ValueError("benchmark graph must contain at least 600 total neurons; increase --max-neurons or add connectomes")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "status": "running", "code": code,
        "training_seeds": seeds, "evaluation_seeds": eval_seeds,
        "config": {"specs": specs, "max_neurons": max_neurons, "train_steps": train_steps,
                   "tasks": tasks, "microsteps": microsteps, "torch_threads": 1, "modes": list(MODES)},
        "evaluation": {"actions": "policy_mean", "learning": False, "episodes_per_task_seed": 1,
                       "state": "fresh trained-state clone and episodic reset per episode",
                       "camera": "deterministic local synthetic stream"},
        "evidence_limit": "Small CPU synthetic/proxy pilot; no general task advantage or biological-fidelity claim.",
        "arms": [],
    }
    episodes, summary = [], {"arms": [], "evidence_limit": manifest["evidence_limit"]}
    _write_json(out / "manifest.json", manifest)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            for seed, graph in graphs.items():
                for task in tasks:
                    for mode in MODES:
                        start = time.perf_counter()
                        config = PlasticityConfig(mode=mode)
                        # Agent construction re-seeds model and action sampling identically.
                        agent = Agent(graph, World.obs_dim, (2, 48, 64), 5, microsteps=microsteps,
                                      seed=seed, plasticity_config=config)
                        arm = {"mode": mode, "task": task, "seed": seed,
                               "plasticity_config": dataclasses.asdict(config),
                               "graph_digest": _graph_digest(graph), "initial_support_digest": _support_digest(agent)}
                        manifest["arms"].append(arm)
                        train_start = time.perf_counter()
                        train_rows = _train(agent, task, seed, train_steps)
                        train_seconds = time.perf_counter() - train_start
                        eval_start = time.perf_counter()
                        eval_rows = evaluate_frozen(agent, graph, task, eval_seeds, train_seed=seed, microsteps=microsteps)
                        eval_seconds = time.perf_counter() - eval_start
                        episodes.extend(train_rows + eval_rows)
                        stats = agent.plasticity.summary()
                        summary["arms"].append({
                            **arm, "candidate_records": len(graph.graft_pre),
                            "active_records": int(agent.plasticity.active_mask.sum()),
                            "swaps": int(stats.get("swaps", 0)), "gate_spread": float(stats.get("gate_spread", 0.0)),
                            "final_support_digest": _support_digest(agent),
                            "train_steps": train_steps, "updates": int(agent.stats["updates"]),
                            "mean_training_return": float(np.mean([row["return"] for row in train_rows])),
                            "mean_evaluation_return": float(np.mean([row["return"] for row in eval_rows])),
                            "training_outcome_rates": _outcome_rates([row for row in train_rows if row["terminal"]]),
                            "evaluation_outcome_rates": _outcome_rates(eval_rows),
                            "train_seconds": train_seconds, "evaluation_seconds": eval_seconds,
                            "wall_seconds": time.perf_counter() - start,
                        })
                        _write_json(out / "manifest.json", manifest)
                        _write_json(out / "episodes.json", episodes)
                        _write_json(out / "summary.json", summary)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        torch.set_num_threads(previous_threads)
        _write_json(out / "manifest.json", manifest)
    return out
