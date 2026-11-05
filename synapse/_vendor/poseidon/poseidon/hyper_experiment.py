"""POSEIDON HYPER EXPERIMENT: Matched-Seed Comparative Assay for HyperAgent v3.0.

Benchmarking Apex HyperAgent against ablated and prior baselines across both:
- Discrete POMDP TidePool with sensor noise and stochastic fog
- Continuous 3D TidePool with hydrodynamic surge and fluid drag

Ablation Arms:
1. hyper_v3: Complete Causal-Quantum Morphogenetic HyperAgent
2. no_quantum: Quantum amplitude interference disabled (classical planning)
3. no_causal: SCM interventional do(a) projection disabled (pure correlation)
4. no_morphogenesis: Neuroplastic hidden layer adaptation disabled
5. cognitive_v2: Supermix Poseidon v2.0 CognitiveAgent champion
6. frozen_tidal: Pure un-augmented frozen baseline policy
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import time
from typing import Any, Callable, Sequence

import torch

from .cognitive_agent import CognitiveAgent
from .core import CoreRuntime, active_core_path
from .genesis_3d import ContinuousTidePool3D, Genesis3DConfig
from .hyper_agent import HyperAgent
from .provenance import assert_source_current
from .world import ACTIONS, N_ACTIONS, OBS_SIZE, TidePool, WORLD_VERSION
from .worldlab import POMDPConfig, POMDPTidePool

HYPER_EXPERIMENT_SCHEMA = "poseidon-hyper-experiment-v3"
HYPER_ARMS = ("hyper_v3", "no_quantum", "no_causal", "no_morphogenesis", "cognitive_v2", "frozen_tidal")


def _run_single_seed_pomdp(
    arm: str,
    seed: int,
    max_steps: int,
    scarcity: float,
    core: CoreRuntime | None = None,
) -> dict[str, Any]:
    core = core or CoreRuntime(active_core_path())
    env = POMDPTidePool(POMDPConfig(scarcity=scarcity, seed=seed), max_steps=max_steps)
    obs = env.observe()

    if arm == "frozen_tidal":
        total_reward = 0.0
        steps = 0
        survived = True
        decisions = []
        for _ in range(max_steps):
            action = core.act(obs)
            decisions.append({"action": action, "override": False})
            step_res = env.step(action)
            obs = step_res.observation
            reward = step_res.reward
            done = step_res.done
            total_reward += reward
            steps += 1
            if done:
                survived = env.health > 0.0
                break
        return {
            "arm": arm,
            "seed": seed,
            "total_reward": float(total_reward),
            "survived": bool(survived),
            "steps": int(steps),
            "overrides": 0,
            "override_rate": 0.0,
            "quantum_coherence": 0.0,
            "active_neurons": 0,
        }

    elif arm == "cognitive_v2":
        agent = CognitiveAgent(core=core)
        res = agent.rollout(env, max_steps=max_steps)
        return {
            "arm": arm,
            "seed": seed,
            "total_reward": float(res["total_reward"]),
            "survived": bool(res["survived"]),
            "steps": int(res["steps"]),
            "overrides": int(res["overrides"]),
            "override_rate": float(res["override_rate"]),
            "quantum_coherence": 0.0,
            "active_neurons": 64,
        }

    # HyperAgent Arms
    agent = HyperAgent(core=core)
    if arm == "no_quantum":
        agent.quantum.enabled = False
    elif arm == "no_causal":
        agent.causeway.enabled = False
    elif arm == "no_morphogenesis":
        agent.molt.max_hidden = agent.molt.min_hidden

    total_reward = 0.0
    steps = 0
    survived = True
    decisions = []

    for _ in range(max_steps):
        decision = agent.act(obs)
        decisions.append(decision.to_dict())

        step_res = env.step(decision.action)
        next_obs = step_res.observation
        reward = step_res.reward
        done = step_res.done
        total_reward += reward
        steps += 1

        agent.observe_transition(obs, decision.action, reward, next_obs, done)
        obs = next_obs
        if done:
            survived = env.health > 0.0
            break

    overrides = sum(1 for d in decisions if d["override"])
    mean_coherence = fmean([d["quantum_coherence"] for d in decisions]) if decisions else 0.0

    return {
        "arm": arm,
        "seed": seed,
        "total_reward": float(total_reward),
        "survived": bool(survived),
        "steps": int(steps),
        "overrides": int(overrides),
        "override_rate": float(overrides / steps) if steps > 0 else 0.0,
        "quantum_coherence": float(mean_coherence),
        "active_neurons": int(agent.molt.current_hidden),
    }


def _run_single_seed_3d(
    arm: str,
    seed: int,
    max_steps: int,
    core: CoreRuntime | None = None,
) -> dict[str, Any]:
    core = core or CoreRuntime(active_core_path())
    env = ContinuousTidePool3D(seed=seed)

    if arm == "frozen_tidal":
        obs = [0.0] * OBS_SIZE
        obs[0] = 1.0
        obs[1] = 0.85
        obs[2] = 0.85
        obs[3] = 0.90
        total_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            action = core.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        return {
            "arm": arm,
            "seed": seed,
            "total_reward": float(total_reward),
            "survived": bool(env.health > 0.0),
            "steps": int(steps),
            "overrides": 0,
            "final_health": float(env.health),
        }

    # HyperAgent arm in 3D
    agent = HyperAgent(core=core)
    if arm == "no_quantum":
        agent.quantum.enabled = False
    elif arm == "no_causal":
        agent.causeway.enabled = False
    elif arm == "no_morphogenesis":
        agent.molt.max_hidden = agent.molt.min_hidden

    res = agent.rollout_3d(env, max_steps=max_steps)
    return {
        "arm": arm,
        "seed": seed,
        "total_reward": float(res["total_reward"]),
        "survived": bool(res["survived"]),
        "steps": int(res["steps"]),
        "overrides": int(res["overrides"]),
        "final_health": float(res["final_health"]),
    }


def run_hyper_experiment(
    seed: int = 42,
    episodes: int = 4,
    max_steps: int = 64,
    scarcity: float = 2.5,
    core: CoreRuntime | None = None,
) -> dict[str, Any]:
    """Runs matched-seed evaluation across all 6 arms in POMDP and 3D substrates."""
    core = core or CoreRuntime(active_core_path())
    seeds = [seed + i * 10007 for i in range(episodes)]

    arm_results_pomdp: dict[str, list[dict[str, Any]]] = {arm: [] for arm in HYPER_ARMS}
    arm_results_3d: dict[str, list[dict[str, Any]]] = {arm: [] for arm in HYPER_ARMS}

    for s in seeds:
        for arm in HYPER_ARMS:
            res_p = _run_single_seed_pomdp(arm, s, max_steps, scarcity, core)
            arm_results_pomdp[arm].append(res_p)
            res_3 = _run_single_seed_3d(arm, s, max_steps, core)
            arm_results_3d[arm].append(res_3)

    summary_pomdp = {}
    for arm, records in arm_results_pomdp.items():
        rewards = [r["total_reward"] for r in records]
        survivals = [1.0 if r["survived"] else 0.0 for r in records]
        summary_pomdp[arm] = {
            "mean_reward": round(fmean(rewards), 4),
            "reward_std": round(stdev(rewards) if len(rewards) > 1 else 0.0, 4),
            "survival_rate": round(fmean(survivals), 4),
            "mean_overrides": round(fmean([r["overrides"] for r in records]), 2),
            "override_rate": round(fmean([r["override_rate"] for r in records]), 4),
        }

    summary_3d = {}
    for arm, records in arm_results_3d.items():
        rewards = [r["total_reward"] for r in records]
        survivals = [1.0 if r["survived"] else 0.0 for r in records]
        healths = [r["final_health"] for r in records]
        summary_3d[arm] = {
            "mean_reward": round(fmean(rewards), 4),
            "reward_std": round(stdev(rewards) if len(rewards) > 1 else 0.0, 4),
            "survival_rate": round(fmean(survivals), 4),
            "mean_final_health": round(fmean(healths), 4),
        }

    payload = {
        "schema": HYPER_EXPERIMENT_SCHEMA,
        "base_seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "seeds": seeds,
        "summary_pomdp": summary_pomdp,
        "summary_3d": summary_3d,
        "records_pomdp": arm_results_pomdp,
        "records_3d": arm_results_3d,
    }

    serialized = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    payload["receipt_hash"] = digest

    receipt_dir = Path("outputs/receipts")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = receipt_dir / f"hyper_experiment_{seed}_{digest[:10]}.json"
    receipt_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["receipt_path"] = str(receipt_file)

    return payload
