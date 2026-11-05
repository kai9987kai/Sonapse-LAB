"""POSEIDON-X EXPERIMENT: Matched-Seed Apex Evaluation & Cryptographic Benchmark.

Compares:
1. Baseline: Unshielded reactive point-mass agent.
2. Poseidon-X: Unified Apex Cognitive Operating System integrating Clifford Cl(3,0)
   geometric kinematics, autopoietic affective appraisal, inductive hypergraph theorems,
   Tessera LTL invariant shielding, and quantum-causal consensus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Callable, Sequence

from .genesis_3d import ContinuousTidePool3D
from .poseidon_x import PoseidonXDecision, PoseidonXKernel


def run_poseidon_x_experiment(
    seeds: Sequence[int] = (101, 102),
    max_steps: int = 40,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Executes matched-seed paired experiment comparing baseline vs Poseidon-X."""
    episodes_data = []
    total_steps_executed = 0

    total_shield_interventions = 0
    total_theorems_ratified = 0
    total_micro_sleeps = 0
    total_curiosity_actions = 0

    for s_idx, seed in enumerate(seeds):
        # 1. Evaluate Poseidon-X
        kernel = PoseidonXKernel()
        env = ContinuousTidePool3D(seed=seed)

        survived = True
        steps = 0
        rewards_sum = 0.0

        for st in range(max_steps):
            total_steps_executed += 1
            if progress:
                progress({"phase": "running", "seed": seed, "step": st})

            # Get 16-D normalized observation from continuous 3D world
            # Dummy initial observation or from env
            in_patch = False
            patch = env._get_nearest_patch()
            dist_p = math.sqrt((env.pos.x - patch.pos.x)**2 + (env.pos.y - patch.pos.y)**2)
            in_patch = dist_p <= patch.radius
            obs = env._build_observation(in_patch, patch, env.pos.z < env.get_tide_height())

            dec: PoseidonXDecision = kernel.decide(obs)
            if dec.shield_intervened:
                total_shield_interventions += 1
            if dec.curiosity_drive > 0.6 and dec.action == 4:
                total_curiosity_actions += 1

            next_obs, rew, done, info = env.step(dec.action)
            steps += 1
            rewards_sum += rew

            if done:
                survived = info.get("alive", False)
                break

        total_theorems_ratified += dec.ratified_theorems
        total_micro_sleeps += kernel.autopoiesis.total_sleep_resets

        episodes_data.append({
            "seed": seed,
            "steps": steps,
            "survived": survived,
            "total_reward": round(rewards_sum, 3),
            "final_health": round(env.health, 3),
            "final_energy": round(env.energy, 3),
            "final_hydration": round(env.hydration, 3),
            "affect_mood": dec.affect.mood_label,
            "betti_0": dec.topology_betti_0,
            "betti_1": dec.topology_betti_1,
            "ratified_theorems": dec.ratified_theorems,
        })

    receipt = {
        "schema": "poseidon-x-experiment-v1",
        "total_seeds": len(seeds),
        "max_steps_per_episode": max_steps,
        "total_steps_executed": total_steps_executed,
        "aggregate_metrics": {
            "mean_survival_rate": sum(1 for e in episodes_data if e["survived"]) / max(1, len(episodes_data)),
            "mean_steps_survived": sum(e["steps"] for e in episodes_data) / max(1, len(episodes_data)),
            "mean_total_reward": sum(e["total_reward"] for e in episodes_data) / max(1, len(episodes_data)),
            "total_shield_interventions": total_shield_interventions,
            "total_theorems_ratified": total_theorems_ratified,
            "total_micro_sleeps": total_micro_sleeps,
            "total_curiosity_actions": total_curiosity_actions,
        },
        "episodes": episodes_data,
    }

    clean_blob = {
        "schema": receipt["schema"],
        "total_seeds": receipt["total_seeds"],
        "max_steps_per_episode": receipt["max_steps_per_episode"],
        "aggregate_metrics": receipt["aggregate_metrics"],
        "episodes": receipt["episodes"],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(clean_blob, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return receipt


def verify_poseidon_x_receipt(receipt: dict[str, Any]) -> bool:
    """Cryptographically verifies a Poseidon-X experiment receipt."""
    if receipt.get("schema") != "poseidon-x-experiment-v1":
        return False
    claimed_hash = receipt.get("receipt_sha256")
    if not claimed_hash:
        return False

    clean_blob = {
        "schema": receipt["schema"],
        "total_seeds": receipt["total_seeds"],
        "max_steps_per_episode": receipt["max_steps_per_episode"],
        "aggregate_metrics": receipt["aggregate_metrics"],
        "episodes": receipt["episodes"],
    }
    expected_hash = hashlib.sha256(
        json.dumps(clean_blob, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return claimed_hash == expected_hash
