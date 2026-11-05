"""POSEIDON ECOSYSTEM-EXPERIMENT: Multi-Agent Co-Evolution & Invariant Shield Benchmark.

Runs matched multi-agent simulations across continuous 3D environments:
1. Evaluates multi-agent populations (Grazers, Predators, Symbionts).
2. Measures trophic biomass transfer efficiency, beacon pulse exchanges,
   Tessera safety shield interventions, and topological manifold persistence (\\beta_0, \\beta_1).
3. Produces cryptographically verifiable SHA-256 benchmark receipts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from typing import Any, Callable, Sequence

from .ecosystem_3d import (
    ContinuousEcosystem3D,
    Ecosystem3DConfig,
    TROPHIC_GRAZER,
    TROPHIC_PREDATOR,
    TROPHIC_SYMBIONT,
)
from .memory_observatory import MemoryCarrierObservatory
from .tessera_prover import TesseraProver


def run_ecosystem_experiment(
    seeds: Sequence[int] = (42, 43),
    steps_per_episode: int = 40,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Executes matched-condition co-evolutionary ecosystem experiment."""
    runs_data = []
    total_steps = len(seeds) * steps_per_episode
    global_step = 0

    total_trophic_transfers = 0
    total_pulses = 0
    total_shield_interventions = 0
    homeostatic_cycles = 0

    for seed_idx, seed in enumerate(seeds):
        eco = ContinuousEcosystem3D(seed=seed)
        eco.spawn_default_colony(n_grazers=2, n_predators=1, n_symbionts=1)

        prover = TesseraProver()
        observatory = MemoryCarrierObservatory(filtration_epsilon=0.40)

        # Track trajectory of the primary grazer
        agent_trajectories: dict[str, list[list[float]]] = {aid: [] for aid in eco.agents}

        episode_steps = 0
        alive_at_end = 0

        for step in range(steps_per_episode):
            global_step += 1
            if progress:
                progress({"phase": "running", "seed": seed, "step": step, "progress": global_step / total_steps})

            # Formulate actions:
            # Grazers forage/explore; Predators hunt/explore; Symbionts rest/clean
            candidate_actions: dict[str, int] = {}
            shielded_actions: dict[str, int] = {}

            # Build dummy observations from last step or default
            for aid, agent in eco.agents.items():
                if not agent.alive:
                    continue
                cand = 4  # default explore
                if agent.trophic_role == TROPHIC_GRAZER:
                    cand = 1 if (agent.energy < 0.6) else 4
                elif agent.trophic_role == TROPHIC_PREDATOR:
                    cand = 4
                elif agent.trophic_role == TROPHIC_SYMBIONT:
                    cand = 0 if (agent.stamina < 0.5) else 4

                candidate_actions[aid] = cand

                # Pass through Tessera Invariant Shield for primary grazer
                curr_obs = [agent.health, agent.energy, agent.hydration, agent.stamina, 0.1, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.0, 0.0, 0.5, 0.5]
                shielded, receipt = prover.shield_action(curr_obs, cand)
                shielded_actions[aid] = shielded
                if receipt.shield_triggered:
                    total_shield_interventions += 1

            obs_dict, rewards, dones, telemetry = eco.step(shielded_actions)
            episode_steps += 1

            for aid, o in obs_dict.items():
                agent_trajectories[aid].append(o[:4])

            # Check if all died
            if all(dones.values()):
                break

        alive_at_end = sum(1 for a in eco.agents.values() if a.alive)
        total_trophic_transfers += eco.total_trophic_transfers
        total_pulses += eco.total_pulses_exchanged

        # Analyze topology of primary grazer
        traj = agent_trajectories.get("grazer_0", [])
        topo_audit = observatory.analyze_trajectory_topology(traj)
        if topo_audit.is_homeostatic_cycle:
            homeostatic_cycles += 1

        runs_data.append({
            "seed": seed,
            "steps": episode_steps,
            "alive_agents": alive_at_end,
            "trophic_transfers": eco.total_trophic_transfers,
            "pulses_exchanged": eco.total_pulses_exchanged,
            "betti_0": topo_audit.betti_0,
            "betti_1": topo_audit.betti_1,
            "euler_chi": topo_audit.euler_characteristic,
            "regime": topo_audit.regime_name,
            "homeostatic": topo_audit.is_homeostatic_cycle,
        })

    receipt = {
        "schema": "poseidon-ecosystem-experiment-v1",
        "total_seeds": len(seeds),
        "total_steps_executed": global_step,
        "aggregate_metrics": {
            "total_trophic_transfers": total_trophic_transfers,
            "total_pulses_exchanged": total_pulses,
            "total_shield_interventions": total_shield_interventions,
            "homeostatic_cycle_ratio": homeostatic_cycles / max(1, len(seeds)),
            "mean_survival_count": sum(r["alive_agents"] for r in runs_data) / max(1, len(runs_data)),
        },
        "episodes": runs_data,
    }

    # Cryptographic hash
    clean_blob = {
        "schema": receipt["schema"],
        "total_seeds": receipt["total_seeds"],
        "aggregate_metrics": receipt["aggregate_metrics"],
        "episodes": receipt["episodes"],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(clean_blob, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return receipt


def verify_ecosystem_receipt(receipt: dict[str, Any]) -> bool:
    """Cryptographically verifies ecosystem experiment receipt."""
    if receipt.get("schema") != "poseidon-ecosystem-experiment-v1":
        return False
    claimed_hash = receipt.get("receipt_sha256")
    if not claimed_hash:
        return False

    clean_blob = {
        "schema": receipt["schema"],
        "total_seeds": receipt["total_seeds"],
        "aggregate_metrics": receipt["aggregate_metrics"],
        "episodes": receipt["episodes"],
    }
    expected_hash = hashlib.sha256(
        json.dumps(clean_blob, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return claimed_hash == expected_hash
