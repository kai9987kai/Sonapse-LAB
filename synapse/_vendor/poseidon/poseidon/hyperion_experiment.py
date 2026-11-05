"""Empirical evaluation and weightless verification for the HYPERION frontier super-controller.

Compares frozen core policy, AURA, METAMORPH, CHIMERA, and HYPERION across paired TidePool seeds.
Generates content-addressed verifiable receipts with SHA-256 seals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .aura import AuraController
from .chimera import ChimeraController
from .hyperion import HyperionController
from .metamorph import MetamorphController
from .world import TidePool, rollout

SCHEMA = "poseidon-hyperion-experiment-v1"


def run_hyperion_experiment(
    core: Any,
    seeds: Optional[List[int]] = None,
    max_steps: int = 48,
    scarcity: float = 2.5,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    seeds = seeds or [303000001, 303000002, 303000003, 303000004]

    controllers = {
        "core": core,
        "aura": AuraController(core=core),
        "metamorph": MetamorphController(core=core),
        "chimera": ChimeraController(core=core),
        "hyperion": HyperionController(core=core),
    }

    episodes: List[Dict[str, Any]] = []
    total_runs = len(seeds) * len(controllers)
    completed = 0

    for seed in seeds:
        for arm_name, controller in controllers.items():
            if hasattr(controller, "reset"):
                controller.reset(scarcity=scarcity)

            ep = rollout(controller, seed=seed, max_steps=max_steps, scarcity=scarcity)

            ep_summary = {
                "arm": arm_name,
                "seed": seed,
                "steps": ep["steps"],
                "survived": ep["survived"],
                "reward": round(float(ep["reward"]), 4),
                "death_reason": ep.get("death_reason"),
            }

            if arm_name == "hyperion" and isinstance(controller, HyperionController):
                ep_summary["quantum_entropy"] = controller.causeway.von_neumann_entropy()
                ep_summary["lattice_coherence"] = controller.lattice.multi_scale_coherence()
                ep_summary["holographic_items"] = controller.hologram.item_count
                ep_summary["mean_sdi"] = controller.ghost.summary()["mean_spectral_divergence"]
                ep_summary["destructive_conflicts"] = len(controller.causeway.detect_destructive_conflicts())
                ep_summary["final_stage"] = controller.molt.stage.value
                ep_summary["sleep_cycles"] = controller.morpheus.total_sleep_cycles
                ep_summary["rem_replays"] = controller.morpheus.total_rem_replays
                ep_summary["phase_gain"] = round(controller.morpheus.accumulated_phase_alignment_gain, 4)
                ep_summary["scarcity_entropy"] = round(controller.prometheus.compute_scarcity_entropy(), 4)
                ep_summary["virtual_capacitor"] = round(controller.intermittent.virtual_capacitor, 4)
                ep_summary["total_harvested_energy"] = round(controller.intermittent.total_harvested_energy, 4)
                ep_summary["resuscitations"] = controller.intermittent.total_resuscitations
                ep_summary["search_entries"] = controller.nexus_search.stats()["indexed_count"]

            episodes.append(ep_summary)
            completed += 1
            if progress:
                progress({"phase": "running", "completed": completed, "total": total_runs, "seed": seed, "arm": arm_name})

    # Aggregate summaries per arm
    arm_stats: Dict[str, Dict[str, Any]] = {}
    for arm in controllers:
        arm_eps = [e for e in episodes if e["arm"] == arm]
        surv_rate = sum(1 for e in arm_eps if e["survived"]) / max(1, len(arm_eps))
        mean_rew = float(np.mean([e["reward"] for e in arm_eps]))
        mean_steps = float(np.mean([e["steps"] for e in arm_eps]))
        arm_stats[arm] = {
            "episodes": len(arm_eps),
            "survival_rate": round(surv_rate, 4),
            "mean_reward": round(mean_rew, 4),
            "mean_steps": round(mean_steps, 2),
        }

    hyp_eps = [e for e in episodes if e["arm"] == "hyperion"]
    mean_qe = float(np.mean([e.get("quantum_entropy", 0.0) for e in hyp_eps]))
    mean_lc = float(np.mean([e.get("lattice_coherence", 0.0) for e in hyp_eps]))
    total_conflicts = sum(e.get("destructive_conflicts", 0) for e in hyp_eps)
    mean_sdi = float(np.mean([e.get("mean_sdi", 0.0) for e in hyp_eps]))
    mean_holo_items = float(np.mean([e.get("holographic_items", 0.0) for e in hyp_eps]))
    total_sleep_cycles = sum(e.get("sleep_cycles", 0) for e in hyp_eps)
    total_rem_replays = sum(e.get("rem_replays", 0) for e in hyp_eps)
    mean_phase_gain = float(np.mean([e.get("phase_gain", 0.0) for e in hyp_eps]))
    mean_scarcity_entropy = float(np.mean([e.get("scarcity_entropy", 0.0) for e in hyp_eps]))
    total_harvested = float(np.sum([e.get("total_harvested_energy", 0.0) for e in hyp_eps]))
    total_resusc = sum(e.get("resuscitations", 0) for e in hyp_eps)

    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "seeds": seeds,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "arm_stats": arm_stats,
        "hyperion_telemetry": {
            "mean_quantum_entropy": round(mean_qe, 6),
            "mean_lattice_coherence": round(mean_lc, 6),
            "total_conflicts_detected": total_conflicts,
            "overall_mean_sdi": round(mean_sdi, 6),
            "mean_holographic_items_stored": round(mean_holo_items, 2),
            "total_sleep_cycles": total_sleep_cycles,
            "total_rem_replays": total_rem_replays,
            "mean_phase_gain": round(mean_phase_gain, 4),
            "mean_scarcity_entropy": round(mean_scarcity_entropy, 4),
            "total_harvested_energy": round(total_harvested, 4),
            "total_resuscitations": total_resusc,
        },
        "episodes": episodes,
    }

    payload = json.dumps({
        "schema": receipt["schema"],
        "seeds": receipt["seeds"],
        "max_steps": receipt["max_steps"],
        "scarcity": receipt["scarcity"],
        "arm_stats": receipt["arm_stats"],
        "hyperion_telemetry": receipt["hyperion_telemetry"],
    }, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    receipt["experiment_id"] = receipt["receipt_sha256"][:32]

    return receipt


def verify_hyperion_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Verify cryptographically sealed HYPERION experiment receipt."""
    if not isinstance(receipt, dict):
        raise ValueError("Receipt must be a JSON object")
    if receipt.get("schema") != SCHEMA:
        raise ValueError(f"Receipt schema mismatch: expected {SCHEMA}")

    payload = json.dumps({
        "schema": receipt["schema"],
        "seeds": receipt["seeds"],
        "max_steps": receipt["max_steps"],
        "scarcity": receipt["scarcity"],
        "arm_stats": receipt["arm_stats"],
        "hyperion_telemetry": receipt["hyperion_telemetry"],
    }, sort_keys=True)
    expected_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if receipt.get("receipt_sha256") != expected_hash:
        raise ValueError("Receipt SHA-256 checksum mismatch")

    return {
        "verified": True,
        "schema": receipt["schema"],
        "experiment_id": receipt["experiment_id"],
        "receipt_sha256": receipt["receipt_sha256"],
        "arm_stats": receipt["arm_stats"],
        "hyperion_telemetry": receipt["hyperion_telemetry"],
    }
