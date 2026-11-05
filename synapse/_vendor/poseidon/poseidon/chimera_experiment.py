"""Empirical evaluation and weightless verification for the CHIMERA super-controller.

Compares frozen core policy, AURA, METAMORPH, and CHIMERA across paired TidePool seeds.
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
from .metamorph import MetamorphController
from .world import TidePool, rollout

SCHEMA = "poseidon-chimera-experiment-v1"


def run_chimera_experiment(
    core: Any,
    seeds: Optional[List[int]] = None,
    max_steps: int = 48,
    scarcity: float = 2.5,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    seeds = seeds or [202000001, 202000002, 202000003, 202000004]

    controllers = {
        "core": core,
        "aura": AuraController(core=core),
        "metamorph": MetamorphController(core=core),
        "chimera": ChimeraController(core=core),
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

            if arm_name == "chimera" and isinstance(controller, ChimeraController):
                ep_summary["quantum_entropy"] = controller.causeway.von_neumann_entropy()
                ep_summary["lattice_coherence"] = controller.lattice.multi_scale_coherence()
                ep_summary["holographic_items"] = controller.hologram.item_count
                ep_summary["mean_sdi"] = controller.ghost.summary()["mean_spectral_divergence"]
                ep_summary["destructive_conflicts"] = len(controller.causeway.detect_destructive_conflicts())
                ep_summary["final_stage"] = controller.molt.stage.value

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

    chimera_eps = [e for e in episodes if e["arm"] == "chimera"]
    mean_qe = float(np.mean([e.get("quantum_entropy", 0.0) for e in chimera_eps]))
    mean_lc = float(np.mean([e.get("lattice_coherence", 0.0) for e in chimera_eps]))
    total_conflicts = sum(e.get("destructive_conflicts", 0) for e in chimera_eps)
    mean_sdi = float(np.mean([e.get("mean_sdi", 0.0) for e in chimera_eps]))
    mean_holo_items = float(np.mean([e.get("holographic_items", 0.0) for e in chimera_eps]))

    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "seeds": seeds,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "arm_stats": arm_stats,
        "chimera_telemetry": {
            "mean_quantum_entropy": round(mean_qe, 6),
            "mean_lattice_coherence": round(mean_lc, 6),
            "total_conflicts_detected": total_conflicts,
            "overall_mean_sdi": round(mean_sdi, 6),
            "mean_holographic_items_stored": round(mean_holo_items, 2),
        },
        "episodes": episodes,
    }

    payload = json.dumps({
        "schema": receipt["schema"],
        "seeds": receipt["seeds"],
        "max_steps": receipt["max_steps"],
        "scarcity": receipt["scarcity"],
        "arm_stats": receipt["arm_stats"],
        "chimera_telemetry": receipt["chimera_telemetry"],
    }, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    receipt["experiment_id"] = receipt["receipt_sha256"][:32]

    return receipt


def verify_chimera_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Verify cryptographically sealed CHIMERA experiment receipt."""
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
        "chimera_telemetry": receipt["chimera_telemetry"],
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
        "chimera_telemetry": receipt["chimera_telemetry"],
    }
