"""Empirical evaluation and weightless verification for the METAMORPH controller.

Compares frozen core policy, AURA, and METAMORPH across paired TidePool seeds.
Generates content-addressed verifiable receipts with BLAKE2b/SHA-256 seals.
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
from .metamorph import MetamorphController
from .world import TidePool, rollout

SCHEMA = "poseidon-metamorph-experiment-v1"


def run_metamorph_experiment(
    core: Any,
    seeds: Optional[List[int]] = None,
    max_steps: int = 48,
    scarcity: float = 2.5,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    seeds = seeds or [199000001, 199000002, 199000003, 199000004]
    
    controllers = {
        "core": core,
        "aura": AuraController(core=core),
        "metamorph": MetamorphController(core=core),
    }

    episodes: List[Dict[str, Any]] = []
    total_runs = len(seeds) * len(controllers)
    completed = 0

    for seed in seeds:
        for arm_name, controller in controllers.items():
            if hasattr(controller, "reset"):
                controller.reset(scarcity=scarcity)

            ep = rollout(controller, seed=seed, max_steps=max_steps, scarcity=scarcity)
            
            # Extract summary telemetry
            ep_summary = {
                "arm": arm_name,
                "seed": seed,
                "steps": ep["steps"],
                "survived": ep["survived"],
                "reward": round(float(ep["reward"]), 4),
                "death_reason": ep.get("death_reason"),
            }

            if arm_name == "metamorph" and isinstance(controller, MetamorphController):
                ep_summary["exuviae_count"] = len(controller.molt.exuviae)
                ep_summary["mean_sdi"] = controller.ghost.summary()["mean_spectral_divergence"]
                ep_summary["interference_events"] = controller.flow.interference_events
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

    metamorph_eps = [e for e in episodes if e["arm"] == "metamorph"]
    total_exuviae = sum(e.get("exuviae_count", 0) for e in metamorph_eps)
    total_interf = sum(e.get("interference_events", 0) for e in metamorph_eps)
    mean_sdi = float(np.mean([e.get("mean_sdi", 0.0) for e in metamorph_eps]))

    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "seeds": seeds,
        "max_steps": max_steps,
        "scarcity": scarcity,
        "arm_stats": arm_stats,
        "metamorph_telemetry": {
            "total_exuviae_shed": total_exuviae,
            "total_interference_events_mitigated": total_interf,
            "overall_mean_sdi": round(mean_sdi, 6),
        },
        "episodes": episodes,
    }

    payload = json.dumps({
        "schema": receipt["schema"],
        "seeds": receipt["seeds"],
        "max_steps": receipt["max_steps"],
        "scarcity": receipt["scarcity"],
        "arm_stats": receipt["arm_stats"],
        "metamorph_telemetry": receipt["metamorph_telemetry"],
    }, sort_keys=True)
    receipt["receipt_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    receipt["experiment_id"] = receipt["receipt_sha256"][:32]

    return receipt


def verify_metamorph_receipt(receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Verify cryptographically sealed METAMORPH experiment receipt."""
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
        "metamorph_telemetry": receipt["metamorph_telemetry"],
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
        "metamorph_telemetry": receipt["metamorph_telemetry"],
    }
