"""Check whether the versioned observation-alias world admits viable policies."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.core import CoreRuntime, active_core_path
from poseidon.research.alias_world_v2 import ObservationAliasTidePoolV2
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes, alias_tidepool_v2_contract


SEED_START = 742000000
SEED_COUNT = 24
MAX_STEPS = 24
CONDITIONS = {
    "id": {"scarcity": 1.0, "sensor_noise_std": 0.0},
    "scarcity": {"scarcity": 2.5, "sensor_noise_std": 0.0},
    "noise": {"scarcity": 1.0, "sensor_noise_std": 0.04},
    "scarcity_noise": {"scarcity": 2.5, "sensor_noise_std": 0.04},
}
ARMS = (
    "frozen_tidal",
    "rest_only",
    "visible_balancer",
    "cue_memory_balancer",
    "guarded_cue_memory_balancer",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choose_action(
    arm: str,
    obs: list[float],
    tick: int,
    remembered_regime: int | None,
    core: CoreRuntime,
) -> int:
    if arm == "frozen_tidal":
        return int(core.act(obs))
    if arm == "rest_only":
        return 0
    if arm == "visible_balancer":
        return 1 if obs[1] <= obs[2] else 2
    if arm in {"cue_memory_balancer", "guarded_cue_memory_balancer"}:
        # The cue is read once from observations, then retained as an explicit
        # one-bit memory. Low resource always takes precedence over reward.
        energy, hydration = float(obs[1]), float(obs[2])
        reserve_floor = 0.30 if arm == "cue_memory_balancer" else 0.50
        if energy <= reserve_floor:
            return 1
        if hydration <= reserve_floor:
            return 2
        if remembered_regime is None:
            return 1 if obs[13] > 0.5 else 2
        return 1 if remembered_regime == 0 else 2
    raise ValueError(f"unknown arm: {arm}")


def _run_episode(
    seed: int,
    regime: int,
    condition: str,
    arm: str,
    core: CoreRuntime,
) -> dict[str, Any]:
    condition_config = CONDITIONS[condition]
    env = ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **condition_config,
    )
    obs = env.reset()
    remembered_regime: int | None = None
    trajectory = []
    total_reward = 0.0

    for tick in range(MAX_STEPS):
        if tick < env.cue_horizon and remembered_regime is None:
            remembered_regime = 0 if obs[13] > 0.5 else 1
        action = _choose_action(arm, obs, tick, remembered_regime, core)
        next_obs, reward, done, info = env.step(action)
        trajectory.append({
            "tick": tick,
            "observation": [float(x) for x in obs],
            "action": int(action),
            "reward": float(reward),
            "next_observation": [float(x) for x in next_obs],
            "info": info,
        })
        total_reward += float(reward)
        obs = next_obs
        if done:
            break

    terminal = trajectory[-1]["info"]
    return {
        "seed": int(seed),
        "regime": int(regime),
        "condition": condition,
        "arm": arm,
        "steps": len(trajectory),
        "survived": not bool(terminal.get("death", False)),
        "total_reward": float(total_reward),
        "terminal_energy": float(env.energy),
        "terminal_hydration": float(env.hydration),
        "minimum_energy": float(min(row["info"]["energy"] for row in trajectory)),
        "minimum_hydration": float(min(row["info"]["hydration"] for row in trajectory)),
        "trajectory": trajectory,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "survived": int(sum(row["survived"] for row in rows)),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
        "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
    }


def run(seed_start: int = SEED_START, seed_count: int = SEED_COUNT) -> dict[str, Any]:
    if seed_start < 0 or seed_count < 2:
        raise ValueError("seed_start must be nonnegative and seed_count at least two")
    contract = alias_tidepool_v2_contract()
    contract.validate()
    core_path = active_core_path().resolve()
    core = CoreRuntime(core_path)

    episodes = []
    for condition in CONDITIONS:
        for seed in range(seed_start, seed_start + seed_count):
            for regime in (0, 1):
                for arm in ARMS:
                    episodes.append(_run_episode(seed, regime, condition, arm, core))

    replay_checks = 0
    for row in episodes:
        replay = _run_episode(row["seed"], row["regime"], row["condition"], row["arm"], core)
        if replay != row:
            raise RuntimeError(
                f"exact replay failed: seed={row['seed']} regime={row['regime']} "
                f"condition={row['condition']} arm={row['arm']}"
            )
        replay_checks += 1

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["condition"], row["arm"])].append(row)
    summaries = {
        condition: {arm: _summary(grouped[(condition, arm)]) for arm in ARMS}
        for condition in CONDITIONS
    }
    viability = {
        "world_version": ALIAS_WORLD_V2_VERSION,
        "balanced_visible_policy_survival_all_conditions": all(
            summaries[condition]["visible_balancer"]["survival_rate"] == 1.0
            for condition in CONDITIONS
        ),
        "cue_memory_policy_survival_all_conditions": all(
            summaries[condition]["cue_memory_balancer"]["survival_rate"] == 1.0
            for condition in CONDITIONS
        ),
        "guarded_cue_memory_policy_survival_all_conditions": all(
            summaries[condition]["guarded_cue_memory_balancer"]["survival_rate"] == 1.0
            for condition in CONDITIONS
        ),
        "rest_only_control_fails_in_all_conditions": all(
            summaries[condition]["rest_only"]["survival_rate"] == 0.0
            for condition in CONDITIONS
        ),
    }
    source_paths = {
        "assay": Path(__file__).resolve(),
        "alias_world_v1": ROOT / "poseidon/research/alias_world.py",
        "alias_world_v2": ROOT / "poseidon/research/alias_world_v2.py",
        "contracts": ROOT / "poseidon/research/contracts.py",
        "world": ROOT / "poseidon/world.py",
        "core_runtime": ROOT / "poseidon/core.py",
    }
    return {
        "schema": "poseidon-alias-world-v2-viability-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment_seeds": list(range(seed_start, seed_start + seed_count)),
            "regimes": [0, 1],
            "conditions": CONDITIONS,
            "arms": list(ARMS),
            "max_steps": MAX_STEPS,
            "maintenance_yield": ObservationAliasTidePoolV2.MAINTENANCE_YIELD,
        "cue_memory_reserve_floors": {
            "unprotected": 0.30,
            "guarded": 0.50,
            "selection_note": "the 0.50 guarded floor was chosen after inspecting only the development pilot failures",
        },
            "same_seed_pairing": "both hidden regimes use each seed; visible pre-cue action and exogenous noise align by event stream",
            "promotion": False,
        },
        "results": {
            "by_condition": summaries,
            "viability_gate": viability,
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
        },
        "episodes": episodes,
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "core_checkpoint_path": str(core_path),
            "core_checkpoint_sha256": _sha256(core_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "limits": [
            "Synthetic environment repair and policy-viability check only; no DV-GEM benefit claim.",
            "The cue-memory policy is a transparent rule baseline, not a trained controller.",
            "Twenty-four seeds are a development audit, not a powered final evaluation.",
            "The rest-only control is intentionally nonviable and only checks failure detection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/research/alias_tidepool_v2_viability_audit_guarded_20261014.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    receipt = run(args.seed_start, args.seed_count)
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": receipt["results"],
        "identity": receipt["identity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
