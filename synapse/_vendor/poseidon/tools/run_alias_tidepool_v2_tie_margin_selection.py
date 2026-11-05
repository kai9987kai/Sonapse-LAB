"""Select the cue-policy tie margin on a development-only seed block."""
from __future__ import annotations

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

from poseidon.research.alias_world_v2 import ObservationAliasTidePoolV2
from poseidon.research.contracts import ALIAS_WORLD_V2_VERSION, canonical_bytes


SEED_START = 742000000
SEED_COUNT = 24
MAX_STEPS = 24
MARGINS = tuple(round(index * 0.05, 2) for index in range(9))
CONDITIONS = {
    "id": {"scarcity": 1.0, "sensor_noise_std": 0.0},
    "scarcity": {"scarcity": 2.5, "sensor_noise_std": 0.0},
    "noise": {"scarcity": 1.0, "sensor_noise_std": 0.04},
    "scarcity_noise": {"scarcity": 2.5, "sensor_noise_std": 0.04},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(seed: int, regime: int, condition: str, margin: float) -> dict[str, Any]:
    env = ObservationAliasTidePoolV2(
        seed=seed,
        max_steps=MAX_STEPS,
        forced_regime=regime,
        **CONDITIONS[condition],
    )
    obs = env.reset()
    cue_regime = 0 if obs[13] > 0.5 else 1
    total_reward = 0.0
    transitions = []
    while not env.done:
        tick = env.tick
        energy, hydration = float(obs[1]), float(obs[2])
        action = 1 if energy <= hydration else 2
        cue_used = tick >= 4 and abs(energy - hydration) <= margin
        if cue_used:
            action = 1 if cue_regime == 0 else 2
        next_obs, reward, _, info = env.step(action)
        transitions.append({
            "tick": tick,
            "action": int(action),
            "cue_used": bool(cue_used),
            "reward": float(reward),
            "energy": float(info["energy"]),
            "hydration": float(info["hydration"]),
            "death": bool(info["death"]),
        })
        total_reward += float(reward)
        obs = next_obs
    terminal = transitions[-1]
    return {
        "seed": int(seed),
        "regime": int(regime),
        "condition": condition,
        "margin": float(margin),
        "steps": len(transitions),
        "survived": not terminal["death"],
        "total_reward": float(total_reward),
        "minimum_energy": float(min(row["energy"] for row in transitions)),
        "minimum_hydration": float(min(row["hydration"] for row in transitions)),
        "cue_influence_steps": int(sum(row["cue_used"] for row in transitions)),
        "trajectory_sha256": hashlib.sha256(canonical_bytes(transitions)).hexdigest(),
    }


def run() -> dict[str, Any]:
    episodes = []
    for margin in MARGINS:
        for condition in CONDITIONS:
            for seed in range(SEED_START, SEED_START + SEED_COUNT):
                for regime in (0, 1):
                    episodes.append(_episode(seed, regime, condition, margin))

    replay_checks = 0
    for row in episodes:
        if _episode(row["seed"], row["regime"], row["condition"], row["margin"]) != row:
            raise RuntimeError(
                f"exact replay failed: margin={row['margin']} condition={row['condition']} "
                f"seed={row['seed']} regime={row['regime']}"
            )
        replay_checks += 1

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["margin"], row["condition"])].append(row)
    by_margin = {}
    for margin in MARGINS:
        by_condition = {}
        for condition in CONDITIONS:
            rows = grouped[(margin, condition)]
            by_condition[condition] = {
                "episodes": len(rows),
                "survived": int(sum(row["survived"] for row in rows)),
                "survival_rate": float(np.mean([row["survived"] for row in rows])),
                "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
                "minimum_energy": float(min(row["minimum_energy"] for row in rows)),
                "minimum_hydration": float(min(row["minimum_hydration"] for row in rows)),
                "mean_cue_influence_steps": float(np.mean([row["cue_influence_steps"] for row in rows])),
            }
        viable = all(by_condition[condition]["survival_rate"] == 1.0 for condition in CONDITIONS)
        by_margin[str(margin)] = {
            "by_condition": by_condition,
            "all_conditions_viable": viable,
            "unweighted_mean_reward_across_conditions": float(
                np.mean([by_condition[condition]["mean_reward"] for condition in CONDITIONS])
            ),
        }
    eligible = [
        margin for margin in MARGINS
        if by_margin[str(margin)]["all_conditions_viable"]
    ]
    selected = max(eligible, key=lambda margin: by_margin[str(margin)]["unweighted_mean_reward_across_conditions"])

    source_paths = {
        "selection_runner": Path(__file__).resolve(),
        "alias_world_v1": ROOT / "poseidon/research/alias_world.py",
        "alias_world_v2": ROOT / "poseidon/research/alias_world_v2.py",
        "contracts": ROOT / "poseidon/research/contracts.py",
        "world": ROOT / "poseidon/world.py",
    }
    return {
        "schema": "poseidon-alias-world-v2-tie-margin-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "world_version": ALIAS_WORLD_V2_VERSION,
            "development_seed_range": [SEED_START, SEED_START + SEED_COUNT - 1],
            "regimes": [0, 1],
            "conditions": CONDITIONS,
            "margins": list(MARGINS),
            "max_steps": MAX_STEPS,
            "cue_horizon": 4,
            "selection_rule": "among margins with 100% survival in every condition, maximize the unweighted mean reward over the four conditions",
            "evaluation_seed_overlap": False,
            "promotion": False,
        },
        "results": {
            "by_margin": by_margin,
            "selected_margin": float(selected),
            "selected_margin_reason": "highest development-block mean reward among margins passing the all-condition survival gate",
            "complete_episode_replay_checks": replay_checks,
            "replay_failures": 0,
        },
        "episodes": episodes,
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "limits": [
            "Development-only heuristic selection; it is not a confirmatory result.",
            "Only nine predeclared margins from 0.00 through 0.40 were compared.",
            "The threshold does not confer safety or transfer guarantees.",
        ],
    }


def main() -> None:
    output = ROOT / "runs/research/alias_tidepool_v2_tie_margin_selection_20261014.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")
    receipt = run()
    receipt["receipt_sha256"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "receipt_sha256": receipt["receipt_sha256"],
        "selected_margin": receipt["results"]["selected_margin"],
        "by_margin": {
            margin: {
                "all_conditions_viable": row["all_conditions_viable"],
                "mean_reward": row["unweighted_mean_reward_across_conditions"],
            }
            for margin, row in receipt["results"]["by_margin"].items()
        },
        "replay_checks": receipt["results"]["complete_episode_replay_checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
