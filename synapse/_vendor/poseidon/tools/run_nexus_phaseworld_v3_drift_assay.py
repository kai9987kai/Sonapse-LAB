"""Measure NEXUS phase tracking after a hidden storm-cycle interval shift."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark_v3 import POMDPPhaseWorldV3


BOOTSTRAP_REPLICATES = 20_000
HORIZON = 128
SHIFT_STEP = 64
ENVIRONMENT_SEEDS = tuple(range(100601, 100665))
AGENT_SEEDS = (56301, 56302)
INITIAL_INTERVAL = 8
INITIAL_DURATION = 6
SHIFTED_INTERVAL = 10
SHIFTED_DURATION = 6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        means[i] = float(np.mean(values[rng.randint(0, len(values), size=len(values))]))
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _action_for_trace(cue: float, memory: dict[str, int], duration: int) -> int:
    if cue > 0.8:
        memory["ticks"] = duration
    if memory["ticks"] > 0:
        memory["ticks"] -= 1
        return 3
    return 0


def _run_episode(
    seed: int,
    arm: str,
    agent_seed: int | None,
    horizon: int,
    shift_step: int,
    initial_interval: int,
    initial_duration: int,
    shifted_interval: int,
    shifted_duration: int,
) -> dict:
    schedule = ((shift_step, shifted_interval, shifted_duration),)
    env = POMDPPhaseWorldV3(
        seed=seed,
        max_steps=horizon,
        storm_interval=initial_interval,
        storm_duration=initial_duration,
        schedule=schedule,
    )
    obs = env.reset()
    memory = {"ticks": 0}
    agent = None
    profile = None
    if arm.startswith("agent_"):
        if agent_seed is None:
            raise ValueError("agent arms require an initialization seed")
        profile = "nexus-phaseworld-v2" if "aligned" in arm else "nexus-phaseworld-v1"
        agent = NexusOmegaAgent(base_seed=agent_seed, dynamics_profile=profile)
        agent.reset()

    observations = [[float(value) for value in obs]]
    actions: list[int] = []
    rewards: list[float] = []
    transitions: list[dict] = []
    for _ in range(horizon):
        step_interval = int(env.storm_interval)
        step_duration = int(env.storm_duration)
        if agent is not None:
            action = int(agent.act(obs))
        elif arm == "trace_memory":
            action = _action_for_trace(float(obs[3]), memory, initial_duration)
        elif arm == "reactive":
            action = 3 if float(obs[3]) > 0.8 else 0
        elif arm == "oracle":
            action = 3 if env.cycle_step >= step_interval - step_duration else 0
        else:
            raise ValueError(f"unknown arm: {arm}")

        next_obs, reward, done, info = env.step(action)
        if agent is not None:
            agent.observe_transition(next_obs, reward, done)
        decision = agent.get_last_decision() if agent is not None else None
        actions.append(action)
        rewards.append(float(reward))
        transitions.append({
            "step": int(info["step"]),
            "is_storm": bool(info["is_storm"]),
            "phase_changed": bool(info.get("phase_changed", False)),
            "phase_change": info.get("phase_change"),
            "storm_interval": step_interval,
            "storm_duration": step_duration,
            "next_storm_interval": int(env.storm_interval),
            "next_storm_duration": int(env.storm_duration),
            "death": bool(info["death"]),
            "truncated": bool(info.get("truncated", done and not info["death"])),
            "done": bool(done),
            "health": float(info["health"]),
            "energy": float(env.energy),
            "hydration": float(env.hydration),
            "health_shock_update": bool(agent.last_health_shock_update) if agent is not None else False,
            "storm_interval_estimate": (
                float(agent.v2_storm_interval_estimate)
                if profile == "nexus-phaseworld-v2" and agent.v2_storm_interval_estimate is not None
                else None
            ),
            "confirmed_storm_interval": (
                float(agent.v2_confirmed_storm_interval)
                if profile == "nexus-phaseworld-v2" and agent.v2_confirmed_storm_interval is not None
                else None
            ),
            "interval_change_candidate": (
                float(agent.v2_interval_change_candidate)
                if profile == "nexus-phaseworld-v2" and agent.v2_interval_change_candidate is not None
                else None
            ),
            "interval_change_streak": (
                int(agent.v2_interval_change_streak)
                if profile == "nexus-phaseworld-v2"
                else None
            ),
            "phase_boundary_latch_active": (
                bool(agent.v2_phase_boundary_latch_active)
                if agent is not None
                else False
            ),
            "storm_duration_estimate": (
                float(agent.v2_storm_duration_estimate)
                if profile == "nexus-phaseworld-v2"
                else None
            ),
            "hazard_guard_active": bool(agent.v2_hazard_guard_active) if agent is not None else False,
            "hazard_guard_override": bool(agent.v2_hazard_guard_override) if agent is not None else False,
            "recovery_admission": bool(getattr(decision, "recovery_admission", False)),
            "action": int(action),
            "version": str(info["version"]),
        })
        observations.append([float(value) for value in next_obs])
        obs = next_obs
        if done:
            break

    terminal = transitions[-1]
    interval_lag = None
    if profile == "nexus-phaseworld-v2":
        for row in transitions:
            if row["step"] > shift_step and row["storm_interval_estimate"] == float(shifted_interval):
                interval_lag = row["step"] - shift_step
                break
    pre = [row for row in transitions if row["step"] <= shift_step]
    post = [row for row in transitions if row["step"] > shift_step]
    return {
        "seed": int(seed),
        "arm": arm,
        "agent_seed": agent_seed,
        "dynamics_profile": profile,
        "schedule": {
            "initial_interval": initial_interval,
            "initial_duration": initial_duration,
            "shift_step": shift_step,
            "shifted_interval": shifted_interval,
            "shifted_duration": shifted_duration,
        },
        "survived": not terminal["death"],
        "death": terminal["death"],
        "truncated": terminal["truncated"],
        "steps": len(actions),
        "total_reward": float(sum(rewards)),
        "pre_shift_reward": float(sum(rewards[:len(pre)])),
        "post_shift_reward": float(sum(rewards[len(pre):])),
        "pre_shift_storm_forages": sum(row["is_storm"] and row["action"] == 0 for row in pre),
        "post_shift_storm_forages": sum(row["is_storm"] and row["action"] == 0 for row in post),
        "interval_adaptation_lag": interval_lag,
        "terminal_health": terminal["health"],
        "terminal_energy": terminal["energy"],
        "terminal_hydration": terminal["hydration"],
        "actions": actions,
        "rewards": rewards,
        "observations": observations,
        "transitions": transitions,
    }


def _summary(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "survival_rate": float(np.mean([row["survived"] for row in rows])),
        "mean_reward": float(np.mean([row["total_reward"] for row in rows])),
        "mean_pre_shift_reward": float(np.mean([row["pre_shift_reward"] for row in rows])),
        "mean_post_shift_reward": float(np.mean([row["post_shift_reward"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_terminal_health": float(np.mean([row["terminal_health"] for row in rows])),
        "mean_terminal_energy": float(np.mean([row["terminal_energy"] for row in rows])),
        "mean_terminal_hydration": float(np.mean([row["terminal_hydration"] for row in rows])),
        "mean_post_shift_storm_forages": float(np.mean([row["post_shift_storm_forages"] for row in rows])),
        "mean_interval_adaptation_lag": (
            float(np.mean([row["interval_adaptation_lag"] for row in rows if row["interval_adaptation_lag"] is not None]))
            if any(row["interval_adaptation_lag"] is not None for row in rows)
            else None
        ),
        "interval_adaptation_coverage": float(np.mean([row["interval_adaptation_lag"] is not None for row in rows])),
    }


def _run_args(
    horizon: int,
    shift_step: int,
    initial_interval: int,
    initial_duration: int,
    shifted_interval: int,
    shifted_duration: int,
) -> dict:
    return {
        "horizon": horizon,
        "shift_step": shift_step,
        "initial_interval": initial_interval,
        "initial_duration": initial_duration,
        "shifted_interval": shifted_interval,
        "shifted_duration": shifted_duration,
    }


def run(
    environment_seeds: tuple[int, ...] = ENVIRONMENT_SEEDS,
    agent_seeds: tuple[int, ...] = AGENT_SEEDS,
    horizon: int = HORIZON,
    shift_step: int = SHIFT_STEP,
    initial_interval: int = INITIAL_INTERVAL,
    initial_duration: int = INITIAL_DURATION,
    shifted_interval: int = SHIFTED_INTERVAL,
    shifted_duration: int = SHIFTED_DURATION,
) -> dict:
    if len(environment_seeds) < 2:
        raise ValueError("at least two environment seeds are required")
    if not 1 <= shift_step < horizon:
        raise ValueError("shift-step must be within the episode horizon")
    if initial_interval < 2 or not 1 <= initial_duration < initial_interval:
        raise ValueError("invalid initial storm schedule")
    if shifted_interval < 2 or not 1 <= shifted_duration < shifted_interval:
        raise ValueError("invalid shifted storm schedule")
    arms = (
        "agent_v1_prior",
        "agent_aligned_prior",
        "trace_memory",
        "reactive",
        "oracle",
    )
    common = _run_args(horizon, shift_step, initial_interval, initial_duration, shifted_interval, shifted_duration)
    episodes: list[dict] = []
    for seed in environment_seeds:
        for agent_seed in agent_seeds:
            episodes.append(_run_episode(seed, "agent_v1_prior", agent_seed, **common))
            episodes.append(_run_episode(seed, "agent_aligned_prior", agent_seed, **common))
        for arm in arms[2:]:
            episodes.append(_run_episode(seed, arm, None, **common))

    replay_verified = 0
    for episode in episodes:
        replay = _run_episode(
            episode["seed"], episode["arm"], episode["agent_seed"], **common
        )
        keys = (
            "survived", "death", "truncated", "steps", "total_reward",
            "pre_shift_reward", "post_shift_reward", "actions", "rewards",
            "observations", "transitions", "interval_adaptation_lag",
        )
        if not all(replay[key] == episode[key] for key in keys):
            raise RuntimeError(f"exact replay failed: seed={episode['seed']} arm={episode['arm']}")
        replay_verified += 1

    by_arm = {arm: [row for row in episodes if row["arm"] == arm] for arm in arms}
    summaries = {arm: _summary(rows) for arm, rows in by_arm.items()}
    paired = {}
    for agent_seed in agent_seeds:
        baseline = [row for row in by_arm["agent_v1_prior"] if row["agent_seed"] == agent_seed]
        candidate = [row for row in by_arm["agent_aligned_prior"] if row["agent_seed"] == agent_seed]
        deltas = np.asarray([b["total_reward"] - a["total_reward"] for a, b in zip(baseline, candidate)])
        paired[str(agent_seed)] = {
            "aligned_minus_v1_reward_mean": float(np.mean(deltas)),
            "aligned_minus_v1_reward_95pct_ci": _bootstrap_ci(deltas, 2_030_001 + agent_seed),
            "v2_only_survival": int(sum(not a["survived"] and b["survived"] for a, b in zip(baseline, candidate))),
            "v1_only_survival": int(sum(a["survived"] and not b["survived"] for a, b in zip(baseline, candidate))),
        }

    source_paths = {
        "assay": Path(__file__).resolve(),
        "agent": ROOT / "poseidon/nexus_omega/agent.py",
        "contracts": ROOT / "poseidon/nexus_omega/contracts.py",
        "memory": ROOT / "poseidon/nexus_omega/memory.py",
        "benchmark_v1": ROOT / "poseidon/nexus_omega/benchmark.py",
        "benchmark_v2": ROOT / "poseidon/nexus_omega/benchmark_v2.py",
        "benchmark_v3": ROOT / "poseidon/nexus_omega/benchmark_v3.py",
        "causal_world_model_v1": ROOT / "poseidon/nexus_omega/causal_world_model.py",
        "causal_world_model_v2": ROOT / "poseidon/nexus_omega/causal_world_model_v2.py",
        "causal_efe": ROOT / "poseidon/nexus_omega/causal_efe.py",
    }
    return {
        "schema": "poseidon-nexus-phaseworld-v3-hidden-schedule-shift-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment_seeds": list(environment_seeds),
            "agent_seeds": list(agent_seeds),
            "horizon": horizon,
            "shift_step": shift_step,
            "initial_schedule": {"interval": initial_interval, "duration": initial_duration},
            "shifted_schedule": {"interval": shifted_interval, "duration": shifted_duration},
            "observation_contract": "same eight channels; cycle and schedule remain hidden except for transient pulse",
            "action_contract": "same four actions; versioned Rest hydration recovery retained",
            "arms": [*arms[:2], *arms[2:]],
            "paired_comparison": "same environment seed and initialization; aligned v2 prior versus legacy v1 prior under the same v3 environment schedule",
            "phase_estimator": "shorter intervals lower the provisional estimate immediately; both directions require two near-matching observations before the confirmed period can tighten the storm-duration window; absent cue near the predicted next pulse latches recovery until a new pulse arrives",
            "primary_metrics": ["survival", "post-shift reward", "storm forage count", "pulse-interval adaptation lag", "terminal vitals"],
            "promotion": False,
        },
        "results": {
            "by_arm": summaries,
            "paired_agent_profile_comparisons": paired,
            "complete_episode_replay_checks": replay_verified,
            "replay_failures": 0,
            "episodes": episodes,
        },
        "limits": [
            "Synthetic one-time interval shift only; duration and resource rules stay fixed.",
            "Two fixed agent initializations do not establish independent model-family replication.",
            "A perfect-information oracle is only a simulator viability check, not a policy baseline.",
            "The aligned v2 controller includes a hand-coded phase guard and is not a learned policy or safety guarantee.",
            "The assay does not promote or activate a runtime controller.",
        ],
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/phaseworld_v3_drift_assay.json")
    parser.add_argument("--seed-start", type=int, default=ENVIRONMENT_SEEDS[0])
    parser.add_argument("--seed-count", type=int, default=len(ENVIRONMENT_SEEDS))
    parser.add_argument("--agent-seeds", type=int, nargs=2, default=AGENT_SEEDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--shift-step", type=int, default=SHIFT_STEP)
    parser.add_argument("--initial-interval", type=int, default=INITIAL_INTERVAL)
    parser.add_argument("--initial-duration", type=int, default=INITIAL_DURATION)
    parser.add_argument("--shifted-interval", type=int, default=SHIFTED_INTERVAL)
    parser.add_argument("--shifted-duration", type=int, default=SHIFTED_DURATION)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {args.output}")
    receipt = run(
        environment_seeds=tuple(range(args.seed_start, args.seed_start + args.seed_count)),
        agent_seeds=tuple(args.agent_seeds),
        horizon=args.horizon,
        shift_step=args.shift_step,
        initial_interval=args.initial_interval,
        initial_duration=args.initial_duration,
        shifted_interval=args.shifted_interval,
        shifted_duration=args.shifted_duration,
    )
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": {key: value for key, value in receipt["results"].items() if key != "episodes"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
