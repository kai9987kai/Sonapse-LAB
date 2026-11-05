"""Compare NEXUS-Ω action scores with exact same-state simulator branches.

This is an offline diagnostic. The behavior policy supplies matched anchor states;
the agent's proposed actions are never executed and branch outcomes never enter
the agent's observations.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poseidon.nexus_omega.agent import NexusOmegaAgent
from poseidon.nexus_omega.benchmark import POMDPPhaseWorld


SEEDS = tuple(range(88101, 88165))
AGENT_SEED = 49001
HORIZON = 32
BOOTSTRAP_REPLICATES = 20_000
ADAPTER_RECEIPT = ROOT / "runs/nexus_omega/contextual_residual_assay.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kendall_tau_b(left: list[float], right: list[float]) -> float:
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_left) * (concordant + discordant + ties_right))
    return (concordant - discordant) / denominator if denominator else 0.0


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        means[i] = np.mean(values[rng.randint(0, len(values), size=len(values))])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_test(values: np.ndarray) -> dict:
    nonzero = [float(x) for x in values if abs(float(x)) > 1e-12]
    n = len(nonzero)
    positive = sum(x > 0 for x in nonzero)
    tail = min(positive, n - positive)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)) if n else 1.0
    return {"n_nonzero": n, "positive": positive, "p_value": p}


def _score_actions(agent: NexusOmegaAgent, obs: np.ndarray) -> tuple[int, int, list[dict]]:
    chosen = int(agent.act(obs))
    decision = agent.get_last_decision()
    if decision is None or len(decision.action_scores) != agent.n_actions:
        raise RuntimeError("planner did not return complete per-action score telemetry")
    scores = sorted(decision.action_scores, key=lambda row: int(row["action"]))
    objective = next(int(row["action"]) for row in scores if row["selected_by_objective"])
    return chosen, objective, scores


def _run_seed(seed: int, agent_seed: int) -> dict:
    env = POMDPPhaseWorld(seed=seed, max_steps=HORIZON)
    obs = env.reset()
    agents = {
        "baseline": NexusOmegaAgent(base_seed=agent_seed),
        "adapter": NexusOmegaAgent(
            base_seed=agent_seed,
            residual_adapter_path=str(ADAPTER_RECEIPT),
        ),
    }
    for agent in agents.values():
        agent.reset()

    behavior_rng = np.random.RandomState(seed + 700_000)
    arms = {name: [] for name in agents}
    behavior_actions: list[int] = []
    branch_lockstep_checks = 0

    for tick in range(HORIZON):
        action_scores: dict[str, tuple[int, int, list[dict]]] = {
            name: _score_actions(agent, obs)
            for name, agent in agents.items()
        }

        branch_outcomes = []
        for action in range(env.n_actions):
            branch = copy.deepcopy(env)
            next_obs, reward, done, info = branch.step(action)
            branch_outcomes.append({
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "death": bool(info["death"]),
                "health": float(info["health"]),
                "is_storm": bool(info["is_storm"]),
                "next_observation": [float(x) for x in next_obs],
            })

        behavior_action = int(behavior_rng.randint(env.n_actions))
        behavior_actions.append(behavior_action)
        actual_next, actual_reward, actual_done, actual_info = env.step(behavior_action)
        branch_actual = branch_outcomes[behavior_action]
        if (
            float(actual_reward) != branch_actual["reward"]
            or bool(actual_done) != branch_actual["done"]
            or bool(actual_info["death"]) != branch_actual["death"]
            or not np.array_equal(actual_next, np.asarray(branch_actual["next_observation"], dtype=np.float32))
        ):
            raise RuntimeError(f"same-state branch did not reproduce actual action at seed={seed}, tick={tick}")
        branch_lockstep_checks += 1

        reward_by_action = [row["reward"] for row in branch_outcomes]
        best_reward = max(reward_by_action)
        step_row = {
            "tick": tick,
            "observation": [float(x) for x in obs],
            "behavior_action": behavior_action,
            "branch_outcomes": branch_outcomes,
        }
        for name, agent in agents.items():
            chosen, objective_action, scores = action_scores[name]
            predicted_rewards = [float(row["predicted_reward"]) for row in scores]
            efe_scores = [float(row["efe"]) for row in scores]
            chosen_reward = reward_by_action[chosen]
            obj_reward = reward_by_action[objective_action]
            arms[name].append({
                "chosen_action": chosen,
                "objective_action": objective_action,
                "chosen_immediate_regret": best_reward - chosen_reward,
                "objective_immediate_regret": best_reward - obj_reward,
                "chosen_reward": chosen_reward,
                "realized_best_actions": [i for i, r in enumerate(reward_by_action) if r == best_reward],
                "efe_tau_b": _kendall_tau_b(efe_scores, reward_by_action),
                "predicted_reward_tau_b": _kendall_tau_b(predicted_rewards, reward_by_action),
                "predicted_reward_mae": float(np.mean(np.abs(np.asarray(predicted_rewards) - np.asarray(reward_by_action)))),
                "action_scores": scores,
            })

            # Keep each model's comparator aligned to the behavior action actually
            # executed, so both receive the exact same observed interaction stream.
            agent.last_action = behavior_action
            agent.factored_world_model.predict_factored(
                np.asarray(obs, dtype=np.float32),
                agent.last_mem,
                behavior_action,
            )

        step_row["branch_action_rewards"] = reward_by_action
        step_row["actual_reward"] = float(actual_reward)
        step_row["actual_done"] = bool(actual_done)
        step_row["branch_matches_actual"] = True
        step_row["arms"] = {name: arms[name][-1] for name in agents}
        # Retain the complete per-step record once, rather than duplicating branch
        # snapshots in each arm.
        if tick == 0:
            rows = []
        rows.append(step_row)

        for agent in agents.values():
            agent.observe_transition(actual_next, actual_reward, actual_done)
        obs = actual_next
        if actual_done:
            break

    return {
        "seed": seed,
        "behavior_actions": behavior_actions,
        "behavior_action_counts": [behavior_actions.count(action) for action in range(env.n_actions)],
        "branch_lockstep_checks": branch_lockstep_checks,
        "steps": len(rows),
        "arms": {
            name: {
                "mean_chosen_immediate_regret": float(np.mean([x["chosen_immediate_regret"] for x in data])),
                "mean_objective_immediate_regret": float(np.mean([x["objective_immediate_regret"] for x in data])),
                "mean_efe_tau_b": float(np.mean([x["efe_tau_b"] for x in data])),
                "mean_predicted_reward_tau_b": float(np.mean([x["predicted_reward_tau_b"] for x in data])),
                "mean_predicted_reward_mae": float(np.mean([x["predicted_reward_mae"] for x in data])),
            }
            for name, data in arms.items()
        },
        "rows": rows,
    }


def run(seed_start: int = SEEDS[0], agent_seed: int = AGENT_SEED) -> dict:
    if type(seed_start) is not int or seed_start < 0 or seed_start + 63 > 2**32 - 1:
        raise ValueError("seed_start must allow 64 uint32 environment seeds")
    if type(agent_seed) is not int or agent_seed < 0:
        raise ValueError("agent_seed must be a non-negative integer")
    seeds = tuple(range(seed_start, seed_start + 64))
    episodes = [_run_seed(seed, agent_seed) for seed in seeds]
    baseline = [episode["arms"]["baseline"] for episode in episodes]
    candidate = [episode["arms"]["adapter"] for episode in episodes]

    def paired(field: str) -> np.ndarray:
        return np.asarray([
            episode["arms"]["adapter"][field] - episode["arms"]["baseline"][field]
            for episode in episodes
        ], dtype=np.float64)

    regret_delta = paired("mean_chosen_immediate_regret")
    efe_tau_delta = paired("mean_efe_tau_b")
    reward_tau_delta = paired("mean_predicted_reward_tau_b")
    pred_mae_delta = paired("mean_predicted_reward_mae")
    paired_results = {}
    for label, values in (
        ("chosen_immediate_regret", regret_delta),
        ("efe_tau_b", efe_tau_delta),
        ("predicted_reward_tau_b", reward_tau_delta),
        ("predicted_reward_mae", pred_mae_delta),
    ):
        paired_results[label] = {
            "adapter_minus_baseline_mean": float(values.mean()),
            "seed_bootstrap_95pct_ci": _bootstrap_ci(values, 918_001 + len(paired_results)),
            "exact_sign_test": _sign_test(values),
        }

    source_paths = {
        "assay": Path(__file__).resolve(),
        "agent": ROOT / "poseidon/nexus_omega/agent.py",
        "causal_efe": ROOT / "poseidon/nexus_omega/causal_efe.py",
        "contracts": ROOT / "poseidon/nexus_omega/contracts.py",
        "causal_world_model": ROOT / "poseidon/nexus_omega/causal_world_model.py",
        "benchmark": ROOT / "poseidon/nexus_omega/benchmark.py",
        "adapter": ROOT / "poseidon/nexus_omega/residual_adapter.py",
    }
    return {
        "schema": "poseidon-nexus-decision-rank-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment": "POMDPPhaseWorld",
            "environment_seeds": list(seeds),
            "agent_seed": agent_seed,
            "horizon": HORIZON,
            "anchors": "same episode states and observations for baseline and adapter; independent seeded uniform-random behavior actions; agents observe the same executed behavior transition",
            "counterfactual": "deep-copy the complete environment at each anchor; execute every action from the identical world and RNG state; verify the executed behavior branch exactly matches the actual transition",
            "primary_outcome": "per-seed mean immediate regret of the proposed action against the best of four exact one-step reward branches",
            "secondary_outcomes": ["EFE versus realized-reward Kendall tau-b", "predicted-reward versus realized-reward Kendall tau-b", "predicted-reward MAE"],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "adapter_receipt_sha256": json.loads(ADAPTER_RECEIPT.read_text(encoding="utf-8"))["receipt_sha256"],
            "runtime_only_diagnostic": True,
            "promotion": False,
        },
        "results": {
            "baseline": {key: float(np.mean([row[key] for row in baseline])) for key in baseline[0]},
            "adapter": {key: float(np.mean([row[key] for row in candidate])) for key in candidate[0]},
            "paired": paired_results,
            "branch_lockstep_checks": int(sum(row["branch_lockstep_checks"] for row in episodes)),
            "branch_lockstep_failures": 0,
            "behavior_action_counts": [
                int(sum(row["behavior_action_counts"][action] for row in episodes))
                for action in range(4)
            ],
            "episodes": episodes,
        },
        "limits": [
            "This evaluates immediate reward ranking under a balanced random behavior-state distribution, not multi-step return or closed-loop policy benefit.",
            "The full hidden simulator state is used only by the offline counterfactual evaluator; the agent receives observations and realized behavior transitions only.",
            "The contextual adapter changes predicted states but not the reward head, so the audit distinguishes reward-head ranking from total EFE ranking.",
            "Repeated anchors within a seed are summarized at seed level; this is one agent initialization and one synthetic world.",
            "Equal behavior trajectories make arm comparisons paired at state/action-stream level, but they do not establish performance under either arm's own induced state distribution.",
        ],
        "identity": {
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "adapter_receipt_file_sha256": _sha256(ADAPTER_RECEIPT),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--agent-seed", type=int, default=AGENT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "runs/nexus_omega" / f"decision_rank_audit_{args.seed_start}_{args.agent_seed}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")
    receipt = run(seed_start=args.seed_start, agent_seed=args.agent_seed)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "receipt_sha256": receipt["receipt_sha256"],
        "protocol": {k: v for k, v in receipt["protocol"].items() if k != "environment_seeds"},
        "results": {k: v for k, v in receipt["results"].items() if k != "episodes"},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
