"""Paired closed-loop control assay for the opt-in NEXUS residual adapter."""
from __future__ import annotations

import argparse
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


SEEDS = tuple(range(74401, 74465))
HORIZON = 32
AGENT_SEED = 44001
BOOTSTRAP_REPLICATES = 20_000
MIN_REWARD_GAIN = 0.05
MAX_SURVIVAL_LOSS = 0.02
ADAPTER_RECEIPT = ROOT / "runs/nexus_omega/contextual_residual_assay.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        means[index] = np.mean(values[rng.randint(0, len(values), size=len(values))])
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _sign_test(values: np.ndarray) -> dict:
    nonzero = [float(value) for value in values if abs(float(value)) > 1e-12]
    n = len(nonzero)
    if not n:
        return {"n_nonzero": 0, "positive": 0, "p_value": 1.0}
    positive = sum(value > 0 for value in nonzero)
    tail = min(positive, n - positive)
    p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n))
    return {"n_nonzero": n, "positive": positive, "p_value": p_value}


def _episode(seed: int, *, use_adapter: bool) -> dict:
    env = POMDPPhaseWorld(seed=seed, max_steps=HORIZON)
    obs = env.reset()
    agent = NexusOmegaAgent(base_seed=AGENT_SEED, residual_adapter_path=str(ADAPTER_RECEIPT) if use_adapter else None)
    agent.reset()
    total_reward = 0.0
    prediction_errors = []
    actions = []
    survived = True
    for _ in range(HORIZON):
        action = int(agent.act(obs))
        actions.append(action)
        next_obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        error = agent.observe_transition(next_obs, reward, done)
        prediction_errors.append(float(np.mean(np.abs(error))))
        obs = next_obs
        if done:
            survived = not bool(info.get("death", False))
            break
    switches = sum(left != right for left, right in zip(actions, actions[1:]))
    return {
        "seed": seed,
        "total_reward": total_reward,
        "survived": survived,
        "steps": len(actions),
        "mean_online_prediction_mae": float(np.mean(prediction_errors)) if prediction_errors else 0.0,
        "action_switches": switches,
        "action_switch_rate": switches / max(1, len(actions) - 1),
        "actions": actions,
    }


def run() -> dict:
    paired = []
    for seed in SEEDS:
        baseline = _episode(seed, use_adapter=False)
        candidate = _episode(seed, use_adapter=True)
        paired.append({
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "reward_delta": candidate["total_reward"] - baseline["total_reward"],
            "survival_delta": int(candidate["survived"]) - int(baseline["survived"]),
            "paired_action_paths_identical": baseline["actions"] == candidate["actions"],
        })

    reward_delta = np.asarray([row["reward_delta"] for row in paired], dtype=np.float64)
    survival_delta = np.asarray([row["survival_delta"] for row in paired], dtype=np.float64)
    reward_ci = _bootstrap_ci(reward_delta, 664401)
    survival_ci = _bootstrap_ci(survival_delta, 664402)
    survival_pairs = {
        "both_survived": sum(row["baseline"]["survived"] and row["candidate"]["survived"] for row in paired),
        "baseline_only_survived": sum(row["baseline"]["survived"] and not row["candidate"]["survived"] for row in paired),
        "candidate_only_survived": sum(not row["baseline"]["survived"] and row["candidate"]["survived"] for row in paired),
        "neither_survived": sum(not row["baseline"]["survived"] and not row["candidate"]["survived"] for row in paired),
    }
    promoted_by_assay = bool(reward_ci[0] > MIN_REWARD_GAIN and survival_ci[0] > -MAX_SURVIVAL_LOSS)
    return {
        "schema": "poseidon-nexus-adapter-control-assay-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "environment": "POMDPPhaseWorld",
            "seeds": list(SEEDS),
            "horizon": HORIZON,
            "paired_control": "same environment seed, same agent base seed, fresh reset per arm; event schedule fixed by the world seed",
            "baseline": "NexusOmegaAgent(base_seed=44001), no contextual residual adapter",
            "candidate": "same agent with receipt-validated contextual residual adapter",
            "primary_outcome": "per-episode total reward delta, candidate minus baseline",
            "admission_rule": f"reward-delta 95% seed-bootstrap lower bound > {MIN_REWARD_GAIN} and survival-delta lower bound > {-MAX_SURVIVAL_LOSS}",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "fresh_seed_evaluation": True,
        },
        "results": {
            "baseline_mean_reward": float(np.mean([row["baseline"]["total_reward"] for row in paired])),
            "candidate_mean_reward": float(np.mean([row["candidate"]["total_reward"] for row in paired])),
            "mean_reward_delta": float(reward_delta.mean()),
            "reward_delta_95pct_ci": reward_ci,
            "reward_delta_exact_sign_test": _sign_test(reward_delta),
            "baseline_survival_rate": float(np.mean([row["baseline"]["survived"] for row in paired])),
            "candidate_survival_rate": float(np.mean([row["candidate"]["survived"] for row in paired])),
            "survival_delta": float(survival_delta.mean()),
            "survival_delta_95pct_ci": survival_ci,
            "survival_pairs": survival_pairs,
            "baseline_mean_online_prediction_mae": float(np.mean([row["baseline"]["mean_online_prediction_mae"] for row in paired])),
            "candidate_mean_online_prediction_mae": float(np.mean([row["candidate"]["mean_online_prediction_mae"] for row in paired])),
            "baseline_mean_action_switch_rate": float(np.mean([row["baseline"]["action_switch_rate"] for row in paired])),
            "candidate_mean_action_switch_rate": float(np.mean([row["candidate"]["action_switch_rate"] for row in paired])),
            "identical_action_paths": sum(row["paired_action_paths_identical"] for row in paired),
            "episodes": paired,
        },
        "decision": {
            "closed_loop_gate_passed": promoted_by_assay,
            "default_promotion": False,
            "reason": "This assay evaluates one fixed synthetic task and one model initialization; default promotion needs broader held-out environments and model seeds." if promoted_by_assay else "Closed-loop outcome gate did not pass; keep adapter opt-in.",
        },
        "identity": {
            "adapter_receipt_sha256": json.loads(ADAPTER_RECEIPT.read_text(encoding="utf-8"))["receipt_sha256"],
            "adapter_receipt_file_sha256": _sha256(ADAPTER_RECEIPT),
            "agent_source_sha256": _sha256(ROOT / "poseidon/nexus_omega/agent.py"),
            "adapter_source_sha256": _sha256(ROOT / "poseidon/nexus_omega/residual_adapter.py"),
            "world_model_source_sha256": _sha256(ROOT / "poseidon/nexus_omega/causal_world_model.py"),
            "benchmark_source_sha256": _sha256(ROOT / "poseidon/nexus_omega/benchmark.py"),
        },
        "limits": [
            "The environment and agent are synthetic research systems.",
            "This compares one fixed agent initialization; repeated seed episodes do not establish generalization across agent weights.",
            "Online prediction errors are on-policy and state distributions can differ between arms; they are descriptive only.",
            "No default promotion follows from this assay, even if its narrow gate passes.",
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/nexus_omega/adapter_control_assay.json")
    args = parser.parse_args()
    receipt = run()
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "receipt_sha256": receipt["receipt_sha256"],
        "results": {key: value for key, value in receipt["results"].items() if key != "episodes"},
        "decision": receipt["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
